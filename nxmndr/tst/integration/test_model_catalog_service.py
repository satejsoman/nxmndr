# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Catalog models (``torchvision:<name>``, ``torchgeo:<name>``) through the real path.

One server process (``tst.support.rpc_service_process``) serves the module; the tests
speak gRPC to it with the canonical client, and its ``RpcWorkerManager`` builds every
model in the spawned PyTorch RPC worker (``_rpc_load``/``_rpc_infer``). Nothing is
mocked. Weights are random, made and saved by each test; nothing is downloaded.

- torchvision resnet18 (3 bands, 5 classes): LoadModel, OpenSession, StreamPredict
  over 4 chips; each tile is the constant argmax class computed with torch on the
  same chip, also with ``return_confidence`` (unary and streamed).
- torchgeo unet (4 bands, 3 classes): label maps equal the direct per-pixel argmax.
- Capabilities ``pytorch_model_classes`` is the catalog of the server's environment.
- A weights file that does not match fails the load with the differing entries;
  unknown and unserved names fail with INVALID_ARGUMENT; an output that fits no
  family is INVALID_ARGUMENT (unary) or a tile error ``inference_failed``.
- Every served name (``catalog.list_model_classes()``) loads without metadata from
  weights built with 5 classes (and 5 bands where the model takes a band count),
  runs one chip, and returns the result family and values of the direct computation.
"""

from __future__ import annotations

import gc
import importlib.util
import inspect

import grpc
import numpy as np
import pytest
import torch
import torchvision

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from nxmndr.models import catalog
from nxmndr.server import dispatch
from nxmndr.tensor_bundle import unpack_tensor_bundle
from tst.integration.test_rpc_pytorch_service import ServiceProcess

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not torch.distributed.is_available(), reason="torch.distributed not available"
    ),
]

HAVE_TORCHGEO = importlib.util.find_spec("torchgeo") is not None
needs_torchgeo = pytest.mark.skipif(
    not HAVE_TORCHGEO, reason="torchgeo is not installed (the nxmndr[geo] extra)"
)
CATALOG_KEYS = ("model_family", "model_class", "num_classes", "in_channels")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    proc = ServiceProcess(tmp_path_factory.mktemp("catalog-server"), "--capacity", "2")
    yield proc
    try:
        stopped = proc.stop()
        assert stopped["children"] == [], stopped
    finally:
        proc.close()


def _spec(path, model_class, task=inference_pb2.TASK_TYPE_UNSPECIFIED, **metadata):
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.PYTORCH,
        source=str(path),
        model_class=model_class,
        task=task,
        name=model_class,
    )
    for key, value in metadata.items():
        spec.metadata.add(key=key, value=str(value))
    return spec


def _array(output, shape, dtype):
    return np.frombuffer(output, dtype=np.dtype(dtype)).reshape(tuple(int(d) for d in shape))


def _unload(client, model_id):
    reply = client._get_stub().UnloadModel(
        inference_pb2.UnloadModelRequest(model_id=model_id), timeout=60
    )
    assert reply.success, reply.message


def _resident(client):
    listed = client._get_stub().ListModels(inference_pb2.ListModelsRequest(), timeout=30)
    return sorted(m.model_id for m in listed.models)


def _stream(client, session_id, chips, prefix, options=None):
    out = {}
    ids = [f"{prefix}{i}" for i in range(len(chips))]
    for resp in client.stream_predict(
        session_id=session_id, samples=chips, tile_ids=ids, options=options
    ):
        meta = dict(resp.metadata)
        assert "error" not in meta, meta
        out[meta["tile_id"]] = (resp, meta)
    assert sorted(out) == sorted(ids)
    return [out[i] for i in ids]


def _direct(model, chip):
    """The model's output on one [H, W, C] chip, computed here with torch."""

    with torch.no_grad():
        out = model(torch.from_numpy(chip).permute(2, 0, 1)[None])
    return (out["out"] if isinstance(out, dict) else out).numpy()


def _softmax(logits):
    exp = np.exp(logits.astype(np.float64) - logits.max())
    return exp / exp.sum()


# ------------------------------------------------------------ (a), (e): resnet18


@pytest.fixture(scope="module")
def resnet18(tmp_path_factory):
    torch.manual_seed(2)
    model = torchvision.models.resnet18(weights=None, num_classes=5).eval()
    path = tmp_path_factory.mktemp("resnet18") / "resnet18.pt"
    torch.save(model.state_dict(), str(path))
    rng = np.random.default_rng(2)
    chips = [(rng.random((64, 64, 3)) * 255 * (k + 1) / 4).astype(np.float32) for k in range(4)]
    logits = [_direct(model, chip)[0] for chip in chips]
    return path, chips, logits


def test_torchvision_resnet18_streams_constant_argmax_tiles(server, resnet18):
    path, chips, logits = resnet18
    expected = [int(np.argmax(v)) for v in logits]
    assert len(set(expected)) > 1, expected  # the chips give different classes
    with server.client() as client:
        loaded = client.load_model_result(
            "",
            _spec(
                path,
                "torchvision:resnet18",
                inference_pb2.CLASSIFICATION,
                in_channels=3,
                num_classes=5,
            ),
        )
        assert {k: loaded.effective_metadata.get(k) for k in CATALOG_KEYS} == {
            "model_family": "torchvision",
            "model_class": "torchvision:resnet18",
            "num_classes": "5",
            "in_channels": "3",
        }
        opened = client.open_session(
            session_id="s-resnet18", spec=inference_pb2.ModelSpec(model_id=loaded.model_id)
        )
        assert opened.status == "ok", opened.error
        tiles = _stream(client, "s-resnet18", chips, "t")
        for (resp, meta), label in zip(tiles, expected):
            assert meta["result_type"] == "segmentation_mask", meta
            assert meta["output_family"] == dispatch.OUTPUT_CLASS_VECTOR
            tile = _array(resp.output, resp.shape, resp.dtype)
            assert tile.dtype == np.uint16 and tile.shape == (64, 64)
            assert (tile == label).all(), (np.unique(tile), label)
        assert client.close_session("s-resnet18").status == "closed"
        _unload(client, loaded.model_id)


def test_a_class_vector_with_return_confidence(server, resnet18):
    path, chips, logits = resnet18
    with server.client() as client:
        model_id = client.load_model_result("", _spec(path, "torchvision:resnet18")).model_id
        result = client.predict(model_id, chips[1], options={"return_confidence": "true"})
        assert (
            result.metadata["payload_format"] == "npz"
            and result.metadata["has_confidence"] == "true"
        )
        bundle = unpack_tensor_bundle(result.output)
        label = int(np.argmax(logits[1]))
        assert (bundle["mask"] == label).all() and bundle["mask"].shape == (64, 64)
        np.testing.assert_allclose(bundle["confidence"], _softmax(logits[1])[label], rtol=1e-4)

        assert (
            client.open_session(
                session_id="s-conf",
                spec=inference_pb2.ModelSpec(model_id=model_id),
                options={"return_confidence": "true"},
            ).status
            == "ok"
        )
        for chip, value, (resp, meta) in zip(chips, logits, _stream(client, "s-conf", chips, "c")):
            bundle = unpack_tensor_bundle(resp.output)
            label = int(np.argmax(value))
            assert (bundle["mask"] == label).all()
            np.testing.assert_allclose(bundle["confidence"], _softmax(value)[label], rtol=1e-4)
            assert float(meta["confidence_min"]) == pytest.approx(_softmax(value)[label], abs=1e-4)
        assert client.close_session("s-conf").status == "closed"
        _unload(client, model_id)


# ------------------------------------------------------------------- (b): unet


@needs_torchgeo
def test_torchgeo_unet_streams_label_maps(server, tmp_path):
    import torchgeo.models

    torch.manual_seed(0)
    model = torchgeo.models.get_model("unet", weights=None, classes=3, in_channels=4).eval()
    path = tmp_path / "unet.pt"
    torch.save(model.state_dict(), str(path))
    rng = np.random.default_rng(0)
    chips = [rng.random((64, 64, 4)).astype(np.float32) * 100 for _ in range(3)]
    expected = [np.argmax(_direct(model, chip)[0], axis=0).astype(np.uint16) for chip in chips]
    assert all(len(np.unique(e)) > 1 for e in expected)
    with server.client() as client:
        spec = _spec(
            path, "torchgeo:unet", inference_pb2.SEGMENTATION, in_channels=4, num_classes=3
        )
        opened = client.open_session(session_id="s-unet", spec=spec)
        assert opened.status == "ok", opened.error
        for (resp, meta), labels in zip(_stream(client, "s-unet", chips, "u"), expected):
            assert meta["output_family"] == dispatch.OUTPUT_LABEL_MAP
            assert meta["result_type"] == "segmentation_mask"
            np.testing.assert_array_equal(_array(resp.output, resp.shape, resp.dtype), labels)
        assert client.close_session("s-unet").status == "closed"


# ------------------------------------------------------------ (c): Capabilities


def test_capabilities_list_the_catalog_of_the_server_environment(server):
    with server.client() as client:
        advertised = client.capabilities()["pytorch_model_classes"].split(",")
    assert advertised == list(catalog.list_model_classes())
    assert "torchvision:resnet18" in advertised
    assert any(n.startswith("torchgeo:") for n in advertised) == HAVE_TORCHGEO


# ------------------------------------------------------------ (d): load errors


def test_a_mismatched_weights_file_fails_the_load_with_the_differing_entries(server, resnet18):
    path = resnet18[0]
    with server.client() as client:
        resident = _resident(client)
        with pytest.raises(InferenceGrpcError) as wrong_classes:
            client.load_model_result("", _spec(path, "torchvision:resnet18", num_classes=7))
        message = str(wrong_classes.value)
        assert "does not match torchvision:resnet18(num_classes=7) (strict load)" in message
        assert "shape mismatch fc.weight file [5, 512] model [7, 512]" in message

        with pytest.raises(InferenceGrpcError) as other_class:
            client.load_model_result("", _spec(path, "torchvision:resnet34"))
        assert "does not match torchvision:resnet34(num_classes=5)" in str(other_class.value)
        assert "missing layer1.2.conv1.weight" in str(other_class.value)

        for model_class, text in (
            ("resnet18", "is not a catalog name"),
            ("torchvision:raft_large", "is not served"),
            ("torchvision:resnet18", "in_channels=4 cannot be applied"),
        ):
            metadata = {"in_channels": 4} if model_class == "torchvision:resnet18" else {}
            with pytest.raises(InferenceGrpcError) as refused:
                client.load_model_result("", _spec(path, model_class, **metadata))
            assert refused.value.code == grpc.StatusCode.INVALID_ARGUMENT, refused.value
            assert text in str(refused.value)
        # None of these loads left a model resident.
        assert _resident(client) == resident


def test_an_output_that_fits_no_family_is_rejected_with_its_shape(server, tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "one_class.pt"
    torch.save(
        torchvision.models.squeezenet1_1(weights=None, num_classes=1).state_dict(), str(path)
    )
    chip = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    with server.client() as client:
        model_id = client.load_model_result("", _spec(path, "torchvision:squeezenet1_1")).model_id
        with pytest.raises(InferenceGrpcError) as unary:
            client.predict(model_id, chip)
        assert unary.value.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "returned shape [1, 1]" in str(unary.value) and "no argmax" in str(unary.value)
        [resp] = list(client.stream_predict(model_id=model_id, samples=[chip], tile_ids=["x"]))
        meta = dict(resp.metadata)
        assert (
            meta["error_code"] == dispatch.ERROR_INFERENCE_FAILED and meta["error_scope"] == "tile"
        )
        assert "returned shape [1, 1]" in meta["error"]
        _unload(client, model_id)


# ---------------------------------------------------------- every served name

# Chip sizes of models with a fixed input size: EarthLoc's default image_size is 320;
# timm's DINOv2 ViTs are built for 518 (both from their builders' defaults). The rest
# run on 224, the size torchvision and timm build their classifiers for.
PROOF_CHIP = {
    "torchgeo:earthloc": 320,
    "torchgeo:vit_base_patch14_dinov2": 518,
    "torchgeo:vit_small_patch14_dinov2": 518,
}
PROOF_CLASSES = 5  # the family defaults are 1000, 21, 1 or 0 classes
PROOF_BANDS = 5  # the family defaults are 3 or 4 bands


def _reference(model_class):
    """The family's model with PROOF_CLASSES classes (and PROOF_BANDS bands where it
    takes a band count), random weights, and the chip size and band count to run."""

    family, name = model_class.split(":", 1)
    if family == "torchvision":
        builder = torchvision.models.get_model_builder(name)
        kwargs = {"weights": None, "num_classes": PROOF_CLASSES}
        if "weights_backbone" in inspect.signature(builder).parameters:
            kwargs["weights_backbone"] = None
        model, bands = torchvision.models.get_model(name, **kwargs), 3
    else:
        import torchgeo.models

        entry = catalog.TORCHGEO_SERVED[name]
        kwargs = {"weights": None}
        if entry.class_arg:
            kwargs[entry.class_arg] = PROOF_CLASSES
        bands = entry.in_channels
        if entry.channel_arg:
            kwargs[entry.channel_arg] = bands = PROOF_BANDS
        model = torchgeo.models.get_model(name, **kwargs)
    return model.eval(), PROOF_CHIP.get(model_class, 224), bands


def _close(got, want):
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * scale)


@pytest.mark.parametrize("model_class", catalog.list_model_classes())
def test_every_served_model_class_loads_and_runs(server, tmp_path, model_class):
    torch.manual_seed(0)
    model, size, bands = _reference(model_class)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), str(path))
    chip = np.random.default_rng(0).random((size, size, bands)).astype(np.float32)
    want = _direct(model, chip)
    del model
    gc.collect()
    family_name, name = model_class.split(":", 1)
    has_head = family_name == "torchvision" or catalog.TORCHGEO_SERVED[name].class_arg is not None
    if not has_head:
        family = dispatch.OUTPUT_EMBEDDINGS
    else:
        family = dispatch.OUTPUT_CLASS_VECTOR if want.ndim == 2 else dispatch.OUTPUT_LABEL_MAP

    with server.client() as client:
        loaded = client.load_model_result("", _spec(path, model_class))  # no metadata: inferred
        assert {k: loaded.effective_metadata.get(k) for k in CATALOG_KEYS} == {
            "model_family": family_name,
            "model_class": model_class,
            "num_classes": str(PROOF_CLASSES) if has_head else "",
            "in_channels": str(bands),
        }
        try:
            result = client.predict(loaded.model_id, chip, options={"return_embeddings": "true"})
        finally:
            _unload(client, loaded.model_id)
    meta = result.metadata
    assert meta["output_family"] == family, meta
    bundle = unpack_tensor_bundle(result.output)
    got = bundle["embeddings"]
    _close(got, want)
    if family == dispatch.OUTPUT_CLASS_VECTOR:
        assert (bundle["mask"] == int(np.argmax(got[0]))).all() and bundle["mask"].shape == (
            size,
            size,
        )
    elif family == dispatch.OUTPUT_LABEL_MAP:
        np.testing.assert_array_equal(bundle["mask"], np.argmax(got[0], axis=0))
    else:
        assert meta["result_type"] == "embeddings" and "mask" not in bundle
