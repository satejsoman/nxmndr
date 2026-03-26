# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import grpc

from nxmndr.inference import inference_pb2
from nxmndr.inference.image_utils import unpack_tensor_bundle
from nxmndr.server import server


class _DummyContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _mk_record(model):
    class _Record:
        def __init__(self):
            self.model = model
            self.backend = "onnx"
            self.metadata = {"task_name": "segmentation"}

    return _Record()


def _mk_manager(model):
    """Create a mock model manager with a single model record."""
    record = _mk_record(model)

    class _Manager:
        def get(self, model_id):
            return record

        def get_model_for_device(self, model_id, device_id):
            return record.model

    return _Manager()


def test_stream_predict_two_samples(monkeypatch):
    svc = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            batch, _, h, w = arr.shape
            logits = np.zeros((batch, 2, h, w), dtype=np.float32)
            logits[:, 1, ...] = 1.0
            return logits

    svc.model_manager = _mk_manager(_Model())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    arr1 = np.zeros((1, 3, 2, 2), dtype=np.float32)
    arr2 = np.ones((1, 3, 2, 2), dtype=np.float32)

    reqs = [
        inference_pb2.StreamPredictRequest(
            model_id="m1",
            chunk=arr1.tobytes(),
            shape=list(arr1.shape),
            dtype=str(arr1.dtype),
            context={"tile_id": "tile1", "session_id": "sessA"},
        ),
        inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": "tile1"}),
        inference_pb2.StreamPredictRequest(
            model_id="m1",
            chunk=arr2.tobytes(),
            shape=list(arr2.shape),
            dtype=str(arr2.dtype),
            context={"tile_id": "tile2", "session_id": "sessA"},
        ),
        inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": "tile2"}),
    ]

    responses = list(svc.StreamPredict(iter(reqs), _DummyContext()))
    assert len(responses) == 2
    for resp in responses:
        assert resp.metadata.get("result_type") == "segmentation_mask"
        # Output may be a packed bundle (mask + confidence) or raw bytes
        if resp.metadata.get("packed_bundle") == "true":
            bundle = unpack_tensor_bundle(resp.output)
            mask = bundle["mask"]
        else:
            mask = np.frombuffer(resp.output, dtype=resp.dtype).reshape(resp.shape)
        assert np.all(mask == 1)
        assert resp.metadata.get("device_id")
        assert resp.metadata.get("session_id") == "sessA"
        assert resp.metadata.get("tile_id") in {"tile1", "tile2"}
        assert resp.end_of_sequence is True


def test_stream_predict_missing_model(monkeypatch):
    svc = server.InferenceService()
    svc.model_manager = type("_", (), {"get": lambda _self, mid: None})()
    req = inference_pb2.StreamPredictRequest(model_id="missing", end_of_sequence=True)
    ctx = _DummyContext()
    responses = list(svc.StreamPredict(iter([req]), ctx))
    assert responses[0].metadata.get("error") == "model_id not found" or ctx.code is not None


def test_stream_predict_backpressure(monkeypatch):
    svc = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            return np.zeros((1, 1, 1, 1), dtype=np.float32)

    svc.model_manager = _mk_manager(_Model())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    # Create many small samples without end_of_sequence until overflow
    reqs = []
    for _ in range(20):
        reqs.append(
            inference_pb2.StreamPredictRequest(
                model_id="m1", chunk=b"0" * 4, shape=[1, 1, 1, 1], dtype="float32"
            )
        )

    ctx = _DummyContext()
    responses = list(svc.StreamPredict(iter(reqs), ctx))
    # Expect an error response due to backpressure
    assert any(r.metadata.get("error", "").startswith("backpressure") for r in responses)


def test_stream_predict_chunk_over_limit(monkeypatch):
    svc = server.InferenceService()
    svc._default_chunk_bytes = 4

    class _Model:
        def predict(self, arr, return_embeddings=False):
            return np.zeros((1, 1, 1, 1), dtype=np.float32)

    svc.model_manager = _mk_manager(_Model())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    class _Ctx:
        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    reqs = [
        inference_pb2.StreamPredictRequest(
            model_id="m1",
            chunk=b"12345678",  # 8 bytes > default 4
            shape=[1, 1, 1, 2],
            dtype="float32",
            context={"session_id": "sessX", "tile_id": "tile-ovr"},
        ),
        inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"session_id": "sessX"}),
    ]

    resps = list(svc.StreamPredict(iter(reqs), _Ctx()))
    assert resps and resps[0].metadata.get("error")
    assert "chunk exceeds" in resps[0].metadata.get("error")
    assert resps[0].metadata.get("session_id") == "sessX"
    assert resps[0].metadata.get("tile_id") == "tile-ovr"


def test_stream_predict_cancel_mid_stream(monkeypatch):
    svc = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            return np.zeros((1, 1, 1, 1), dtype=np.float32)

    svc.model_manager = _mk_manager(_Model())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    class _CancelCtx:
        def __init__(self):
            self.calls = 0

        def cancelled(self):
            self.calls += 1
            return self.calls > 1

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    reqs = [
        inference_pb2.StreamPredictRequest(
            model_id="m1",
            chunk=b"abcd",
            shape=[1, 1, 1, 1],
            dtype="float32",
            context={"session_id": "sessC"},
        ),
        inference_pb2.StreamPredictRequest(
            model_id="m1",
            chunk=b"efgh",
            shape=[1, 1, 1, 1],
            dtype="float32",
            context={"session_id": "sessC"},
        ),
    ]

    resps = list(svc.StreamPredict(iter(reqs), _CancelCtx()))
    assert resps and resps[0].metadata.get("error") == "client_cancelled"
    assert resps[0].metadata.get("session_id") == "sessC"


def test_session_lifecycle_open_close_cancel(monkeypatch):
    svc = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            return np.zeros((1, 1, 1, 1), dtype=np.float32)

    svc.model_manager = _mk_manager(_Model())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    open_req = inference_pb2.OpenSessionRequest(
        session_id="sessL",
        spec=inference_pb2.ModelSpec(model_id="m1", format=inference_pb2.ONNX, source="ignored"),
    )
    open_resp = svc.OpenSession(open_req, _DummyContext())
    assert open_resp.status == "ok"
    assert open_resp.session_id == "sessL"
    assert svc._sessions["sessL"]["model_id"] == "m1"

    # Simulate some tiles then close
    svc._sessions["sessL"]["ok_tiles"] = 3
    svc._sessions["sessL"]["failed_tiles"] = 1
    svc._sessions["sessL"]["errors"] = ["oops"]
    close_resp = svc.CloseSession(
        inference_pb2.CloseSessionRequest(session_id="sessL"), _DummyContext()
    )
    assert close_resp.status == "closed"
    assert close_resp.summary.ok_tiles == 3
    assert close_resp.summary.failed_tiles == 1
    assert "oops" in list(close_resp.summary.errors)
    assert "sessL" not in svc._sessions

    # Cancel on unknown -> not_found
    cancel_ctx = _DummyContext()
    cancel_resp = svc.CancelSession(
        inference_pb2.CancelSessionRequest(session_id="sessL"), cancel_ctx
    )
    assert cancel_resp.status == "not_found"
    assert cancel_ctx.code == grpc.StatusCode.NOT_FOUND
