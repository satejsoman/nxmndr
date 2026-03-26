# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from pathlib import Path

import numpy as np

from nxmndr.inference import (
    InferenceSession,
    RemoteInferenceProvider,
    inference_pb2,
    inference_pb2_grpc,
)
from nxmndr.models import OnnxModelSpec
from tst.integration.remote_test_utils import RemoteTestServer
from tst.example_model.modeling_exampleconv import export

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Local server helper logic removed; using acquire_channel_or_server from remote_test_utils


def test_remote_onnx_inference(tmp_path):
    # Ensure test model artifacts exist (reuse if already present)
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "tst" / "example_model"
    model_path = model_dir / "example_model.onnx"
    if not model_path.exists():
        export(model_dir)
    logger.info(f"Model path: {model_path}")
    assert model_path.exists()

    # Start server directly (ephemeral port) and obtain bound port
    with RemoteTestServer() as channel:
        provider = RemoteInferenceProvider(channel)
        spec = OnnxModelSpec(model_path=str(model_path), name="remote-test")
        provider.load_spec(spec)
        assert provider.model_id, "model_id should be set after load"
        session = InferenceSession(spec, provider)
        arr = np.zeros((1, 3, 32, 32), dtype=np.float32)
        out = session.run(arr)
        assert out is not None and isinstance(out, np.ndarray)

        # List models via raw stub
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        listed = stub.ListModels(inference_pb2.ListModelsRequest())
        ids = [m.model_id for m in listed.models]
        assert provider.model_id in ids

        # Capabilities
        caps = stub.Capabilities(inference_pb2.CapabilitiesRequest())
        assert any(c.key == "cuda" for c in caps.capabilities)

        # Unload model
        unload_resp = stub.UnloadModel(inference_pb2.UnloadModelRequest(model_id=provider.model_id))
        assert unload_resp.success
    # server auto-stopped by context manager


def test_load_model_spec():
    repo_root = Path.cwd()
    model_path = repo_root / "tst" / "example_model" / "example_model.onnx"
    logger.info(f"Testing model spec loading with path: {model_path}")

    # This is a simpler test that just tests model spec creation
    spec = OnnxModelSpec(model_path=str(model_path), name="test-spec")
    assert spec.model_path == str(model_path)
    assert spec.name == "test-spec"
    logger.info("Model spec created successfully")


if __name__ == "__main__":
    test_remote_onnx_inference()
