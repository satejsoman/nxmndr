# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json

import numpy as np

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2, inference_pb2_grpc


def test_predict_attaches_sam_options(monkeypatch):
    captured = {}

    class _FakeStub:
        def Predict(self, request, timeout=None):
            captured["request"] = request
            return inference_pb2.PredictResponse(
                output=np.zeros((1,), dtype=np.uint8).tobytes(),
                shape=[1],
                dtype="uint8",
                metadata={},
            )

    def _fake_stub_ctor(channel):
        captured["channel"] = channel
        return _FakeStub()

    class _FakeFuture:
        def result(self, timeout=None):
            return None

    class _FakeChannel:
        def close(self):
            captured["closed"] = True

    def _fake_channel_ready_future(channel):
        return _FakeFuture()

    def _fake_insecure_channel(endpoint, options=None):
        captured["endpoint"] = endpoint
        captured["options"] = options
        return _FakeChannel()

    monkeypatch.setattr(inference_pb2_grpc, "InferenceServiceStub", _fake_stub_ctor)
    monkeypatch.setattr("grpc.channel_ready_future", _fake_channel_ready_future)
    monkeypatch.setattr("grpc.insecure_channel", _fake_insecure_channel)

    client = InferenceGrpcClient("localhost:50051", timeout=5)

    sam_points = [[[[10, 20]]], [[[30, 40]]]]
    sam_labels = [1, 0]
    sam_bbox = [5, 5, 50, 50]

    result = client.predict(
        model_id="sam-model",
        tensor=np.zeros((1, 3, 2, 2), dtype=np.uint8),
        options={
            "sam_text_prompt": "tree",
            "sam_input_points": json.dumps(sam_points),
            "sam_input_labels": json.dumps(sam_labels),
            "sam_input_bbox": json.dumps(sam_bbox),
            "task_type": "segmentation",
        },
    )

    req = captured["request"]
    assert req.model_id == "sam-model"
    assert req.options["sam_text_prompt"] == "tree"
    assert json.loads(req.options["sam_input_points"]) == sam_points
    assert json.loads(req.options["sam_input_labels"]) == sam_labels
    assert json.loads(req.options["sam_input_bbox"]) == sam_bbox
    assert req.options["task_type"] == "segmentation"

    assert isinstance(result.output, (bytes, bytearray))
    assert tuple(result.shape) == (1,)
    assert result.dtype == "uint8"
    assert result.metadata == {}
