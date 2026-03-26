# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json

import numpy as np

from nxmndr.inference import inference_pb2
from nxmndr.server import server


class _DummyContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class _DummyRecord:
    def __init__(self):
        self.model = None
        self.backend = "onnx"
        self.metadata = {"repo_id": "facebook/sam3", "token": "token"}


class _DummyManager:
    def __init__(self, record):
        self._record = record

    def get(self, model_id):
        return self._record

    def get_model_for_device(self, model_id, device_id):
        return self._record.model if self._record else None


def test_predict_routes_sam_prompts(monkeypatch):
    service = server.InferenceService()
    record = _DummyRecord()
    service.model_manager = _DummyManager(record)

    captured = {}

    def _fake_is_sam_model(model_id, model_spec):
        captured["is_sam_model"] = (model_id, model_spec)
        return True

    def _fake_handle_sam_inference(
        model_id, model_spec, image_array, options, device, logger, token=None
    ):
        captured["options"] = dict(options)
        captured["image_shape"] = image_array.shape
        return np.array([[1, 2], [3, 4]], dtype=np.uint16)

    monkeypatch.setattr(server, "is_sam_model", _fake_is_sam_model)
    monkeypatch.setattr(server, "handle_sam_inference", _fake_handle_sam_inference)

    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    req = inference_pb2.PredictRequest(
        model_id="sam-model",
        input=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )
    req.options["sam_text_prompt"] = "tree"
    req.options["sam_input_points"] = json.dumps([[[[10, 20]]]])
    req.options["sam_input_labels"] = json.dumps([1])
    req.options["sam_input_bbox"] = json.dumps([5, 5, 50, 50])
    req.options["task_type"] = "segmentation"

    response = service.Predict(req, _DummyContext())

    assert captured["is_sam_model"][0] == "sam-model"
    assert captured["options"]["sam_text_prompt"] == "tree"
    assert json.loads(captured["options"]["sam_input_points"]) == [[[[10, 20]]]]
    assert json.loads(captured["options"]["sam_input_labels"]) == [1]
    assert json.loads(captured["options"]["sam_input_bbox"]) == [5, 5, 50, 50]
    assert captured["image_shape"] == (2, 2, 3)

    output = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert output.shape == (2, 2)
    assert output.dtype == np.uint16
    assert np.array_equal(output, np.array([[1, 2], [3, 4]], dtype=np.uint16))
    assert response.metadata.get("result_type") == "segmentation_mask"


def test_predict_segmentation_mask_non_sam(monkeypatch):
    service = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            # Return a 2D mask directly; prepare_segmentation_mask will accept (H, W)
            return np.ones((2, 2), dtype=np.uint16)

    class _Record:
        def __init__(self):
            self.model = _Model()
            self.backend = "onnx"
            self.metadata = {"task_name": "segmentation"}

    service.model_manager = _DummyManager(_Record())

    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    req = inference_pb2.PredictRequest(
        model_id="seg-model",
        input=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )
    req.options["task_type"] = "segmentation"

    response = service.Predict(req, _DummyContext())

    mask = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert mask.shape == (2, 2)
    assert response.metadata.get("result_type") == "segmentation_mask"
    # Since channel 1 > channel 0 everywhere, mask should be all ones
    assert np.all(mask == 1)


def test_predict_segmentation_mask_from_logits(monkeypatch):
    service = server.InferenceService()

    class _Model:
        def predict(self, arr, return_embeddings=False):
            # Return logits shaped (N, C, H, W); channel 1 wins everywhere.
            return np.concatenate(
                [
                    np.zeros((1, 1, 2, 2), dtype=np.float32),
                    np.ones((1, 1, 2, 2), dtype=np.float32),
                ],
                axis=1,
            )

    class _Record:
        def __init__(self):
            self.model = _Model()
            self.backend = "onnx"
            self.metadata = {"task_name": "segmentation"}

    service.model_manager = _DummyManager(_Record())
    monkeypatch.setattr(server, "is_sam_model", lambda mid, meta: False)

    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    req = inference_pb2.PredictRequest(
        model_id="seg-model",
        input=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )
    req.options["task_type"] = "segmentation"

    response = service.Predict(req, _DummyContext())

    mask = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert mask.shape == (2, 2)
    assert response.metadata.get("result_type") == "segmentation_mask"
    assert np.all(mask == 1)
