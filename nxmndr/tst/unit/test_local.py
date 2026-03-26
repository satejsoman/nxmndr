# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from pathlib import Path

import numpy as np

from nxmndr.inference import InferenceSession, LocalInferenceProvider
from nxmndr.models import PytorchModelSpec
from tst.example_model.modeling_exampleconv import ExampleModel, export as export_example

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cwd = Path(__file__).parent
example_dir = cwd.parent / "example_model"
weights_path = example_dir / "example_model.pth"


def serialize_model_spec():
    if not weights_path.exists():
        example_dir.mkdir(parents=True, exist_ok=True)
        export_example(example_dir)
    return PytorchModelSpec(model_class=ExampleModel, model_path=str(weights_path), name="example")


def test_load_model_spec():
    provider = LocalInferenceProvider()
    model_spec = serialize_model_spec()
    session = InferenceSession(model_spec, provider)
    assert session.model is not None
    logger.info(f"Model loaded: {session.model}")


def test_predict():
    provider = LocalInferenceProvider()
    model_spec = serialize_model_spec()
    session = InferenceSession(model_spec, provider)
    arr = np.zeros((1, 3, 32, 32), dtype=np.float32)
    output = session.run(arr)
    assert output.shape[0] == 1 and output.shape[1] == 10
    logger.info(f"Predict response shape: {output.shape}, dtype: {output.dtype}")


def test_has_gpu():
    provider = LocalInferenceProvider()
    assert isinstance(provider.has_gpu(), bool)
    logger.info(f"Server has GPU: {provider.has_gpu()}")


if __name__ == "__main__":
    test_load_model_spec()
    test_predict()
    test_has_gpu()
