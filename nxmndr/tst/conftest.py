# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Top-level test configuration.

Auto-generates synthetic test model weights (PyTorch, ONNX, HuggingFace)
before the test session starts so binary model files do not need to be
checked into git.
"""

from pathlib import Path


def pytest_configure(config):
    """Generate test model artifacts if they are missing."""
    tst_dir = Path(__file__).parent
    example_dir = tst_dir / "example_model"
    integration_dir = tst_dir / "integration" / "example_model"

    from example_model.modeling_exampleconv import export

    # Generate models for unit tests
    export(example_dir)
    # Generate models for integration tests
    export(integration_dir)
