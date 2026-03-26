# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from nxmndr.inference.inference import load_huggingface
from nxmndr.inference import RemoteInferenceProvider, LocalInferenceProvider
from nxmndr.models import HuggingFaceModelSpec


def test_hf_loader_processor_fallback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    # Minimal config files to satisfy transformers path lookups
    (repo / "config.json").write_text(
        '{"model_type": "exampleconv", "architectures": ["ExampleConvModel"]}'
    )
    (repo / "preprocessor_config.json").write_text(
        '{"processor_class": "ExampleImageProcessor", "auto_map": {"AutoImageProcessor": "modeling_exampleconv.ExampleImageProcessor"}}'
    )
    (repo / "pytorch_model.bin").write_bytes(b"")

    # Force snapshot_download to return our local repo and AutoProcessor to fail so fallback is used.
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(repo))

    class DummyModel:
        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **k: DummyModel())

    def _fail_auto_processor(*args, **kwargs):
        raise ValueError("no auto processor")

    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", _fail_auto_processor)
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", lambda *a, **k: "processor-ok"
    )

    spec = HuggingFaceModelSpec(
        repo_id="dummy",
        filename="pytorch_model.bin",
        local_files_only=True,
        cache_dir=str(tmp_path / "cache"),
    )
    provider = LocalInferenceProvider()

    hf_model = load_huggingface(spec, provider, session=None)

    assert getattr(hf_model, "processor", None) == "processor-ok"
    assert getattr(hf_model, "model", None) is not None


def test_remote_predict_requires_model_id():
    class _DummyChan:
        def unary_unary(self, *args, **kwargs):
            return lambda *a, **k: None

        def stream_stream(self, *args, **kwargs):
            return lambda *a, **k: None

    provider = RemoteInferenceProvider(grpc_channel=_DummyChan())
    provider.stub = type("Stub", (), {"Predict": lambda *a, **k: None})()
    provider.model_id = None
    with pytest.raises(RuntimeError, match="model_id"):
        provider.predict(np.zeros((1, 1)))
