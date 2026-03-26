# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from unittest.mock import patch, Mock

from nxmndr.huggingface.search import HFModelSearchResult
from nxmndr.models.models import HuggingFaceModelSpec


def _entry(model_id, siblings):
    return {"modelId": model_id, "siblings": [{"rfilename": s} for s in siblings]}


def test_single_result_prefers_model_safetensors():
    entry = _entry("org/model-a", ["config.json", "model.safetensors", "pytorch_model.bin"])
    res = HFModelSearchResult.from_dict(entry)
    assert res is not None
    spec = res.to_spec()
    assert isinstance(spec, HuggingFaceModelSpec)
    assert spec.repo_id == "org/model-a"
    assert spec.filename == "model.safetensors"


def test_single_result_fallback_pytorch_bin():
    entry = _entry("org/model-b", ["README.md", "pytorch_model.bin"])
    res = HFModelSearchResult.from_dict(entry)
    spec = res.to_spec()
    assert spec.filename == "pytorch_model.bin"


def test_result_without_weights_skipped_when_required():
    entry = _entry("org/model-c", ["README.md", "config.json"])
    res = HFModelSearchResult.from_dict(entry)
    spec = res.to_spec(require_weights=True)
    assert spec is None


def test_result_without_weights_included_when_not_required():
    entry = _entry("org/model-d", ["README.md", "tokenizer.json"])
    res = HFModelSearchResult.from_dict(entry)
    spec = res.to_spec(require_weights=False)
    assert spec is not None
    assert spec.filename == "pytorch_model.bin"  # default fallback


def test_fetch_files_success():
    raw = _entry("org/model-e", ["pytorch_model.bin"])  # initial minimal siblings
    result = HFModelSearchResult.from_dict(raw)
    assert result is not None

    mock_json = {
        "modelId": "org/model-e",
        "siblings": [
            {"rfilename": "model.safetensors"},
            {"rfilename": "config.json"},
        ],
    }

    with patch("nxmndr.huggingface.search.requests.get") as mock_get:
        mock_resp = Mock()
        mock_resp.json.return_value = mock_json
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        files = result.fetch_files(timeout=1)
        assert "model.safetensors" in files
        assert result.siblings[0]["rfilename"] == "model.safetensors"


def test_fetch_files_timeout():
    raw = _entry("org/model-f", [])
    result = HFModelSearchResult.from_dict(raw)
    assert result is not None

    with patch(
        "nxmndr.huggingface.search.requests.get", side_effect=__import__("requests").Timeout
    ):
        files = result.fetch_files(timeout=0)
        assert files == []
