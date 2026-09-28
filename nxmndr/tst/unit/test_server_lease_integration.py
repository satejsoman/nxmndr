# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Wave-1 interface requests between chunk 1 (server.py) and chunk 1a (managers.py).

Each test confirms one request against the real ModelManager, which the wave-1
branches could only test against an in-test fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nxmndr.models import sam as sam_support
from nxmndr.server import dispatch, managers, server
from tst.unit.model_cache_test_utils import FakeClock


def _record(model_spec, key, metadata, device_plan):
    return managers.ModelRecord(
        model=object(), backend="onnx", metadata=dict(metadata), spec=model_spec, device_models={}
    )


def test_loader_first_positional_argument_is_the_model_spec():
    """Chunk 1 request (a): loader(model_spec, key, metadata, device_plan), positional."""

    seen = []

    def loader(*args, **kwargs):
        seen.append((args, kwargs))
        return _record(*args)

    manager = managers.ModelManager(None, capacity=2, loader=loader)
    spec_a, spec_b = object(), object()
    key_a = managers.ModelCacheKey(format="onnx", source="a")
    key_b = managers.ModelCacheKey(format="onnx", source="b")
    plan = [{"id": "cpu:0"}]

    a = manager.load(spec_a, key=key_a, metadata={"name": "a"}, device_plan=plan).model_id
    manager.open_session("s1", model_spec=spec_b, key=key_b)

    (args_a, kwargs_a), (args_b, kwargs_b) = seen
    assert args_a[0] is spec_a and args_a[1] == key_a and args_a[3] == plan and kwargs_a == {}
    assert args_a[2] == {"name": "a", "model_id": a}
    assert args_b[0] is spec_b and args_b[1] == key_b and kwargs_b == {}
    record = manager.get(a)
    assert record.model_id == a and record.key == key_a  # assigned after the loader returns


@pytest.mark.parametrize("reason", managers.SESSION_CLOSE_REASONS)
def test_session_state_is_open_then_the_close_reason_then_unknown(reason):
    """Chunk 1 request (b): 'open', the tombstone reason while it lives, else 'unknown'."""

    clock = FakeClock()
    manager = managers.ModelManager(None, capacity=2, session_ttl_s=10.0, clock=clock, loader=_record)
    mid = manager.load(object(), key=managers.ModelCacheKey(format="onnx", source="a")).model_id

    assert manager.session_state("s1") == "unknown"
    manager.open_session("s1", model_id=mid)
    assert manager.session_state("s1") == "open"
    assert manager.close_session("s1", reason=reason) is True
    assert manager.session_state("s1") == reason
    assert manager.close_session("s1", reason="closed") is False  # closed: False, no raise
    clock.advance(10.5)
    assert manager.session_state("s1") == "unknown"  # the tombstone expired
    assert manager.close_session("s1", reason="closed") is False  # unknown: False, no raise


def test_model_manager_capacity_and_ttl_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NXMNDR_MODEL_CACHE_CAPACITY", "3")
    monkeypatch.setenv("NXMNDR_SESSION_TTL_SECONDS", "7")
    svc = server.InferenceService(model_cache_dir=tmp_path / "cache")
    assert isinstance(svc.model_manager, managers.ModelManager)
    assert svc.model_manager.stats().capacity == 3
    assert svc.model_manager._ttl == 7.0

    monkeypatch.delenv("NXMNDR_MODEL_CACHE_CAPACITY")
    monkeypatch.delenv("NXMNDR_SESSION_TTL_SECONDS")
    svc = server.InferenceService(model_cache_dir=tmp_path / "cache")
    assert svc.model_manager.stats().capacity == 10  # plan default
    assert svc.model_manager._ttl == 3600.0


def test_sam_variant_factory_gets_the_record_auth_token(monkeypatch):
    """The SAM path reads ModelRecord.auth_token, not a token on the spec or in metadata."""

    tokens = []

    def load_sam_variant(variant, capability, device, token=None):
        tokens.append(token)
        return sam_support.SamVariantResource("tracker-model", "tracker-processor")

    monkeypatch.setattr(sam_support, "load_sam_variant", load_sam_variant)
    record = managers.ModelRecord(
        model=SimpleNamespace(model="sam3-model", processor="sam3-processor"),
        backend="huggingface",
        metadata={},
        spec=SimpleNamespace(token="spec-token"),
        auth_token="record-token",
    )
    lease = SimpleNamespace(
        record=record,
        model_for_device=lambda device_id: record.model,
        resource=lambda name, factory: factory(),
    )
    capability = SimpleNamespace(record_variant=sam_support.SAM_VARIANT_TEXT)
    get_variant = dispatch._sam_variant_getter(lease, capability, "cpu:0", "cpu")

    assert get_variant(sam_support.SAM_VARIANT_TEXT) == ("sam3-model", "sam3-processor")
    assert get_variant(sam_support.SAM_VARIANT_GEOMETRY) == ("tracker-model", "tracker-processor")
    assert tokens == ["record-token"]
