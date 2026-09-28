# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model cache leases through the real server, client and gRPC transport.

The two real-gRPC cases of the plan's chunk 1a Done list: with a tiny capacity,
a model pinned by an open session stays resident and usable while other models
load and evict each other, and becomes evictable after its last lease ends; and a
prediction still running when its session closes keeps its record alive until the
call returns, after which the record is disposed exactly once.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from tst.support.grpc_harness import running_server, wait_until
from tst.support.stream_doubles import GatedModel, SegLogitsModel
from tst.unit.model_cache_test_utils import CountingRecord

pytestmark = pytest.mark.integration

CHIP = np.ones((4, 4, 3), np.float32)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


class _Loader:
    """Builds one dispose-counting record per load from a per-source model factory."""

    def __init__(self, factories):
        self.factories = factories
        self.records = defaultdict(list)
        self.lock = threading.Lock()

    def __call__(self, model_spec, key, metadata, device_plan):
        record = CountingRecord(
            model=self.factories[model_spec.model_path](),
            backend="onnx",
            metadata=dict(metadata),
            spec=model_spec,
        )
        with self.lock:
            self.records[model_spec.model_path].append(record)
        return record


def _load(client, source):
    return client.load_model("ignored", {"format": "onnx", "source": source, "task": "segmentation"})


def _open(client, session_id, model_id):
    resp = client.open_session(session_id=session_id, spec=inference_pb2.ModelSpec(model_id=model_id))
    assert resp.status == "ok", resp.error
    return resp


def _predict(client, session_id):
    [resp] = list(client.stream_predict(session_id=session_id, samples=[CHIP], tile_ids=["t0"]))
    assert resp.metadata.get("error") is None, resp.metadata
    return resp


def test_session_pinned_model_survives_while_other_models_load_and_evict(tmp_path):
    sources = ("double://pinned", "double://m1", "double://m2", "double://m3")
    loader = _Loader({source: SegLogitsModel for source in sources})
    with running_server(tmp_path, loader=loader, capacity=2) as h, h.client() as c:
        pinned = _load(c, "double://pinned")
        _open(c, "s-pin", pinned)

        m1 = _load(c, "double://m1")  # fills the second slot
        m2 = _load(c, "double://m2")  # evicts m1, the only unpinned record
        assert h.manager.get(m1) is None and h.manager.get(m2) is not None
        m3 = _load(c, "double://m3")  # evicts m2
        assert h.manager.get(m2) is None and h.manager.get(m3) is not None
        assert h.manager.get(pinned) is not None and h.manager.pin_count(pinned) == 1

        assert _predict(c, "s-pin").metadata["model_id"] == pinned  # resident and usable
        _load(c, "double://m1")  # a reload evicts m3, never the pinned record
        assert h.manager.get(m3) is None and h.manager.get(pinned) is not None
        assert loader.records["double://pinned"][0].dispose_calls == 0
        assert [r.dispose_calls for r in loader.records["double://m1"]] == [1, 0]
        assert [r.dispose_calls for r in loader.records["double://m2"]] == [1]
        assert [r.dispose_calls for r in loader.records["double://m3"]] == [1]

        c.close_session("s-pin")  # the last lease on the pinned model ends
        assert wait_until(lambda: h.manager.pin_count(pinned) == 0)
        _load(c, "double://m2")  # the pinned model is now the least recently used
        assert h.manager.get(pinned) is None
        assert loader.records["double://pinned"][0].dispose_calls == 1
        assert len(loader.records["double://pinned"]) == 1  # loaded once throughout


@pytest.mark.parametrize("ending", ["close", "cancel"])
def test_running_prediction_keeps_its_record_until_the_call_returns(tmp_path, ending):
    gated = GatedModel()
    loader = _Loader({"double://gated": lambda: gated, "double://other": SegLogitsModel})
    with (
        running_server(tmp_path, loader=loader, capacity=1) as h,
        h.client() as c,
        h.client() as control,
    ):
        mid = _load(c, "double://gated")
        _open(c, "s-run", mid)
        [record] = loader.records["double://gated"]
        call = c.stream_predict(session_id="s-run", samples=[CHIP], tile_ids=["t0"])
        got, errors = [], []

        def consume():
            try:
                got.extend(call)
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert gated.entered.wait(10)  # the prediction is running

        if ending == "close":
            assert control.close_session("s-run").status == "closed"
        else:
            assert control.cancel_session("s-run").status == "cancelled"
        assert h.manager.pin_count(mid) == 1  # the execution lease outlives the session
        with pytest.raises(InferenceGrpcError) as err:
            _load(control, "double://other")  # the only slot is held by the running call
        assert err.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert h.manager.get(mid) is record and record.dispose_calls == 0

        gated.release.set()
        consumer.join(timeout=10)
        assert errors == [] and [r.metadata.get("error") for r in got] == [None]
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)
        assert record.dispose_calls == 0  # released, not yet evicted

        _load(control, "double://other")  # now the record can be evicted
        assert h.manager.get(mid) is None
        assert record.dispose_calls == 1
    assert record.dispose_calls == 1  # server shutdown does not dispose it again
