# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RpcWorkerManager and the PyTorch record, without a real RPC group.

- Every ``rpc_sync`` call passes the worker function object, never its name
  (wave-5 review W3: PyTorch rejects a string with "function should be callable").
- A failed ``_rpc_stop`` is logged, then the child is terminated (W3).
- A generic PyTorch record owns its worker-side model as an aux resource: disposing
  the record frees that model exactly once (W4).
- A catalog model the worker cannot build is returned as a refusal, raised in the
  server as the ``CatalogError`` with its one-line message, and answered with
  INVALID_ARGUMENT (FAILED_PRECONDITION for a missing family package), never INTERNAL.
The real worker, rendezvous and routing run in tst/integration/test_rpc_pytorch_service.py.
"""

from __future__ import annotations

import logging

import grpc
import pytest

from nxmndr.models import PytorchModelSpec, catalog
from nxmndr.server import managers, rpc_worker
from tst.support.rpc_models import AddConst


class _Proc:
    def __init__(self):
        self.pid = 4242
        self.exitcode = None
        self.alive = True
        self.calls = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.calls.append("join")

    def terminate(self):
        self.calls.append("terminate")
        self.alive = False

    def kill(self):
        self.calls.append("kill")
        self.alive = False


def _running_manager(monkeypatch):
    manager = managers.RpcWorkerManager()
    manager._state = "running"
    manager._proc = _Proc()
    left = []
    monkeypatch.setattr(managers.RpcWorkerManager, "_leave_group", staticmethod(lambda graceful: left.append(graceful)))
    return manager, left


def test_rpc_calls_pass_the_worker_functions_not_their_names(monkeypatch, tmp_path):
    calls = []

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        calls.append((to, func, args))
        if func is rpc_worker._rpc_load:
            return {"model_id": args[0], "device": "cpu", "fingerprint": "sha256:x"}
        return True

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, left = _running_manager(monkeypatch)
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"), name="w")

    assert manager.load("m1", spec)["model_id"] == "m1"
    manager.infer("m1", "tensor", device_hint="cpu", timeout=5.0)
    assert manager.unload("m1") is True
    manager.status()
    manager.stop()

    funcs = [func for _, func, _ in calls]
    assert funcs == [
        rpc_worker._rpc_load,
        rpc_worker._rpc_infer,
        rpc_worker._rpc_unload,
        rpc_worker._rpc_status,
        rpc_worker._rpc_stop,
    ]
    assert all(callable(f) and not isinstance(f, str) for f in funcs)
    assert all(to == managers.RPC_WORKER_NAME for to, _, _ in calls)
    assert calls[0][2] == ("m1", {"model_class": AddConst, "model_path": str(tmp_path / "w.pt"), "name": "w"})
    assert calls[1][2] == ("m1", "tensor", "cpu")
    assert left == [True]  # a live worker gets a graceful RPC shutdown


def test_a_failed_rpc_stop_is_logged_then_the_worker_is_terminated(monkeypatch, caplog):
    def failing_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        assert func is rpc_worker._rpc_stop
        raise RuntimeError("worker unreachable")

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", failing_rpc_sync)
    manager, left = _running_manager(monkeypatch)
    proc = manager._proc
    with caplog.at_level(logging.WARNING):
        manager.stop()
    assert any(
        "_rpc_stop failed" in r.getMessage() and r.exc_info and "worker unreachable" in str(r.exc_info[1])
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert "terminate" in proc.calls
    assert left == [False]  # no graceful shutdown with an unreachable peer
    assert manager._state == "stopped" and not manager.is_alive()


def test_an_unregistered_model_class_is_refused_before_any_worker_starts(monkeypatch, tmp_path):
    manager = managers.RpcWorkerManager()
    monkeypatch.setattr(manager, "_start_locked", lambda: pytest.fail("worker started"))
    spec = PytorchModelSpec(model_class="not_registered", model_path=str(tmp_path / "w.pt"))
    with pytest.raises(ValueError, match="not registered"):
        manager.load("m1", spec)


class _Workers:
    def __init__(self):
        self.loaded, self.unloaded = [], []

    def load(self, model_id, spec):
        self.loaded.append((model_id, spec))
        return {"model_id": model_id, "device": "cpu", "fingerprint": f"sha256:{model_id}"}

    def unload(self, model_id):
        self.unloaded.append(model_id)
        return True


def test_a_pytorch_record_owns_its_worker_model_and_frees_it_once(tmp_path):
    workers = _Workers()
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"))
    record = managers.build_model_record(
        None, spec, None, {"model_id": "abc", "task": 0}, None, rpc_workers=workers
    )
    assert workers.loaded == [("abc", spec)]
    assert record.backend == "pytorch" and record.model is None
    assert record.metadata["rpc_fingerprint"] == "sha256:abc"
    handle = record._resource(managers.RPC_MODEL_RESOURCE, lambda: pytest.fail("created twice"))
    assert handle.model_id == "abc"
    record.dispose()
    record.dispose()
    assert workers.unloaded == ["abc"]


def test_a_pytorch_spec_without_the_rpc_worker_fails_to_load(tmp_path):
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"))
    with pytest.raises(ValueError, match="RPC worker"):
        managers.build_model_record(None, spec, None, {"model_id": "abc"}, None)


def test_the_worker_registry_refuses_a_second_model_under_one_id(tmp_path):
    import torch

    path = tmp_path / "w.pt"
    torch.save({"bias": torch.tensor(1.0)}, str(path))
    payload = {"model_class": AddConst, "model_path": str(path), "name": "w"}
    try:
        info = rpc_worker._rpc_load("unit-m1", payload)
        assert info["device"] == "cpu" and info["fingerprint"].startswith("sha256:")
        with pytest.raises(ValueError, match="already loaded"):
            rpc_worker._rpc_load("unit-m1", payload)
        out = rpc_worker._rpc_infer("unit-m1", torch.zeros(2), "cpu")
        assert out.tolist() == [1.0, 1.0]
        assert rpc_worker._rpc_unload("unit-m1") is True
        with pytest.raises(rpc_worker.UnknownModelError):
            rpc_worker._rpc_infer("unit-m1", torch.zeros(2), "cpu")
        assert rpc_worker._rpc_unload("unit-m1") is False
    finally:
        rpc_worker._rpc_unload("unit-m1")


def test_the_worker_refuses_a_catalog_model_it_cannot_build(tmp_path):
    import torch
    import torchvision

    path = tmp_path / "r18.pt"
    torch.save(torchvision.models.resnet18(weights=None, num_classes=5).state_dict(), str(path))
    payload = {
        "model_class": "torchvision:resnet18",
        "model_path": str(path),
        "name": "r18",
        "catalog_args": {"num_classes": 7},
    }
    before = rpc_worker._rpc_status()["load_calls"]
    try:
        reply = rpc_worker._rpc_load("unit-refused", payload)
        assert set(reply) == {"model_id", "refused"}
        refused = reply["refused"]
        assert refused["unavailable"] is False
        assert refused["message"].startswith(f"weights file {path} does not match ")
        assert "\n" not in refused["message"]
        assert refused["traceback"].startswith("Traceback")
        assert "CatalogError" in refused["traceback"]
        status = rpc_worker._rpc_status()
        assert status["load_calls"] == before + 1
        assert "unit-refused" not in status["models"]
        # Nothing was kept under the ID: a model that fits loads under it.
        payload["catalog_args"] = {}
        assert "refused" not in rpc_worker._rpc_load("unit-refused", payload)
        assert rpc_worker._rpc_status()["load_calls"] == before + 2
    finally:
        rpc_worker._rpc_unload("unit-refused")


def test_a_refusal_is_raised_in_the_server_as_the_catalog_error(monkeypatch, caplog, tmp_path):
    replies = []

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        assert func is rpc_worker._rpc_load
        return {"model_id": args[0], "refused": replies.pop(0)}

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, _ = _running_manager(monkeypatch)
    spec = PytorchModelSpec(
        model_class="torchvision:resnet18", model_path=str(tmp_path / "w.pt"), name="w"
    )
    replies.append({"message": "does not match", "unavailable": False, "traceback": "Traceback X"})
    with caplog.at_level(logging.DEBUG, logger=managers.logger.name):
        with pytest.raises(catalog.CatalogError) as refused:
            manager.load("m1", spec)
    assert type(refused.value) is catalog.CatalogError and str(refused.value) == "does not match"
    assert any(
        r.levelno == logging.DEBUG and "Traceback X" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]

    replies.append({"message": "needs torchgeo", "unavailable": True, "traceback": ""})
    with pytest.raises(catalog.CatalogUnavailableError, match="needs torchgeo"):
        manager.load("m2", spec)


def test_a_catalog_refusal_is_a_request_error_on_the_wire():
    from nxmndr.server.server import InferenceService

    def failed_load(cause):
        try:
            try:
                raise cause
            except Exception as exc:
                raise managers.ModelLoadError(f"loading model m1 (pytorch) failed: {exc}") from exc
        except managers.ModelLoadError as exc:
            return exc

    mismatch = failed_load(catalog.CatalogError("weights file w.pt does not match x"))
    assert InferenceService._cache_error_status(mismatch) == grpc.StatusCode.INVALID_ARGUMENT
    assert InferenceService._cache_error_details(mismatch) == "weights file w.pt does not match x"
    missing = failed_load(catalog.CatalogUnavailableError("torchgeo:unet needs torchgeo"))
    assert InferenceService._cache_error_status(missing) == grpc.StatusCode.FAILED_PRECONDITION
    assert InferenceService._cache_error_details(missing) == "torchgeo:unet needs torchgeo"
    other = failed_load(RuntimeError("worker gone"))
    assert InferenceService._cache_error_status(other) == grpc.StatusCode.INTERNAL
    assert InferenceService._cache_error_details(other) == str(other)


def _listener():
    import socket

    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.1)
    return listener, listener.getsockname()[1]


def test_the_ready_handshake_accepts_only_its_child_and_keeps_its_deadline():
    import socket
    import threading
    import time

    listener, port = _listener()
    proc = _Proc()
    stop = threading.Event()

    def stray():  # a local process that keeps connecting with the wrong line
        while not stop.is_set():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1) as conn:
                    conn.sendall(b"ready 1\n")
            except OSError:
                pass
            time.sleep(0.01)

    thread = threading.Thread(target=stray, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(managers.RpcWorkerStartError, match="within 0.5 s"):
            managers.RpcWorkerManager._await_ready(proc, listener, 0.5)
        assert time.monotonic() - started < 2.0
    finally:
        stop.set()
        thread.join(timeout=5)

    with socket.create_connection(("127.0.0.1", port), timeout=1) as conn:
        conn.sendall(f"ready {proc.pid}\n".encode("ascii"))
        managers.RpcWorkerManager._await_ready(proc, listener, 5.0)  # returns: its own child
    proc.alive = False
    with pytest.raises(managers.RpcWorkerStartError, match="exited with code None"):
        managers.RpcWorkerManager._await_ready(proc, listener, 5.0)
    listener.close()
