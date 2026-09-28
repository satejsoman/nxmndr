# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RpcWorkerManager starts its PyTorch RPC worker under the ``spawn`` start method.

``spawn`` is the default on macOS and Windows; it pickles the process target, so a
nested function fails in the parent with "Can't pickle local object".
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from multiprocessing.reduction import ForkingPickler

import pytest

from nxmndr.models import PytorchModelSpec
from nxmndr.server import managers
from tst.support.rpc_probe import MARKER_ENV, MarkerModel

pytestmark = pytest.mark.integration


class _RecordedProcess:
    """Stands in for multiprocessing.Process and keeps what it was given."""

    made = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args, self.kwargs = target, tuple(args), dict(kwargs or {})
        _RecordedProcess.made.append(self)

    def start(self):
        pass

    def is_alive(self):
        return False


def test_worker_target_and_arguments_pickle_for_spawn(tmp_path, monkeypatch):
    _RecordedProcess.made.clear()
    monkeypatch.setattr(managers.multiprocessing, "Process", _RecordedProcess)
    spec = PytorchModelSpec(model_class=MarkerModel, model_path=str(tmp_path / "w.pth"))
    managers.RpcWorkerManager().ensure_worker(spec, master_addr="127.0.0.1", master_port=29555)

    [proc] = _RecordedProcess.made
    payload = bytes(ForkingPickler.dumps((proc.target, proc.args, proc.kwargs)))
    target, args, kwargs = pickle.loads(payload)
    assert target is managers._run_rpc_worker
    assert args[1:] == ("127.0.0.1", 29555) and args[0].model_class is MarkerModel


def test_worker_process_starts_and_runs_under_spawn(tmp_path, monkeypatch):
    marker = tmp_path / "rpc-worker-ran"
    monkeypatch.setenv(MARKER_ENV, str(marker))
    monkeypatch.setattr(
        managers.multiprocessing, "Process", multiprocessing.get_context("spawn").Process
    )
    spec = PytorchModelSpec(model_class=MarkerModel, model_path=str(tmp_path / "w.pth"))
    manager = managers.RpcWorkerManager()
    manager.ensure_worker(spec, master_addr="127.0.0.1", master_port=29555)
    proc = manager._proc
    try:
        proc.join(timeout=120)
        assert proc.exitcode == 0, f"spawned worker exit code {proc.exitcode}"
        assert marker.read_text(encoding="utf-8") == f"pid={proc.pid}"
        assert proc.pid != os.getpid()
    finally:
        manager.stop()  # terminates only this manager's own child if still alive
