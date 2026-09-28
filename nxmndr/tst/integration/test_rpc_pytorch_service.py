# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generic PyTorch models through the real gRPC service and the real PyTorch RPC worker.

Each test starts a fresh server process (``tst.support.rpc_service_process``;
torch joins one RPC group per process) and talks to it only over gRPC with the
canonical client. Nothing is mocked: the server's ``RpcWorkerManager`` starts a
spawned worker, joins its RPC group and calls ``_rpc_load``/``_rpc_infer``/
``_rpc_unload`` with the function objects. The model is ``AddConst`` (input plus
the scalar in its weights file), so each model's output names the model.

Covers the wave-5 review: W2 (the first OpenSession starts the worker; startup
failures are bounded and clean up the child), W3 (unary Predict and StreamPredict
run through the real RPC invocation) and W4 (two models loaded at once route by
identity under concurrent pinned sessions; eviction and unload free the model in
the worker; a reload gets a new identity).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np
import pytest
import torch

from nxmndr.client import InferenceGrpcClient, InferenceGrpcError
from nxmndr.inference import inference_pb2
from tst.support import rpc_models

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed not available"),
]

ROOT = Path(__file__).resolve().parents[2]  # the nxmndr directory
CHIP = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)


class ServiceProcess:
    """One ``rpc_service_process`` server; ``command`` sends a line and returns its reply."""

    def __init__(self, tmp_path: Path, *args: str):
        self.log_path = tmp_path / "rpc_service_process.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")) if p
        )
        for var in ("MASTER_ADDR", "MASTER_PORT", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
            env.pop(var, None)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tst.support.rpc_service_process", *args],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            env=env,
        )
        self._replies: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        hello = self._reply(timeout=180)
        self.endpoint = f"127.0.0.1:{hello['port']}"

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith("@@ "):
                self._replies.put(json.loads(line[3:]))

    def _reply(self, timeout=120):
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"no reply from the server process; log:\n{self.log_tail()}")

    def command(self, text: str, timeout=120) -> dict:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return self._reply(timeout)

    def client(self) -> InferenceGrpcClient:
        return InferenceGrpcClient(self.endpoint, timeout=120, max_attempts=1)

    def log_tail(self, n=4000) -> str:
        self._log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:]

    def stop(self) -> dict:
        reply = self.command("stop", timeout=120)
        assert self.proc.wait(timeout=60) == 0, self.log_tail()
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self._log.close()


@pytest.fixture
def service_process(tmp_path):
    made = []

    def start(*args):
        made.append(ServiceProcess(tmp_path, *args))
        return made[-1]

    yield start
    for proc in made:
        proc.close()


def _spec(path) -> inference_pb2.ModelSpec:
    return inference_pb2.ModelSpec(
        format=inference_pb2.PYTORCH,
        source=str(path),
        model_class=rpc_models.ADD_CONST_CLASS,
        name=Path(path).name,
    )


def _array(output, shape, dtype):
    return np.frombuffer(output, dtype=np.dtype(dtype)).reshape(tuple(int(d) for d in shape))


def _predict(client, model_id, chip=CHIP):
    result = client.predict(model_id, chip)
    assert result.metadata.get("model_type") == "pytorch", result.metadata
    return _array(result.output, result.shape, result.dtype)


def _stream(client, session_id, chips, prefix):
    out = {}
    ids = [f"{prefix}{i}" for i in range(len(chips))]
    for resp in client.stream_predict(session_id=session_id, samples=chips, tile_ids=ids):
        meta = dict(resp.metadata)
        assert "error" not in meta, meta
        out[meta["tile_id"]] = _array(resp.output, resp.shape, resp.dtype)
    assert sorted(out) == sorted(ids)
    return [out[i] for i in ids]


def _open(client, session_id, **spec_fields):
    resp = client.open_session(session_id=session_id, spec=inference_pb2.ModelSpec(**spec_fields))
    assert resp.status == "ok", resp.error
    return resp


def _pytorch_model_ids(client):
    listed = client._get_stub().ListModels(inference_pb2.ListModelsRequest(), timeout=30)
    return sorted(m.model_id for m in listed.models if m.format == inference_pb2.PYTORCH)


def _worker_models(proc) -> dict:
    status = proc.command("status")
    assert "worker" in status, status
    return status["worker"]["models"]


def test_first_generic_pytorch_open_session_starts_the_worker_and_serves_both_rpcs(
    tmp_path, service_process
):
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process()
    with proc.client() as client:
        started = time.monotonic()
        opened = client.open_session(session_id="s-first", spec=_spec(weights))
        elapsed = time.monotonic() - started
        assert opened.status == "ok", f"{opened.error}\n{proc.log_tail()}"
        assert opened.model_cache_hit is False
        assert elapsed < 60.0, elapsed  # RPC_STARTUP_TIMEOUT_SECONDS bounds each start phase

        [model_id] = _pytorch_model_ids(client)
        models = _worker_models(proc)
        assert list(models) == [model_id]
        assert models[model_id]["device"] == "cpu"

        tiles = _stream(client, "s-first", [CHIP, CHIP * 2], "t")
        np.testing.assert_array_equal(tiles[0], CHIP + 1)
        np.testing.assert_array_equal(tiles[1], CHIP * 2 + 1)
        unary = _predict(client, model_id)
        assert unary.dtype == np.float32
        np.testing.assert_array_equal(unary, CHIP + 1)
        # Predict records the record's metadata: it names the worker's model.
        [entry] = [e for e in client.list_model_registry() if e.model_id == model_id]
        assert entry.metadata["rpc_fingerprint"] == models[model_id]["fingerprint"]
        assert client.close_session("s-first").status == "closed"

    stopped = proc.stop()
    assert stopped["state"] == "stopped" and stopped["children"] == [], stopped


def test_two_models_route_by_identity_across_sessions_eviction_and_reload(tmp_path, service_process):
    path_a = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    path_b = rpc_models.save_add_const(tmp_path / "b.pt", 2.0)
    path_c = rpc_models.save_add_const(tmp_path / "c.pt", 3.0)
    proc = service_process("--capacity", "2")
    with proc.client() as client:
        id_a = client.load_model_result("", _spec(path_a)).model_id
        id_b = client.load_model_result("", _spec(path_b)).model_id
        assert id_a != id_b
        models = _worker_models(proc)
        assert set(models) == {id_a, id_b}
        assert models[id_a]["fingerprint"] != models[id_b]["fingerprint"]

        _open(client, "s-a", model_id=id_a)
        _open(client, "s-b", model_id=id_b)

        # Concurrent pinned sessions, plus unary calls on both models meanwhile.
        chips = [CHIP + k for k in range(4)]
        results, errors = {}, []
        barrier = threading.Barrier(3)

        def run_session(session_id):
            try:
                with proc.client() as own:
                    barrier.wait(timeout=30)
                    results[session_id] = _stream(own, session_id, chips, f"{session_id}-")
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=run_session, args=(sid,)) for sid in ("s-a", "s-b")]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=30)
        for _ in range(3):
            np.testing.assert_array_equal(_predict(client, id_a), CHIP + 1)
            np.testing.assert_array_equal(_predict(client, id_b), CHIP + 2)
        for thread in threads:
            thread.join(timeout=120)
        assert not errors, errors
        for chip, got_a, got_b in zip(chips, results["s-a"], results["s-b"]):
            np.testing.assert_array_equal(got_a, chip + 1)
            np.testing.assert_array_equal(got_b, chip + 2)

        # Both slots pinned: a third model is refused; nothing changes beneath the sessions.
        with pytest.raises(InferenceGrpcError) as refused:
            client.load_model_result("", _spec(path_c))
        assert refused.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert set(_worker_models(proc)) == {id_a, id_b}

        # A unpinned and least recently used: loading C evicts it and frees it in the worker.
        assert client.close_session("s-a").status == "closed"
        id_c = client.load_model_result("", _spec(path_c)).model_id
        assert set(_worker_models(proc)) == {id_b, id_c}
        with pytest.raises(InferenceGrpcError) as gone:
            client.predict(id_a, CHIP)
        assert gone.value.code == grpc.StatusCode.NOT_FOUND
        np.testing.assert_array_equal(_stream(client, "s-b", [CHIP], "s-b-again-")[0], CHIP + 2)
        np.testing.assert_array_equal(_predict(client, id_c), CHIP + 3)

        # Reloading A gives a new identity (C, unpinned, is evicted; B stays pinned).
        id_a2 = client.load_model_result("", _spec(path_a)).model_id
        assert id_a2 not in (id_a, id_b, id_c)
        models = _worker_models(proc)
        assert set(models) == {id_b, id_a2}
        np.testing.assert_array_equal(_predict(client, id_a2), CHIP + 1)
        np.testing.assert_array_equal(_predict(client, id_b), CHIP + 2)

        # Unload frees the worker-side model too.
        assert client.close_session("s-b").status == "closed"
        unload = client._get_stub().UnloadModel(
            inference_pb2.UnloadModelRequest(model_id=id_b), timeout=30
        )
        assert unload.success, unload.message
        assert set(_worker_models(proc)) == {id_a2}

    stopped = proc.stop()  # the cache shutdown frees A2 before the worker stops
    assert stopped["children"] == [], stopped


def test_worker_start_failures_are_bounded_clean_up_and_allow_a_retry(tmp_path, service_process):
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process("--target", "crash", "--startup-timeout", "30")
    with proc.client() as client:
        # A child that exits before it is ready fails the first OpenSession at once.
        started = time.monotonic()
        with pytest.raises(InferenceGrpcError) as crashed:
            client.open_session(session_id="s-crash", spec=_spec(weights))
        assert time.monotonic() - started < 20.0
        assert crashed.value.code == grpc.StatusCode.INTERNAL
        assert "exited with code 3" in str(crashed.value), str(crashed.value)
        status = proc.command("status")
        assert status["state"] == "idle" and status["children"] == [], status

        # A child that never reports ready fails after the startup timeout and is terminated.
        proc.command("timeout 3")
        proc.command("target silent")
        started = time.monotonic()
        with pytest.raises(InferenceGrpcError) as silent:
            client.load_model_result("", _spec(weights))
        waited = time.monotonic() - started
        assert 3.0 <= waited < 3.0 + 3 * 2.0 + 10.0, waited  # timeout, then bounded terminate
        assert "not ready to join its RPC group within 3 s" in str(silent.value), str(silent.value)
        status = proc.command("status")
        assert status["state"] == "idle" and status["children"] == [], status

        # Neither failure entered the torch rendezvous, so the next load starts a worker.
        proc.command("timeout 60")
        proc.command("target real")
        retried = client.open_session(session_id="s-retry", spec=_spec(weights))
        assert retried.status == "ok", retried.error
        [model_id] = _pytorch_model_ids(client)
        np.testing.assert_array_equal(_predict(client, model_id), CHIP + 1)

    stopped = proc.stop()
    assert stopped["children"] == [], stopped
