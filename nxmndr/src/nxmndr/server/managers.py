# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Server-side managers for model lifecycle and RPC worker orchestration.

This module extracts logic from server/server.py to improve separation of concerns.
The bounded model cache and its session/execution leases live in ``model_cache``;
the lease API names stay importable from here (rebuild contract, chunk 1a).
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import threading
import time
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import torch.distributed as dist
import torch.distributed.rpc as torch_rpc

from ..logging import get_logger
from .model_cache import (
    SESSION_CLOSE_REASONS,
    CacheExhaustedError,
    CacheShutdownError,
    CacheStats,
    ExecutionLease,
    LoadResult,
    ModelCache,
    ModelCacheError,
    ModelCacheKey,
    ModelInUseError,
    ModelLoadError,
    ModelRecord,
    SessionClosedError,
    SessionConflictError,
    SessionLease,
    UnknownModelError,
    UnknownSessionError,
    cache_key_from_spec,
)

logger = get_logger(__name__)


def _load_model_object(model_spec, provider):
    """Construct the model for ``model_spec`` without the process-global model cache.

    ``InferenceSession`` consults ``nxmndr.memory.get_model_cache()``, a second
    LRU keyed only by spec class, path/repo and name. Going through it here would
    keep evicted models alive and hand back a model loaded for another revision
    or token, so the local path calls the registered loader directly.
    """

    from ..inference import InferenceSession  # registers the built-in loaders
    from ..models import get_loader
    from ..validation import validate_model_spec

    if hasattr(provider, "load_spec"):
        # Remote and RPC providers load through the provider, not in this process.
        return InferenceSession(model_spec, provider).model
    validate_model_spec(model_spec)
    loader = get_loader(model_spec)
    if loader is None:
        raise ValueError(f"No registered model loader for spec type: {type(model_spec).__name__}")
    return loader(model_spec, provider, SimpleNamespace(model_spec=model_spec))


def _build_pytorch_record(rpc_workers, model_spec, metadata: Dict[str, object]) -> ModelRecord:
    """A generic PyTorch record: its model lives in the RPC worker under the record's model ID.

    The record owns that worker-side model as the aux resource ``RPC_MODEL_RESOURCE``,
    so evicting, unloading or shutting down the record frees it in the worker.
    """

    if rpc_workers is None:
        raise ValueError(
            "generic PyTorch models run in the server's PyTorch RPC worker; this "
            "ModelManager has none (rpc_workers)"
        )
    model_id = str(metadata.get("model_id") or "")
    if not model_id:
        raise ValueError("a PyTorch record needs the model_id the cache assigned")
    info = rpc_workers.load(model_id, model_spec)
    try:
        record_metadata = dict(metadata)
        record_metadata["rpc_device"] = str(info.get("device", ""))
        record_metadata["rpc_fingerprint"] = str(info.get("fingerprint", ""))
        record = ModelRecord(None, "pytorch", record_metadata, model_spec)
        handle = _RpcModelHandle(rpc_workers, model_id, info)
        record._resource(RPC_MODEL_RESOURCE, lambda: handle)
    except BaseException:
        rpc_workers.unload(model_id)
        raise
    logger.info(
        "ModelManager loaded model_id=%s type=PytorchModelSpec in the RPC worker on %s (%s)",
        model_id,
        handle.device,
        handle.fingerprint,
    )
    return record


def build_model_record(
    provider,
    model_spec,
    key: ModelCacheKey,
    metadata: Dict[str, object],
    device_plan: Optional[List[Dict[str, str]]] = None,
    *,
    rpc_workers: Optional["RpcWorkerManager"] = None,
) -> ModelRecord:
    """Default loader: load ``model_spec``, optionally replicating to multiple devices.

    Args:
        provider: The inference provider models are loaded for
        model_spec: The model specification to load
        key: The cache key of the spec (unused here; part of the loader signature)
        metadata: Record metadata; ``metadata["model_id"]`` is the assigned model_id
        device_plan: Optional list of device dicts with 'id' keys (e.g., [{'id': 'cuda:0'}, {'id': 'cuda:1'}])
        rpc_workers: The server's ``RpcWorkerManager``; generic PyTorch specs load there.
    """
    model_type = model_spec.__class__.__name__
    model_id = metadata.get("model_id", "")

    if model_type == "PytorchModelSpec":
        return _build_pytorch_record(rpc_workers, model_spec, metadata)
    if model_type not in ("OnnxModelSpec", "HuggingFaceModelSpec", "UltralyticsModelSpec"):
        raise ValueError(f"Unsupported spec type {model_type}")
    model = _load_model_object(model_spec, provider)
    backend = model_type[:-9].lower()  # "onnx", "huggingface" or "ultralytics"
    record = ModelRecord(model, backend, metadata, model_spec)

    # Multi-device replication for ONNX/HuggingFace models
    if device_plan and len(device_plan) > 0:
        primary_device = device_plan[0].get("id", "cpu")
        record.device_models[primary_device] = model

        # Replicate to additional devices
        for device_info in device_plan[1:]:
            device_id = device_info.get("id", "cpu")
            try:
                if backend == "huggingface" and hasattr(model, "to"):
                    # For HuggingFace models, create a copy and move to device
                    import copy

                    device_model = copy.deepcopy(model)
                    device_model.to(device_id)
                    record.device_models[device_id] = device_model
                    logger.info(f"Replicated model {model_id} to {device_id}")
                else:
                    # For ONNX, we typically can't replicate easily
                    # Just reference the same model for now
                    record.device_models[device_id] = model
            except Exception as e:
                logger.warning(f"Failed to replicate model to {device_id}: {e}")
                record.device_models[device_id] = model
    logger.info(
        f"ModelManager loaded model_id={model_id} type={model_type} devices={list(record.device_models.keys())}"
    )
    return record


class ModelManager(ModelCache):
    """Bounded LRU of loaded models with session and execution leases.

    Supports loading models to multiple devices for parallel inference. ``loader``
    defaults to ``build_model_record`` bound to ``provider`` and ``rpc_workers`` (the
    server's ``RpcWorkerManager``, needed for generic PyTorch specs); tests inject
    their own.
    """

    def __init__(
        self,
        provider,
        *,
        capacity: int = 10,
        session_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        loader: Optional[Callable[..., ModelRecord]] = None,
        rpc_workers: Optional["RpcWorkerManager"] = None,
    ):
        self._provider = provider
        default_loader = partial(build_model_record, provider, rpc_workers=rpc_workers)
        super().__init__(
            loader=loader if loader is not None else default_loader,
            capacity=capacity,
            session_ttl_s=session_ttl_s,
            clock=clock,
        )


# The PyTorch RPC worker's names in its two-rank torch.distributed.rpc group.
RPC_DRIVER_NAME = "driver"
RPC_WORKER_NAME = "worker"
# Bound on starting the PyTorch RPC worker, applied twice: to the child's readiness
# handshake and to the torch rendezvous that follows it. 60 s is torch's default RPC
# timeout (torch.distributed.rpc.constants.DEFAULT_RPC_TIMEOUT_SEC), the bound the RPC
# group already applied to every call; a warm start took 0.5 s on macOS arm64.
RPC_STARTUP_TIMEOUT_SECONDS = 60.0
# Stopping the worker: the _rpc_stop call, the graceful RPC shutdown and each join
# of the child get this long (the values the earlier stop() used).
RPC_STOP_TIMEOUT_SECONDS = 2.0
# Aux resource name of a PyTorch record's model in the RPC worker.
RPC_MODEL_RESOURCE = "pytorch_rpc_model"


class RpcWorkerStartError(RuntimeError):
    """The PyTorch RPC worker could not be started, or can no longer be used."""


# torch.distributed and torch.distributed.rpc are process-global, and a process
# cannot join a second group after its first (torch numbers its store prefixes per
# process, so a fresh child never meets a second attempt). So one manager per
# process may enter the torch rendezvous, once.
_PROCESS_LOCK = threading.Lock()
_process_owner: Optional["RpcWorkerManager"] = None


def _free_port(addr: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((addr, 0))
        return sock.getsockname()[1]


def _terminate(proc) -> None:
    """End a child process: wait, terminate, then kill, each bounded."""

    if proc is None or proc.pid is None:  # never started
        return
    proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)


def _run_rpc_worker(
    master_addr: str, master_port: int, ready_port: int, startup_timeout_s: float
) -> None:
    """Process target of the PyTorch RPC worker.

    Module level so the ``spawn`` start method can pickle it; a nested function
    cannot be pickled. The worker module is imported in the child only. The worker
    starts with no model: the server loads each model by its model ID.
    """

    from .rpc_worker import run_worker

    run_worker(
        RPC_WORKER_NAME,
        None,
        rank=1,
        world_size=2,
        master_addr=master_addr,
        master_port=master_port,
        ready_port=ready_port,
        startup_timeout_s=startup_timeout_s,
    )


class _RpcModelHandle:
    """Aux resource of a PyTorch ModelRecord: its model in the RPC worker.

    ``dispose()`` (called once, when the record is evicted, unloaded or shut down)
    frees that model in the worker.
    """

    def __init__(self, workers: "RpcWorkerManager", model_id: str, info: Dict[str, object]):
        self.workers = workers
        self.model_id = model_id
        self.device = str(info.get("device", ""))
        self.fingerprint = str(info.get("fingerprint", ""))

    def infer(self, tensor, *, device_hint: str, timeout: float):
        return self.workers.infer(self.model_id, tensor, device_hint=device_hint, timeout=timeout)

    def dispose(self) -> None:
        self.workers.unload(self.model_id)


class RpcWorkerManager:
    """The PyTorch RPC worker of one server: start, per-model load/infer/unload, stop.

    Start order: the child process starts first and reports over a loopback socket
    that it is about to rendezvous; only then does this process join the two-rank
    group as the driver, so both ranks are present. Each phase is bounded by
    ``startup_timeout_s``. A child that exits or never reports fails the load with
    ``RpcWorkerStartError`` and is terminated; a later load tries again. A failure
    inside the torch rendezvous also terminates the child, but torch cannot rejoin a
    group in this process, so later loads fail until the server restarts; so does a
    worker that exits after it started.
    """

    # The child's process target; tests replace it with a module-level function.
    worker_target = staticmethod(_run_rpc_worker)

    def __init__(
        self,
        *,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        startup_timeout_s: float = RPC_STARTUP_TIMEOUT_SECONDS,
    ):
        self._lock = threading.Lock()
        self._proc = None
        self._state = "idle"  # idle -> running -> stopped; "broken" after a failed rendezvous
        self._error = ""
        self._addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
        self._port = int(master_port or os.environ.get("MASTER_PORT", "0") or 0)
        self.startup_timeout_s = float(startup_timeout_s)

    # ---- start ------------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the worker and join its RPC group once; raise ``RpcWorkerStartError``."""

        with self._lock:
            if self._state == "running":
                if self._proc is not None and self._proc.is_alive():
                    return
                raise RpcWorkerStartError(
                    "the PyTorch RPC worker exited (exit code "
                    f"{getattr(self._proc, 'exitcode', None)}); torch cannot rejoin an RPC "
                    "group in this process, so restart the server to use PyTorch models"
                )
            if self._state == "stopped":
                raise RpcWorkerStartError(
                    "the PyTorch RPC worker was stopped; restart the server to use PyTorch models"
                )
            if self._state == "broken":
                raise RpcWorkerStartError(
                    f"the PyTorch RPC group of this server failed to start ({self._error}); "
                    "restart the server to use PyTorch models"
                )
            self._start_locked()

    def _start_locked(self) -> None:
        global _process_owner
        with _PROCESS_LOCK:
            if _process_owner is not None and _process_owner is not self:
                raise RpcWorkerStartError(
                    "another server in this process already runs the PyTorch RPC worker; "
                    "torch allows one RPC group per process"
                )
            if dist.is_initialized() or torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
                raise RpcWorkerStartError(
                    "this process already belongs to a torch.distributed group; the PyTorch "
                    "RPC worker needs a process of its own"
                )
            timeout = self.startup_timeout_s
            addr = self._addr
            port = self._port or _free_port(addr)
            listener = socket.create_server((addr, 0))
            listener.settimeout(0.1)
            # spawn: the child must not inherit this process's gRPC and torch threads.
            proc = multiprocessing.get_context("spawn").Process(
                target=type(self).worker_target,
                args=(addr, port, listener.getsockname()[1], timeout),
                daemon=True,
                name="nxmndr-pytorch-rpc-worker",
            )
            try:
                proc.start()
                self._await_ready(proc, listener, timeout)
            except BaseException:
                _terminate(proc)
                raise
            finally:
                listener.close()
            _process_owner = self  # from here on this process has used its one attempt
            init_method = f"tcp://{addr}:{port}"
            try:
                dist.init_process_group(
                    backend="gloo",
                    rank=0,
                    world_size=2,
                    init_method=init_method,
                    timeout=timedelta(seconds=timeout),
                )
                opts = torch_rpc.TensorPipeRpcBackendOptions(
                    init_method=init_method, rpc_timeout=timeout
                )
                torch_rpc.init_rpc(RPC_DRIVER_NAME, rank=0, world_size=2, rpc_backend_options=opts)
            except BaseException as exc:
                self._state = "broken"
                self._error = f"{type(exc).__name__}: {exc}"
                _terminate(proc)
                self._leave_group(graceful=False)
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker did not complete the rendezvous at {init_method} "
                    f"within {timeout:g} s: {self._error}"
                ) from exc
            self._proc = proc
            self._state = "running"
        logger.info("PyTorch RPC worker pid=%s joined the RPC group at %s", proc.pid, init_method)

    @staticmethod
    def _await_ready(proc, listener, timeout: float) -> None:
        """Wait for the child's ``ready <pid>`` line; fail fast if it exits."""

        deadline = time.monotonic() + timeout
        while True:
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                if not proc.is_alive():
                    raise RpcWorkerStartError(
                        f"the PyTorch RPC worker exited with code {proc.exitcode} before it "
                        "was ready to join its RPC group"
                    ) from None
                if time.monotonic() >= deadline:
                    raise RpcWorkerStartError(
                        f"the PyTorch RPC worker was not ready to join its RPC group within "
                        f"{timeout:g} s"
                    ) from None
                continue
            with conn:
                conn.settimeout(max(0.1, deadline - time.monotonic()))
                try:
                    line = conn.recv(64).decode("ascii", "replace").split()
                except OSError:
                    line = []
            if line == ["ready", str(proc.pid)]:
                return

    # ---- models -------------------------------------------------------------

    def load(self, model_id: str, model_spec) -> Dict[str, object]:
        """Build ``model_spec`` in the worker under ``model_id`` (starting the worker first).

        Returns the worker's ``{"model_id", "device", "fingerprint"}``.
        """

        model_class = getattr(model_spec, "model_class", None)
        if isinstance(model_class, str) or not callable(model_class):
            raise ValueError(
                f"PyTorch model_class {model_class!r} is not registered in this server "
                "(nxmndr.models.register_pytorch_model)"
            )
        self.ensure_started()
        from .rpc_worker import _rpc_load

        payload = {
            "model_class": model_class,
            "model_path": model_spec.model_path,
            "name": getattr(model_spec, "name", None),
        }
        return torch_rpc.rpc_sync(RPC_WORKER_NAME, _rpc_load, args=(model_id, payload))

    def infer(self, model_id: str, tensor, *, device_hint: str = "cpu", timeout: float):
        """Run the worker's model ``model_id`` (``rpc_worker.UnknownModelError`` if absent)."""

        if not self.is_alive():
            raise RpcWorkerStartError("the PyTorch RPC worker is not running")
        from .rpc_worker import _rpc_infer

        return torch_rpc.rpc_sync(
            RPC_WORKER_NAME, _rpc_infer, args=(model_id, tensor, device_hint), timeout=timeout
        )

    def unload(self, model_id: str) -> bool:
        """Free the worker's model ``model_id``. False if the worker or the model is gone."""

        if not self.is_alive():
            logger.info("PyTorch RPC worker not running; nothing to free for model %s", model_id)
            return False
        from .rpc_worker import _rpc_unload

        try:
            return bool(torch_rpc.rpc_sync(RPC_WORKER_NAME, _rpc_unload, args=(model_id,)))
        except Exception:
            logger.warning("freeing model %s in the PyTorch RPC worker failed", model_id, exc_info=True)
            return False

    def status(self) -> Dict[str, object]:
        """The worker's ``_rpc_status()``: state, loaded model IDs with device and fingerprint."""

        if not self.is_alive():
            raise RpcWorkerStartError("the PyTorch RPC worker is not running")
        from .rpc_worker import _rpc_status

        return torch_rpc.rpc_sync(RPC_WORKER_NAME, _rpc_status, args=())

    # ---- stop ---------------------------------------------------------------

    @staticmethod
    def _leave_group(*, graceful: bool) -> None:
        try:
            if torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
                torch_rpc.shutdown(graceful=graceful, timeout=RPC_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("PyTorch RPC shutdown failed", exc_info=True)
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            logger.warning("destroying the torch.distributed process group failed", exc_info=True)

    def stop(self):
        with self._lock:
            proc, self._proc = self._proc, None
            if self._state == "running":
                self._state = "stopped"
                alive = proc is not None and proc.is_alive()
                if alive:
                    from .rpc_worker import _rpc_stop

                    try:
                        torch_rpc.rpc_sync(
                            RPC_WORKER_NAME, _rpc_stop, args=(), timeout=RPC_STOP_TIMEOUT_SECONDS
                        )
                    except Exception:
                        alive = False
                        logger.warning(
                            "_rpc_stop failed; terminating the PyTorch RPC worker", exc_info=True
                        )
                self._leave_group(graceful=alive)
        _terminate(proc)

    def shutdown(self):
        """Public shutdown hook used by atexit to ensure worker exits."""

        self.stop()

    def is_alive(self) -> bool:
        proc = self._proc
        return self._state == "running" and proc is not None and proc.is_alive()


def shutdown_process_rpc_worker() -> None:
    """Stop the PyTorch RPC worker this process started, if any (the server's atexit hook)."""

    owner = _process_owner
    if owner is not None:
        owner.shutdown()


__all__ = [
    "SESSION_CLOSE_REASONS",
    "ModelCacheError",
    "CacheExhaustedError",
    "ModelInUseError",
    "UnknownModelError",
    "UnknownSessionError",
    "SessionConflictError",
    "SessionClosedError",
    "ModelLoadError",
    "CacheShutdownError",
    "ModelCacheKey",
    "cache_key_from_spec",
    "ModelRecord",
    "LoadResult",
    "SessionLease",
    "ExecutionLease",
    "CacheStats",
    "ModelManager",
    "build_model_record",
    "RpcWorkerManager",
    "RpcWorkerStartError",
    "RPC_MODEL_RESOURCE",
    "shutdown_process_rpc_worker",
]
