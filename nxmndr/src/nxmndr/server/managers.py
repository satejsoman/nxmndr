# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Server-side managers for model lifecycle and RPC worker orchestration.

This module extracts logic from server/server.py to improve separation of concerns.
The bounded model cache and its session/execution leases live in ``model_cache``;
the lease API names stay importable from here (rebuild contract, chunk 1a).
"""

from __future__ import annotations

import multiprocessing
import time
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

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


def build_model_record(
    provider,
    model_spec,
    key: ModelCacheKey,
    metadata: Dict[str, object],
    device_plan: Optional[List[Dict[str, str]]] = None,
) -> ModelRecord:
    """Default loader: load ``model_spec``, optionally replicating to multiple devices.

    Args:
        provider: The inference provider models are loaded for
        model_spec: The model specification to load
        key: The cache key of the spec (unused here; part of the loader signature)
        metadata: Record metadata; ``metadata["model_id"]`` is the assigned model_id
        device_plan: Optional list of device dicts with 'id' keys (e.g., [{'id': 'cuda:0'}, {'id': 'cuda:1'}])
    """
    model_type = model_spec.__class__.__name__
    model_id = metadata.get("model_id", "")

    if model_type == "PytorchModelSpec":
        # Defer actual model object creation to RPC worker path
        return ModelRecord(None, "pytorch", metadata, model_spec)
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
    defaults to ``build_model_record`` bound to ``provider``; tests inject their own.
    """

    def __init__(
        self,
        provider,
        *,
        capacity: int = 10,
        session_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        loader: Optional[Callable[..., ModelRecord]] = None,
    ):
        self._provider = provider
        super().__init__(
            loader=loader if loader is not None else partial(build_model_record, provider),
            capacity=capacity,
            session_ttl_s=session_ttl_s,
            clock=clock,
        )


def _run_rpc_worker(model_spec, master_addr: str, master_port: int) -> None:
    """Process target of the PyTorch RPC worker.

    Module level so the ``spawn`` start method (the default on macOS and Windows)
    can pickle it; a nested function cannot be pickled. The worker module is
    imported in the child only.
    """

    from .rpc_worker import run_worker

    run_worker(
        "worker",
        model_spec,
        rank=1,
        world_size=2,
        master_addr=master_addr,
        master_port=master_port,
    )


class RpcWorkerManager:
    """Manage lifecycle of a single RPC worker process for PyTorch models."""

    def __init__(self):
        self._proc: Optional[multiprocessing.Process] = None

    def ensure_worker(
        self, model_spec, *, master_addr: str = "127.0.0.1", master_port: int = 29500
    ):
        if self._proc is None or not self._proc.is_alive():
            self._proc = multiprocessing.Process(
                target=_run_rpc_worker,
                args=(model_spec, master_addr, master_port),
                daemon=True,
            )
            self._proc.start()
            logger.info("RpcWorkerManager started worker process")

    def stop(self):
        if self._proc and self._proc.is_alive():
            try:
                torch_rpc.rpc_sync("worker", "_rpc_stop", args=(), timeout=2.0)
            except Exception:
                pass
            self._proc.terminate()
            self._proc.join(timeout=2)
            self._proc = None

    def shutdown(self):
        """Public shutdown hook used by atexit to ensure worker exits."""

        self.stop()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()


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
]
