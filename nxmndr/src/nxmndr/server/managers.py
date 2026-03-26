# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Server-side managers for model lifecycle and RPC worker orchestration.

This module extracts logic from server/server.py to improve separation of concerns.
"""

from __future__ import annotations

import multiprocessing
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.distributed.rpc as torch_rpc

from ..inference import InferenceSession
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRecord:
    model: Optional[object]
    backend: str
    metadata: Dict[str, object]
    spec: Optional[object] = None
    # Multi-device support: device_id -> model instance
    device_models: Dict[str, object] = field(default_factory=dict)


class ModelManager:
    """Manage model specifications, loading, unloading, and metadata tracking.

    Supports loading models to multiple devices for parallel inference.
    """

    def __init__(self, provider):
        self._provider = provider
        self._models: Dict[str, ModelRecord] = {}
        self._temp_artifacts: set[str] = set()

    def load_spec(
        self,
        model_spec,
        metadata: Optional[Dict[str, object]] = None,
        device_plan: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Load a model spec, optionally replicating to multiple devices.

        Args:
            model_spec: The model specification to load
            metadata: Optional metadata dict
            device_plan: Optional list of device dicts with 'id' keys (e.g., [{'id': 'cuda:0'}, {'id': 'cuda:1'}])

        Returns:
            The assigned model_id
        """
        model_type = model_spec.__class__.__name__
        model_id = uuid.uuid4().hex
        record_meta = dict(metadata or {})
        record_meta.setdefault("model_id", model_id)

        if model_type == "PytorchModelSpec":
            # Defer actual model object creation to RPC worker path
            self._models[model_id] = ModelRecord(None, "pytorch", record_meta, model_spec)
        elif model_type in ("OnnxModelSpec", "HuggingFaceModelSpec"):
            session = InferenceSession(model_spec, self._provider)
            backend = model_type[:-9].lower()
            record = ModelRecord(session.model, backend, record_meta, model_spec)

            # Multi-device replication for ONNX/HuggingFace models
            if device_plan and len(device_plan) > 0:
                primary_device = device_plan[0].get("id", "cpu")
                record.device_models[primary_device] = session.model

                # Replicate to additional devices
                for device_info in device_plan[1:]:
                    device_id = device_info.get("id", "cpu")
                    try:
                        if backend == "huggingface" and hasattr(session.model, "to"):
                            # For HuggingFace models, create a copy and move to device
                            import copy

                            device_model = copy.deepcopy(session.model)
                            device_model.to(device_id)
                            record.device_models[device_id] = device_model
                            logger.info(f"Replicated model {model_id} to {device_id}")
                        else:
                            # For ONNX, we typically can't replicate easily
                            # Just reference the same model for now
                            record.device_models[device_id] = session.model
                    except Exception as e:
                        logger.warning(f"Failed to replicate model to {device_id}: {e}")
                        record.device_models[device_id] = session.model

            self._models[model_id] = record
        else:
            raise ValueError(f"Unsupported spec type {model_type}")
        logger.info(
            f"ModelManager loaded model_id={model_id} type={model_type} devices={list((self._models[model_id].device_models or {}).keys())}"
        )
        return model_id

    def get_model_for_device(self, model_id: str, device_id: str) -> Optional[object]:
        """Get the model instance for a specific device.

        Falls back to primary model if device-specific instance not available.
        """
        record = self._models.get(model_id)
        if record is None:
            return None

        # Try device-specific model first
        if record.device_models and device_id in record.device_models:
            return record.device_models[device_id]

        # Fall back to primary model
        return record.model

    def register_temp_artifact(self, path: str) -> None:
        self._temp_artifacts.add(path)

    def unload(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            remaining_onnx = any(rec.backend == "onnx" for rec in self._models.values())
            if not remaining_onnx:
                self._cleanup_temp_artifacts()
            return True
        return False

    def _cleanup_temp_artifacts(self):
        for p in list(self._temp_artifacts):
            try:
                Path(p).unlink(missing_ok=True)
                logger.debug(f"Cleaned temp artifact {p}")
            except OSError:
                pass
            finally:
                self._temp_artifacts.discard(p)

    def list_models(self):
        return self._models.copy()

    def get(self, model_id: str):
        return self._models.get(model_id)


class RpcWorkerManager:
    """Manage lifecycle of a single RPC worker process for PyTorch models."""

    def __init__(self):
        self._proc: Optional[multiprocessing.Process] = None

    def ensure_worker(
        self, model_spec, *, master_addr: str = "127.0.0.1", master_port: int = 29500
    ):
        if self._proc is None or not self._proc.is_alive():

            def _start():
                from .rpc_worker import run_worker

                run_worker(
                    "worker",
                    model_spec,
                    rank=1,
                    world_size=2,
                    master_addr=master_addr,
                    master_port=master_port,
                )

            self._proc = multiprocessing.Process(target=_start, daemon=True)
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
