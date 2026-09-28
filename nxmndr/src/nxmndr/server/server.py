# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NXMNDR gRPC Inference Server.

The default execution path runs the Torch/ONNX inference engine over gRPC on port 50051.
This is the recommended deployment mode and works out-of-the-box with logging levels
controlled by the ``PYTHONLOGLEVEL`` environment variable.

Optional components:
- Azure OpenAI proxy HTTP server (aiohttp) enabled via ``--azure-proxy``.

Examples:
    python server/server.py                             # gRPC inference server on port 50051
    python server/server.py --grpc-port 6000            # gRPC server on custom port
    python server/server.py --azure-proxy               # Azure proxy HTTP server on port 8080
    python server/server.py --azure-proxy --http-port 9000  # Azure proxy HTTP server on port 9000

Streaming (``StreamPredict``) contract v1:

- A session stream names a session opened with ``OpenSession`` in
  ``context["session_id"]``; an unknown ID fails the stream with NOT_FOUND. A
  sessionless stream names ``model_id`` instead.
- Every message of a tile carries ``context["tile_id"]``; the tile's first message
  may carry per-tile options as ``context["opt.<name>"]``. Effective options are the
  session options updated by that tile's own options.
- Each tile gets exactly one response. Response metadata carries ``error``,
  ``error_code`` and ``error_scope`` on failure: ``tile`` scope fails one tile and the
  stream continues; ``stream`` scope ends the stream.
- Limits: a message chunk over ``max_chunk_bytes`` and a tile over ``max_tile_bytes``
  are tile-scope failures. The number of chunks per tile is not limited. The server
  processes tiles in arrival order and answers each before reading on, so a client
  window of ``max_inflight`` outstanding tiles cannot deadlock.
"""

import asyncio
import hashlib
import json
import os
import inspect
import logging
import sys
import threading
import time
import uuid
import random
import re
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import grpc
import atexit
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as torch_rpc
from aiohttp import web

from ..logging import get_logger

from ..inference import LocalInferenceProvider, inference_pb2, inference_pb2_grpc
from ..models import Model
from ..models.sam import resolve_sam_capability
from . import dispatch
from . import managers
from .azure_openai_proxy import AzureOpenAIProxy, setup_azure_proxy_routes
from .managers import RpcWorkerManager

# The model cache and session/execution lease API (ModelManager, cache_key_from_spec
# and the ModelCacheError family) is used through the ``managers`` module at call
# time, so processes that import this package only for other modules (for example
# the PyTorch RPC worker) do not depend on it.

if __package__ in {None, ""}:  # pragma: no cover - direct script execution support
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from nxmndr.nxmndr.server.registry import ModelRegistryStore  # type: ignore[import]
else:
    from .registry import ModelRegistryStore  # type: ignore[import]

# Logging is initialized via nxmndr.logging; PYTHONLOGLEVEL controls the default level.
logger = get_logger("nxmndr.server")

GRPC_MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128MB ceiling for large tensor payloads
RPC_READY_TIMEOUT_SECONDS = 5.0  # Timeout for RPC worker readiness checks
RPC_INFER_TIMEOUT_SECONDS = 5.0  # Timeout for individual RPC inference calls
DEFAULT_MODEL_CACHE_CAPACITY = 10  # plan: LRU model cache capacity (NXMNDR_MODEL_CACHE_CAPACITY)
SERVER_STOP_GRACE_SECONDS = 5.0  # grace for in-flight RPCs, then cache shutdown drains leases
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]


def _shutdown_rpc_driver_if_needed():
    """Best-effort RPC shutdown to avoid hanging pytest due to TensorPipe threads."""

    try:
        if torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
            torch_rpc.shutdown()
    except Exception:
        pass
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Swallow to avoid masking teardown
        pass

    try:
        # Attempt to stop worker process if manager exists on any instantiated service
        svc = getattr(_shutdown_rpc_driver_if_needed, "_svc_ref", None)
        if svc and getattr(svc, "rpc_manager", None):
            svc.rpc_manager.shutdown()
    except Exception:
        pass


atexit.register(_shutdown_rpc_driver_if_needed)


def _await_rpc_worker_ready(timeout: float = 5.0, poll: float = 0.1) -> bool:
    """Poll the RPC worker status until it reports ready or timeout."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            from .rpc_worker import _rpc_status

            resp = torch_rpc.rpc_sync("worker", _rpc_status, args=(), timeout=poll)
            if resp.get("status") == "ok" and resp.get("model_loaded"):
                return True
        except Exception:
            time.sleep(poll)
    return False


# Invocation metadata keys whose values are never logged (credential-like).
_CREDENTIAL_METADATA_KEY = re.compile(
    r"authorization|cookie|token|key|secret|password", re.IGNORECASE
)


def _loggable_metadata(metadata) -> Dict[str, object]:
    """Invocation metadata with the value of every credential-like key redacted."""

    return {
        key: "<redacted>" if _CREDENTIAL_METADATA_KEY.search(key) else value
        for key, value in (metadata or [])
    }


class LoggingInterceptor(grpc.ServerInterceptor):
    """Interceptor to log all incoming gRPC requests (credential values redacted)."""

    def intercept_service(self, continuation, handler_call_details):
        logger.debug(
            "gRPC request: method=%s, metadata=%s",
            handler_call_details.method,
            _loggable_metadata(handler_call_details.invocation_metadata),
        )
        return continuation(handler_call_details)


class _LoadRequestError(Exception):
    """A ModelSpec that cannot be loaded as given (maps to a gRPC status)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class _PreparedLoad:
    model_spec: object
    key: object
    metadata: Dict[str, object]
    device_plan: Optional[List[Dict[str, object]]]
    model_type: str


@dataclass
class _SessionBook:
    """Server-side bookkeeping for one session: negotiated limits, options, summary.

    The session's lifecycle (open/closed, model pin) lives in the ModelManager.
    """

    session_id: str
    model_id: str
    max_inflight: int
    max_chunk_bytes: int
    max_tile_bytes: int
    options: Dict[str, str]
    total_tiles: Optional[int]
    created_at: float
    ok_tiles: int = 0
    failed_tiles: int = 0
    errors: List[str] = field(default_factory=list)
    seen_tiles: set = field(default_factory=set)
    closed_seen_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class _TileBuffer:
    """Bytes and first-message metadata of the tile currently being received."""

    def __init__(self, tile_id: str):
        self.tile_id = tile_id
        self.data = bytearray()
        self.shape = None
        self.dtype = None
        self.options: Dict[str, str] = {}
        self.reported = False  # a (failure) response was already sent for this tile


class _StreamEnd(Exception):
    """Internal: the stream ends after the given stream-scope response."""

    def __init__(self, response, *, status=None, details=""):
        super().__init__(details)
        self.response = response
        self.status = status
        self.details = details


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(
        self,
        *,
        model_cache_dir: str | Path | None = None,
        max_cores: int = 4,
        model_manager=None,
    ):
        # Cap max_cores at a minimum of 1
        self._max_cores = max(1, int(max_cores or 4))

        self.provider = LocalInferenceProvider()
        self._session_ttl_seconds = int(os.environ.get("NXMNDR_SESSION_TTL_SECONDS", "3600"))
        self._model_cache_capacity = int(
            os.environ.get("NXMNDR_MODEL_CACHE_CAPACITY", str(DEFAULT_MODEL_CACHE_CAPACITY))
        )
        self.model_manager = model_manager or managers.ModelManager(
            self.provider,
            capacity=self._model_cache_capacity,
            session_ttl_s=float(self._session_ttl_seconds),
        )
        self.rpc_manager = RpcWorkerManager()
        self.logger = logger
        self.rpc_worker = None  # deprecated in favor of rpc_manager
        self._rpc_driver_initialized = False
        self._rpc_master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self._rpc_master_port = int(
            os.environ.get("MASTER_PORT", str(random.randint(40000, 50000)))
        )
        # Allow atexit to access this instance for cleanup
        setattr(_shutdown_rpc_driver_if_needed, "_svc_ref", self)
        artifact_root = None
        if model_cache_dir:
            expanded = os.path.expandvars(os.path.expanduser(str(model_cache_dir)))
            artifact_root = Path(expanded)
        self.registry_store = ModelRegistryStore(artifact_root=artifact_root)
        self._model_cache_dir = self.registry_store.artifact_root()

        # Set global model cache directory on provider config for server-wide caching
        # All model types (HuggingFace, TorchHub, PyTorch, ONNX) can use subdirectories
        if self._model_cache_dir:
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.provider.config.inference.model_cache_dir = str(self._model_cache_dir)
            self.logger.info("Model cache directory: %s", self._model_cache_dir)

        # Device planning: prefer GPUs up to max_cores, fall back to CPU threads otherwise.
        self._device_plan = []
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            selected = max(1, min(self._max_cores, gpu_count))
            self._device_plan = [
                {"type": "gpu", "ordinal": idx, "id": f"cuda:{idx}"} for idx in range(selected)
            ]
        else:
            cpu_count = os.cpu_count() or 1
            selected = max(1, min(self._max_cores, cpu_count))
            self._device_plan = [
                {"type": "cpu", "ordinal": idx, "id": f"cpu:{idx}"} for idx in range(selected)
            ]
        self._device_cursor = 0
        # Session bookkeeping (limits, options, summary); lifecycle lives in model_manager.
        self._sessions: Dict[str, _SessionBook] = {}
        self._sessions_lock = threading.Lock()
        self._last_session_cleanup = time.time()
        self._default_max_inflight = int(os.environ.get("NXMNDR_STREAM_MAX_INFLIGHT", "16"))
        self._default_chunk_bytes = int(
            os.environ.get("NXMNDR_STREAM_MAX_CHUNK_BYTES", str(8 * 1024 * 1024))
        )
        self._default_tile_bytes = int(
            os.environ.get("NXMNDR_STREAM_MAX_TILE_BYTES", str(GRPC_MAX_MESSAGE_LENGTH))
        )

    def _cleanup_stale_sessions(self) -> int:
        """Expire idle sessions in the model manager and drop stale bookkeeping.

        Returns the number of sessions expired by this call.
        """
        now = time.time()
        # Only run cleanup every 60 seconds at most
        if now - self._last_session_cleanup < 60:
            return 0
        self._last_session_cleanup = now

        expired = self.model_manager.expire_sessions()
        for sid in expired:
            self.logger.info(f"Expired idle session: {sid}")

        with self._sessions_lock:
            books = list(self._sessions.items())
        for sid, book in books:
            if self._session_state(sid) == "open":
                continue
            if book.closed_seen_at is None:
                book.closed_seen_at = now
            elif now - book.closed_seen_at > self._session_ttl_seconds:
                with self._sessions_lock:
                    self._sessions.pop(sid, None)
        return len(expired)

    def _session_state(self, session_id: str) -> str:
        try:
            return self.model_manager.session_state(session_id)
        except managers.UnknownSessionError:
            return "unknown"

    def _next_device(self):
        if not self._device_plan:
            return {"type": "cpu", "ordinal": 0, "id": "cpu:0"}
        device = self._device_plan[self._device_cursor % len(self._device_plan)]
        self._device_cursor += 1
        return device

    def _device_plan_messages(self):
        return [
            inference_pb2.DeviceInfo(
                id=d.get("id", ""),
                type=d.get("type", ""),
                ordinal=int(d.get("ordinal", 0)),
                weight=int(d.get("weight", 1)),
            )
            for d in self._device_plan
        ]

    # ---- Lifecycle ----
    def _prepare_load(self, spec_msg, corr_id: str) -> _PreparedLoad:
        """Turn a wire ModelSpec into what ModelManager.load/open_session need.

        Raises _LoadRequestError for requests that cannot be loaded as given.
        """
        if spec_msg.format not in (
            inference_pb2.PYTORCH,
            inference_pb2.ONNX,
            inference_pb2.HUGGINGFACE,
            inference_pb2.TORCHHUB,
        ):
            raise _LoadRequestError(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported model format enum")
        # Map enum int -> registry type string expected by model_spec_from_json
        format_map = {
            inference_pb2.PYTORCH: "pytorch",
            inference_pb2.ONNX: "onnx",
            inference_pb2.HUGGINGFACE: "huggingface",
            inference_pb2.TORCHHUB: "torchhub",
        }
        fmt_str = format_map[spec_msg.format]
        # Determine artifact path (write ONNX artifact bytes if provided); special handling for huggingface
        model_path = spec_msg.source or ""
        metadata_updates = {}
        artifact_sha = ""
        if spec_msg.artifact:
            artifact_bytes = spec_msg.artifact
            artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
            existing_entry = self.registry_store.find_by_artifact_sha(artifact_sha)
            cached_path = None
            if existing_entry:
                cached_candidate = existing_entry.metadata.get("artifact_path")
                if cached_candidate and Path(cached_candidate).exists():
                    cached_path = Path(cached_candidate)
            if cached_path is None:
                ext_hint = {
                    "onnx": "onnx",
                    "pytorch": "pt",
                    "torchhub": "pt",
                    "huggingface": "bin",
                }.get(fmt_str, "bin")
                target_path = self.registry_store.resolve_artifact_path(artifact_sha, ext_hint)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.exists() or target_path.stat().st_size != len(artifact_bytes):
                    target_path.write_bytes(artifact_bytes)
                cached_path = target_path
            model_path = str(cached_path)
            metadata_updates["artifact_path"] = str(cached_path)
            metadata_updates["artifact_sha256"] = artifact_sha
            if spec_msg.checksum:
                metadata_updates["checksum"] = spec_msg.checksum
        if fmt_str == "huggingface":
            # For HuggingFace we serialize repo_id explicitly (avoid misuse of model_path)
            # Let load_huggingface and AutoModel.from_pretrained handle weight file resolution
            if not spec_msg.source:
                raise _LoadRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT, "HuggingFace models require source (repo_id)"
                )
            spec_dict = {
                "type": "huggingface",
                "repo_id": spec_msg.source,
                "name": spec_msg.name or spec_msg.source,
            }
            # Same normalization as cache_key_from_spec, so the key and the load agree.
            revision = (spec_msg.version or "").strip()
            if revision:
                spec_dict["revision"] = revision
            token = (
                spec_msg.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            )
            if token:
                # The token lives only in the spec (HuggingFaceModelSpec.token, repr=False),
                # never in record metadata, cache keys, the registry file or logs.
                spec_dict["token"] = token
                self.logger.debug("HuggingFace token provided", extra={"corr_id": corr_id})
            else:
                self.logger.warning(
                    "No HuggingFace token in request for gated repo %s",
                    spec_msg.source,
                    extra={"corr_id": corr_id},
                )
        else:
            spec_dict = {
                "type": fmt_str,
                "model_path": model_path,
                "name": spec_msg.name or "",
            }
            if fmt_str in ("pytorch", "torchhub") and spec_msg.model_class:
                spec_dict["model_class"] = spec_msg.model_class
        if fmt_str == "pytorch":
            # Start RPC driver before spawning worker to avoid rendezvous hangs
            try:
                self._ensure_rpc_driver()
                if not _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS):
                    # Worker might not yet be started; allow LoadModel to continue and worker to start.
                    pass
            except Exception as exc:
                raise _LoadRequestError(
                    grpc.StatusCode.INTERNAL, "failed to initialize RPC driver"
                ) from exc

        model_spec = Model.model_spec_from_json(json.dumps(spec_dict))
        model_type = model_spec.__class__.__name__
        try:
            task_name = inference_pb2.TaskType.Name(spec_msg.task)
        except ValueError:
            task_name = "TASK_TYPE_UNSPECIFIED"
        model_metadata = {
            "task": int(spec_msg.task),
            "task_name": task_name,
            "format": fmt_str,
            "name": spec_msg.name or "",
            "source": spec_msg.source or "",
        }
        model_metadata.update(metadata_updates)
        if model_type == "PytorchModelSpec":
            self.rpc_manager.ensure_worker(
                model_spec,
                master_addr=self._rpc_master_addr,
                master_port=self._rpc_master_port,
            )
            device_plan = None
        elif model_type in ("OnnxModelSpec", "HuggingFaceModelSpec"):
            # Pass device plan for multi-device model replication
            device_plan = self._device_plan
        else:
            raise _LoadRequestError(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported spec type")
        key = managers.cache_key_from_spec(spec_msg, artifact_sha256=artifact_sha)
        return _PreparedLoad(model_spec, key, model_metadata, device_plan, model_type)

    @staticmethod
    def _cache_error_status(exc):
        if isinstance(exc, managers.CacheExhaustedError):
            return grpc.StatusCode.RESOURCE_EXHAUSTED
        if isinstance(exc, managers.ModelInUseError):
            return grpc.StatusCode.FAILED_PRECONDITION
        if isinstance(exc, managers.CacheShutdownError):
            return grpc.StatusCode.UNAVAILABLE
        if isinstance(exc, managers.SessionConflictError):
            return grpc.StatusCode.ALREADY_EXISTS
        if isinstance(exc, managers.SessionClosedError):
            return grpc.StatusCode.FAILED_PRECONDITION
        if isinstance(exc, (managers.UnknownModelError, managers.UnknownSessionError)):
            return grpc.StatusCode.NOT_FOUND
        return grpc.StatusCode.INTERNAL

    def _capability_metadata(self, model_id: str):
        record = self.model_manager.get(model_id)
        if record is None:
            return []
        capability = resolve_sam_capability(record.model, record.spec)
        if capability is None:
            return []
        return [inference_pb2.MetadataEntry(key="capability.sam", value=capability.family)]

    def LoadModel(self, request, context):
        corr_id = uuid.uuid4().hex[:8]
        self.logger.info("LoadModel called", extra={"corr_id": corr_id})
        try:
            prepared = self._prepare_load(request.spec, corr_id)
            result = self.model_manager.load(
                prepared.model_spec,
                key=prepared.key,
                metadata=prepared.metadata,
                device_plan=prepared.device_plan,
                overwrite=bool(request.overwrite),
            )
            model_id = result.model_id
            self.logger.info(
                f"Model loaded model_id={model_id} type={prepared.model_type} "
                f"cache_hit={result.cache_hit}",
                extra={"corr_id": corr_id},
            )
            self._record_registry_entry(model_id, prepared.metadata)
            return inference_pb2.LoadModelResponse(
                success=True,
                model_id=model_id,
                message="loaded",
                effective_metadata=[
                    inference_pb2.MetadataEntry(
                        key="model_cache_hit", value=str(result.cache_hit).lower()
                    )
                ]
                + self._capability_metadata(model_id),
            )
        except _LoadRequestError as e:
            context.set_code(e.code)
            context.set_details(str(e))
            return inference_pb2.LoadModelResponse(success=False, model_id="", message=str(e))
        except managers.ModelCacheError as e:
            self.logger.warning("LoadModel failed: %s", e, extra={"corr_id": corr_id})
            context.set_code(self._cache_error_status(e))
            context.set_details(str(e))
            return inference_pb2.LoadModelResponse(success=False, model_id="", message=str(e))
        except Exception as e:
            self.logger.exception("LoadModel failed", extra={"corr_id": corr_id})
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.LoadModelResponse(success=False, model_id="", message=str(e))

    def UnloadModel(self, request, context):
        mid = request.model_id
        try:
            unloaded = self.model_manager.unload(mid)
        except managers.ModelInUseError as e:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(e))
            return inference_pb2.UnloadModelResponse(success=False, message=f"model in use: {e}")
        if unloaded:
            return inference_pb2.UnloadModelResponse(success=True, message=f"unloaded {mid}")
        return inference_pb2.UnloadModelResponse(success=False, message="unknown model_id")

    def ListModels(self, request, context):
        infos = []
        for mid, record in self.model_manager.list_models().items():
            model_obj = record.model
            mtype = record.backend
            metadata = record.metadata or {}
            format_enum = inference_pb2.MODEL_FORMAT_UNSPECIFIED
            if mtype == "pytorch":
                format_enum = inference_pb2.PYTORCH
            elif mtype == "onnx":
                format_enum = inference_pb2.ONNX
            elif mtype == "huggingface":
                format_enum = inference_pb2.HUGGINGFACE
            task_enum = inference_pb2.TASK_TYPE_UNSPECIFIED
            task_value = metadata.get("task")
            if task_value is not None:
                try:
                    if isinstance(task_value, int):
                        task_enum = task_value
                    elif isinstance(task_value, str):
                        task_enum = inference_pb2.TaskType.Value(task_value.upper())
                    else:
                        task_enum = inference_pb2.TaskType.Value(str(task_value))
                except ValueError:
                    task_enum = inference_pb2.TASK_TYPE_UNSPECIFIED
            infos.append(
                inference_pb2.ModelInfo(
                    model_id=mid,
                    name=mtype,
                    format=format_enum,
                    task=task_enum,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    loaded=model_obj is not None or mtype == "pytorch",
                )
            )
        return inference_pb2.ListModelsResponse(models=infos)

    # ---- Prediction ----
    def _ensure_pytorch_worker(self, record) -> Optional[str]:
        """Make the PyTorch RPC worker usable; return an error label if it is not."""
        if record.backend != "pytorch":
            return None
        if not self._rpc_driver_initialized:
            self._ensure_rpc_driver()
            if not _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS):
                return "rpc_worker_not_ready"
        if not self.rpc_manager.is_alive():
            try:
                self.rpc_manager.ensure_worker(
                    record.spec,
                    master_addr=self._rpc_master_addr,
                    master_port=self._rpc_master_port,
                )
                _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS)
            except Exception:
                return "rpc_worker_not_available"
        return None

    def _dispatch(self, lease, input_array, options) -> "dispatch.DispatchResult":
        """Run the shared dispatcher on the next device of the plan."""
        device_sel = self._next_device()
        device_id = device_sel.get("id", "cpu")
        torch_device = device_id if torch.cuda.is_available() else "cpu"
        model_obj = lease.model_for_device(device_id)
        backend = lease.record.backend

        def infer(array, return_embeddings):
            if backend == "pytorch":
                input_tensor = torch.from_numpy(np.array(array, copy=True)).float()
                output = torch_rpc.rpc_sync(
                    "worker", "_rpc_infer", args=(input_tensor,), timeout=RPC_INFER_TIMEOUT_SECONDS
                )
                if torch.is_tensor(output):
                    output = output.detach().cpu().numpy()
                return output
            return model_obj.predict(array, return_embeddings=return_embeddings)

        result = dispatch.run_prediction(
            lease,
            image=input_array,
            options=options,
            device_id=device_id,
            torch_device=torch_device,
            infer=infer,
            logger=self.logger,
        )
        result.metadata.update(
            {
                "model_type": backend,
                "model_id": lease.model_id,
                "device_id": device_sel.get("id"),
                "device_type": device_sel.get("type"),
            }
        )
        return result

    def Predict(self, request, context):
        corr_id = uuid.uuid4().hex[:8]
        self.logger.info(
            f"Predict called model_id={request.model_id} shape={list(request.shape)} dtype={request.dtype}",
            extra={"corr_id": corr_id},
        )
        start_total = time.time()
        try:
            lease = self.model_manager.acquire_execution(model_id=request.model_id)
        except (managers.UnknownModelError, ValueError):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("model_id not found")
            return inference_pb2.PredictResponse(
                metadata={"error": "model_not_found", "corr_id": corr_id}
            )
        except managers.ModelCacheError as e:
            context.set_code(self._cache_error_status(e))
            context.set_details(str(e))
            return inference_pb2.PredictResponse(metadata={"error": str(e), "corr_id": corr_id})

        try:
            record = lease.record
            worker_error = self._ensure_pytorch_worker(record)
            if worker_error:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(worker_error.replace("_", " "))
                return inference_pb2.PredictResponse(
                    metadata={"error": worker_error, "corr_id": corr_id}
                )
            input_array = dispatch.decode_input(request.input, tuple(request.shape), request.dtype)
            result = self._dispatch(lease, input_array, dict(request.options))
            meta = dict(result.metadata)
            meta["corr_id"] = corr_id
            meta["latency_total_ms"] = f"{(time.time() - start_total) * 1000.0:.2f}"
            self._record_registry_entry(request.model_id, record.metadata or {})
            return inference_pb2.PredictResponse(
                output=result.output,
                shape=result.shape,
                dtype=result.dtype,
                metadata=meta,
            )
        except dispatch.DispatchError as e:
            self.logger.warning("Predict rejected: %s", e, extra={"corr_id": corr_id})
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return inference_pb2.PredictResponse(
                metadata={"error": str(e), "error_code": e.code, "corr_id": corr_id}
            )
        except Exception as e:
            self.logger.exception("Predict failed", extra={"corr_id": corr_id})
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse(
                metadata={
                    "error": str(e),
                    "error_code": dispatch.ERROR_INFERENCE_FAILED,
                    "corr_id": corr_id,
                }
            )
        finally:
            lease.release()

    def _ensure_rpc_driver(self):
        """Initialize RPC driver for PyTorch RPC worker if not already running."""

        if self._rpc_driver_initialized:
            return

        addr = os.environ.get("MASTER_ADDR", self._rpc_master_addr)
        port = int(os.environ.get("MASTER_PORT", self._rpc_master_port))
        self._rpc_master_addr, self._rpc_master_port = addr, port
        init_method = f"tcp://{addr}:{port}"

        try:
            os.environ.setdefault("MASTER_ADDR", addr)
            os.environ.setdefault("MASTER_PORT", str(port))

            if dist.is_available() and not dist.is_initialized():
                dist.init_process_group(
                    backend="gloo", rank=0, world_size=2, init_method=init_method
                )

            if not torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
                opts = torch_rpc.TensorPipeRpcBackendOptions(init_method=init_method)
                torch_rpc.init_rpc("driver", rank=0, world_size=2, rpc_backend_options=opts)

            self._rpc_driver_initialized = True
        except Exception:
            self.logger.exception("Failed to initialize RPC driver")
            raise

    # ---- Streaming ----
    @staticmethod
    def _stream_response(session_id, meta, corr_id):
        meta = dict(meta)
        meta["corr_id"] = corr_id
        if session_id:
            meta["session_id"] = session_id
        return inference_pb2.StreamPredictResponse(metadata=meta, end_of_sequence=True)

    def _stream_error(self, book, session_id, corr_id, code, message):
        """A stream-scope failure: no tile_id; the host marks every unresolved tile error."""
        if book is not None:
            with book.lock:
                book.errors.append(f"{code}: {message}")
        meta = {
            "error": message,
            "error_code": code,
            "error_scope": dispatch.SCOPE_STREAM,
        }
        return self._stream_response(session_id, meta, corr_id)

    def _tile_error(self, book, session_id, corr_id, tile_id, code, message):
        """A tile-scope failure: only this tile failed; the stream continues."""
        with book.lock:
            book.failed_tiles += 1
            book.errors.append(f"{tile_id}: {code}: {message}")
        meta = {
            "error": message,
            "error_code": code,
            "error_scope": dispatch.SCOPE_TILE,
            "tile_id": tile_id,
        }
        return self._stream_response(session_id, meta, corr_id)

    def _begin_stream(self, first_request, session_id: str):
        """Acquire the execution lease and limits for a stream. Raises manager errors."""
        if session_id:
            lease = self.model_manager.acquire_execution(session_id=session_id)
            with self._sessions_lock:
                book = self._sessions.get(session_id)
            if book is None:  # pragma: no cover - session opened on the manager directly
                book = self._new_book(session_id, lease.model_id, {}, None, 0, 0)
                with self._sessions_lock:
                    self._sessions.setdefault(session_id, book)
            return lease, book
        model_id = first_request.model_id
        if not model_id:
            raise managers.UnknownModelError("model_id is required for a stream without session_id")
        lease = self.model_manager.acquire_execution(model_id=model_id)
        book = self._new_book("", model_id, {}, None, 0, 0)
        return lease, book

    def _new_book(
        self, session_id, model_id, options, total_tiles, max_inflight, max_chunk_bytes,
        max_tile_bytes=0,
    ):
        default_tile_bytes = min(self._default_tile_bytes, GRPC_MAX_MESSAGE_LENGTH)
        return _SessionBook(
            session_id=session_id,
            model_id=model_id,
            max_inflight=int(max_inflight or self._default_max_inflight),
            max_chunk_bytes=int(max_chunk_bytes or self._default_chunk_bytes),
            max_tile_bytes=int(max_tile_bytes or default_tile_bytes),
            options=dispatch.session_inference_options(options),
            total_tiles=total_tiles,
            created_at=time.time(),
        )

    def _start_tile(self, tile, context_map, book):
        """Record a tile's first message; return (code, message) if the tile already failed."""
        with book.lock:
            duplicate = tile.tile_id in book.seen_tiles
            book.seen_tiles.add(tile.tile_id)
        if duplicate:
            return dispatch.ERROR_DUPLICATE_TILE_ID, f"tile {tile.tile_id} was already sent"
        try:
            decoded = dispatch.decode_tile_context(context_map)
        except dispatch.MalformedOptionsError as exc:
            return exc.code, str(exc)
        tile.options = decoded.options
        return None

    def _add_chunk(self, tile, req, book):
        """Append one data message to the tile; return (code, message) on a tile failure."""
        chunk = req.chunk or b""
        if len(chunk) > book.max_chunk_bytes:
            return (
                dispatch.ERROR_CHUNK_TOO_LARGE,
                f"chunk exceeds max_chunk_bytes: {len(chunk)} > {book.max_chunk_bytes}",
            )
        if len(tile.data) + len(chunk) > book.max_tile_bytes:
            return (
                dispatch.ERROR_TILE_TOO_LARGE,
                f"tile exceeds max_tile_bytes: {len(tile.data) + len(chunk)} > {book.max_tile_bytes}",
            )
        shape = tuple(int(d) for d in req.shape) if req.shape else None
        dtype = req.dtype or None
        if shape is not None:
            if tile.shape is None:
                tile.shape = shape
            elif shape != tile.shape:
                return dispatch.ERROR_MALFORMED_PAYLOAD, "shape changed within a tile"
        if dtype is not None:
            if tile.dtype is None:
                tile.dtype = dtype
            elif dtype != tile.dtype:
                return dispatch.ERROR_MALFORMED_PAYLOAD, "dtype changed within a tile"
        tile.data.extend(chunk)
        return None

    def _run_tile(self, lease, book, session_id, tile, corr_id):
        """Dispatch one complete tile and return its response (success or tile error)."""
        try:
            input_array = dispatch.decode_input(bytes(tile.data), tile.shape, tile.dtype)
            options = dispatch.effective_tile_options(book.options, tile.options)
            result = self._dispatch(lease, input_array, options)
        except dispatch.DispatchError as exc:
            return self._tile_error(book, session_id, corr_id, tile.tile_id, exc.code, str(exc))
        except Exception as exc:
            self.logger.exception("StreamPredict tile %s failed", tile.tile_id, extra={"corr_id": corr_id})
            return self._tile_error(
                book, session_id, corr_id, tile.tile_id, dispatch.ERROR_INFERENCE_FAILED, str(exc)
            )
        with book.lock:
            book.ok_tiles += 1
            done = book.ok_tiles + book.failed_tiles
        meta = dict(result.metadata)
        meta["tile_id"] = tile.tile_id
        if book.total_tiles:
            meta["progress"] = f"{done}/{book.total_tiles}"
        response = self._stream_response(session_id, meta, corr_id)
        response.output = result.output
        response.shape.extend(result.shape)
        response.dtype = result.dtype
        return response

    def _check_session_open(self, book, session_id, corr_id):
        """None while the session is open, else the stream-scope error to send."""
        state = self._session_state(session_id)
        if state == "open":
            return None
        return self._session_closed_error(book, session_id, corr_id, state)

    def _session_closed_error(self, book, session_id, corr_id, state=None):
        state = state or self._session_state(session_id)
        code = dispatch.ERROR_CANCELLED if state == "cancelled" else dispatch.ERROR_SESSION_NOT_OPEN
        return self._stream_error(book, session_id, corr_id, code, f"session {session_id} is {state}")

    def StreamPredict(self, request_iterator, context):
        """Bidirectional streaming prediction; see the module docstring for the v1 contract.

        Each tile is dispatched through the same code path as unary Predict. A session
        stream holds one execution lease on the session's model from the first message
        until the stream ends, checks that the session is still open before every
        message, and closes the session with reason ``disconnected`` if the client
        goes away or ``failed`` after a stream-scope server error.
        """

        corr_id = uuid.uuid4().hex[:8]
        self.logger.info("StreamPredict called", extra={"corr_id": corr_id})

        lease = None
        book = None
        session_id = ""
        outcome = "disconnected"  # until the request stream ends or a stream error is sent
        tile = None
        try:
            for req in request_iterator:
                ctx = dict(req.context)
                if lease is None:
                    session_id = ctx.get(dispatch.CONTEXT_SESSION_ID, "")
                    try:
                        lease, book = self._begin_stream(req, session_id)
                    except managers.UnknownSessionError:
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(f"unknown_session: {session_id}")
                        outcome = "rejected"
                        yield self._stream_error(
                            None, session_id, corr_id, dispatch.ERROR_UNKNOWN_SESSION,
                            f"session {session_id} was not opened with OpenSession",
                        )
                        return
                    except managers.SessionClosedError:
                        outcome = "rejected"
                        yield self._session_closed_error(None, session_id, corr_id)
                        return
                    except managers.UnknownModelError:
                        outcome = "rejected"
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details("model_id not found")
                        return
                    except managers.ModelCacheError as exc:
                        outcome = "rejected"
                        context.set_code(self._cache_error_status(exc))
                        context.set_details(str(exc))
                        return
                    if session_id and req.model_id and req.model_id != lease.model_id:
                        outcome = "failed"
                        yield self._stream_error(
                            book, session_id, corr_id, dispatch.ERROR_MALFORMED_PAYLOAD,
                            "model_id does not match the session's model",
                        )
                        return
                    worker_error = self._ensure_pytorch_worker(lease.record)
                    if worker_error:
                        outcome = "failed"
                        yield self._stream_error(
                            book, session_id, corr_id, dispatch.ERROR_INTERNAL, worker_error
                        )
                        return

                if session_id:
                    msg_sid = ctx.get(dispatch.CONTEXT_SESSION_ID, "")
                    if msg_sid and msg_sid != session_id:
                        outcome = "failed"
                        yield self._stream_error(
                            book, session_id, corr_id, dispatch.ERROR_MALFORMED_PAYLOAD,
                            "session_id changed within a stream",
                        )
                        return
                    closed = self._check_session_open(book, session_id, corr_id)
                    if closed is not None:
                        outcome = "failed"
                        yield closed
                        return

                tile_id = ctx.get(dispatch.CONTEXT_TILE_ID, "")
                if not tile_id and tile is not None and req.end_of_sequence and not req.chunk:
                    tile_id = tile.tile_id  # legacy end-of-sequence message without tile_id
                if not tile_id:
                    outcome = "failed"
                    yield self._stream_error(
                        book, session_id, corr_id, dispatch.ERROR_MISSING_TILE_ID,
                        "message without context['tile_id']",
                    )
                    return

                if tile is not None and tile.tile_id != tile_id:
                    if not tile.reported:
                        yield self._tile_error(
                            book, session_id, corr_id, tile.tile_id,
                            dispatch.ERROR_MALFORMED_PAYLOAD, "tile ended without end_of_sequence",
                        )
                    tile = None
                if tile is None:
                    tile = _TileBuffer(tile_id)
                    failure = self._start_tile(tile, ctx, book)
                    if failure is not None:
                        tile.reported = True
                        yield self._tile_error(book, session_id, corr_id, tile_id, *failure)

                if not tile.reported and (req.chunk or not req.end_of_sequence):
                    failure = self._add_chunk(tile, req, book)
                    if failure is not None:
                        tile.reported = True
                        tile.data = bytearray()  # drop the partial tile; drain until its EOS
                        yield self._tile_error(book, session_id, corr_id, tile_id, *failure)

                if req.end_of_sequence:
                    if not tile.reported:
                        is_active = getattr(context, "is_active", None)
                        if callable(is_active) and not is_active():
                            return  # the client is gone: do not run inference for nobody
                        if session_id:
                            try:
                                self.model_manager.touch_session(session_id)
                            except (managers.SessionClosedError, managers.UnknownSessionError):
                                # Closed or cancelled after the check above: no dispatch.
                                outcome = "failed"
                                yield self._session_closed_error(book, session_id, corr_id)
                                return
                        yield self._run_tile(lease, book, session_id, tile, corr_id)
                    tile = None

            if tile is not None and not tile.reported:
                yield self._tile_error(
                    book, session_id, corr_id, tile.tile_id,
                    dispatch.ERROR_MALFORMED_PAYLOAD, "stream ended before end_of_sequence",
                )
            outcome = "completed"
        except grpc.RpcError:
            # The client cancelled or disconnected while the server waited for a message.
            outcome = "disconnected"
        except Exception as exc:
            self.logger.exception("StreamPredict failed", extra={"corr_id": corr_id})
            outcome = "failed"
            yield self._stream_error(book, session_id, corr_id, dispatch.ERROR_INTERNAL, str(exc))
        finally:
            if lease is not None:
                lease.release()
                if session_id:
                    if outcome == "failed":
                        self.model_manager.close_session(session_id, reason="failed")
                    elif outcome == "disconnected":
                        self.model_manager.close_session(session_id, reason="disconnected")
            self.logger.info(
                "StreamPredict ended: %s", outcome, extra={"corr_id": corr_id}
            )

    # ---- Registry RPCs ----
    def ListModelRegistry(self, request, context):
        entries = self.registry_store.list_entries(request.provider_id or None)
        response_entries = []
        for entry in entries:
            response_entries.append(
                inference_pb2.ModelRegistryEntry(
                    entry_id=entry.entry_id,
                    model_id=entry.model_id,
                    display_name=entry.display_name,
                    source=entry.source,
                    task=entry.task,
                    format=entry.format,
                    last_used_epoch_ms=entry.last_used_epoch_ms,
                    metadata=entry.metadata,
                )
            )
        return inference_pb2.ListModelRegistryResponse(entries=response_entries)

    def EvictModelRegistryEntry(self, request, context):
        if not request.entry_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("entry_id is required")
            return inference_pb2.EvictModelRegistryResponse(
                success=False, message="missing entry_id"
            )
        removed = self.registry_store.evict(request.entry_id)
        if not removed:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("entry_id not found")
            return inference_pb2.EvictModelRegistryResponse(
                success=False, message="entry not found"
            )
        return inference_pb2.EvictModelRegistryResponse(success=True, message="evicted")

    # ---- Sessions ----
    def OpenSession(self, request, context):
        corr_id = uuid.uuid4().hex[:8]
        self.logger.info("OpenSession called", extra={"corr_id": corr_id})

        # Opportunistically clean up stale sessions
        cleaned = self._cleanup_stale_sessions()
        if cleaned:
            self.logger.debug(f"Expired {cleaned} idle sessions", extra={"corr_id": corr_id})

        session_id = request.session_id or uuid.uuid4().hex

        def _fail(code, message, status="error"):
            context.set_code(code)
            context.set_details(message)
            return inference_pb2.OpenSessionResponse(
                session_id=session_id, status=status, error=message
            )

        transport = request.transport
        max_inflight = min(
            transport.max_inflight or self._default_max_inflight, self._default_max_inflight
        )
        # A requested chunk size is honored as sent, so the negotiated value is exactly
        # what the client asked for; without one the server default applies.
        max_chunk_bytes = transport.chunk_bytes or self._default_chunk_bytes
        if transport.chunk_bytes < 0 or transport.chunk_bytes > GRPC_MAX_MESSAGE_LENGTH:
            return _fail(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"transport.chunk_bytes {transport.chunk_bytes} is outside 0..{GRPC_MAX_MESSAGE_LENGTH} "
                "(Capabilities stream_max_chunk_bytes)",
            )
        max_tile_bytes = min(self._default_tile_bytes, GRPC_MAX_MESSAGE_LENGTH)
        if "max_tile_bytes" in request.options:
            try:
                requested = int(request.options["max_tile_bytes"])
            except ValueError:
                return _fail(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"options.max_tile_bytes {request.options['max_tile_bytes']!r} is not an integer",
                )
            if requested <= 0:
                return _fail(grpc.StatusCode.INVALID_ARGUMENT, "options.max_tile_bytes must be > 0")
            max_tile_bytes = min(max_tile_bytes, requested)

        spec = request.spec
        if not spec.model_id and not spec.source and not spec.artifact:
            return _fail(grpc.StatusCode.INVALID_ARGUMENT, "session requires model_id or spec.source")

        try:
            lease = None
            if spec.model_id:
                try:
                    lease = self.model_manager.open_session(session_id, model_id=spec.model_id)
                except managers.UnknownModelError:
                    if not spec.source and not spec.artifact:
                        raise
            if lease is None:
                prepared = self._prepare_load(spec, corr_id)
                lease = self.model_manager.open_session(
                    session_id,
                    model_spec=prepared.model_spec,
                    key=prepared.key,
                    metadata=prepared.metadata,
                    device_plan=prepared.device_plan,
                )
                self._record_registry_entry(lease.model_id, prepared.metadata)
        except _LoadRequestError as e:
            return _fail(e.code, str(e))
        except managers.ModelCacheError as e:
            self.logger.warning("OpenSession failed: %s", e, extra={"corr_id": corr_id})
            return _fail(self._cache_error_status(e), str(e))
        except Exception as e:
            self.logger.exception("OpenSession failed", extra={"corr_id": corr_id})
            return _fail(grpc.StatusCode.INTERNAL, str(e))

        with self._sessions_lock:
            book = self._sessions.get(session_id) if lease.reused else None
            if book is None:
                total_tiles = request.manifest.total_tiles if request.HasField("manifest") else 0
                book = self._new_book(
                    session_id,
                    lease.model_id,
                    dict(request.options),
                    int(total_tiles) if total_tiles else None,
                    max_inflight,
                    max_chunk_bytes,
                    max_tile_bytes,
                )
                self._sessions[session_id] = book

        return inference_pb2.OpenSessionResponse(
            session_id=session_id,
            device_plan=self._device_plan_messages(),
            model_cache_hit=bool(lease.cache_hit),
            max_inflight=int(book.max_inflight),
            max_tile_bytes=int(book.max_tile_bytes),
            status="ok",
        )

    def CloseSession(self, request, context):
        session_id = request.session_id
        if not session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return inference_pb2.CloseSessionResponse(status="error", error="missing session_id")

        closed = self.model_manager.close_session(session_id, reason="closed")
        with self._sessions_lock:
            book = self._sessions.pop(session_id, None)
        state = self._session_state(session_id)
        if not closed and book is None and state == "unknown":
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("session_id not found")
            return inference_pb2.CloseSessionResponse(status="not_found", error="session not found")

        # A session that was cancelled, failed or disconnected keeps that status.
        status_label = "closed" if closed else state
        summary = inference_pb2.SessionSummary(session_id=session_id)
        if book is not None:
            with book.lock:
                summary.ok_tiles = int(book.ok_tiles)
                summary.failed_tiles = int(book.failed_tiles)
                summary.duration_ms = int((time.time() - book.created_at) * 1000)
                summary.errors.extend(book.errors)
        return inference_pb2.CloseSessionResponse(status=status_label, summary=summary)

    def CancelSession(self, request, context):
        session_id = request.session_id
        if not session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return inference_pb2.CancelSessionResponse(status="error", error="missing session_id")

        cancelled = self.model_manager.close_session(session_id, reason="cancelled")
        with self._sessions_lock:
            book = self._sessions.get(session_id)
        state = self._session_state(session_id)
        if not cancelled and book is None and state == "unknown":
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("session_id not found")
            return inference_pb2.CancelSessionResponse(
                status="not_found", error="session not found"
            )
        if cancelled and book is not None:
            with book.lock:
                book.errors.append(request.reason or "client_cancelled")
        # The active stream of this session stops before its next message or dispatch.
        return inference_pb2.CancelSessionResponse(status="cancelled" if cancelled else state)

    # ---- Helpers ----
    def _record_registry_entry(self, model_id: str, metadata: dict) -> None:
        if not model_id:
            return
        metadata = metadata or {}
        try:
            display_name = (
                str(metadata.get("name"))
                if metadata.get("name")
                else str(metadata.get("source") or model_id)
            )
            source = str(metadata.get("source") or "")
            task_label = metadata.get("task_name") or metadata.get("task") or ""
            if isinstance(task_label, int):
                try:
                    task_label = inference_pb2.TaskType.Name(task_label)
                except ValueError:
                    task_label = str(task_label)
            elif task_label:
                task_label = str(task_label)
            format_label = str(metadata.get("format") or "")
            meta_payload = {}
            for key, value in metadata.items():
                if value is None or key == "token":  # never persist credentials to the registry file
                    continue
                try:
                    meta_payload[str(key)] = str(value)
                except Exception:  # pragma: no cover - defensive
                    meta_payload[str(key)] = ""

            self.registry_store.record_usage(
                model_id=model_id,
                display_name=display_name,
                source=source,
                task=str(task_label or ""),
                model_format=format_label,
                metadata=meta_payload,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to update registry entry: %s", exc)

    # ---- Introspection ----
    def Capabilities(self, request, context):
        caps = [
            inference_pb2.CapabilityInfo(key="cuda", value=str(torch.cuda.is_available()).lower()),
            inference_pb2.CapabilityInfo(key="max_cores", value=str(self._max_cores)),
            inference_pb2.CapabilityInfo(
                key="stream_context_version", value=dispatch.STREAM_CONTEXT_VERSION
            ),
            inference_pb2.CapabilityInfo(
                key="stream_default_chunk_bytes", value=str(self._default_chunk_bytes)
            ),
            inference_pb2.CapabilityInfo(
                key="stream_max_chunk_bytes", value=str(GRPC_MAX_MESSAGE_LENGTH)
            ),
            inference_pb2.CapabilityInfo(
                key="stream_max_tile_bytes",
                value=str(min(self._default_tile_bytes, GRPC_MAX_MESSAGE_LENGTH)),
            ),
            inference_pb2.CapabilityInfo(
                key="stream_max_inflight", value=str(self._default_max_inflight)
            ),
        ]
        if self._device_plan:
            caps.append(
                inference_pb2.CapabilityInfo(
                    key="device_plan",
                    value=json.dumps(self._device_plan),
                )
            )
        return inference_pb2.CapabilitiesResponse(capabilities=caps)

    def Health(self, request, context):
        return inference_pb2.HealthResponse(ready=True, message="ok")

    def shutdown(self, grace: float | None = SERVER_STOP_GRACE_SECONDS) -> None:
        """Close all sessions and dispose every model after running predictions drain."""
        self.model_manager.shutdown(drain_timeout_s=grace)


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    model_cache_dir: str | Path | None = None,
    max_cores: int = 4,
):
    """Build, bind and start the gRPC inference server without blocking.

    Args:
            host: address to bind; loopback by default. ``serve()`` passes ``"[::]"``.
                An IPv6 address needs brackets.
            port: TCP port; 0 picks a free ephemeral port.
            model_cache_dir, max_cores: as for ``InferenceService``.

    Returns:
            ``(server, bound_port, service)``: the started ``grpc.Server``, the port it
            is bound to and its ``InferenceService``. Stop it with ``stop_server``.

    Raises:
            RuntimeError: the address cannot be bound (grpcio raises it).
    """
    cpu_count = os.cpu_count() or 1
    server_workers = max(1, min(int(max_cores or 4), cpu_count))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_workers),
        interceptors=[LoggingInterceptor()],
        options=GRPC_OPTIONS,
    )
    service = InferenceService(model_cache_dir=model_cache_dir, max_cores=max_cores)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    try:
        bound_port = server.add_insecure_port(f"{host}:{port}")
    except BaseException:
        service.shutdown(grace=0)
        raise
    server.start()
    return server, bound_port, service


def stop_server(server, service, *, grace: float | None = SERVER_STOP_GRACE_SECONDS) -> None:
    """Stop a server from ``create_server``: gRPC first, then the model cache.

    New RPCs are refused and running ones get ``grace`` seconds (``server.stop``).
    Then ``service.model_manager.shutdown(drain_timeout_s=grace)`` closes every
    session, waits at most ``grace`` seconds for execution leases to drain and
    disposes every model. ``grace=None`` aborts running RPCs at once and waits for
    the leases without limit.
    """
    server.stop(grace).wait()
    service.shutdown(grace=grace)


def serve(
    port: int = 50051,
    *,
    model_cache_dir: str | Path | None = None,
    startup_event=None,
    stop_event=None,
    max_cores: int = 4,
    stop_grace: float = SERVER_STOP_GRACE_SECONDS,
) -> int:
    """Start the gRPC inference server.

    Args:
            port: TCP port to bind (50051 default). Use 0 for ephemeral.
            startup_event: threading.Event that will be set once the server is started.
            stop_event: threading.Event; if provided, the server thread will block until it is set then shut down.
            stop_grace: seconds running RPCs get at shutdown; the model cache then drains
                execution leases for at most the same time before disposing models.

    Returns:
            The actual bound port (useful if port=0 was passed).
    """
    server, bound_port, service = create_server(
        "[::]", port, model_cache_dir=model_cache_dir, max_cores=max_cores
    )
    logger.info(f"gRPC inference server started on port {bound_port}")
    if startup_event is not None:
        startup_event.set()
    # Backward compatibility: discover stop_event from caller frame if not explicitly passed
    if stop_event is None:
        frame = inspect.currentframe().f_back
        stop_event = frame.f_locals.get("stop_event", None)
    try:
        if stop_event is not None:
            stop_event.wait()
            logging.info("Shutting down gRPC server after test.")
        else:
            server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server interrupted")
    finally:
        stop_server(server, service, grace=stop_grace)
    return bound_port


async def create_unified_web_app(
    *, include_azure_proxy=False, model_cache_dir: str | Path | None = None
):
    """Create a unified web application with inference endpoints and optionally Azure proxy endpoints."""
    app = web.Application()

    # Add basic inference endpoints (converted from gRPC)
    inference_service = InferenceService(model_cache_dir=model_cache_dir)

    async def health_check(request):
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "service": "nxmndr-inference"})

    async def list_models_http(request):
        """HTTP endpoint to list available models."""
        try:
            # Create a mock gRPC request
            grpc_request = inference_pb2.ListModelsRequest()
            response = inference_service.ListModels(grpc_request, None)

            # Convert gRPC response to JSON
            models_data = []
            for model in response.models:
                models_data.append(
                    {
                        "model_id": model.model_id,
                        "name": model.name,
                        "status": model.status,
                    }
                )

            return web.json_response({"models": models_data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # Add basic inference routes under /inference prefix
    app.router.add_get("/health", health_check)
    app.router.add_get("/inference/health", health_check)  # Also available under inference prefix
    app.router.add_get("/inference/models", list_models_http)

    # Add a root endpoint that shows available services
    async def root_handler(request):
        services = {
            "inference": {
                "description": "ML model inference service",
                "endpoints": [
                    "GET /inference/health - Health check",
                    "GET /inference/models - List available models",
                ],
            }
        }

        if include_azure_proxy:
            services["azure_proxy"] = {
                "description": "Azure OpenAI API proxy service",
                "endpoints": [
                    "POST /openai/deployments/{deployment}/chat/completions - Chat completions",
                    "POST /openai/deployments/{deployment}/images/edits - Image editing",
                    "POST /generate/image/{deployment} - Unified image generation",
                    "GET /models - List proxy models",
                    "GET /azure/health - Azure proxy health",
                ],
            }

        return web.json_response({"server": "NXMNDR Unified Server", "services": services})

    app.router.add_get("/", root_handler)

    # Add Azure proxy endpoints if requested
    if include_azure_proxy:
        logger.info("Adding Azure OpenAI proxy endpoints...")

        azure_proxy = AzureOpenAIProxy()
        setup_azure_proxy_routes(app, azure_proxy)

        logger.info("Azure OpenAI proxy endpoints added successfully")

    return app


async def run_unified_server(
    host="0.0.0.0",
    port=8080,
    *,
    include_azure_proxy=False,
    model_cache_dir: str | Path | None = None,
):
    """Run the optional unified web server with inference and (optionally) Azure proxy endpoints."""
    # Create the unified web application
    app = await create_unified_web_app(
        include_azure_proxy=include_azure_proxy, model_cache_dir=model_cache_dir
    )

    # Log available endpoints
    logger.info(f"Starting unified web server on {host}:{port}")
    logger.info("Available services:")
    logger.info("  📊 Inference Service:")
    logger.info("    GET  /health - Global health check")
    logger.info("    GET  /inference/health - Inference health check")
    logger.info("    GET  /inference/models - List inference models")

    if include_azure_proxy:
        logger.info("  🔗 Azure OpenAI Proxy Service:")
        logger.info("    POST /openai/deployments/{deployment}/chat/completions - Chat completions")
        logger.info("    POST /openai/deployments/{deployment}/images/edits - Image editing")
        logger.info("    POST /generate/image/{deployment} - Unified image generation")
        logger.info("    GET  /models - List proxy models")
        logger.info("    GET  /azure/health - Azure proxy health")

    logger.info("  📋 Discovery:")
    logger.info("    GET  / - Service discovery endpoint")

    # Start the web server
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info("Unified server started successfully")

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down unified server...")
    finally:
        await runner.cleanup()


def _configure_cudnn() -> None:
    """Disable cuDNN when the bundled cuDNN cannot serve the attached GPUs.

    torch >= 2.9 ships cuDNN >= 9.11, which has no kernels for SM < 7.5 (Volta and older);
    convolutions then fail with "unable to find an engine". Falling back to the native CUDA
    kernels keeps inference on the GPU. Override with NXMNDR_CUDNN=0|1.
    """
    override = os.environ.get("NXMNDR_CUDNN")
    if override is not None:
        torch.backends.cudnn.enabled = override.strip() not in {"0", "false", "no", "off"}
        logger.info("cuDNN %s by NXMNDR_CUDNN", "enabled" if torch.backends.cudnn.enabled else "disabled")
        return
    try:
        if not torch.cuda.is_available():
            return
        compiled = torch._C._cudnn.getCompileVersion() if hasattr(torch._C, "_cudnn") else None
        min_cc = min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
        if compiled and (compiled[0], compiled[1]) >= (9, 11) and min_cc < (7, 5):
            torch.backends.cudnn.enabled = False
            logger.warning(
                "cuDNN %s does not support SM %d.%d GPUs; cuDNN disabled, using native CUDA kernels",
                ".".join(str(v) for v in compiled), min_cc[0], min_cc[1],
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cuDNN capability check skipped: %s", exc)


def main():
    """Main CLI entry point. Defaults to gRPC; Azure proxy HTTP server is opt-in."""
    import argparse

    parser = argparse.ArgumentParser(description="NXMNDR gRPC inference server")
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="Port for the gRPC inference server (default: 50051)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for the Azure proxy HTTP server when enabled (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the Azure proxy HTTP server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--azure-proxy",
        action="store_true",
        help="Run both gRPC server and Azure OpenAI proxy HTTP server",
    )
    parser.add_argument(
        "--model-cache-dir",
        default="",
        help="Directory where downloaded model artifacts are stored "
        "(default: $NXMNDR_CACHE_DIR/models, else ~/.cache/nxmndr/models)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=4,
        help="Maximum worker slots (GPUs preferred; CPUs used if no GPU).",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.setLevel(getattr(logging, args.log_level))

    _configure_cudnn()

    if args.azure_proxy:
        # Run both gRPC and HTTP servers
        import threading

        logger.info(
            f"Starting both gRPC (port {args.grpc_port}) and HTTP (port {args.http_port}) servers"
        )

        # Start HTTP server in background thread
        http_thread = threading.Thread(
            target=lambda: asyncio.run(
                run_unified_server(
                    host=args.host,
                    port=args.http_port,
                    include_azure_proxy=True,
                    model_cache_dir=args.model_cache_dir or None,
                )
            ),
            daemon=True,
        )
        http_thread.start()

        # Run gRPC server in main thread
        try:
            serve(
                port=args.grpc_port,
                model_cache_dir=args.model_cache_dir or None,
                max_cores=args.max_cores,
            )
        except KeyboardInterrupt:
            logger.info("Servers stopped")
        except Exception as e:
            logger.error(f"Server error: {e}")
            exit(1)
    else:
        logger.info(f"Starting gRPC inference server on port {args.grpc_port}")
        serve(
            port=args.grpc_port,
            model_cache_dir=args.model_cache_dir or None,
            max_cores=args.max_cores,
        )


if __name__ == "__main__":
    main()
