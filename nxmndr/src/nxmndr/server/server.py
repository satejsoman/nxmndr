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
"""

import asyncio
import hashlib
import json
import os
import inspect
import logging
import sys
import time
import uuid
import random
from concurrent import futures
from pathlib import Path

import grpc
import atexit
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as torch_rpc
from aiohttp import web

from ..logging import get_logger

from ..inference import LocalInferenceProvider, inference_pb2, inference_pb2_grpc
from ..inference.image_utils import (
    masks_to_bounding_boxes,
    pack_tensor_bundle,
    prepare_segmentation_mask,
    prepare_segmentation_mask_with_confidence,
)
from ..models import Model
from ..models.sam import is_sam_model, handle_sam_inference
from .azure_openai_proxy import AzureOpenAIProxy, setup_azure_proxy_routes
from .managers import ModelManager, RpcWorkerManager

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
DEFAULT_BATCH_SIZE = 1  # Default batch size for single-sample prediction
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


class LoggingInterceptor(grpc.ServerInterceptor):
    """Interceptor to log all incoming gRPC requests."""

    def intercept_service(self, continuation, handler_call_details):
        logger.debug(
            f"gRPC request: method={handler_call_details.method}, "
            f"metadata={dict(handler_call_details.invocation_metadata or [])}"
        )
        return continuation(handler_call_details)


# Helper: deserialize input_data (bytes) to numpy array
def deserialize_input(input_bytes, shape, dtype):
    arr = np.frombuffer(input_bytes, dtype=dtype)
    return arr.reshape(shape)


# Helper: serialize numpy array to bytes
def serialize_output(output):
    return output.tobytes()


def _parse_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, *, model_cache_dir: str | Path | None = None, max_cores: int = 4):
        # Cap max_cores at a minimum of 1
        self._max_cores = max(1, int(max_cores or 4))

        self.provider = LocalInferenceProvider()
        self.model_manager = ModelManager(self.provider)
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
        # Session registry: session_id -> {model_id, created_at}
        self._sessions: dict[str, dict] = {}
        self._session_ttl_seconds = int(os.environ.get("NXMNDR_SESSION_TTL_SECONDS", "3600"))
        self._last_session_cleanup = time.time()
        self._default_max_inflight = int(os.environ.get("NXMNDR_STREAM_MAX_INFLIGHT", "16"))
        self._default_chunk_bytes = int(
            os.environ.get("NXMNDR_STREAM_MAX_CHUNK_BYTES", str(8 * 1024 * 1024))
        )
        self._default_tile_bytes = int(
            os.environ.get("NXMNDR_STREAM_MAX_TILE_BYTES", str(GRPC_MAX_MESSAGE_LENGTH))
        )

    def _cleanup_stale_sessions(self) -> int:
        """Remove sessions that have exceeded TTL. Returns count of cleaned sessions."""
        now = time.time()
        # Only run cleanup every 60 seconds at most
        if now - self._last_session_cleanup < 60:
            return 0
        self._last_session_cleanup = now

        stale_ids = []
        for sid, state in list(self._sessions.items()):
            created = state.get("created_at", 0)
            if now - created > self._session_ttl_seconds:
                stale_ids.append(sid)

        for sid in stale_ids:
            self._sessions.pop(sid, None)
            self.logger.info(f"Cleaned up stale session: {sid}")

        return len(stale_ids)

    def _next_device(self):
        if not self._device_plan:
            return {"type": "cpu", "ordinal": 0, "id": "cpu:0"}
        device = self._device_plan[self._device_cursor % len(self._device_plan)]
        self._device_cursor += 1
        return device

    # ---- Lifecycle ----
    def LoadModel(self, request, context):
        corr_id = uuid.uuid4().hex[:8]
        self.logger.info("LoadModel called", extra={"corr_id": corr_id})
        try:
            # Early validation of format
            spec_msg = request.spec
            if spec_msg.format not in (
                inference_pb2.PYTORCH,
                inference_pb2.ONNX,
                inference_pb2.HUGGINGFACE,
                inference_pb2.TORCHHUB,
            ):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Unsupported model format enum")
                return inference_pb2.LoadModelResponse(
                    success=False, model_id="", message="unsupported format"
                )
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
                    if not target_path.exists() or target_path.stat().st_size != len(
                        artifact_bytes
                    ):
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
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("HuggingFace models require source (repo_id)")
                    return inference_pb2.LoadModelResponse(
                        success=False, model_id="", message="missing repo_id"
                    )
                spec_dict = {
                    "type": "huggingface",
                    "repo_id": spec_msg.source,
                    "name": spec_msg.name or spec_msg.source,
                }
                token = (
                    spec_msg.token
                    or os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_TOKEN")
                )
                if token:
                    spec_dict["token"] = token
                    self.logger.debug(
                        "HuggingFace token provided (length: %d)",
                        len(token),
                        extra={"corr_id": corr_id},
                    )
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
                except Exception:
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("failed to initialize RPC driver")
                    return inference_pb2.LoadModelResponse(
                        success=False, model_id="", message="rpc_driver_init_failed"
                    )
            import json as _json  # local import for clarity

            model_spec = Model.model_spec_from_json(_json.dumps(spec_dict))
            model_id = spec_msg.model_id or uuid.uuid4().hex
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
            # Copy token from spec_dict if present (for HuggingFace gated repos)
            if "token" in spec_dict:
                model_metadata["token"] = spec_dict["token"]
            model_metadata.update(metadata_updates)
            if model_type == "PytorchModelSpec":
                self.rpc_manager.ensure_worker(
                    model_spec,
                    master_addr=self._rpc_master_addr,
                    master_port=self._rpc_master_port,
                )
                model_id = self.model_manager.load_spec(model_spec, metadata=model_metadata)
            elif model_type in ("OnnxModelSpec", "HuggingFaceModelSpec"):
                # Pass device plan for multi-device model replication
                model_id = self.model_manager.load_spec(
                    model_spec,
                    metadata=model_metadata,
                    device_plan=self._device_plan,
                )
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Unsupported spec type")
                return inference_pb2.LoadModelResponse(
                    success=False, model_id="", message="unsupported spec type"
                )
            self.logger.info(
                f"Model loaded model_id={model_id} type={model_type}",
                extra={"corr_id": corr_id},
            )
            self._record_registry_entry(model_id, model_metadata)
            return inference_pb2.LoadModelResponse(
                success=True, model_id=model_id, message="loaded"
            )
        except Exception as e:
            self.logger.exception("LoadModel failed", extra={"corr_id": corr_id})
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.LoadModelResponse(success=False, model_id="", message=str(e))

    def UnloadModel(self, request, context):
        mid = request.model_id
        if self.model_manager.unload(mid):
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
    def Predict(self, request, context):
        corr_id = uuid.uuid4().hex[:8]
        self.logger.info(
            f"Predict called model_id={request.model_id} shape={list(request.shape)} dtype={request.dtype}",
            extra={"corr_id": corr_id},
        )
        start_total = time.time()
        record = self.model_manager.get(request.model_id)
        if record is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("model_id not found")
            return inference_pb2.PredictResponse(
                metadata={"error": "model_not_found", "corr_id": corr_id}
            )
        model_obj = record.model
        mtype = record.backend
        model_metadata = record.metadata or {}

        # PyTorch: ensure RPC worker ready
        if mtype == "pytorch" and not self._rpc_driver_initialized:
            self._ensure_rpc_driver()
            if not _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS):
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("RPC worker not ready")
                return inference_pb2.PredictResponse(
                    metadata={"error": "rpc_worker_not_ready", "corr_id": corr_id}
                )
        if mtype == "pytorch" and not self.rpc_manager.is_alive():
            try:
                self.rpc_manager.ensure_worker(
                    record.spec,
                    master_addr=self._rpc_master_addr,
                    master_port=self._rpc_master_port,
                )
                _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS)
            except Exception:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("RPC worker not available")
                return inference_pb2.PredictResponse(
                    metadata={"error": "rpc_worker_not_available", "corr_id": corr_id}
                )

        # Check if client requested embeddings
        options = request.options or {}
        return_embeddings = _parse_bool(options.get("return_embeddings"))

        device_sel = self._next_device()
        device_id = device_sel.get("id", "cpu")

        # Get device-specific model if available
        device_model = self.model_manager.get_model_for_device(request.model_id, device_id)
        if device_model is not None:
            model_obj = device_model

        try:
            input_array = deserialize_input(
                request.input, shape=tuple(request.shape), dtype=request.dtype
            )

            start_infer = time.time()

            # Check if model supports SAM-style prompting and delegate
            if is_sam_model(request.model_id, model_metadata):
                # Delegate to SAM handler (returns None if no prompts)
                token = (
                    model_metadata.get("token")
                    or os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_TOKEN")
                )
                if not token:
                    self.logger.warning("No HF token in model_metadata for SAM inference")
                output = handle_sam_inference(
                    model_id=request.model_id,
                    model_spec=model_metadata,
                    image_array=input_array,
                    options=options,
                    device=device_sel.get("id") if torch.cuda.is_available() else "cpu",
                    logger=self.logger,
                    token=token,
                )

                # If no SAM prompts were provided, fall through to standard inference
                if output is None:
                    if mtype == "pytorch":
                        input_tensor = torch.from_numpy(input_array).float()
                        output = torch_rpc.rpc_sync(
                            "worker", "_rpc_infer", args=(input_tensor,), timeout=RPC_INFER_TIMEOUT_SECONDS
                        )
                        if torch.is_tensor(output):
                            output = output.detach().cpu().numpy()
                    else:
                        output = model_obj.predict(input_array, return_embeddings=return_embeddings)
            else:
                # Normal inference path (no SAM prompts)
                if mtype == "pytorch":
                    input_tensor = torch.from_numpy(input_array).float()
                    output = torch_rpc.rpc_sync(
                        "worker", "_rpc_infer", args=(input_tensor,), timeout=RPC_INFER_TIMEOUT_SECONDS
                    )
                    if torch.is_tensor(output):
                        output = output.detach().cpu().numpy()
                else:
                    # Pass return_embeddings to predict method
                    output = model_obj.predict(input_array, return_embeddings=return_embeddings)

            # Handle dict output from models that return multiple outputs
            embeddings_from_model = None
            if isinstance(output, dict):
                if isinstance(output, dict):
                    # Extract embeddings if present
                    if "embeddings" in output:
                        embeddings_from_model = output["embeddings"]

                    # Extract output if present, otherwise use embeddings as output
                    if "output" in output:
                        output = output["output"]
                        if embeddings_from_model is not None:
                            self.logger.debug(
                                "Model returned both output and embeddings. Output shape: %s, Embeddings shape: %s",
                                output.shape,
                                embeddings_from_model.shape,
                            )
                    elif embeddings_from_model is not None:
                        output = embeddings_from_model  # Use embeddings as primary output if no other output

                if isinstance(output, list) and len(output) == 1:
                    output = output[0]
                if torch.is_tensor(output):
                    output = output.detach().cpu().numpy()
            infer_ms = (time.time() - start_infer) * 1000.0
            if not isinstance(output, np.ndarray):
                output = np.array(output)
            base_output = np.ascontiguousarray(output)

            # Store embeddings for later bundling if they were returned by the model
            if embeddings_from_model is not None:
                if torch.is_tensor(embeddings_from_model):
                    embeddings_from_model = embeddings_from_model.detach().cpu().numpy()
                if not isinstance(embeddings_from_model, np.ndarray):
                    embeddings_from_model = np.array(embeddings_from_model)
                embeddings_from_model = np.ascontiguousarray(embeddings_from_model)

            options = request.options or {}
            task_type = (options.get("task_type") or "").strip().lower()
            self.logger.info(
                f"[Server Predict] Received task_type from options: {task_type!r}",
                extra={"corr_id": corr_id},
            )
            if not task_type:
                task_hint = model_metadata.get("task_name")
                if isinstance(task_hint, str) and task_hint:
                    task_type = task_hint.lower()
                else:
                    raw_task = model_metadata.get("task")
                    if isinstance(raw_task, int) and raw_task:
                        try:
                            task_type = inference_pb2.TaskType.Name(raw_task).lower()
                        except ValueError:
                            task_type = ""
                self.logger.info(
                    f"[Server Predict] task_type after fallback: {task_type!r}",
                    extra={"corr_id": corr_id},
                )
            return_embeddings = _parse_bool(options.get("return_embeddings"))
            embedding_only = task_type in {
                "embedding",
                "embeddings",
                "feature",
                "features",
            }
            geotransform_opt = options.get("geotransform")
            projection_opt = (
                options.get("projection") or options.get("crs") or options.get("crs_wkt")
            )

            response_array = base_output
            meta = {
                "corr_id": corr_id,
                "latency_infer_ms": f"{infer_ms:.2f}",
                "model_type": mtype,
                "model_id": request.model_id,
                "device_id": device_sel.get("id"),
                "device_type": device_sel.get("type"),
            }

            if task_type:
                meta["task_type"] = task_type

            model_task_name = model_metadata.get("task_name")
            if model_task_name:
                meta.setdefault("model_task", model_task_name)

            if embedding_only:
                meta["result_type"] = "embeddings"
                meta["return_embeddings"] = "true"
                meta["embeddings_available"] = "true"
                meta["embeddings_shape"] = json.dumps(list(base_output.shape))
                meta["embeddings_dtype"] = str(base_output.dtype)

            if embedding_only:
                bundle_payload = {"embeddings": base_output}
            else:
                # Use embeddings from model if available, otherwise use base_output if requested
                if embeddings_from_model is not None:
                    bundle_payload = (
                        {"embeddings": embeddings_from_model} if return_embeddings else None
                    )
                else:
                    bundle_payload = {"embeddings": base_output} if return_embeddings else None

            if not embedding_only and task_type == "segmentation":
                self.logger.info(
                    f"[Server Predict] Applying prepare_segmentation_mask for task_type='segmentation', base_output shape: {base_output.shape}",
                    extra={"corr_id": corr_id},
                )
                try:
                    masks = prepare_segmentation_mask(base_output)
                    response_array = masks
                    meta["result_type"] = "segmentation_mask"
                    meta["mask_shape"] = str(list(masks.shape))
                    meta["mask_dtype"] = str(masks.dtype)
                    self.logger.info(
                        f"[Server Predict] prepare_segmentation_mask succeeded, masks shape: {masks.shape}, dtype: {masks.dtype}",
                        extra={"corr_id": corr_id},
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning(
                        f"[Server Predict] prepare_segmentation_mask failed: {exc}",
                        extra={"corr_id": corr_id},
                    )
                    meta["result_type"] = "raw"
                    meta["task_warning"] = f"segmentation_fallback:{exc}"
                    response_array = output
            elif not embedding_only and task_type in {
                "object_detection",
                "object-detection",
                "detection",
            }:
                try:
                    masks = prepare_segmentation_mask(base_output)
                    boxes = masks_to_bounding_boxes(masks)
                    response_array = boxes
                    meta["result_type"] = "bounding_boxes"
                    meta["boxes_count"] = str(boxes.shape[0])
                    meta["box_format"] = "batch_index,class_id,x_min,y_min,x_max,y_max"
                except Exception as exc:  # pragma: no cover - defensive
                    meta["result_type"] = "raw"
                    meta["task_warning"] = f"detection_fallback:{exc}"
                    response_array = output
            else:
                self.logger.info(
                    f"[Server Predict] NOT applying segmentation mask. embedding_only={embedding_only}, task_type={task_type!r}, response_array shape: {response_array.shape}",
                    extra={"corr_id": corr_id},
                )
                if task_type:
                    meta.setdefault("result_type", "raw")

            bundle_requested = return_embeddings and not embedding_only

            if geotransform_opt:
                meta.setdefault("geotransform", str(geotransform_opt))
            if projection_opt:
                meta.setdefault("projection", str(projection_opt))
            if return_embeddings or embedding_only:
                meta["return_embeddings"] = "true"

            if bundle_requested:
                bundle = bundle_payload or {"embeddings": base_output}
                result_type = meta.get("result_type", "raw")
                if result_type == "segmentation_mask":
                    bundle["mask"] = np.ascontiguousarray(response_array)
                elif result_type == "bounding_boxes":
                    bundle["detections"] = np.ascontiguousarray(response_array)
                bundle_bytes = pack_tensor_bundle(bundle)
                response_array = np.frombuffer(bundle_bytes, dtype=np.uint8)
                meta["payload_format"] = "npz"
                meta["bundle_size_bytes"] = str(len(bundle_bytes))
                meta["bundle_keys"] = ",".join(sorted(bundle.keys()))
                meta["embeddings_available"] = "true"
                meta["embeddings_shape"] = json.dumps(list(base_output.shape))
                meta["embeddings_dtype"] = str(base_output.dtype)
            elif embedding_only and bundle_payload is not None:
                bundle_bytes = pack_tensor_bundle(bundle_payload)
                response_array = np.frombuffer(bundle_bytes, dtype=np.uint8)
                meta["payload_format"] = "npz"
                meta["bundle_size_bytes"] = str(len(bundle_bytes))
                meta["bundle_keys"] = ",".join(sorted(bundle_payload.keys()))

            response_array = np.ascontiguousarray(response_array)
            meta.setdefault("result_type", "raw")
            total_ms = (time.time() - start_total) * 1000.0
            meta["latency_total_ms"] = f"{total_ms:.2f}"

            self._record_registry_entry(request.model_id, model_metadata)

            return inference_pb2.PredictResponse(
                output=serialize_output(response_array),
                shape=list(response_array.shape),
                dtype=str(response_array.dtype),
                metadata=meta,
            )
        except Exception as e:
            self.logger.exception("Predict failed", extra={"corr_id": corr_id})
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse(metadata={"error": str(e), "corr_id": corr_id})

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

    def StreamPredict(self, request_iterator, context):
        """Streaming prediction supporting chunked input and streamed output.

        Supports multiple logical samples per stream, delimited by end_of_sequence flags.
        Enforces basic size limits and respects client cancellation.

        Batch Processing:
            When batch_size > 1 is specified via session options or context,
            tiles are accumulated and processed together for efficiency.
            Results are still yielded individually to maintain streaming semantics.
        """

        corr_id = uuid.uuid4().hex[:8]
        self.logger.info("StreamPredict called", extra={"corr_id": corr_id})

        buffer = []
        shape = None
        dtype = None
        options = {}
        model_id = None
        model_obj = None
        mtype = None
        model_metadata = None
        record = None
        session_id = None
        session_state = None
        max_tile_bytes = self._default_tile_bytes
        max_inflight = self._default_max_inflight
        max_chunk_bytes = self._default_chunk_bytes
        last_tile_id = None

        # Batching state
        batch_size = DEFAULT_BATCH_SIZE
        pending_tiles = []  # List of (tile_id, input_bytes, shape, dtype, options)

        def _process_batch(tiles):
            """Process multiple tiles in a batch and yield individual responses.

            Args:
                tiles: List of (tile_id, input_bytes, shape, dtype, tile_options) tuples

            Yields:
                StreamPredictResponse for each tile in the batch
            """
            nonlocal model_id, model_obj, mtype, model_metadata, session_state, session_id

            if not tiles:
                return

            device_sel = self._next_device()
            device_id = device_sel.get("id", "cpu")

            # Get device-specific model if available
            device_model = self.model_manager.get_model_for_device(model_id, device_id)
            effective_model = device_model if device_model is not None else model_obj

            task_type = (options.get("task_type") or model_metadata.get("task_name") or "").lower()

            # Process each tile (future: could batch into single model call for compatible models)
            for tile_id, input_bytes, tile_shape, tile_dtype, tile_opts in tiles:
                try:
                    if len(input_bytes) > max_tile_bytes:
                        meta = {
                            "error": f"tile exceeds max bytes: {len(input_bytes)} > {max_tile_bytes}",
                            "corr_id": corr_id,
                        }
                        if session_id:
                            meta["session_id"] = session_id
                        if tile_id:
                            meta["tile_id"] = tile_id
                        if session_state is not None:
                            session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                            session_state.setdefault("errors", []).append(meta["error"])
                        yield inference_pb2.StreamPredictResponse(
                            metadata=meta,
                            end_of_sequence=True,
                        )
                        continue

                    input_array = deserialize_input(input_bytes, shape=tile_shape, dtype=tile_dtype)

                    # Inference
                    if mtype == "pytorch":
                        input_tensor = torch.from_numpy(input_array).float()
                        output = torch_rpc.rpc_sync(
                            "worker", "_rpc_infer", args=(input_tensor,), timeout=RPC_INFER_TIMEOUT_SECONDS
                        )
                        if torch.is_tensor(output):
                            output = output.detach().cpu().numpy()
                    else:
                        output = effective_model.predict(input_array, return_embeddings=False)

                    if not isinstance(output, np.ndarray):
                        output = np.array(output)

                    meta = {
                        "corr_id": corr_id,
                        "model_type": mtype,
                        "model_id": model_id,
                        "device_id": device_sel.get("id"),
                        "device_type": device_sel.get("type"),
                        "batch_size": str(len(tiles)),
                    }
                    if session_id:
                        meta["session_id"] = session_id
                    if tile_id:
                        meta["tile_id"] = tile_id
                    if session_state:
                        total_tiles = session_state.get("total_tiles")
                        done = (
                            session_state.get("ok_tiles", 0)
                            + session_state.get("failed_tiles", 0)
                            + 1
                        )
                        if total_tiles:
                            meta["progress"] = f"{done}/{total_tiles}"

                    response_array = output
                    confidence_array = None
                    if task_type == "segmentation":
                        try:
                            # Use confidence-preserving version
                            masks, confidence_array = prepare_segmentation_mask_with_confidence(
                                output
                            )
                            response_array = masks
                            meta["result_type"] = "segmentation_mask"
                            meta["mask_shape"] = str(list(masks.shape))
                            meta["mask_dtype"] = str(masks.dtype)
                            # Include confidence stats in metadata
                            if confidence_array is not None:
                                meta["confidence_min"] = f"{float(confidence_array.min()):.4f}"
                                meta["confidence_max"] = f"{float(confidence_array.max()):.4f}"
                                meta["confidence_mean"] = f"{float(confidence_array.mean()):.4f}"
                                meta["has_confidence"] = "true"
                        except Exception as exc:
                            meta["result_type"] = "raw"
                            meta["task_warning"] = f"segmentation_fallback:{exc}"

                    # Pack response with optional confidence data
                    response_array = np.ascontiguousarray(response_array)
                    if session_state is not None:
                        session_state["ok_tiles"] = session_state.get("ok_tiles", 0) + 1

                    # If confidence available, pack both mask and confidence into bundle
                    if confidence_array is not None:
                        bundle = pack_tensor_bundle(
                            {
                                "mask": response_array,
                                "confidence": np.ascontiguousarray(confidence_array),
                            }
                        )
                        meta["packed_bundle"] = "true"
                        yield inference_pb2.StreamPredictResponse(
                            output=bundle,
                            shape=list(response_array.shape),
                            dtype=str(response_array.dtype),
                            end_of_sequence=True,
                            metadata=meta,
                        )
                    else:
                        yield inference_pb2.StreamPredictResponse(
                            output=serialize_output(response_array),
                            shape=list(response_array.shape),
                            dtype=str(response_array.dtype),
                            end_of_sequence=True,
                            metadata=meta,
                        )
                except Exception as exc:
                    if session_state is not None:
                        session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                        session_state.setdefault("errors", []).append(str(exc))
                    error_meta = {"error": str(exc), "corr_id": corr_id}
                    if session_id:
                        error_meta["session_id"] = session_id
                    if tile_id:
                        error_meta["tile_id"] = tile_id
                    yield inference_pb2.StreamPredictResponse(
                        metadata=error_meta,
                        end_of_sequence=True,
                    )

        def _flush_sample(tile_id: str | None = None):
            nonlocal \
                buffer, \
                shape, \
                dtype, \
                options, \
                model_id, \
                model_obj, \
                mtype, \
                model_metadata, \
                session_state

            if not buffer:
                return None
            if not shape or not dtype:
                raise ValueError("missing shape or dtype in stream")

            device_sel = self._next_device()
            device_id = device_sel.get("id", "cpu")

            # Get device-specific model if available
            device_model = self.model_manager.get_model_for_device(model_id, device_id)
            effective_model = device_model if device_model is not None else model_obj

            input_bytes = b"".join(buffer)
            if len(input_bytes) > max_tile_bytes:
                meta = {
                    "error": f"stream sample exceeds max message limit: {len(input_bytes)} > {max_tile_bytes}",
                    "corr_id": corr_id,
                }
                if session_id:
                    meta["session_id"] = session_id
                if tile_id:
                    meta["tile_id"] = tile_id
                if session_state is not None:
                    session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                    session_state.setdefault("errors", []).append(meta["error"])
                buffer = []
                return inference_pb2.StreamPredictResponse(
                    metadata=meta,
                    end_of_sequence=True,
                )

            input_array = deserialize_input(input_bytes, shape=shape, dtype=dtype)

            # Inference using device-specific model
            if mtype == "pytorch":
                input_tensor = torch.from_numpy(input_array).float()
                output = torch_rpc.rpc_sync(
                    "worker", "_rpc_infer", args=(input_tensor,), timeout=RPC_INFER_TIMEOUT_SECONDS
                )
                if torch.is_tensor(output):
                    output = output.detach().cpu().numpy()
            else:
                output = effective_model.predict(input_array, return_embeddings=False)

            if not isinstance(output, np.ndarray):
                output = np.array(output)

            task_type = (options.get("task_type") or model_metadata.get("task_name") or "").lower()
            meta = {
                "corr_id": corr_id,
                "model_type": mtype,
                "model_id": model_id,
                "device_id": device_sel.get("id"),
                "device_type": device_sel.get("type"),
            }
            if session_id:
                meta["session_id"] = session_id
            if tile_id:
                meta["tile_id"] = tile_id
            if session_state:
                total_tiles = session_state.get("total_tiles")
                done = session_state.get("ok_tiles", 0) + session_state.get("failed_tiles", 0) + 1
                if total_tiles:
                    meta["progress"] = f"{done}/{total_tiles}"

            response_array = output
            if task_type == "segmentation":
                try:
                    masks = prepare_segmentation_mask(output)
                    response_array = masks
                    meta["result_type"] = "segmentation_mask"
                    meta["mask_shape"] = str(list(masks.shape))
                    meta["mask_dtype"] = str(masks.dtype)
                except Exception as exc:
                    meta["result_type"] = "raw"
                    meta["task_warning"] = f"segmentation_fallback:{exc}"

            response_array = np.ascontiguousarray(response_array)
            resp = inference_pb2.StreamPredictResponse(
                output=serialize_output(response_array),
                shape=list(response_array.shape),
                dtype=str(response_array.dtype),
                end_of_sequence=True,
                metadata=meta,
            )
            buffer = []
            return resp

        try:
            for req in request_iterator:
                if hasattr(context, "cancelled") and callable(getattr(context, "cancelled")):
                    cancelled_flag = False
                    try:
                        cancelled_flag = bool(context.cancelled())
                    except Exception:
                        cancelled_flag = False
                    if cancelled_flag:
                        # Flush any pending batch before cancellation
                        if pending_tiles:
                            for resp in _process_batch(pending_tiles):
                                yield resp
                            pending_tiles = []
                        raise RuntimeError("client_cancelled")
                if model_id is None:
                    model_id = req.model_id
                    session_id = req.context.get("session_id") if req.context else None
                    # Extract batch_size from context
                    if req.context and req.context.get("batch_size"):
                        try:
                            batch_size = max(1, int(req.context.get("batch_size")))
                        except (ValueError, TypeError):
                            batch_size = DEFAULT_BATCH_SIZE
                    if session_id:
                        session_state = self._sessions.get(session_id)
                        if session_state is None:
                            session_state = {
                                "model_id": model_id,
                                "created_at": time.time(),
                                "max_inflight": max_inflight,
                                "max_chunk_bytes": max_chunk_bytes,
                                "max_tile_bytes": max_tile_bytes,
                                "batch_size": batch_size,
                                "ok_tiles": 0,
                                "failed_tiles": 0,
                                "errors": [],
                            }
                            self._sessions[session_id] = session_state
                        max_inflight = int(session_state.get("max_inflight", max_inflight))
                        max_tile_bytes = int(session_state.get("max_tile_bytes", max_tile_bytes))
                        max_chunk_bytes = int(session_state.get("max_chunk_bytes", max_chunk_bytes))
                        batch_size = int(session_state.get("batch_size", batch_size))
                    record = self.model_manager.get(model_id)
                    if record is None:
                        raise RuntimeError("model_id not found")
                    model_obj = record.model
                    mtype = record.backend
                    model_metadata = record.metadata or {}
                    if mtype == "pytorch":
                        if not self._rpc_driver_initialized:
                            self._ensure_rpc_driver()
                        if not self.rpc_manager.is_alive():
                            self.rpc_manager.ensure_worker(
                                record.spec,
                                master_addr=self._rpc_master_addr,
                                master_port=self._rpc_master_port,
                            )
                        if not _await_rpc_worker_ready(timeout=RPC_READY_TIMEOUT_SECONDS):
                            raise RuntimeError("RPC worker not ready")
                if shape is None:
                    shape = tuple(req.shape) if req.shape else None
                if dtype is None:
                    dtype = req.dtype or "float32"
                if not options and req.context:
                    options = dict(req.context)

                if req.context and req.context.get("tile_id"):
                    last_tile_id = req.context.get("tile_id")

                chunk = req.chunk or b""
                if len(chunk) > max_chunk_bytes:
                    if session_state is not None:
                        session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                        session_state.setdefault("errors", []).append(
                            f"chunk_over_limit:{len(chunk)}"
                        )
                    error_meta = {
                        "error": f"chunk exceeds max_chunk_bytes: {len(chunk)} > {max_chunk_bytes}",
                        "corr_id": corr_id,
                    }
                    if session_id:
                        error_meta["session_id"] = session_id
                    if last_tile_id:
                        error_meta["tile_id"] = last_tile_id
                    resp = inference_pb2.StreamPredictResponse(
                        metadata=error_meta,
                        end_of_sequence=True,
                    )
                    # count as failed
                    if session_state is not None:
                        session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 0
                    yield resp
                    buffer = []
                    shape = None
                    dtype = None
                    options = {}
                    return

                buffer.append(chunk)

                if len(buffer) > max_inflight:
                    error_meta = {
                        "error": "backpressure: too many buffered chunks",
                        "corr_id": corr_id,
                    }
                    if session_id:
                        error_meta["session_id"] = session_id
                    if last_tile_id:
                        error_meta["tile_id"] = last_tile_id
                    resp = inference_pb2.StreamPredictResponse(
                        metadata=error_meta, end_of_sequence=True
                    )
                    if session_state is not None:
                        session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                        session_state.setdefault("errors", []).append(error_meta["error"])
                    yield resp
                    buffer = []
                    shape = None
                    dtype = None
                    options = {}
                    return

                if req.end_of_sequence:
                    tile_id = None
                    if req.context and req.context.get("tile_id"):
                        tile_id = req.context.get("tile_id")
                    if tile_id is None:
                        tile_id = last_tile_id
                    last_tile_id = tile_id or last_tile_id

                    # Accumulate tile for batching
                    input_bytes = b"".join(buffer)
                    tile_shape = shape
                    tile_dtype = dtype
                    tile_opts = dict(options)
                    pending_tiles.append((tile_id, input_bytes, tile_shape, tile_dtype, tile_opts))

                    # Reset buffer for next tile
                    buffer = []
                    shape = None
                    dtype = None
                    options = {}

                    # Process batch when full
                    if len(pending_tiles) >= batch_size:
                        try:
                            for result in _process_batch(pending_tiles):
                                yield result
                        except Exception as exc:
                            if session_state is not None:
                                session_state["failed_tiles"] = session_state.get(
                                    "failed_tiles", 0
                                ) + len(pending_tiles)
                                session_state.setdefault("errors", []).append(str(exc))
                            raise
                        finally:
                            pending_tiles = []
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception("StreamPredict failed", extra={"corr_id": corr_id})
            # Flush pending tiles before error response
            if pending_tiles:
                try:
                    for result in _process_batch(pending_tiles):
                        yield result
                except Exception:
                    pass
                pending_tiles = []
            if session_state is not None:
                session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                session_state.setdefault("errors", []).append(str(exc))
            error_meta = {"error": str(exc), "corr_id": corr_id}
            if session_id:
                error_meta["session_id"] = session_id
            if last_tile_id:
                error_meta["tile_id"] = last_tile_id
            # Clear buffer to avoid double-flush after error
            buffer = []
            shape = None
            dtype = None
            options = {}
            yield inference_pb2.StreamPredictResponse(metadata=error_meta, end_of_sequence=True)
            return

        # Flush any pending batch tiles
        if pending_tiles:
            try:
                for result in _process_batch(pending_tiles):
                    yield result
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.exception(
                    "StreamPredict failed flushing pending batch", extra={"corr_id": corr_id}
                )
                if session_state is not None:
                    session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + len(
                        pending_tiles
                    )
                    session_state.setdefault("errors", []).append(str(exc))
                error_meta = {"error": str(exc), "corr_id": corr_id}
                if session_id:
                    error_meta["session_id"] = session_id
                yield inference_pb2.StreamPredictResponse(metadata=error_meta, end_of_sequence=True)
            finally:
                pending_tiles = []

        # Flush any trailing buffer without end_of_sequence (legacy support)
        if buffer:
            # Add to pending and process
            input_bytes = b"".join(buffer)
            pending_tiles.append((last_tile_id, input_bytes, shape, dtype, dict(options)))
            try:
                for result in _process_batch(pending_tiles):
                    yield result
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.exception(
                    "StreamPredict failed in trailing flush", extra={"corr_id": corr_id}
                )
                if session_state is not None:
                    session_state["failed_tiles"] = session_state.get("failed_tiles", 0) + 1
                    session_state.setdefault("errors", []).append(str(exc))
                error_meta = {"error": str(exc), "corr_id": corr_id}
                if session_id:
                    error_meta["session_id"] = session_id
                if last_tile_id:
                    error_meta["tile_id"] = last_tile_id
                yield inference_pb2.StreamPredictResponse(metadata=error_meta, end_of_sequence=True)
                return
        # Reset buffer
        buffer = []

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
            self.logger.debug(f"Cleaned {cleaned} stale sessions", extra={"corr_id": corr_id})

        session_id = request.session_id or uuid.uuid4().hex
        transport = request.transport
        max_inflight = transport.max_inflight or self._default_max_inflight
        max_chunk_bytes = transport.chunk_bytes or self._default_chunk_bytes
        max_tile_bytes = min(self._default_tile_bytes, GRPC_MAX_MESSAGE_LENGTH)
        if request.options.get("max_tile_bytes"):
            try:
                max_tile_bytes = min(max_tile_bytes, int(request.options.get("max_tile_bytes")))
            except Exception:
                pass

        if session_id in self._sessions:
            state = self._sessions[session_id]
            device_plan = [
                inference_pb2.DeviceInfo(
                    id=d.get("id", ""),
                    type=d.get("type", ""),
                    ordinal=int(d.get("ordinal", 0)),
                    weight=int(d.get("weight", 1)),
                )
                for d in self._device_plan
            ]
            return inference_pb2.OpenSessionResponse(
                session_id=session_id,
                device_plan=device_plan,
                model_cache_hit=True,
                max_inflight=int(state.get("max_inflight", max_inflight)),
                max_tile_bytes=int(state.get("max_tile_bytes", max_tile_bytes)),
                status="ok",
            )

        spec = request.spec
        if not spec.model_id and not spec.source:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session requires model_id or spec.source")
            return inference_pb2.OpenSessionResponse(
                session_id=session_id, status="error", error="missing model"
            )

        model_cache_hit = False
        model_id = spec.model_id or ""
        record = self.model_manager.get(model_id) if model_id else None
        if record is not None:
            model_cache_hit = True
            model_id = model_id or record.model_id if hasattr(record, "model_id") else spec.model_id
        else:

            class _ContextProxy:
                def __init__(self):
                    self.code = None
                    self.details = None

                def set_code(self, code):
                    self.code = code

                def set_details(self, details):
                    self.details = details

            proxy_ctx = _ContextProxy()
            load_resp = self.LoadModel(inference_pb2.LoadModelRequest(spec=spec), proxy_ctx)
            if not getattr(load_resp, "success", False):
                context.set_code(proxy_ctx.code or grpc.StatusCode.INTERNAL)
                context.set_details(proxy_ctx.details or load_resp.message or "load failed")
                return inference_pb2.OpenSessionResponse(
                    session_id=session_id,
                    status="error",
                    error=proxy_ctx.details or load_resp.message or "load failed",
                )
            model_id = load_resp.model_id

        device_plan = [
            inference_pb2.DeviceInfo(
                id=d.get("id", ""),
                type=d.get("type", ""),
                ordinal=int(d.get("ordinal", 0)),
                weight=int(d.get("weight", 1)),
            )
            for d in self._device_plan
        ]

        total_tiles = request.manifest.total_tiles if request.manifest else None
        self._sessions[session_id] = {
            "model_id": model_id,
            "created_at": time.time(),
            "max_inflight": int(max_inflight),
            "max_chunk_bytes": int(max_chunk_bytes),
            "max_tile_bytes": int(max_tile_bytes),
            "total_tiles": int(total_tiles) if total_tiles else None,
            "ok_tiles": 0,
            "failed_tiles": 0,
            "errors": [],
            "status": "open",
        }

        return inference_pb2.OpenSessionResponse(
            session_id=session_id,
            device_plan=device_plan,
            model_cache_hit=model_cache_hit,
            max_inflight=int(max_inflight),
            max_tile_bytes=int(max_tile_bytes),
            status="ok",
        )

    def CloseSession(self, request, context):
        session_id = request.session_id
        if not session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return inference_pb2.CloseSessionResponse(status="error", error="missing session_id")

        state = self._sessions.pop(session_id, None)
        if not state:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("session_id not found")
            return inference_pb2.CloseSessionResponse(status="not_found", error="session not found")

        # Mark terminal status before summarizing and removing
        if state.get("status") != "cancelled":
            state["status"] = "closed"

        duration_ms = int((time.time() - state.get("created_at", time.time())) * 1000)
        summary = inference_pb2.SessionSummary(
            session_id=session_id,
            ok_tiles=int(state.get("ok_tiles", 0)),
            failed_tiles=int(state.get("failed_tiles", 0)),
            duration_ms=duration_ms,
            errors=list(state.get("errors", [])),
        )
        status_label = state.get("status") or "closed"
        return inference_pb2.CloseSessionResponse(status=status_label, summary=summary)

    def CancelSession(self, request, context):
        session_id = request.session_id
        if not session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return inference_pb2.CancelSessionResponse(status="error", error="missing session_id")

        state = self._sessions.get(session_id)
        if not state:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("session_id not found")
            return inference_pb2.CancelSessionResponse(
                status="not_found", error="session not found"
            )

        state["status"] = "cancelled"
        reason = request.reason or "client_cancelled"
        state.setdefault("errors", []).append(reason)
        return inference_pb2.CancelSessionResponse(status="cancelled")

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
                if value is None:
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


def serve(
    port: int = 50051,
    *,
    model_cache_dir: str | Path | None = None,
    startup_event=None,
    stop_event=None,
    max_cores: int = 4,
) -> int:
    """Start the gRPC inference server.

    Args:
            port: TCP port to bind (50051 default). Use 0 for ephemeral.
            startup_event: threading.Event that will be set once the server is started.
            stop_event: threading.Event; if provided, the server thread will block until it is set then shut down.

    Returns:
            The actual bound port (useful if port=0 was passed).
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

    bound_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC inference server started on port {bound_port}")
    if startup_event is not None:
        startup_event.set()
    # Backward compatibility: discover stop_event from caller frame if not explicitly passed
    if stop_event is None:
        frame = inspect.currentframe().f_back
        stop_event = frame.f_locals.get("stop_event", None)
    if stop_event is not None:
        stop_event.wait()
        logging.info("Shutting down gRPC server after test.")
        server.stop(0)
    else:
        server.wait_for_termination()
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
        help="Directory where downloaded model artifacts are stored (default: ~/.cache/nxmndr/models)",
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
