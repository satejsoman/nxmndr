# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The PyTorch RPC worker: a child process that holds generic PyTorch models.

The server's ``RpcWorkerManager`` (``nxmndr.server.managers``) starts one worker per
server process and calls these functions over ``torch.distributed.rpc``. RPC calls
pass the function objects below, never their names (``rpc_sync`` rejects a string).

Models are routed by identity: the worker keeps a registry keyed by the model cache
record's model ID. ``_rpc_load(model_id, spec)`` builds one model, ``_rpc_infer``
runs exactly that model (``UnknownModelError`` if it is not loaded), ``_rpc_unload``
frees it, ``_rpc_stop`` ends the worker. One ID never names two models.
"""

import hashlib
import os
import socket
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from ..logging import get_logger

logger = get_logger(__name__)

# The model that ``run_worker(model_spec=...)`` preloads and ``_rpc_load_model``
# replaces: the single slot of ``TorchRpcInferenceProvider``. The server's model IDs
# are 32-character hex strings, so they never collide with it.
DEFAULT_MODEL_ID = "default"


class UnknownModelError(LookupError):
    """``_rpc_infer`` named a model ID that this worker has not loaded (or has unloaded)."""


class _LoadedModel:
    __slots__ = ("model", "device", "fingerprint")

    def __init__(self, model, device: str, fingerprint: str):
        self.model = model
        self.device = device
        self.fingerprint = fingerprint


# Use global state for RPC worker since each RPC call might be in different threads
_worker_state = {
    "models": {},  # model_id -> _LoadedModel
    "loading": set(),  # model IDs whose _rpc_load is running
    "spec": None,
    "initialized": False,
    "stop_requested": False,
    "pg_initialized": False,
    "rpc_initialized": False,
}
_models_lock = threading.Lock()


def _get_worker_state():
    """Get the global worker state."""
    return _worker_state


def _spec_value(spec, name, default=None):
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _load_model(model_spec, device: str = "cpu"):
    """Build ``model_spec``: a registered class, or a catalog name (``nxmndr.models.catalog``)."""

    model_class = _spec_value(model_spec, "model_class")
    target = "cpu"
    if device and device.startswith("cuda") and torch.cuda.is_available():
        target = device
    model_path = _spec_value(model_spec, "model_path")
    if isinstance(model_class, str):
        from ..models import catalog

        model = catalog.build(model_class, model_path, _spec_value(model_spec, "catalog_args"))
    elif callable(model_class):
        model = model_class()
        model.load_state_dict(
            torch.load(model_path, map_location=target, weights_only=True), strict=False
        )
    else:
        raise TypeError(
            f"model_class {model_class!r} is not a PyTorch model class or a catalog name; the "
            "server resolves ModelSpec.model_class by the names registered with "
            "register_pytorch_model, then by nxmndr.models.catalog"
        )
    if target != "cpu":
        model.to(target)
    model.eval()
    return model


def _model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def _fingerprint(model) -> str:
    """sha256 of the model class and every state-dict entry (name, dtype, shape, bytes)."""

    digest = hashlib.sha256()
    cls = type(model)
    digest.update(f"{cls.__module__}.{cls.__qualname__}".encode("utf-8"))
    digest.update(str(getattr(model, "catalog_name", "")).encode("utf-8"))
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"|{name}|{value.dtype}|{tuple(value.shape)}|".encode("utf-8"))
        if value.numel():
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _register(model_id: str, model) -> dict:
    entry = _LoadedModel(model, _model_device(model), _fingerprint(model))
    with _models_lock:
        _worker_state["models"][model_id] = entry
    info = {"model_id": model_id, "device": entry.device, "fingerprint": entry.fingerprint}
    if hasattr(model, "catalog_info"):  # a catalog model: its family, class and arguments
        info["catalog"] = model.catalog_info()
    return info


def _rpc_ping():
    """Ping function to check if worker is alive."""
    return {"status": "ok", "timestamp": time.time()}


def _rpc_load(model_id, spec, device="cpu"):
    """Build the model of ``spec`` under ``model_id``; return its device and fingerprint.

    ``spec`` has ``model_class`` (a registered class itself, or a catalog name with its
    ``catalog_args``), ``model_path`` and ``name``, as a mapping or as attributes. A
    model ID that is loaded or loading is refused, so one ID never names two models.
    A catalog model's result also has ``catalog`` (``CatalogModel.catalog_info``).
    """

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    with _models_lock:
        if model_id in _worker_state["models"] or model_id in _worker_state["loading"]:
            raise ValueError(f"model {model_id} is already loaded in this PyTorch RPC worker")
        _worker_state["loading"].add(model_id)
    try:
        info = _register(model_id, _load_model(spec, device))
    finally:
        with _models_lock:
            _worker_state["loading"].discard(model_id)
    logger.info(
        "Loaded model %s (%s) on %s, %s",
        model_id, _spec_value(spec, "name") or "unnamed", info["device"], info["fingerprint"],
    )
    return info


def _rpc_unload(model_id):
    """Free the model of ``model_id``. True if it was loaded."""

    with _models_lock:
        entry = _worker_state["models"].pop(model_id, None)
    if entry is None:
        return False
    on_cuda = entry.device.startswith("cuda")
    del entry
    if on_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Unloaded model %s", model_id)
    return True


def _rpc_load_model(spec_dict):
    """Load a model into the ``DEFAULT_MODEL_ID`` slot (``TorchRpcInferenceProvider``).

    Args:
        spec_dict: Dictionary containing all model information including:
            - model_class: The actual model class (serialized by torch.distributed.rpc)
            - model_path: Path to model weights
            - name: Model name
            - type: Model type (e.g., 'pytorch')
    """
    logger.info(f"Loading model on RPC worker: {spec_dict.get('name', 'unnamed')}")

    try:
        state = _get_worker_state()

        # Extract model information
        model_class = spec_dict["model_class"]
        model_path = spec_dict["model_path"]
        model_name = spec_dict.get("name", "unnamed")

        # Instantiate the model
        model = model_class()

        # Load weights if path exists
        if model_path and Path(model_path).exists():
            import torch

            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")

        # Set to evaluation mode
        model.eval()

        # Store in the provider's slot
        _register(DEFAULT_MODEL_ID, model)
        state["spec"] = spec_dict
        state["initialized"] = True

        logger.info(f"Successfully loaded model {model_name} on RPC worker")
        return {"success": True, "message": f"Model {model_name} loaded successfully"}

    except Exception as e:
        error_msg = f"Failed to load model on RPC worker: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


def _rpc_infer(model_id, tensor, device_hint="cpu"):
    with _models_lock:
        entry = _worker_state["models"].get(model_id)
    if entry is None:
        raise UnknownModelError(f"model {model_id} is not loaded in the PyTorch RPC worker")
    model = entry.model
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")
    if device_hint.startswith("cuda") and torch.cuda.is_available() and model_device.type == "cuda":
        tensor = tensor.to(model_device, non_blocking=True)
    with torch.no_grad():
        out = model(tensor)
    if torch.is_tensor(out) and out.device.type != "cpu":
        out = out.detach().cpu()
    return out


def _rpc_stop():
    state = _get_worker_state()
    state["stop_requested"] = True
    return {"status": "stopping"}


def _rpc_status():
    state = _get_worker_state()
    with _models_lock:
        models = {
            mid: {"device": entry.device, "fingerprint": entry.fingerprint}
            for mid, entry in state["models"].items()
        }
    return {
        "status": "ok" if state.get("rpc_initialized") else "initializing",
        "model_loaded": bool(models),
        "models": models,
        "stop_requested": state.get("stop_requested"),
        "pid": os.getpid(),
    }


def _init_dist(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize torch.distributed process group in a forward compatible way.

    PyTorch 2.0 deprecates directly constructing ProcessGroup objects; use init_process_group.
    Idempotent: safe to call multiple times inside same process.
    """
    state = _get_worker_state()
    if state.get("pg_initialized", False):
        return
    if dist.is_available() and not dist.is_initialized():
        logger.info(
            "Initializing process group backend=%s rank=%d world_size=%d", backend, rank, world_size
        )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    state["pg_initialized"] = True


def _signal_ready(master_addr: str, ready_port: int, timeout: float) -> None:
    """Tell the parent this process is about to rendezvous (loopback TCP, one line)."""

    with socket.create_connection((master_addr, ready_port), timeout=timeout) as conn:
        conn.sendall(f"ready {os.getpid()}\n".encode("ascii"))


def run_worker(
    worker_name,
    model_spec=None,
    rank=1,
    world_size=2,
    master_addr="127.0.0.1",
    master_port=29500,
    backend: str = "gloo",
    *,
    ready_port: int = 0,
    startup_timeout_s: float = 0.0,
):
    """Entry point for RPC worker process.

    - Sets env vars for rendezvous
    - Preloads ``model_spec`` into the ``DEFAULT_MODEL_ID`` slot when given
    - With ``ready_port``, tells the parent that it is about to rendezvous
    - Initializes torch.distributed process group (forward compatible API), bounded by
      ``startup_timeout_s`` when given
    - Starts RPC framework
    - Serves ``_rpc_*`` calls until ``_rpc_stop`` or until its parent process is gone
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    state = _get_worker_state()
    parent_pid = os.getppid()
    if model_spec is not None:
        logger.info("Loading model for worker %s", worker_name)
        _register(DEFAULT_MODEL_ID, _load_model(model_spec))
    logger.info(
        "Ensuring process group then initializing RPC (name=%s rank=%d world_size=%d)",
        worker_name,
        rank,
        world_size,
    )
    init_method = f"tcp://{master_addr}:{master_port}"
    pg_kwargs = {}
    opts = rpc.TensorPipeRpcBackendOptions(init_method=init_method)
    if startup_timeout_s:
        pg_kwargs["timeout"] = timedelta(seconds=float(startup_timeout_s))
        opts.rpc_timeout = float(startup_timeout_s)
    if ready_port:
        _signal_ready(master_addr, int(ready_port), float(startup_timeout_s or 60.0))
    if dist.is_available() and not dist.is_initialized():
        logger.info("Initializing process group (worker) backend=gloo %s", init_method)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size, init_method=init_method, **pg_kwargs
        )
    rpc.init_rpc(worker_name, rank=rank, world_size=world_size, rpc_backend_options=opts)
    logger.info("RPC initialized for worker %s", worker_name)
    state["rpc_initialized"] = True
    orphaned = False
    try:
        while True:
            if state.get("stop_requested", False):
                logger.info("Stop requested via _rpc_stop; exiting loop")
                break
            if os.getppid() != parent_pid:
                logger.warning("Parent process %d is gone; exiting", parent_pid)
                orphaned = True
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down RPC worker %s", worker_name)
        try:
            # A graceful shutdown waits for the driver's own; an orphan has no driver.
            rpc.shutdown(graceful=not orphaned)
        finally:
            if dist.is_available() and dist.is_initialized():  # best-effort cleanup
                try:
                    dist.destroy_process_group()
                    logger.info("Destroyed process group")
                except Exception:
                    logger.exception("Failed destroying process group")
