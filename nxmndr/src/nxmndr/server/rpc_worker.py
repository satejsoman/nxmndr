# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from ..logging import get_logger

logger = get_logger(__name__)

# Use global state for RPC worker since each RPC call might be in different threads
_worker_state = {
    "model": None,
    "spec": None,
    "initialized": False,
    "stop_requested": False,
    "pg_initialized": False,
    "rpc_initialized": False,
}


def _get_worker_state():
    """Get the global worker state."""
    return _worker_state


def _load_model(model_spec, device: str = "cpu"):
    model = model_spec.model_class()
    target = "cpu"
    if device and device.startswith("cuda") and torch.cuda.is_available():
        target = device
    model.load_state_dict(torch.load(model_spec.model_path, map_location=target, weights_only=True), strict=False)
    if target != "cpu":
        model.to(target)
    model.eval()
    return model


def _rpc_ping():
    """Ping function to check if worker is alive."""
    return {"status": "ok", "timestamp": time.time()}


def _rpc_load_model(spec_dict):
    """Load a model on the RPC worker from a complete specification.

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

        # Store in global state
        state["model"] = model
        state["spec"] = spec_dict
        state["initialized"] = True

        logger.info(f"Successfully loaded model {model_name} on RPC worker")
        return {"success": True, "message": f"Model {model_name} loaded successfully"}

    except Exception as e:
        error_msg = f"Failed to load model on RPC worker: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


def _rpc_infer(tensor, device_hint="cpu"):
    state = _get_worker_state()
    if state["model"] is None:
        raise RuntimeError("Model not initialized on RPC worker")
    try:
        model_device = next(state["model"].parameters()).device
    except Exception:
        model_device = torch.device("cpu")
    if device_hint.startswith("cuda") and torch.cuda.is_available() and model_device.type == "cuda":
        tensor = tensor.to(model_device, non_blocking=True)
    with torch.no_grad():
        out = state["model"](tensor)
    if torch.is_tensor(out) and out.device.type != "cpu":
        out = out.detach().cpu()
    return out


def _rpc_stop():
    state = _get_worker_state()
    state["stop_requested"] = True
    return {"status": "stopping"}


def _rpc_status():
    state = _get_worker_state()
    return {
        "status": "ok" if state.get("rpc_initialized") else "initializing",
        "model_loaded": state.get("model") is not None,
        "stop_requested": state.get("stop_requested"),
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


def run_worker(
    worker_name,
    model_spec,
    rank=1,
    world_size=2,
    master_addr="127.0.0.1",
    master_port=29500,
    backend: str = "gloo",
):
    """Entry point for RPC worker process.

    - Sets env vars for rendezvous
    - Initializes torch.distributed process group (forward compatible API)
    - Starts RPC framework
    - Loads model into global singleton
    - Blocks indefinitely until external termination
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    state = _get_worker_state()
    logger.info("Loading model for worker %s", worker_name)
    state["model"] = _load_model(model_spec)
    logger.info(
        "Model loaded; ensuring process group then initializing RPC (name=%s rank=%d world_size=%d)",
        worker_name,
        rank,
        world_size,
    )
    init_method = f"tcp://{master_addr}:{master_port}"
    if dist.is_available() and not dist.is_initialized():
        logger.info("Initializing process group (worker) backend=gloo %s", init_method)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size, init_method=init_method
        )
    opts = rpc.TensorPipeRpcBackendOptions(init_method=init_method)
    rpc.init_rpc(worker_name, rank=rank, world_size=world_size, rpc_backend_options=opts)
    logger.info("RPC initialized for worker %s", worker_name)
    state["rpc_initialized"] = True
    try:
        while True:
            if state.get("stop_requested", False):
                logger.info("Stop requested via _rpc_stop; exiting loop")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down RPC worker %s", worker_name)
        try:
            rpc.shutdown()
        finally:
            if dist.is_available() and dist.is_initialized():  # best-effort cleanup
                try:
                    dist.destroy_process_group()
                    logger.info("Destroyed process group")
                except Exception:
                    logger.exception("Failed destroying process group")
