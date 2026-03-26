# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for Torch RPC inference provider end-to-end flow.

Focus:
    * Worker process lifecycle (spawn -> ping -> infer -> shutdown)
    * Lazy driver RPC initialization
    * Proper teardown (remote _rpc_stop + torch_rpc.shutdown)

Notes:
    * Skips if torch.distributed is unavailable (e.g., minimal CPU-only build missing RPC backend).
    * Uses a dynamically chosen free port to avoid collisions in CI.
"""

import logging
import multiprocessing
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.rpc as torch_rpc

from nxmndr.inference import InferenceSession, TorchRpcInferenceProvider
from nxmndr.models import PytorchModelSpec, register_pytorch_model
from nxmndr.server.rpc_worker import run_worker
from tst.example_model.modeling_exampleconv import ExampleModel

logging.basicConfig(
    level=logging.INFO, format="[TEST %(asctime)s] %(processName)s %(levelname)s %(message)s"
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_worker(spec_dict):
    """Worker process entrypoint wrapper.

    Creates a PytorchModelSpec and invokes run_worker with supplied rendezvous info.
    """
    logging.info("[worker] starting: %s", spec_dict)
    spec = PytorchModelSpec(
        model_class=ExampleModel,
        model_path=str(spec_dict["model_path"]),
        name=spec_dict.get("name"),
    )
    try:
        run_worker(
            worker_name="worker",
            model_spec=spec,
            rank=1,
            world_size=2,
            master_addr=spec_dict["addr"],
            master_port=spec_dict["port"],
        )
    except Exception:  # pragma: no cover - defensive
        logging.exception("[worker] exception")
        raise


@pytest.fixture(scope="module")
def rpc_environment():
    """Spawn RPC worker process and provide rendezvous info.

    Skips if example model weights missing.
    """
    model_path = Path(__file__).parent.parent / "example_model" / "example_model.pth"
    if not model_path.exists():  # pragma: no cover - environment expectation
        pytest.skip("example_model.pth missing; run test that exports it first.")

    # Register model class (idempotent) for spec JSON round-trips (future usage)
    register_pytorch_model("ExampleModel", ExampleModel)

    addr = "127.0.0.1"
    port = _find_free_port()
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)

    spec_dict = {
        "model_path": model_path,
        "name": "rpc-test",
        "addr": addr,
        "port": port,
    }
    worker_proc = multiprocessing.Process(target=_start_worker, args=(spec_dict,), daemon=True)
    worker_proc.start()

    yield {
        "model_path": model_path,
        "addr": addr,
        "port": port,
        "worker_proc": worker_proc,
    }

    # Teardown worker process & driver RPC agent if still active
    try:
        if torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
            try:
                torch_rpc.shutdown()
            except Exception:  # pragma: no cover - defensive
                logging.exception("driver RPC shutdown error")
        if worker_proc.is_alive():
            worker_proc.terminate()
            worker_proc.join(timeout=2)
            if worker_proc.is_alive():
                worker_proc.kill()
    finally:
        logging.info("RPC test environment teardown complete")


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed not available in this build"
)
def test_rpc_provider_initialization(rpc_environment):
    """End-to-end inference through TorchRpcInferenceProvider.

    Validates: ping(), session creation, inference output shape.
    """
    spec = PytorchModelSpec(
        model_class=ExampleModel, model_path=str(rpc_environment["model_path"]), name="rpc-test"
    )
    provider = TorchRpcInferenceProvider(
        worker_name="worker",
        master_addr=rpc_environment["addr"],
        master_port=rpc_environment["port"],
    )
    ping = provider.ping()
    assert ping.get("status") == "ok", f"ping failed: {ping}"
    assert ping.get("model_loaded") is True, "model not loaded on worker yet"

    session = InferenceSession(spec, provider)
    arr = np.zeros((1, 3, 32, 32), dtype=np.float32)
    out = session.run(arr)
    assert isinstance(out, np.ndarray), "Output not converted to ndarray"
    assert out.shape[0] == 1, f"Unexpected batch dimension in output: {out.shape}"
    provider.close()  # Ensure clean driver shutdown (fixture handles worker)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-vv"])
