# nxmndr

(named after / pronounced: **Anaximander**)

Backend and experimentation ground for multi-format vision inference (PyTorch, ONNX, HuggingFace) with local, remote (gRPC), and Torch RPC execution paths.

## Installation

### Editable Install (Recommended for Development)

Create a virtual environment (conda or venv) and install in editable mode:

```bash
# Using conda
conda create -n nxmndr python=3.11 -y
conda activate nxmndr

# Install with desired extras
pip install -e .                    # core only
pip install -e .[server]            # + server deps (aiohttp, azure-identity, openai)
pip install -e .[dev]               # + dev tools (pytest, ruff, mypy, etc.)
pip install -e .[server,dev]        # all extras
```

### Using requirements.txt Files

For users preferring flat requirements files:

```bash
pip install -r requirements.txt              # core runtime deps
pip install -r requirements-server.txt       # server deployment deps
pip install -r requirements-dev.txt          # dev tools (includes requirements.txt)
```

> **Note:** `pyproject.toml` is the source of truth for dependencies. The requirements files are maintained for legacy workflows.

### CUDA Note

If you need CUDA, install the appropriate Torch wheel *before* installing `nxmndr` so the dependency (`torch>=2.1.0`) is satisfied by the GPU build and not replaced by a CPU wheel.

Preferred order (example for CUDA 12.1):

```bash
# 1. Install a CUDA-enabled PyTorch wheel
pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Install nxmndr with desired extras (will reuse existing torch)
pip install -e .[server]
```

Verification:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Console Scripts

The package installs convenient entry points:

| Command | Purpose |
|---------|---------|
| `nxmndr-server` | Unified HTTP server (inference + optional Azure proxy) |
| `python -m nxmndr` | Multi-command launcher (see `--help`) |

### Examples Directory

Example scripts live under `examples/` (not installed as console scripts):
* `examples/client.py` (gRPC client utility)
* `examples/example_chatgpt_vision.py` (Azure ChatGPT Vision usage)

Legacy direct invocation via `python -m server.server` still works when running from source.

## Quickstart (Local PyTorch)

```python
from nxmndr.inference import InferenceSession, LocalInferenceProvider
from nxmndr.models import PytorchModelSpec
import torch.nn as nn

# Define a simple model (or import your own)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

spec = PytorchModelSpec(model_class=MyModel, model_path='tst/example_model/example_model.pth', name='example')
provider = LocalInferenceProvider()
session = InferenceSession(spec, provider)

import numpy as np
inp = np.zeros((1,3,32,32), dtype=np.float32)
out = session.run(inp)
print(out.shape)
```

## Remote Inference (gRPC)

1. Start the server (HTTP unified):

```bash
nxmndr-server --port 8080 --azure-proxy   # optional Azure proxy endpoints
```

Or start legacy gRPC server only:

```bash
python -m server.server --grpc-only
```

2. Load an ONNX model and predict:

```python
import grpc, numpy as np
from nxmndr.inference import InferenceSession, RemoteInferenceProvider
from nxmndr.models import OnnxModelSpec

channel = grpc.insecure_channel('localhost:50051')
provider = RemoteInferenceProvider(channel)
spec = OnnxModelSpec(model_path='tst/example_model/example_model.onnx', name='remote-test')
provider.load_spec(spec)
session = InferenceSession(spec, provider)
out = session.run(np.zeros((1,3,32,32), dtype=np.float32))
print(out.shape)
```

## Torch RPC (Experimental)

Used for remote execution of PyTorch models via `torch.distributed.rpc`. A worker process is spawned server-side for PyTorch specs, or started by hand, on the server's host or another one (external worker mode, below).

### External Worker Mode (worker on another host)

1. Start the server with `NXMNDR_RPC_EXTERNAL_WORKER=1`. It spawns no worker. `MASTER_ADDR` is the address it listens on (default `0.0.0.0`) and `MASTER_PORT` the rendezvous port (default `29500`).
2. On the worker host:

```bash
python -m nxmndr.server.rpc_worker --master <server address> --port <MASTER_PORT> --device cuda:0   # or cpu, mps
```

The worker waits until the server loads its first PyTorch model, reports on `MASTER_PORT + 1`, joins the two-rank RPC group at `MASTER_ADDR:MASTER_PORT`, and exits when the server stops it. If no worker reports within 60 s, `LoadModel` and `OpenSession` fail with `FAILED_PRECONDITION` naming the address; the next load waits again.

- Weights load on the worker host: the spec's `source` path must exist there (as for a remote nxmndr server). A client that uploads `artifact` bytes gets a path in the server's cache, which the worker host must also see.
- The worker host needs nxmndr with the same torch version, and the model code: the module of a class registered with `register_pytorch_model` must import there, and a catalog model needs its family package.
- Network: TCP between the hosts on `MASTER_PORT`, `MASTER_PORT + 1` and the ports gloo and TensorPipe open. Those transports pick an interface from the hostname; where it resolves to a loopback address (`::1` on the macOS test host; Debian-family hosts often map it to `127.0.1.1`), set `GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` to the LAN interface on both hosts. `torch.distributed.rpc` has no authentication and runs pickled calls: use it only on a trusted network.
- One worker per server run. A server that dies without stopping its worker leaves it running: stop it and start a new one. A worker that dies disables PyTorch models until the server restarts.

### Torch RPC Execution Flow

This path lets the driver process (rank 0) perform forward passes on a remote worker process (rank 1) without standing up a full gRPC model server for PyTorch weights.

High-level lifecycle:

1. Worker process spawn
	- Test harness (or future orchestrator) launches a Python process running `server.rpc_worker.run_worker(...)`.
	- Environment variables `MASTER_ADDR` / `MASTER_PORT` identify the rendezvous endpoint.
	- A minimal readiness sentinel file is written immediately so the driver can proceed (acts only as a process-start signal, not model-ready guarantee).
2. Model load (worker)
	- Worker constructs model from `PytorchModelSpec` and loads state dict to CPU (or GPU if later extended).
3. RPC initialization (worker)
	- Worker calls `rpc.init_rpc(worker_name, rank=1, world_size=2, backend=TensorPipe)` using an explicit `TensorPipeRpcBackendOptions(init_method=tcp://host:port)`.
	- This explicit init method prevents deprecated implicit ProcessGroup creation.
4. Driver lazy initialization
	- `TorchRpcInferenceProvider.ping()` or first `predict()` triggers `_init_rpc()` on the driver if not already initialized.
	- Driver initializes a matching RPC agent (`rank=0`).
5. Liveness (ping)
	- Driver repeatedly calls `rpc_sync(worker, _rpc_ping)` until it receives `{"status": "ok"}`.
	- `_rpc_ping` is intentionally trivial and does not depend on model load; success implies fabric is established.
6. Inference call
	- `InferenceSession.run()` -> provider.predict -> `rpc_sync(worker, _rpc_infer, tensor)`.
	- Driver-side `_rpc_infer` delegates dynamically into the worker module's `_rpc_infer` (ensures single implementation of the forward path on the worker).
7. Shutdown / cleanup
	- `provider.close()` issues a best-effort `_rpc_stop` RPC; worker loop exits, then driver calls `torch_rpc.shutdown()`.
	- Fixture teardown (tests) redundantly ensures shutdown and kills the worker process if still alive.

### Key Components

File / Symbol | Role
--------------|-----
`nxmndr/inference/inference.py: TorchRpcInferenceProvider` | Driver-side abstraction providing `ping()`, `predict()`, `close()`
`inference/inference.py: _rpc_ping` | Simple remote liveness probe
`inference/inference.py: _rpc_infer` | Delegation shim calling worker's `_rpc_infer`
`inference/inference.py: _rpc_stop` | Remote stop signal (delegates to worker)
`server/rpc_worker.py: run_worker` | Worker entrypoint (model load + RPC init + main loop)
`server/rpc_worker.py: _rpc_infer` | Actual model forward pass on the worker
`server/rpc_worker.py: _rpc_stop` | Sets a flag to break worker loop cleanly
`tst/test_rpc_pytorch.py` | Orchestrates end‑to‑end flow in tests

### Synchronization Strategy

- Readiness file: Only asserts the worker process started execution. It no longer waits for 'ready' content before proceeding; liveness is confirmed via ping.
- Ping loop: Ensures the RPC fabric is active before first inference call, removing prior races around import timing.

### Error Handling & Resilience (Current State)

- If `_rpc_infer` executes before the model is loaded (should not happen in current flow), a `RuntimeError` is raised from the worker.
- Ping retries for up to 5s with short backoff; raises `TimeoutError` if fabric not ready.
- Shutdown is best-effort; failures in `_rpc_stop` do not prevent driver shutdown.

### Known Limitations / Future Work

- The spawned worker builds models on CPU; an external worker uses its `--device`.
- No batching / streaming; each call is a single `rpc_sync` invocation.
- Error metadata is plain exceptions; could wrap into structured envelopes.
- Readiness file could be replaced with a pipe/queue IPC for cleaner startup semantics.

### Minimal Example (Conceptual)

```python
from nxmndr.inference import InferenceSession, TorchRpcInferenceProvider
from nxmndr.models import PytorchModelSpec
from my_models import MyNet

spec = PytorchModelSpec(model_class=MyNet, model_path='model.pth', name='net')
provider = TorchRpcInferenceProvider(worker_name='worker', master_addr='127.0.0.1', master_port=59391)
provider.ping()  # blocks until RPC worker reachable
session = InferenceSession(spec, provider)
out = session.run(np.zeros((1,3,32,32), dtype=np.float32))
provider.close()
```

### Rationale vs gRPC Path

Use Torch RPC when:
* You need low-overhead Python ↔ Python tensor forwarding without serializing to protobuf.
* You control both driver and worker environment tightly.

Use gRPC when:
* Language interop or deployment boundary crossing is required.
* You need model lifecycle management, multiple model formats, or network security layers.


## Protobuf API (High-Level)

Service: `InferenceService`

Endpoints:
- `LoadModel(LoadModelRequest) -> LoadModelResponse` : Uploads or references a model artifact; returns `model_id`.
- `ListModels(ListModelsRequest) -> ListModelsResponse` : Enumerate loaded models.
- `Predict(PredictRequest) -> PredictResponse` : Run a batch prediction (includes `model_id`).
- `UnloadModel(UnloadModelRequest) -> UnloadModelResponse` : Remove a model.
- `Capabilities` / `Health` : Introspection & readiness.
- `StreamPredict` : Bidirectional streaming for tiled/chunked inference with session support.

### Model Formats
Supported enums: `PYTORCH`, `ONNX`, `HUGGINGFACE`, `TORCHHUB`.

## Testing

Tests live in `tst/`:

```bash
pytest -q
```

Key tests:
- `tst/unit/test_local.py` local PyTorch
- `tst/integration/test_remote_onnx.py` remote ONNX load/predict/unload
- `tst/integration/test_server.py` lifecycle & capabilities
- `tst/integration/test_rpc_pytorch.py` Torch RPC smoke test
- `tst/integration/test_stream_predict.py` streaming prediction

## Roadmap (Excerpt)
- Structured error envelopes (use `ErrorStatus`).
- Model warm-up and metrics.
- Artifact checksum validation & allowlist for safe `model_class` usage.
- Multi-GPU load balancing improvements.

## See also
- https://github.com/0xk1h0/ONNX_gRPC
- https://github.com/ikeboo/ezonnx
