# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np

from nxmndr.inference import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cwd = Path(__file__).parent
stop_event = threading.Event()


# Example: serialize a PytorchModelSpec (customize as needed)
def build_model_spec():
    # For new proto we construct a ModelSpec message directly.
    pth = cwd / "example_model.pth"
    if not pth.exists():
        from tst.example_model.modeling_exampleconv import export

        export(cwd)
    return inference_pb2.ModelSpec(
        model_id="",
        format=inference_pb2.PYTORCH,
        name="example",
        source=str(pth),
        model_class="ExampleModel",
    )


def main():
    # Start the server as a subprocess and capture logs
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd.parent),
        text=True,
        bufsize=1,
    )

    def print_server_logs():
        for line in server_proc.stdout:
            logger.info(f"[SERVER] {line.rstrip()}")

    log_thread = threading.Thread(target=print_server_logs, daemon=True)
    log_thread.start()
    # Wait for the server to be ready (retry loop)
    max_attempts = 10
    for attempt in range(max_attempts):
        channel = grpc.insecure_channel("localhost:50051")
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            break
        except grpc.FutureTimeoutError:
            if attempt == max_attempts - 1:
                server_proc.terminate()
                raise RuntimeError("Failed to connect to gRPC server after waiting.")
            time.sleep(1)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # 1. Load model
    spec_msg = build_model_spec()
    load_response = stub.LoadModel(inference_pb2.LoadModelRequest(spec=spec_msg))
    logger.info(f"LoadModel response: {load_response}")
    if not load_response.success:
        raise RuntimeError(f"Load failed: {load_response.message}")
    model_id = load_response.model_id

    # 2. Prepare dummy input (adjust shape/dtype as needed)
    arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
    predict_request = inference_pb2.PredictRequest(
        model_id=model_id, input=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
    )
    predict_response = stub.Predict(predict_request)
    # Deserialize output
    output = np.frombuffer(predict_response.output, dtype=predict_response.dtype).reshape(
        predict_response.shape
    )
    logger.info(f"Predict response shape: {output.shape}, dtype: {output.dtype}")

    # 3. Capabilities & Health
    caps = stub.Capabilities(inference_pb2.CapabilitiesRequest())
    logger.info(f"Capabilities: {dict((c.key, c.value) for c in caps.capabilities)}")
    health = stub.Health(inference_pb2.HealthRequest())
    logger.info(f"Health: ready={health.ready}, message={health.message}")


if __name__ == "__main__":
    server_proc = None
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        stop_event.set()
        if server_proc is not None:
            server_proc.terminate()
