# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Example client script for the gRPC inference service.

Moved from nxmndr/client.py to examples/client.py.

Run via module:
    python examples/client.py --format onnx --model-path tst/example_model/example_model.onnx --shape 1 3 32 32
Or installed console script:
    nxmndr-client --format onnx --model-path tst/example_model/example_model.onnx --shape 1 3 32 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import grpc
import numpy as np

from nxmndr.inference import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_spec(args) -> inference_pb2.ModelSpec:
    fmt_map = {
        "pytorch": inference_pb2.PYTORCH,
        "onnx": inference_pb2.ONNX,
        "torchhub": inference_pb2.TORCHHUB,
        "huggingface": inference_pb2.HUGGINGFACE,
    }
    model_format = fmt_map.get(args.format.lower(), inference_pb2.MODEL_FORMAT_UNSPECIFIED)
    source = args.model_path or args.repo or args.repo_id or ""
    artifact = b""
    if model_format == inference_pb2.ONNX and args.model_path:
        with open(args.model_path, "rb") as f:
            artifact = f.read()
    return inference_pb2.ModelSpec(
        model_id="",
        format=model_format,
        name=args.name or Path(source).stem,
        source=source,
        model_class=args.model_class or "",
        artifact=artifact,
    )


def load_and_predict(args):
    channel = grpc.insecure_channel(args.address)
    grpc.channel_ready_future(channel).result(timeout=5)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    spec = build_spec(args)
    resp = stub.LoadModel(inference_pb2.LoadModelRequest(spec=spec))
    if not resp.success:
        raise SystemExit(f"Load failed: {resp.message}")
    model_id = resp.model_id
    shape = tuple(args.shape)
    if not shape:
        raise SystemExit("--shape required (e.g. --shape 1 3 32 32)")
    arr = np.zeros(shape, dtype=np.float32)
    pred = stub.Predict(
        inference_pb2.PredictRequest(
            model_id=model_id, input=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
        )
    )
    if not pred.output:
        raise SystemExit("Empty output from server")
    out = np.frombuffer(pred.output, dtype=pred.dtype).reshape(pred.shape)
    logger.info(f"Prediction output shape={out.shape} dtype={out.dtype}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="gRPC Inference Client (examples)")
    p.add_argument("--address", default="localhost:50051", help="gRPC server address host:port")
    p.add_argument(
        "--mode", default="grpc", choices=["grpc"], help="Only grpc mode is currently supported"
    )
    p.add_argument(
        "--format", required=True, help="Model format: pytorch|onnx|torchhub|huggingface"
    )
    p.add_argument("--model-path", help="Path to local model artifact (pth or onnx)")
    p.add_argument("--model-class", help="PyTorch model class name (if required)")
    p.add_argument("--repo", help="TorchHub repo (e.g. pytorch/vision)")
    p.add_argument("--repo-id", help="HuggingFace repo id")
    p.add_argument("--name", help="Logical model name")
    p.add_argument(
        "--shape", nargs="+", type=int, default=[], help="Input tensor shape (e.g. 1 3 224 224)"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if args.mode == "grpc":
        load_and_predict(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
