# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""High-level gRPC client for the nxmndr inference service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import grpc
import numpy as np
import time

from .inference import inference_pb2, inference_pb2_grpc

# Match server's message size limits
GRPC_MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128MB
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]


class InferenceGrpcError(RuntimeError):
    """Raised when a gRPC request to the inference service fails."""


@dataclass
class PredictResult:
    """Container for Predict responses returned by the inference service."""

    output: bytes
    shape: Sequence[int]
    dtype: str
    metadata: Mapping[str, str]


@dataclass
class ModelRegistryEntry:
    """Container for model registry entries returned by the inference service."""

    entry_id: str
    model_id: str
    display_name: str
    source: str
    task: str
    format: str
    metadata: Mapping[str, str]
    last_used_epoch_ms: int


class InferenceGrpcClient:
    """Simple wrapper around the nxmndr inference gRPC service.

    Maintains a persistent channel with lazy reconnection on failure.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: Optional[float] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        # Strip http:// or https:// prefix if present (gRPC doesn't use these)
        endpoint = endpoint.replace("https://", "").replace("http://", "")
        self._endpoint = endpoint
        self._timeout = timeout
        self._credentials = credentials
        self._max_attempts = max(1, int(max_attempts or 1))
        self._backoff_seconds = max(0.0, float(backoff_seconds or 0.0))
        # Persistent channel (created lazily)
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[inference_pb2_grpc.InferenceServiceStub] = None

    def _get_channel(self) -> grpc.Channel:
        """Get or create a persistent channel."""
        if self._channel is None:
            self._channel = self._create_channel()
        return self._channel

    def _get_stub(self) -> inference_pb2_grpc.InferenceServiceStub:
        """Get or create the service stub."""
        if self._stub is None:
            self._stub = inference_pb2_grpc.InferenceServiceStub(self._get_channel())
        return self._stub

    def _reset_channel(self) -> None:
        """Reset channel on failure for reconnection."""
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
        self._channel = None
        self._stub = None

    def _create_channel(self) -> grpc.Channel:
        if self._credentials is not None:
            return grpc.secure_channel(self._endpoint, self._credentials, options=GRPC_OPTIONS)
        return grpc.insecure_channel(self._endpoint, options=GRPC_OPTIONS)

    def _with_retry(self, func):
        last_err = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return func()
            except grpc.RpcError as err:  # pragma: no cover - network dependent
                last_err = err
                # Reset channel on connection errors for reconnection
                self._reset_channel()
                if attempt >= self._max_attempts:
                    break
                time.sleep(self._backoff_seconds * attempt)
        if last_err:
            details = getattr(last_err, "details", None)
            message = details() if callable(details) else str(details or last_err)
            raise InferenceGrpcError(message) from last_err
        raise InferenceGrpcError("request failed")

    def close(self) -> None:
        """Close the persistent channel."""
        self._reset_channel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def load_model(
        self,
        model_id: str,
        spec: Mapping[str, str],
    ) -> str:
        """Load a model on the remote inference service.

        Args:
            model_id: Unique identifier for the model
            spec: Model specification dict with keys like:
                - format: Model format (huggingface, pytorch, onnx, torchhub)
                - source: Model source (repo_id for huggingface, path for others)
                - task: Task type (segmentation, detection, classification, etc)
                - name: Display name for the model

        Returns:
            Server-assigned model_id for use in predict calls
        """
        if not model_id:
            raise ValueError("model_id is required")
        if not spec:
            raise ValueError("spec is required")

        format_map = {
            "huggingface": inference_pb2.HUGGINGFACE,
            "pytorch": inference_pb2.PYTORCH,
            "onnx": inference_pb2.ONNX,
            "torchhub": inference_pb2.TORCHHUB,
        }

        task_map = {
            "segmentation": inference_pb2.SEGMENTATION,
            "detection": inference_pb2.OBJECT_DETECTION,
            "object_detection": inference_pb2.OBJECT_DETECTION,
            "classification": inference_pb2.CLASSIFICATION,
            "embedding": inference_pb2.EMBEDDING,
        }

        format_str = str(spec.get("format", "")).lower()
        task_str = str(spec.get("task", "")).lower()

        format_enum = format_map.get(format_str, inference_pb2.HUGGINGFACE)
        task_enum = task_map.get(task_str, inference_pb2.TASK_TYPE_UNSPECIFIED)

        source_value = str(spec.get("source", ""))
        name_value = str(spec.get("name", source_value))
        token_value = str(spec.get("token", ""))

        model_class_value = str(spec.get("model_class", ""))

        model_spec = inference_pb2.ModelSpec(
            model_id=str(model_id),
            format=format_enum,
            source=source_value,
            task=task_enum,
            name=name_value,
            model_class=model_class_value,
        )

        if token_value:
            model_spec.token = token_value

        request = inference_pb2.LoadModelRequest(spec=model_spec)

        def _call():
            stub = self._get_stub()
            return stub.LoadModel(request, timeout=self._timeout)

        response = self._with_retry(_call)

        if not response.success:
            raise InferenceGrpcError(f"LoadModel failed: {response.message}")

        return response.model_id

    def predict(
        self,
        model_id: str,
        tensor: np.ndarray,
        *,
        options: Optional[Mapping[str, str]] = None,
    ) -> PredictResult:
        """Run a single Predict call against the remote inference service."""

        if not model_id:
            raise ValueError("model_id is required")
        if tensor is None:
            raise ValueError("tensor is required")

        array = np.ascontiguousarray(tensor)
        request = inference_pb2.PredictRequest(
            model_id=str(model_id),
            input=array.tobytes(),
            shape=[int(dim) for dim in array.shape],
            dtype=str(array.dtype),
        )

        if options:
            for key, value in options.items():
                if value is None:
                    continue
                request.options[str(key)] = str(value)

        def _call():
            stub = self._get_stub()
            return stub.Predict(request, timeout=self._timeout)

        response = self._with_retry(_call)

        return PredictResult(
            output=bytes(response.output or b""),
            shape=tuple(int(dim) for dim in response.shape),
            dtype=response.dtype or str(array.dtype),
            metadata=dict(response.metadata),
        )

    # ---- Streaming & Sessions ----
    def open_session(
        self,
        *,
        session_id: Optional[str] = None,
        spec: Optional[inference_pb2.ModelSpec] = None,
        manifest: Optional[inference_pb2.TileManifest] = None,
        transport: Optional[inference_pb2.TransportCaps] = None,
        options: Optional[Mapping[str, str]] = None,
    ) -> inference_pb2.OpenSessionResponse:
        session_id = session_id or ""
        req = inference_pb2.OpenSessionRequest(
            session_id=session_id,
        )
        if spec:
            req.spec.CopyFrom(spec)
        if manifest:
            req.manifest.CopyFrom(manifest)
        if transport:
            req.transport.CopyFrom(transport)
        if options:
            for k, v in options.items():
                req.options[str(k)] = str(v)

        def _call():
            stub = self._get_stub()
            return stub.OpenSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def close_session(
        self, session_id: str, *, force: bool = False
    ) -> inference_pb2.CloseSessionResponse:
        req = inference_pb2.CloseSessionRequest(session_id=str(session_id or ""), force=force)

        def _call():
            stub = self._get_stub()
            return stub.CloseSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def cancel_session(
        self, session_id: str, *, reason: str = ""
    ) -> inference_pb2.CancelSessionResponse:
        req = inference_pb2.CancelSessionRequest(
            session_id=str(session_id or ""), reason=str(reason or "")
        )

        def _call():
            stub = self._get_stub()
            return stub.CancelSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def stream_predict(
        self,
        *,
        model_id: str,
        samples: Iterable[np.ndarray],
        session_id: Optional[str] = None,
        tile_ids: Optional[Iterable[str]] = None,
        chunk_bytes: Optional[int] = None,
        dtype: Optional[str] = None,
        cancel_fn=None,
        on_response=None,
        on_error=None,
        on_tile_start=None,
    ) -> Iterable[inference_pb2.StreamPredictResponse]:
        """Client-side streaming inference with optional chunking and session/tile metadata.

        Args:
            model_id: model identifier.
            samples: iterable of numpy arrays representing logical tiles/samples.
            session_id: optional session identifier to bind stream to.
            tile_ids: optional iterable of tile ids aligned with samples.
            chunk_bytes: if set, will chunk the sample bytes into pieces <= chunk_bytes.
            dtype: override dtype string; defaults to array dtype.
            cancel_fn: optional callable returning True to abort streaming; will close channel.
            on_response: optional callback(resp) invoked per response.
            on_error: optional callback(err_msg) invoked if the stream errors.
            on_tile_start: optional callback(tile_id) invoked when a tile starts being sent.
        """

        model_id = str(model_id or "")
        if not model_id:
            raise ValueError("model_id is required")

        def _iter_requests():
            tid_iter = iter(tile_ids) if tile_ids is not None else None
            for sample in samples:
                arr = np.ascontiguousarray(sample)
                tile_id = next(tid_iter) if tid_iter is not None else None

                # Notify that this tile is starting to be processed
                if on_tile_start and tile_id:
                    try:
                        on_tile_start(tile_id)
                    except Exception:
                        pass

                raw = arr.tobytes()
                shape_list = [int(dim) for dim in arr.shape]
                dtype_str = dtype or str(arr.dtype)

                chunks: list[bytes] = []
                if chunk_bytes and chunk_bytes > 0:
                    for i in range(0, len(raw), int(chunk_bytes)):
                        chunks.append(raw[i : i + int(chunk_bytes)])
                else:
                    chunks.append(raw)

                for c in chunks:
                    req = inference_pb2.StreamPredictRequest(
                        model_id=model_id,
                        chunk=c,
                        shape=shape_list,
                        dtype=dtype_str,
                    )
                    if session_id or tile_id:
                        ctx = req.context
                        if session_id:
                            ctx["session_id"] = str(session_id)
                        if tile_id:
                            ctx["tile_id"] = str(tile_id)
                    yield req

                # end-of-sequence marker for this sample
                eos = inference_pb2.StreamPredictRequest(end_of_sequence=True)
                if session_id or tile_id:
                    ctx = eos.context
                    if session_id:
                        ctx["session_id"] = str(session_id)
                    if tile_id:
                        ctx["tile_id"] = str(tile_id)
                yield eos

        try:
            stub = self._get_stub()
            resp_iter = stub.StreamPredict(_iter_requests(), timeout=self._timeout)
            for resp in resp_iter:
                if cancel_fn and cancel_fn():
                    break
                if on_response:
                    try:
                        on_response(resp)
                    except Exception:
                        pass
                yield resp
        except grpc.RpcError as err:  # pragma: no cover - network dependent
            self._reset_channel()  # Reset for reconnection on next call
            msg = getattr(err, "details", None)
            msg = msg() if callable(msg) else str(msg or err)
            if on_error:
                try:
                    on_error(msg)
                except Exception:
                    pass
            raise InferenceGrpcError(msg) from err

    def list_model_registry(
        self, provider_id: Optional[str] = None
    ) -> Sequence[ModelRegistryEntry]:
        """Fetch registry entries stored on the remote inference service."""

        request = inference_pb2.ListModelRegistryRequest()
        if provider_id:
            request.provider_id = str(provider_id)

        try:
            stub = self._get_stub()
            response = stub.ListModelRegistry(request, timeout=self._timeout)
        except grpc.RpcError as err:  # pragma: no cover - requires network
            self._reset_channel()
            details = getattr(err, "details", None)
            message = details() if callable(details) else str(details or err)
            raise InferenceGrpcError(message) from err

        return [
            ModelRegistryEntry(
                entry_id=str(entry.entry_id or ""),
                model_id=str(entry.model_id or ""),
                display_name=str(entry.display_name or ""),
                source=str(entry.source or ""),
                task=str(entry.task or ""),
                format=str(entry.format or ""),
                metadata=dict(entry.metadata),
                last_used_epoch_ms=int(entry.last_used_epoch_ms or 0),
            )
            for entry in response.entries
        ]

    def evict_model_registry_entry(self, entry_id: str, provider_id: Optional[str] = None) -> bool:
        """Remove an existing registry entry on the remote inference service."""

        entry_id = str(entry_id or "").strip()
        if not entry_id:
            raise ValueError("entry_id is required")

        request = inference_pb2.EvictModelRegistryRequest(entry_id=entry_id)
        if provider_id:
            request.provider_id = str(provider_id)

        try:
            stub = self._get_stub()
            response = stub.EvictModelRegistryEntry(request, timeout=self._timeout)
        except grpc.RpcError as err:  # pragma: no cover - requires network
            self._reset_channel()
            details = getattr(err, "details", None)
            message = details() if callable(details) else str(details or err)
            raise InferenceGrpcError(message) from err

        return bool(response.success)


__all__ = [
    "InferenceGrpcClient",
    "PredictResult",
    "InferenceGrpcError",
    "ModelRegistryEntry",
    "inference_pb2",
    "inference_pb2_grpc",
]
