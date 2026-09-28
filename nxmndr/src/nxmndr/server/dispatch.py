# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared prediction dispatch for unary ``Predict`` and every ``StreamPredict`` tile.

One code path turns (leased model record, decoded input, effective options) into a
response array plus metadata: SAM prompt routing, the unprompted fallback, task
shaping (segmentation masks and confidence, detection boxes, embeddings) and NPZ
tensor bundles. The gRPC layer only decodes transport, holds leases and adds
correlation metadata, so unary and streaming results cannot drift apart.

Stream context v1 (``StreamPredictRequest.context``) is also decoded here:
``session_id`` and ``tile_id`` are reserved, per-tile options use the ``opt.``
prefix, and every other unprefixed key is reserved and never used as an option.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..inference import inference_pb2
from ..inference.image_utils import (
    masks_to_bounding_boxes,
    prepare_segmentation_mask,
    prepare_segmentation_mask_with_confidence,
)
from ..models import sam as sam_support
from ..models import ultralytics_yolo
from ..tensor_bundle import pack_tensor_bundle

# ---------------------------------------------------------------- stream context v1

STREAM_CONTEXT_VERSION = "1"
CONTEXT_SESSION_ID = "session_id"
CONTEXT_TILE_ID = "tile_id"
CONTEXT_BATCH_SIZE = "batch_size"  # legacy; reserved, never an option
RESERVED_CONTEXT_KEYS = (CONTEXT_SESSION_ID, CONTEXT_TILE_ID, CONTEXT_BATCH_SIZE)
TILE_OPTION_PREFIX = "opt."
SESSION_CONTROL_OPTION_KEYS = ("max_tile_bytes", "batch_size")
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# ------------------------------------------------------------------ error codes

ERROR_MISSING_TILE_ID = "missing_tile_id"
ERROR_DUPLICATE_TILE_ID = "duplicate_tile_id"
ERROR_UNKNOWN_SESSION = "unknown_session"
ERROR_SESSION_NOT_OPEN = "session_not_open"
ERROR_MALFORMED_PAYLOAD = "malformed_payload"
ERROR_MALFORMED_OPTIONS = "malformed_options"
ERROR_CHUNK_TOO_LARGE = "chunk_too_large"
ERROR_TILE_TOO_LARGE = "tile_too_large"
ERROR_BACKPRESSURE = "backpressure"
ERROR_RESOURCE_EXHAUSTED = "resource_exhausted"
ERROR_INFERENCE_FAILED = "inference_failed"
ERROR_CANCELLED = "cancelled"
ERROR_INTERNAL = "internal"
SCOPE_TILE = "tile"
SCOPE_STREAM = "stream"


class DispatchError(Exception):
    """A request-level failure with a wire ``error_code``."""

    code = ERROR_INTERNAL


class MalformedOptionsError(DispatchError):
    code = ERROR_MALFORMED_OPTIONS


class MalformedPayloadError(DispatchError):
    code = ERROR_MALFORMED_PAYLOAD


@dataclass
class DecodedContext:
    session_id: str
    tile_id: str
    options: Dict[str, str]
    unknown_keys: Tuple[str, ...]


def validate_option_name(name: str) -> str:
    if not isinstance(name, str) or not _OPTION_NAME.match(name):
        raise MalformedOptionsError(f"option name {name!r} must match {_OPTION_NAME.pattern}")
    if name in RESERVED_CONTEXT_KEYS:
        raise MalformedOptionsError(f"option name {name!r} is a reserved context key")
    return name


def decode_tile_context(context: Mapping[str, str]) -> DecodedContext:
    """Split one message's context into reserved IDs, ``opt.`` options and ignored keys."""

    options: Dict[str, str] = {}
    unknown: List[str] = []
    for key, value in context.items():
        if key.startswith(TILE_OPTION_PREFIX):
            options[validate_option_name(key[len(TILE_OPTION_PREFIX):])] = value
        elif key not in RESERVED_CONTEXT_KEYS:
            unknown.append(key)
    return DecodedContext(
        session_id=context.get(CONTEXT_SESSION_ID, ""),
        tile_id=context.get(CONTEXT_TILE_ID, ""),
        options=options,
        unknown_keys=tuple(sorted(unknown)),
    )


def session_inference_options(session_options: Mapping[str, str]) -> Dict[str, str]:
    """``OpenSessionRequest.options`` minus the session-control keys."""

    return {k: v for k, v in session_options.items() if k not in SESSION_CONTROL_OPTION_KEYS}


def effective_tile_options(
    session_options: Mapping[str, str], tile_options: Mapping[str, str]
) -> Dict[str, str]:
    """Session options updated by this tile's own options; nothing carries between tiles."""

    merged = session_inference_options(session_options)
    merged.update(tile_options)
    return merged


# ------------------------------------------------------------------ input decoding


def decode_input(data: bytes, shape: Optional[Sequence[int]], dtype: Optional[str]) -> np.ndarray:
    """Decode wire bytes; the declared shape and dtype must describe the bytes exactly."""

    if shape is None or not dtype:
        raise MalformedPayloadError("missing shape or dtype")
    try:
        dt = np.dtype(dtype)
    except TypeError as exc:
        raise MalformedPayloadError(f"unknown dtype {dtype!r}") from exc
    if dt.hasobject or dt.kind not in "biufc":
        raise MalformedPayloadError(f"unsupported dtype {dtype!r}")
    dims = tuple(int(d) for d in shape)
    if any(d < 0 for d in dims):
        raise MalformedPayloadError(f"negative dimension in shape {list(dims)}")
    expected = int(np.prod(dims, dtype=np.int64)) * dt.itemsize
    if len(data) != expected:
        raise MalformedPayloadError(
            f"payload has {len(data)} bytes but shape {list(dims)} dtype {dt.name} needs {expected}"
        )
    return np.frombuffer(data, dtype=dt.newbyteorder("<")).reshape(dims)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def option_bool(options: Mapping[str, str], name: str) -> bool:
    raw = options.get(name)
    if raw is None:
        return False
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise MalformedOptionsError(f"option {name}={raw!r} must be true or false")


# ------------------------------------------------------------------ dispatch


@dataclass
class DispatchResult:
    array: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def output(self) -> bytes:
        return self.array.tobytes()

    @property
    def shape(self) -> List[int]:
        return [int(d) for d in self.array.shape]

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)


def _task_type(options: Mapping[str, str], model_metadata: Mapping[str, object]) -> str:
    task_type = str(options.get("task_type") or "").strip().lower()
    if task_type:
        return task_type
    task_hint = model_metadata.get("task_name")
    if isinstance(task_hint, str) and task_hint:
        return task_hint.lower()
    raw_task = model_metadata.get("task")
    if isinstance(raw_task, int) and raw_task:
        try:
            return inference_pb2.TaskType.Name(raw_task).lower()
        except ValueError:
            return ""
    return ""


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    return np.ascontiguousarray(value)


def shape_result(
    output,
    *,
    options: Mapping[str, str],
    model_metadata: Mapping[str, object],
    logger=None,
    instance_masks: bool = False,
) -> DispatchResult:
    """Task shaping and serialization shared by every prediction path.

    ``instance_masks``: ``output`` is an ``(N, H, W)`` instance stack that stays as it
    is (no argmax, no squeeze) as a ``segmentation_mask`` result.
    """

    embeddings_from_model = None
    if isinstance(output, dict):
        if "embeddings" in output:
            embeddings_from_model = output["embeddings"]
        if "output" in output:
            output = output["output"]
        elif embeddings_from_model is not None:
            output = embeddings_from_model
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
    base_output = _to_numpy(output)
    if embeddings_from_model is not None:
        embeddings_from_model = _to_numpy(embeddings_from_model)

    task_type = _task_type(options, model_metadata)
    return_embeddings = option_bool(options, "return_embeddings")
    return_confidence = option_bool(options, "return_confidence")
    embedding_only = task_type in {"embedding", "embeddings", "feature", "features"}

    meta: Dict[str, str] = {}
    if task_type:
        meta["task_type"] = task_type
    model_task_name = model_metadata.get("task_name")
    if model_task_name:
        meta["model_task"] = str(model_task_name)

    response_array = base_output
    confidence = None
    if embedding_only:
        meta["result_type"] = "embeddings"
    elif task_type == "segmentation":
        try:
            if instance_masks:
                masks = np.ascontiguousarray(base_output)
            elif return_confidence:
                masks, confidence = prepare_segmentation_mask_with_confidence(base_output)
            else:
                masks = prepare_segmentation_mask(base_output)
            response_array = masks
            meta["result_type"] = "segmentation_mask"
            meta["mask_shape"] = str(list(masks.shape))
            meta["mask_dtype"] = str(masks.dtype)
        except Exception as exc:
            if logger:
                logger.warning("prepare_segmentation_mask failed: %s", exc)
            meta["result_type"] = "raw"
            meta["task_warning"] = f"segmentation_fallback:{exc}"
            confidence = None
    elif task_type in {"object_detection", "object-detection", "detection"}:
        try:
            boxes = masks_to_bounding_boxes(prepare_segmentation_mask(base_output))
            response_array = boxes
            meta["result_type"] = "bounding_boxes"
            meta["boxes_count"] = str(boxes.shape[0])
            meta["box_format"] = "batch_index,class_id,x_min,y_min,x_max,y_max"
        except Exception as exc:
            meta["result_type"] = "raw"
            meta["task_warning"] = f"detection_fallback:{exc}"

    geotransform_opt = options.get("geotransform")
    projection_opt = options.get("projection") or options.get("crs") or options.get("crs_wkt")
    if geotransform_opt:
        meta["geotransform"] = str(geotransform_opt)
    if projection_opt:
        meta["projection"] = str(projection_opt)

    bundle: Optional[Dict[str, np.ndarray]] = None
    if embedding_only:
        bundle = {"embeddings": base_output}
    elif return_embeddings or confidence is not None:
        bundle = {}
        if return_embeddings:
            bundle["embeddings"] = (
                embeddings_from_model if embeddings_from_model is not None else base_output
            )
        result_type = meta.get("result_type", "raw")
        if result_type == "segmentation_mask":
            bundle["mask"] = np.ascontiguousarray(response_array)
            if confidence is not None:
                bundle["confidence"] = np.ascontiguousarray(confidence)
                meta["has_confidence"] = "true"
                if confidence.size:
                    meta["confidence_min"] = f"{float(confidence.min()):.4f}"
                    meta["confidence_max"] = f"{float(confidence.max()):.4f}"
                    meta["confidence_mean"] = f"{float(confidence.mean()):.4f}"
        elif result_type == "bounding_boxes":
            bundle["detections"] = np.ascontiguousarray(response_array)

    if return_embeddings or embedding_only:
        meta["return_embeddings"] = "true"
        embeddings = bundle.get("embeddings") if bundle else None
        if embeddings is not None:
            meta["embeddings_available"] = "true"
            meta["embeddings_shape"] = json.dumps(list(embeddings.shape))
            meta["embeddings_dtype"] = str(embeddings.dtype)

    if bundle:
        bundle_bytes = pack_tensor_bundle(bundle)
        response_array = np.frombuffer(bundle_bytes, dtype=np.uint8)
        meta["payload_format"] = "npz"
        meta["bundle_size_bytes"] = str(len(bundle_bytes))
        meta["bundle_keys"] = ",".join(sorted(bundle.keys()))
    else:
        meta["payload_format"] = "raw"

    meta.setdefault("result_type", "raw")
    return DispatchResult(np.ascontiguousarray(response_array), meta)


def _sam_variant_getter(lease, capability, device_id: str, torch_device: str):
    def get_variant(variant: str):
        if variant == capability.record_variant:
            model_obj = lease.model_for_device(device_id)
            return model_obj.model, model_obj.processor
        # The record's credential (ModelRecord.auth_token, set by the model cache from
        # the load request); never taken from metadata, which holds no token.
        token = lease.record.auth_token or None

        def factory():
            return sam_support.load_sam_variant(variant, capability, torch_device, token=token)

        resource = lease.resource(f"{variant}@{torch_device}", factory)
        return resource.model, resource.processor

    return get_variant


# The embedding-only task names of shape_result.
_EMBEDDING_TASKS = {"embedding", "embeddings", "feature", "features"}


def _run_instance_prediction(model_obj, image, options, model_metadata, logger) -> DispatchResult:
    """An instance-mask model (``ultralytics_yolo``): always a ``segmentation_mask``
    result holding the ``(N, H, W)`` uint8 instance stack at the chip's size.

    Embeddings and per-pixel confidence do not exist for these models, so
    ``return_embeddings``, ``return_confidence`` and an embedding task are refused
    (``malformed_options``) instead of being ignored. Option ``drop_group_masks``
    (boolean, default true) controls the group-mask filter.
    """

    if (
        option_bool(options, "return_embeddings")
        or option_bool(options, "return_confidence")
        or _task_type(options, model_metadata) in _EMBEDDING_TASKS
    ):
        raise MalformedOptionsError(
            f"{ultralytics_yolo.ULTRALYTICS_YOLO} models return instance masks only; "
            "embeddings and confidence are not available"
        )
    drop = True
    if ultralytics_yolo.OPTION_DROP_GROUP_MASKS in options:
        drop = option_bool(options, ultralytics_yolo.OPTION_DROP_GROUP_MASKS)
    start = time.monotonic()
    try:
        masks = model_obj.predict(image, drop_group_masks=drop)
    except ultralytics_yolo.InstanceInputError as exc:
        raise MalformedPayloadError(str(exc)) from exc
    infer_ms = (time.monotonic() - start) * 1000.0
    shaping = dict(options)
    shaping["task_type"] = "segmentation"
    result = shape_result(
        masks, options=shaping, model_metadata=model_metadata, logger=logger, instance_masks=True
    )
    result.metadata["latency_infer_ms"] = f"{infer_ms:.2f}"
    return result


def run_prediction(
    lease,
    *,
    image: np.ndarray,
    options: Mapping[str, str],
    device_id: str,
    torch_device: str,
    infer: Callable[[np.ndarray, bool], object],
    logger=None,
) -> DispatchResult:
    """Run one prediction on a leased model record.

    ``infer(array, return_embeddings)`` runs the record's own model (ONNX, Hugging
    Face or the PyTorch RPC worker). SAM 3 records with a prompt go to the shared SAM
    handler instead; SAM 3 records without a prompt use ``infer`` (unprompted mode).
    """

    record = lease.record
    model_metadata = record.metadata or {}
    if record.backend == ultralytics_yolo.ULTRALYTICS_BACKEND:
        return _run_instance_prediction(
            lease.model_for_device(device_id), image, options, model_metadata, logger
        )
    capability = sam_support.resolve_sam_capability(record.model, record.spec)

    start = time.monotonic()
    output = None
    sam_mode = ""
    if capability is not None:
        try:
            prompt = sam_support.parse_sam_prompt(options)
        except sam_support.SamPromptError as exc:
            raise MalformedOptionsError(str(exc)) from exc
        if prompt.has_prompt:
            getter = _sam_variant_getter(lease, capability, device_id, torch_device)
            try:
                output = sam_support.handle_sam_inference(
                    image, prompt, getter, device=torch_device, logger=logger
                )
            except sam_support.SamPromptError as exc:
                raise MalformedOptionsError(str(exc)) from exc
            except sam_support.SamInputError as exc:
                raise MalformedPayloadError(str(exc)) from exc
            sam_mode = "geometry" if prompt.has_geometry else "text"
    if output is None:
        output = infer(image, option_bool(options, "return_embeddings"))
    infer_ms = (time.monotonic() - start) * 1000.0

    result = shape_result(output, options=options, model_metadata=model_metadata, logger=logger)
    result.metadata["latency_infer_ms"] = f"{infer_ms:.2f}"
    if capability is not None:
        result.metadata["sam_prompt"] = sam_mode or "none"
    return result


__all__ = [
    "STREAM_CONTEXT_VERSION",
    "RESERVED_CONTEXT_KEYS",
    "TILE_OPTION_PREFIX",
    "SESSION_CONTROL_OPTION_KEYS",
    "SCOPE_TILE",
    "SCOPE_STREAM",
    "DispatchError",
    "MalformedOptionsError",
    "MalformedPayloadError",
    "DecodedContext",
    "DispatchResult",
    "decode_tile_context",
    "session_inference_options",
    "effective_tile_options",
    "decode_input",
    "option_bool",
    "shape_result",
    "run_prediction",
]
