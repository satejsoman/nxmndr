# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared session protocol types for tile-level inference sessions.

Defines the tile-level dataclasses and status types that flow between
clients and the inference server during session-based inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

SCHEMA_VERSION = "v1"

TileStatus = Literal["pending", "in_flight", "ok", "error", "cancelled"]


@dataclass
class TileEnvelope:
    """A single tile sent from client to server for inference."""

    tile_id: str
    bbox: Tuple[float, float, float, float]
    shape: Tuple[int, ...]
    crs: str = ""
    dtype: str = "float32"
    payload_uri: Optional[str] = None
    payload_bytes: Optional[bytes] = None
    options: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class TileProgress:
    """Status update for a single tile (returned by predict or emitted during processing)."""

    tile_id: str
    status: TileStatus = "pending"
    error: Optional[str] = None
    device_id: Optional[str] = None
    mask_bytes: Optional[bytes] = None
    mask_shape: Optional[Tuple[int, ...]] = None
    mask_dtype: Optional[str] = None


@dataclass
class ClientSessionStart:
    """Request payload to start a new inference session."""

    model_id: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    tile_count: int = 0
    max_cores: int = 0


@dataclass
class ClientSessionInfo:
    """Response payload after a session is created."""

    session_id: str
    allocated_devices: List[str] = field(default_factory=list)
    server_pool_size: int = 0
    suggested_client_pool: int = 1
    metadata_preamble: Dict[str, Any] = field(default_factory=dict)


def validate_tile_envelope(tile: TileEnvelope) -> List[str]:
    """Return a list of validation errors (empty if the envelope is valid)."""
    errors: List[str] = []
    if not tile.tile_id:
        errors.append("tile_id is required")
    if not tile.bbox or len(tile.bbox) != 4:
        errors.append("bbox must be a 4-tuple (minx, miny, maxx, maxy)")
    if not tile.shape or len(tile.shape) < 2:
        errors.append("shape must have at least 2 dimensions")
    if tile.payload_bytes is None and tile.payload_uri is None:
        errors.append("either payload_bytes or payload_uri is required")
    return errors


__all__ = [
    "SCHEMA_VERSION",
    "ClientSessionInfo",
    "ClientSessionStart",
    "TileEnvelope",
    "TileProgress",
    "TileStatus",
    "validate_tile_envelope",
]
