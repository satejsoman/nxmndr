# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Centralized constants for model file extensions and related utilities."""

from __future__ import annotations

import os
from typing import FrozenSet, Optional, Tuple

# Supported model weight file extensions (order matters for preference)
PYTORCH_EXTENSIONS: FrozenSet[str] = frozenset({".pt", ".pth"})
SAFETENSORS_EXTENSIONS: FrozenSet[str] = frozenset({".safetensors"})
ONNX_EXTENSIONS: FrozenSet[str] = frozenset({".onnx"})
CHECKPOINT_EXTENSIONS: FrozenSet[str] = frozenset({".ckpt"})
LEGACY_EXTENSIONS: FrozenSet[str] = frozenset({".bin"})

# All supported model extensions
MODEL_EXTENSIONS: FrozenSet[str] = (
    PYTORCH_EXTENSIONS
    | SAFETENSORS_EXTENSIONS
    | ONNX_EXTENSIONS
    | CHECKPOINT_EXTENSIONS
    | LEGACY_EXTENSIONS
)

# Extension preference order for selecting weight files
# Higher index = lower preference
EXTENSION_PREFERENCE_ORDER: Tuple[str, ...] = (
    ".safetensors",  # safe & memory mapped, preferred
    ".bin",  # legacy pytorch
    ".pt",  # occasional pytorch
    ".pth",  # pytorch checkpoints
    ".ckpt",  # converted checkpoints
    ".onnx",  # ONNX format
)

# File dialog filter for model selection
MODEL_FILE_FILTER = "Model Files (*.pt *.pth *.onnx *.safetensors *.bin *.ckpt);;All Files (*.*)"


def is_model_file(path: str) -> bool:
    """Check if a file path has a supported model extension.

    Args:
        path: File path to check.

    Returns:
        True if the file has a supported model extension.
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in MODEL_EXTENSIONS


def get_model_type(path: str) -> Optional[str]:
    """Determine the model type from file extension.

    Args:
        path: File path to check.

    Returns:
        Model type string: 'pytorch', 'onnx', 'safetensors', 'checkpoint',
        'legacy', or None if not a recognized model file.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in PYTORCH_EXTENSIONS:
        return "pytorch"
    elif ext in ONNX_EXTENSIONS:
        return "onnx"
    elif ext in SAFETENSORS_EXTENSIONS:
        return "safetensors"
    elif ext in CHECKPOINT_EXTENSIONS:
        return "checkpoint"
    elif ext in LEGACY_EXTENSIONS:
        return "legacy"
    return None


def is_pytorch_model(path: str) -> bool:
    """Check if path is a PyTorch model file (.pt, .pth)."""
    _, ext = os.path.splitext(path)
    return ext.lower() in PYTORCH_EXTENSIONS


def is_onnx_model(path: str) -> bool:
    """Check if path is an ONNX model file (.onnx)."""
    _, ext = os.path.splitext(path)
    return ext.lower() in ONNX_EXTENSIONS


def is_safetensors_model(path: str) -> bool:
    """Check if path is a safetensors model file (.safetensors)."""
    _, ext = os.path.splitext(path)
    return ext.lower() in SAFETENSORS_EXTENSIONS


def get_extension_rank(ext: str) -> int:
    """Get the preference rank for an extension (lower is better).

    Args:
        ext: Extension with leading dot (e.g., '.pt').

    Returns:
        Rank index, or 1000 if extension is not recognized.
    """
    ext = ext.lower()
    try:
        return EXTENSION_PREFERENCE_ORDER.index(ext)
    except ValueError:
        return 1000
