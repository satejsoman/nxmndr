# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Memory management utilities for efficient tensor operations and caching.

This module provides tensor optimization and model caching.
"""

import threading
from collections import OrderedDict
from typing import Any, Optional

import torch

from .logging import get_logger

logger = get_logger(__name__)


class ModelCache:
    """LRU cache for loaded models to avoid repeated loading.

    Supports pinning models to prevent eviction during active sessions.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._pinned: set[str] = set()

        logger.info(f"Initialized model cache with capacity {max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get a model from the cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                model = self._cache.pop(key)
                self._cache[key] = model
                logger.debug(f"Model cache hit: {key}")
                return model

        logger.debug(f"Model cache miss: {key}")
        return None

    def put(self, key: str, model: Any) -> None:
        """Put a model in the cache."""
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest unpinned (LRU)
                self._evict_one()

            self._cache[key] = model
            logger.debug(f"Added model to cache: {key}")

    def _evict_one(self) -> bool:
        """Evict the oldest unpinned entry. Returns True if eviction occurred."""
        # Find oldest unpinned entry
        for oldest_key in list(self._cache.keys()):
            if oldest_key not in self._pinned:
                self._cache.pop(oldest_key)
                logger.debug(f"Evicted model from cache: {oldest_key}")
                return True
        # All entries are pinned, cannot evict
        logger.warning("Cannot evict: all cached models are pinned")
        return False

    def pin(self, key: str) -> bool:
        """Pin a model to prevent eviction. Returns True if model exists and was pinned."""
        with self._lock:
            if key in self._cache:
                self._pinned.add(key)
                logger.debug(f"Pinned model: {key}")
                return True
            logger.warning(f"Cannot pin model not in cache: {key}")
            return False

    def unpin(self, key: str) -> bool:
        """Unpin a model to allow eviction. Returns True if model was unpinned."""
        with self._lock:
            if key in self._pinned:
                self._pinned.discard(key)
                logger.debug(f"Unpinned model: {key}")
                return True
            return False

    def is_pinned(self, key: str) -> bool:
        """Check if a model is pinned."""
        with self._lock:
            return key in self._pinned

    def clear(self) -> None:
        """Clear all models from the cache."""
        with self._lock:
            self._cache.clear()
            self._pinned.clear()
            logger.info("Cleared model cache")

    def size(self) -> int:
        """Get the current number of cached models."""
        return len(self._cache)

    def keys(self):
        """Get all cached model keys."""
        return list(self._cache.keys())


def optimize_tensor_copy(
    source: torch.Tensor, target_device: str, non_blocking: bool = True
) -> torch.Tensor:
    """
    Optimize tensor copying between devices.

    Args:
        source: Source tensor
        target_device: Target device string (e.g., 'cuda:0', 'cpu')
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Tensor on target device
    """
    if str(source.device) == target_device:
        return source

    logger.debug(f"Copying tensor from {source.device} to {target_device}")
    return source.to(target_device, non_blocking=non_blocking)


# Global model cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
        logger.info("Initialized global model cache")
    return _model_cache
