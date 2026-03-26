# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unified registry system for model specifications, loaders, and model classes.

This module consolidates the various registration patterns used throughout
the codebase into a single, consistent registry system.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar
from ..logging import get_logger

# Type variables for generic registry
SpecType = TypeVar("SpecType")
LoaderType = Callable[[Any, Any, Any], Any]  # (spec, provider, session) -> Model

logger = get_logger(__name__)


class ModelRegistry:
    """Centralized registry for model specifications, loaders, and PyTorch model classes."""

    def __init__(self):
        self._specs: Dict[str, Type] = {}
        self._loaders: Dict[Type, LoaderType] = {}
        self._model_classes: Dict[str, Type] = {}

    def register_spec(self, name: str, spec_cls: Type, loader_fn: LoaderType) -> None:
        """Register a model specification with its corresponding loader.

        Args:
            name: String identifier for the spec type (e.g., 'pytorch', 'onnx')
            spec_cls: ModelSpec subclass
            loader_fn: Function to load models from this spec type
        """
        if name in self._specs:
            logger.warning(f"Overriding existing spec registration for '{name}'")

        self._specs[name] = spec_cls
        self._loaders[spec_cls] = loader_fn
        logger.debug(f"Registered spec '{name}' -> {spec_cls.__name__}")

    def register_model_class(self, name: str, model_cls: Type) -> None:
        """Register a PyTorch model class by name.

        Args:
            name: String identifier for the model class
            model_cls: PyTorch nn.Module subclass
        """
        if name in self._model_classes:
            logger.warning(f"Overriding existing model class registration for '{name}'")

        self._model_classes[name] = model_cls
        logger.debug(f"Registered model class '{name}' -> {model_cls.__name__}")

    def get_spec_class(self, name: str) -> Optional[Type]:
        """Get a spec class by name."""
        return self._specs.get(name)

    def get_loader(self, spec: Any) -> Optional[LoaderType]:
        """Get the loader function for a given spec instance."""
        for spec_cls, loader in self._loaders.items():
            if isinstance(spec, spec_cls):
                return loader
        return None

    def get_model_class(self, name: str) -> Optional[Type]:
        """Get a PyTorch model class by name."""
        return self._model_classes.get(name)

    def list_specs(self) -> Dict[str, Type]:
        """Get all registered spec types."""
        return self._specs.copy()

    def list_model_classes(self) -> Dict[str, Type]:
        """Get all registered model classes."""
        return self._model_classes.copy()


# Global registry instance
registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global registry instance."""
    return registry


# Convenience functions for backward compatibility
def register_model_spec(type_name: str, spec_cls: Type, loader_fn: LoaderType) -> None:
    """Register a model specification and its loader."""
    registry.register_spec(type_name, spec_cls, loader_fn)


def register_pytorch_model(name: str, cls_obj: Type) -> None:
    """Register a PyTorch model class by name."""
    registry.register_model_class(name, cls_obj)


def get_loader(spec: Any) -> Optional[LoaderType]:
    """Get the loader function for a given spec."""
    return registry.get_loader(spec)


def get_spec_class(type_name: str) -> Optional[Type]:
    """Get a spec class by type name."""
    return registry.get_spec_class(type_name)


def get_model_class(name: str) -> Optional[Type]:
    """Get a PyTorch model class by name."""
    return registry.get_model_class(name)
