# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Exception hierarchy for the inference system using standard library exceptions.

This module defines a consistent exception hierarchy for all inference-related errors,
providing better error handling and debugging capabilities.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Base exception for all inference-related errors."""

    def __init__(
        self, message: str, cause: Optional[Exception] = None, context: Optional[dict] = None
    ):
        super().__init__(message)
        self.cause = cause
        self.context = context or {}

        # Log the error for debugging
        logger.error(
            f"InferenceError: {message}",
            extra={"cause": str(cause) if cause else None, "context": context},
        )


class ModelLoadError(InferenceError):
    """Raised when model loading fails."""

    pass


class ModelNotFoundError(FileNotFoundError, InferenceError):
    """Raised when a model file or specification cannot be found."""

    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        message = f"Model not found: {model_path}"
        super().__init__(message, cause, {"model_path": model_path})


class PredictionError(InferenceError):
    """Raised when model prediction fails."""

    pass


class ResourceError(OSError, InferenceError):
    """Raised when system resources are unavailable (memory, GPU, etc.)."""

    pass


class ValidationError(ValueError, InferenceError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        invalid_value: Any = None,
        expected: Any = None,
        cause: Optional[Exception] = None,
    ):
        context = {}
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if expected is not None:
            context["expected"] = str(expected)
        super().__init__(message, cause, context)


class ConfigurationError(ValueError, InferenceError):
    """Raised when configuration is invalid."""

    pass


class InferenceConnectionError(ConnectionError, InferenceError):  # type: ignore
    """Raised when remote connection fails (renamed to avoid shadowing built-in)."""

    pass


class InferenceTimeoutError(TimeoutError, InferenceError):  # type: ignore
    """Raised when operations timeout (renamed to avoid shadowing built-in)."""

    pass


class RegistryError(LookupError, InferenceError):
    """Raised when registry lookup fails."""

    pass


def handle_inference_error(func):
    """Decorator to wrap functions with consistent error handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except InferenceError:
            # Re-raise our custom exceptions as-is
            raise
        except FileNotFoundError as e:
            # Convert standard library exceptions to our hierarchy
            raise ModelNotFoundError(str(e), cause=e) from e
        except ValueError as e:
            raise ValidationError(str(e), cause=e) from e
        except OSError as e:
            raise ResourceError(str(e), cause=e) from e
        except Exception as e:
            # Catch-all for unexpected errors
            raise InferenceError(f"Unexpected error in {func.__name__}: {str(e)}", cause=e) from e

    return wrapper
