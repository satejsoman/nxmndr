# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ChatGPT Vision inference submodule.

This module provides ChatGPT Vision API integration for image analysis tasks.
All API credential management is handled within the inference provider.
"""

from .models import (
    ChatGptVisionModel,
    ChatGptVisionModelSpec,
    Gpt4VisionModelSpec,
    Gpt5VisionModelSpec,
    ModelEndpointSpec,
)
from .prompts import build_prompt, get_task_prompt
from .provider import AzureChatGptVisionProvider, OpenAIChatGptVisionProvider


def load_chatgpt_vision(spec, provider, session):
    """Load ChatGPT Vision model."""
    from .provider import ChatGptVisionInferenceProvider

    if not isinstance(provider, ChatGptVisionInferenceProvider):
        raise ValueError("ChatGptVisionModelSpec requires ChatGptVisionInferenceProvider subclass")
    return ChatGptVisionModel(spec, provider)


def register_chatgpt_vision():
    """Register ChatGPT Vision with the model registry."""
    from ..models import get_registry

    registry = get_registry()
    registry.register_spec("chatgpt_vision", ChatGptVisionModelSpec, load_chatgpt_vision)


__all__ = [
    "ChatGptVisionModelSpec",
    "Gpt4VisionModelSpec",
    "Gpt5VisionModelSpec",
    "ChatGptVisionModel",
    "OpenAIChatGptVisionProvider",
    "AzureChatGptVisionProvider",
    "ModelEndpointSpec",
    "get_task_prompt",
    "build_prompt",
]
