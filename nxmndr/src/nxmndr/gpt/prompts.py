# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Task-specific prompts and prompt building utilities for ChatGPT Vision.
"""

from typing import Optional

from nxmndr.tasks import Task
from ..logging import get_logger

logger = get_logger(__name__)


# Task-specific prompts optimized for ChatGPT Vision
TASK_PROMPTS = {
    Task.IMAGE_CLASSIFICATION: (
        "Analyze this image and identify the main objects, subjects, or categories present. "
        "Provide a clear, structured list of what you see, focusing on the most prominent "
        "and identifiable elements. Be specific and concise."
    ),
    Task.OBJECT_DETECTION: (
        "Examine this image and identify all distinct objects you can see. "
        "For each object, describe what it is and approximately where it's located "
        "in the image (e.g., 'top-left', 'center', 'bottom-right'). "
        "Be thorough and systematic in your analysis."
    ),
    Task.SEGMENTATION: (
        "Analyze this image and describe the different regions, areas, or segments you can identify. "
        "Focus on distinct visual regions, textures, colors, or spatial divisions. "
        "Describe how the image is composed and what different areas contain."
    ),
    Task.EMBEDDING_VISUALIZATION: (
        "Describe this image in detail, focusing on visual features that would be important "
        "for machine learning analysis. Include information about colors, textures, shapes, "
        "patterns, composition, and any distinctive visual characteristics."
    ),
}


# Default prompt when no task is specified
DEFAULT_PROMPT = (
    "Please analyze this image and describe what you see. "
    "Provide a clear, detailed description of the contents, objects, scenes, "
    "and any notable features or characteristics."
)


def get_task_prompt(task: Task) -> str:
    """
    Get the default prompt for a specific task.

    Args:
        task: The Task enum value

    Returns:
        The corresponding prompt string
    """
    return TASK_PROMPTS.get(task, DEFAULT_PROMPT)


def build_prompt(
    task: Optional[Task] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Build a complete prompt for ChatGPT Vision analysis.

    Args:
        task: Optional task type for prompt selection
        custom_prompt: Optional custom prompt (takes precedence over task)
        system_prompt: Optional system-level instructions to prepend

    Returns:
        Complete formatted prompt string
    """
    # Determine base prompt
    if custom_prompt:
        base_prompt = custom_prompt
        logger.debug("Using custom prompt")
    elif task:
        base_prompt = get_task_prompt(task)
        logger.debug(f"Using task-specific prompt for {task.value}")
    else:
        base_prompt = DEFAULT_PROMPT
        logger.debug("Using default prompt")

    # Build complete prompt
    parts = []

    if system_prompt:
        parts.append(f"System instructions: {system_prompt}")

    parts.append(base_prompt)

    complete_prompt = " ".join(parts)

    logger.debug(f"Built prompt: {complete_prompt[:100]}...")
    return complete_prompt


def create_custom_classification_prompt(categories: list[str]) -> str:
    """
    Create a custom classification prompt for specific categories.

    Args:
        categories: List of category names to classify into

    Returns:
        Custom classification prompt
    """
    if not categories:
        return get_task_prompt(Task.IMAGE_CLASSIFICATION)

    categories_str = ", ".join(f"'{cat}'" for cat in categories)

    return (
        f"Analyze this image and classify its contents into one or more of these categories: "
        f"{categories_str}. For each relevant category, explain why the image fits that "
        f"classification. If the image doesn't clearly fit any category, describe what "
        f"you see and suggest the closest match."
    )


def create_detection_prompt_with_objects(target_objects: list[str]) -> str:
    """
    Create a detection prompt focused on specific objects.

    Args:
        target_objects: List of object types to look for

    Returns:
        Custom detection prompt
    """
    if not target_objects:
        return get_task_prompt(Task.OBJECT_DETECTION)

    objects_str = ", ".join(f"'{obj}'" for obj in target_objects)

    return (
        f"Examine this image specifically looking for these objects: {objects_str}. "
        f"For each object you find, describe what it is, where it's located in the image, "
        f"and any relevant details about its appearance or condition. Also mention if any "
        f"of the target objects are clearly absent from the image."
    )


def create_comparison_prompt(reference_description: str) -> str:
    """
    Create a prompt for comparing with a reference description.

    Args:
        reference_description: Description to compare against

    Returns:
        Comparison prompt
    """
    return (
        f"Analyze this image and compare it with this reference description: "
        f'"{reference_description}". Identify similarities and differences, '
        f"noting what matches, what's different, and what additional details "
        f"you can observe in the current image."
    )
