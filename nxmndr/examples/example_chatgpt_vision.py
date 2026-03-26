#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Azure OpenAI ChatGPT Vision inference provider examples.

This demonstrates how to use the ChatGPT Vision API integration for image analysis
tasks using Azure OpenAI with Entra ID authentication. This is the recommended
approach for enterprise deployments with proper security and cost management.

Requires:
- Azure OpenAI resource with GPT-4 Vision model deployment
- Entra ID authentication (Azure CLI login or service principal)
- ENDPOINT_URL and DEPLOYMENT_NAME environment variables
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from nxmndr.logging import get_logger
from nxmndr.gpt import (
    AzureChatGptVisionProvider,
    Gpt5VisionModelSpec,
    ChatGptVisionModelSpec,
    get_task_prompt,
)
from nxmndr.inference import InferenceSession
from nxmndr.tasks import Task

logger = get_logger(__name__)


def example_azure_setup():
    """Demonstrate Azure OpenAI provider setup options."""
    logger.info("Azure OpenAI Setup Examples")

    # Method 1: Environment variables (recommended)
    logger.info("Method 1: Environment Variables")
    logger.info("Required environment variables:")
    logger.info("export ENDPOINT_URL='https://your-resource.openai.azure.com/'")
    logger.info("export DEPLOYMENT_NAME='gpt-4-vision-preview'")

    if os.getenv("ENDPOINT_URL"):
        try:
            provider = AzureChatGptVisionProvider()
            logger.info(f"Azure provider initialized with endpoint: {provider.endpoint_url}")
            logger.info(f"Using deployment: {provider.deployment_name}")
        except Exception as e:
            logger.error(f"Azure provider initialization failed: {e}")
    else:
        logger.warning("ENDPOINT_URL not set - using example values")

    # Method 2: Explicit parameters
    logger.info("Method 2: Explicit Parameters")
    try:
        provider = AzureChatGptVisionProvider(
            endpoint_url="https://example.openai.azure.com/", deployment_name="gpt-4-vision-preview"
        )
        logger.info("Azure provider created with explicit parameters")
        logger.info(f"Endpoint: {provider.endpoint_url}")
        logger.info(f"Deployment: {provider.deployment_name}")
    except Exception as e:
        logger.error(f"Explicit setup failed: {e}")

    # Authentication info
    logger.info("Authentication Requirements")
    logger.info("Azure OpenAI uses Entra ID (Azure AD) authentication:")
    logger.info("1. Azure CLI: az login")
    logger.info("2. Service Principal: Set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")
    logger.info("3. Managed Identity: Available automatically in Azure resources")
    logger.info("4. VS Code: Sign in to Azure account")


def example_basic_usage():
    """Basic image description example using Azure OpenAI."""
    logger.info("Basic Image Description with Azure OpenAI")

    # Load the bomas.png image
    test_image_path = "bomas.png"
    if Path(test_image_path).exists():
        test_image = np.array(Image.open(test_image_path).convert("RGB"))
    else:
        logger.warning(f"{test_image_path} not found, using placeholder")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Set up ChatGPT Vision model spec
    spec = Gpt5VisionModelSpec(
        model_name="gpt-5", max_completion_tokens=500, temperature=1.0, name="image-analyzer"
    )

    # Create Azure provider (endpoint and deployment from environment)
    provider = AzureChatGptVisionProvider(
        endpoint_url=os.getenv("ENDPOINT_URL", "https://your-resource.openai.azure.com/"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "gpt-5"),
    )

    # Create inference session
    session = InferenceSession(spec, provider)

    # Run inference with custom prompt
    try:
        result = session.run(test_image, prompt="Describe what you see in this image in detail.")
        logger.info(f"Azure OpenAI Response: {result}")
        return result
    except Exception as e:
        logger.error(f"Basic usage failed: {e}")
        raise


def example_task_based_analysis():
    """Task-based image analysis examples using Azure OpenAI."""
    logger.info("Task-Based Analysis with Azure OpenAI")

    # Load the bomas.png image
    test_image_path = "bomas.png"
    if Path(test_image_path).exists():
        test_image = np.array(Image.open(test_image_path).convert("RGB"))
    else:
        logger.warning(f"{test_image_path} not found, using placeholder")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Set up model with system prompt
    spec = ChatGptVisionModelSpec(
        model_name="gpt-4o",
        system_prompt="You are an expert image analyst. Be precise and detailed.",
        max_tokens=800,
        temperature=0.1,
    )

    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)

    # Test different tasks
    tasks_to_test = [Task.IMAGE_CLASSIFICATION, Task.OBJECT_DETECTION, Task.SEGMENTATION]

    for task in tasks_to_test:
        logger.info(f"Testing task: {task.value.upper()}")
        logger.debug(f"Default prompt: {get_task_prompt(task)}")

        try:
            result = session.run(test_image, task=task)
            logger.info(f"Task {task.value} result: {result}")
        except Exception as e:
            logger.error(f"Task {task.value} failed: {e}")


def example_with_real_image():
    """Example using a real image file if available."""
    logger.info("Real Image Analysis")

    # Look for any image files in the project
    image_paths = list(Path(".").glob("**/*.{jpg,jpeg,png,bmp}"))

    if not image_paths:
        logger.warning("No image files found in project directory")
        return

    image_path = image_paths[0]
    logger.info(f"Analyzing image: {image_path}")

    spec = ChatGptVisionModelSpec(model_name="gpt-4o", max_tokens=1000)

    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)

    # Load and analyze the image
    try:
        result = session.run(str(image_path), task=Task.IMAGE_CLASSIFICATION)
        logger.info(f"Image analysis result: {result}")
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")


def example_custom_prompts():
    """Example with custom domain-specific prompts."""
    logger.info("Custom Domain Prompts")

    test_image_path = "bomas.png"
    if Path(test_image_path).exists():
        test_image = np.array(Image.open(test_image_path).convert("RGB"))
    else:
        logger.warning(f"{test_image_path} not found, using placeholder")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    spec = ChatGptVisionModelSpec(model_name="gpt-4o")
    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)

    # Custom medical analysis prompt
    medical_prompt = (
        "Analyze this image as if it were a medical scan or diagnostic image. "
        "Identify any areas of concern, anomalies, or notable features. "
        "Be thorough but note that this is for educational purposes only."
    )

    # Custom architectural analysis prompt
    architectural_prompt = (
        "Examine this image from an architectural perspective. "
        "Identify structural elements, design styles, materials, "
        "and comment on the spatial composition and aesthetic qualities."
    )

    # Custom security analysis prompt
    security_prompt = (
        "Analyze this image from a security perspective. "
        "Identify potential security risks, access points, surveillance coverage, "
        "and recommend security improvements."
    )

    prompts = {
        "Medical Analysis": medical_prompt,
        "Architectural Analysis": architectural_prompt,
        "Security Assessment": security_prompt,
    }

    for name, prompt in prompts.items():
        logger.info(f"Running analysis: {name}")
        try:
            result = session.run(test_image, prompt=prompt)
            logger.info(f"{name} result: {result}")
        except Exception as e:
            logger.error(f"{name} failed: {e}")


def example_batch_processing():
    """Example of processing multiple images in batch."""
    logger.info("Batch Image Processing")

    # Look for multiple image files
    image_paths = list(Path(".").glob("**/*.{jpg,jpeg,png,bmp}"))[:3]  # Limit to 3 for demo

    if not image_paths:
        logger.warning("No image files found for batch processing")
        return

    spec = ChatGptVisionModelSpec(model_name="gpt-4o", max_tokens=200, temperature=0.0)
    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)

    results = {}
    for image_path in image_paths:
        logger.info(f"Processing image: {image_path.name}")
        try:
            result = session.run(
                str(image_path),
                prompt="Provide a brief description of this image in 1-2 sentences.",
            )
            results[image_path.name] = result
            logger.info(f"Successfully processed {image_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            results[image_path.name] = f"Error: {e}"

    logger.info(f"Batch processing complete. Processed {len(results)} images")
    return results


def example_enterprise_deployment():
    """Demonstrate enterprise deployment patterns with Azure OpenAI."""
    logger.info("Enterprise Deployment Patterns")

    # Example 1: Service principal authentication
    logger.info("Service Principal Authentication Pattern")
    logger.info("Set these environment variables for production:")
    logger.info("AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")

    # Example 2: Multiple deployment configurations
    deployment_configs = [
        {
            "name": "primary",
            "endpoint": "https://primary.openai.azure.com/",
            "deployment": "gpt-4-vision",
        },
        {
            "name": "secondary",
            "endpoint": "https://secondary.openai.azure.com/",
            "deployment": "gpt-4o-vision",
        },
        {
            "name": "dev",
            "endpoint": "https://dev.openai.azure.com/",
            "deployment": "gpt-4-vision-dev",
        },
    ]

    for config in deployment_configs:
        logger.info(f"Configuration: {config['name']}")
        logger.info(f"  Endpoint: {config['endpoint']}")
        logger.info(f"  Deployment: {config['deployment']}")

        try:
            AzureChatGptVisionProvider(
                endpoint_url=config["endpoint"], deployment_name=config["deployment"]
            )
            logger.info(f"  Status: Successfully configured {config['name']}")
        except Exception as e:
            logger.error(f"  Status: Failed to configure {config['name']}: {e}")

    # Example 3: Cost optimization settings
    logger.info("Cost Optimization Configuration")
    optimized_spec = ChatGptVisionModelSpec(
        model_name="gpt-4o",
        max_tokens=150,  # Reduced for cost control
        temperature=0.0,  # Deterministic responses
        system_prompt="Provide concise, focused analysis.",
    )
    logger.info(
        f"Optimized spec: max_tokens={optimized_spec.max_tokens}, temperature={optimized_spec.temperature}"
    )


def example_monitoring_and_logging():
    """Demonstrate monitoring and logging best practices."""
    logger.info("Monitoring and Logging Examples")

    test_image_path = "bomas.png"
    if Path(test_image_path).exists():
        test_image = np.array(Image.open(test_image_path).convert("RGB"))
    else:
        logger.warning(f"{test_image_path} not found, using placeholder")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Enhanced logging configuration
    spec = ChatGptVisionModelSpec(model_name="gpt-4o", max_tokens=300, name="production-analyzer")

    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)

    # Example with detailed logging
    import time

    start_time = time.time()

    try:
        logger.info(f"Starting inference request for {test_image}")
        logger.debug(f"Model config: {spec}")

        result = session.run(test_image, prompt="Analyze this image for production monitoring.")

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Inference completed successfully in {duration:.2f} seconds")
        logger.info(f"Response length: {len(result)} characters")
        logger.debug(f"Full response: {result}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error(f"Inference failed after {duration:.2f} seconds: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


def example_error_handling():
    """Demonstrate error handling scenarios with Azure provider."""
    logger.info("Error Handling Examples for Azure OpenAI")

    # Test without endpoint URL
    logger.info("Testing missing endpoint URL scenario")
    try:
        # Clear endpoint temporarily
        original_endpoint = os.environ.get("ENDPOINT_URL")
        if "ENDPOINT_URL" in os.environ:
            del os.environ["ENDPOINT_URL"]

        provider = AzureChatGptVisionProvider()
        spec = ChatGptVisionModelSpec()
        session = InferenceSession(spec, provider)

        # This should fail
        session.run(np.zeros((100, 100, 3)), prompt="Test")

    except Exception as e:
        logger.info(f"Expected error caught: {e}")
    finally:
        # Restore endpoint URL
        if original_endpoint:
            os.environ["ENDPOINT_URL"] = original_endpoint

    # Test with invalid image data
    logger.info("Testing invalid image data scenario")
    try:
        spec = ChatGptVisionModelSpec()
        provider = AzureChatGptVisionProvider(
            endpoint_url="https://test.openai.azure.com/", deployment_name="gpt-4-vision"
        )
        session = InferenceSession(spec, provider)

        # Invalid image input
        session.run("not an image", prompt="Test")
    except Exception as e:
        logger.info(f"Expected error caught: {e}")

    # Test with invalid deployment name
    logger.info("Testing invalid deployment name scenario")
    try:
        spec = ChatGptVisionModelSpec()
        provider = AzureChatGptVisionProvider(
            endpoint_url=os.getenv("ENDPOINT_URL", "https://test.openai.azure.com/"),
            deployment_name="nonexistent-deployment",
        )
        session = InferenceSession(spec, provider)

        # Load image as numpy array for API call
        if Path("bomas.png").exists():
            test_img = np.array(Image.open("bomas.png").convert("RGB"))
        else:
            test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        session.run(test_img, prompt="Test")
    except Exception as e:
        logger.info(f"Expected deployment error caught: {e}")


if __name__ == "__main__":
    # Check Azure environment variables
    has_azure = bool(os.getenv("ENDPOINT_URL"))

    if not has_azure:
        logger.warning("Azure OpenAI not configured")
        logger.info("Required environment variables:")
        logger.info("export ENDPOINT_URL='https://your-resource.openai.azure.com/'")
        logger.info("export DEPLOYMENT_NAME='gpt-4-vision-preview'")

    # Check for test images
    if not Path("bomas.png").exists():
        logger.warning("bomas.png not found. Some examples may fail")

    # Run examples
    logger.info("Starting Azure OpenAI ChatGPT Vision Examples")

    try:
        # Always show Azure setup information
        example_azure_setup()

        # Run Azure examples if configured
        if has_azure:
            example_basic_usage()
        else:
            logger.warning("Skipping live examples due to missing Azure configuration")

        logger.info("All examples completed successfully")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        logger.info("Requirements checklist:")
        logger.info("1. Azure OpenAI resource with GPT-5 Vision deployment")
        logger.info("2. ENDPOINT_URL and DEPLOYMENT_NAME environment variables")
        logger.info("3. Required dependencies: pip install openai azure-identity Pillow")
        logger.info("4. Azure authentication (az login or service principal)")
        logger.info("5. Test images in the current directory")
        raise
