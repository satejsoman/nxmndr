# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SAM (Segment Anything Model) family handling for server-side inference.

This module centralizes all SAM-specific logic including SAM3, SAM3Tracker, and other variants.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import torch


def is_sam_model(model_id: str, model_spec: Dict[str, Any]) -> bool:
    """Detect if a model is a SAM-style model by repo name or metadata."""
    repo_name = model_spec.get("repo_id") or model_spec.get("name") or model_id or ""
    return "sam" in repo_name.lower()


def handle_sam_inference(
    model_id: str,
    model_spec: Dict[str, Any],
    image_array: np.ndarray,
    options: Dict[str, Any],
    device: str = "cuda",
    logger=None,
    token: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Handle SAM inference routing and execution.

    Parses SAM prompts from options and routes to appropriate model.
    Geometry (points/bbox) takes precedence over text prompts.
    Returns None if no prompts are provided (caller should use standard inference).

    Args:
        model_id: Model identifier
        model_spec: Model metadata dict
        image_array: Input image as numpy array
        options: Request options dict containing SAM prompts
        device: Device to run on
        logger: Optional logger instance
        token: Optional HuggingFace API token for gated repos

    Returns:
        Numpy array of masks, or None if no prompts provided
    """
    import json
    from PIL import Image as PILImage

    # Parse SAM prompts from options
    text_prompt = options.get("sam_text_prompt", "").strip() or None
    input_points = None
    input_bbox = None
    conf_threshold = float(options.get("sam_conf_threshold", 0.5))
    mask_threshold = float(options.get("sam_mask_threshold", 0.5))

    if "sam_input_points" in options:
        try:
            input_points = json.loads(options["sam_input_points"])
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse sam_input_points: {e}")

    if "sam_input_bbox" in options:
        try:
            input_bbox = json.loads(options["sam_input_bbox"])
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse sam_input_bbox: {e}")

    # Determine if we have geometric prompts (takes precedence)
    has_geometry = input_points or input_bbox

    # Return None if no prompts - caller should use standard inference
    if not has_geometry and not text_prompt:
        return None

    # Convert input array to PIL Image
    if image_array.ndim == 3:
        image = PILImage.fromarray(image_array.astype("uint8"))
    else:
        image = PILImage.fromarray(image_array.astype("uint8"), mode="L")

    if has_geometry:
        # Use SAM3Tracker for geometric prompts
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        if logger:
            logger.info(
                f"Loading SAM3Tracker for geometric prompts: points={bool(input_points)}, bbox={bool(input_bbox)}"
            )
            if not token:
                logger.warning("No HF token provided for SAM3Tracker loading")

        # Load model
        model = Sam3TrackerModel.from_pretrained(
            model_spec.get("repo_id", "facebook/sam3"), token=token
        ).to(device)
        processor = Sam3TrackerProcessor.from_pretrained(
            model_spec.get("repo_id", "facebook/sam3"), token=token
        )

        # Convert bbox to points if provided
        if input_bbox and not input_points:
            x_min, y_min, x_max, y_max = input_bbox
            input_points = [
                [[[int(x_min), int(y_min)]]],
                [[[int(x_max), int(y_min)]]],
                [[[int(x_max), int(y_max)]]],
                [[[int(x_min), int(y_max)]]],
            ]
            if logger:
                logger.info(f"Converted bbox {input_bbox} to 4 corner points")

        results = run_sam3_tracker_inference(
            model, processor, image, input_points=input_points, device=device
        )

        if logger:
            logger.info(f"SAM3Tracker inference completed: output shape={results['masks'].shape}")

        return results["masks"]

    elif text_prompt:
        # Use SAM3 for text-based prompts
        from transformers import Sam3Model, Sam3Processor

        if logger:
            logger.info(f"Loading SAM3 for text prompt: {text_prompt!r}")
            if not token:
                logger.warning("No HF token provided for SAM3 loading")

        # Load model
        model = Sam3Model.from_pretrained(
            model_spec.get("repo_id", "facebook/sam3"), token=token
        ).to(device)
        processor = Sam3Processor.from_pretrained(
            model_spec.get("repo_id", "facebook/sam3"), token=token
        )

        results = run_sam3_text_inference(
            model,
            processor,
            image,
            text_prompt,
            device=device,
            threshold=conf_threshold,
            mask_threshold=mask_threshold,
            logger=logger,
        )

        if logger:
            logger.info(f"SAM3 text inference completed: output shape={results['masks'].shape}")

        return results["masks"]

    else:
        raise ValueError("No SAM prompts provided (need text_prompt, input_points, or input_bbox)")


def generate_grid_points(
    tile_shape: Tuple[int, int], num_points: int = 16
) -> List[List[List[List[int]]]]:
    """
    Generate num_points uniformly spaced points for a tile of shape (H, W).
    Returns: list of points [[[x, y]]] for SAM3Tracker
    """
    h, w = tile_shape[:2]
    grid_size = int(np.sqrt(num_points))
    x_coords = np.linspace(0, w - 1, grid_size, dtype=int)
    y_coords = np.linspace(0, h - 1, grid_size, dtype=int)
    grid_points = []
    for y in y_coords:
        for x in x_coords:
            grid_points.append([[[int(x), int(y)]]])
    return grid_points


def run_sam3_text_inference(
    model,
    processor,
    image,
    text_prompt: str,
    device: str = "cuda",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    logger=None,
) -> Dict[str, Any]:
    """
    Run SAM3 text-based inference.

    Args:
        model: SAM3Model instance
        processor: SAM3Processor instance
        image: PIL Image or numpy array
        text_prompt: Text prompt for segmentation
        device: Device to run on
        threshold: Detection threshold
        mask_threshold: Mask threshold

    Returns:
        Dict with 'masks', 'boxes', 'scores'
    """
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"].cpu().numpy() if len(results["masks"]) > 0 else np.array([])
    boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
    scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])
    if len(masks) > 1:
        # drop whole-region "group" masks that contain other instances (see drop_group_masks)
        kept = drop_group_masks(masks, logger=logger)
        masks, boxes, scores = masks[kept], boxes[kept], scores[kept]

    return {"masks": masks, "boxes": boxes, "scores": scores}


def run_sam3_tracker_inference(
    model,
    processor,
    image,
    input_points: Optional[List[List[List[List[int]]]]] = None,
    device: str = "cuda",
    num_points: int = 16,
) -> Dict[str, Any]:
    """
    Run SAM3Tracker point-based inference.

    Args:
        model: SAM3TrackerModel instance
        processor: SAM3TrackerProcessor instance
        image: PIL Image or numpy array
        input_points: List of points in format [[[x, y]]] or None to generate grid
        device: Device to run on
        num_points: Number of grid points to generate if input_points is None

    Returns:
        Dict with 'masks' (numpy array)
    """
    from PIL import Image as PILImage

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)

    # Generate grid points if not provided
    if input_points is None:
        input_points = generate_grid_points(image.size[::-1], num_points=num_points)

    # Create labels (all positive clicks)
    input_labels = [[[1]] for _ in input_points]

    # Create batch of same image repeated for each point
    batch_images = [image] * len(input_points)

    # Process inputs
    tracker_inputs = processor(
        images=batch_images,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        tracker_outputs = model(**tracker_inputs, multimask_output=False)

    # Post-process masks
    batch_masks = processor.post_process_masks(
        tracker_outputs.pred_masks.cpu(), tracker_inputs["original_sizes"]
    )

    # Extract masks
    all_masks = []
    for i in range(len(batch_masks)):
        mask = batch_masks[i][0, 0].numpy()  # Get first object, first mask
        all_masks.append(mask)

    return {
        "masks": np.array(all_masks) if all_masks else np.array([]),
    }


def drop_group_masks(
    masks: np.ndarray,
    min_contained: int = 2,
    overlap: float = 0.8,
    mask_logits: Optional[np.ndarray] = None,
    logger=None,
) -> np.ndarray:
    """Indices of masks to keep after dropping "group" masks that contain other instances.

    For a concept like "field", SAM 3 returns both per-parcel instances and a whole-region mask
    (the union of all parcels). The region mask has the HIGHEST concept score, so it cannot be
    removed by score; it is identified structurally: a mask is dropped when at least
    ``min_contained`` other masks lie inside it (``overlap`` fraction of their pixels). On the
    case-study chips the region mask contained 4 to 56 parcels and covered 100% of the chip,
    parcels contained none (mean in-mask logit 1.3-1.5 for the region vs 1.7-5.5 for parcels).

    Args:
        masks: (N, H, W) binary masks (bool or 0/1).
        min_contained: drop a mask that contains at least this many other masks.
        overlap: fraction of the contained mask's pixels that must lie inside the container.
        mask_logits: optional (N, H, W) mask logits, only used to log the in-mask mean logit.
    A contained mask that itself covers >= ``overlap`` of the container is treated as a duplicate
    rather than a group member, so the smallest non-empty mask is never dropped.
    Returns:
        1-D array of kept indices (in the original order).
    """
    n = len(masks)
    if n < 2:
        return np.arange(n)
    binm = [np.asarray(m) > 0.5 for m in masks]
    areas = np.array([int(b.sum()) for b in binm])
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if areas[i] == 0:
            continue
        contained = 0
        for j in range(n):
            if j == i or areas[j] == 0 or areas[j] > areas[i]:
                continue
            inter = np.logical_and(binm[i], binm[j]).sum()
            # j inside i, but i not inside j: mutual containment is a duplicate detection, not a group
            if inter >= overlap * areas[j] and inter < overlap * areas[i]:
                contained += 1
                if contained >= min_contained:
                    break
        if contained >= min_contained:
            keep[i] = False
            if logger:
                extra = ""
                if mask_logits is not None:
                    extra = f", mean in-mask logit {float(np.asarray(mask_logits[i])[binm[i]].mean()):.2f}"
                logger.info(
                    "SAM3: dropping group mask %d (area %.2f of chip, contains >=%d instances%s)",
                    i, areas[i] / binm[i].size, contained, extra,
                )
    return np.flatnonzero(keep)


def select_masks_by_area_and_iou(
    masks: np.ndarray,
    order: np.ndarray,
    min_area: float = 0.001,
    max_area: float = 0.5,
    iou: float = 0.5,
) -> List[int]:
    """Score-free instance selection: walk ``masks`` in ``order`` (best first), drop masks whose
    area fraction is outside [min_area, max_area] (specks and whole-chip masks) and masks with
    IoU > ``iou`` against an already kept mask. Returns the kept indices in ``order``.
    """
    binm = [np.asarray(m) > 0.5 for m in masks]
    total = float(binm[0].size) if len(binm) else 1.0
    kept: List[int] = []
    for i in order:
        a = binm[i].sum() / total
        if a < min_area or a > max_area:
            continue
        ok = True
        for j in kept:
            inter = np.logical_and(binm[i], binm[j]).sum()
            union = np.logical_or(binm[i], binm[j]).sum()
            if union and inter / union > iou:
                ok = False
                break
        if ok:
            kept.append(int(i))
    return kept


def filter_overlapping_masks(masks: np.ndarray, keep_smaller: bool = True) -> np.ndarray:
    """
    Remove overlapping masks, optionally keeping the smaller one in each overlapping pair.

    Args:
        masks: Array of masks with shape (N, H, W)
        keep_smaller: If True, keep smaller mask; if False, keep larger mask

    Returns:
        Filtered array of masks
    """
    if len(masks) == 0:
        return masks

    keep = [True] * len(masks)
    areas = [np.sum(mask > 0.5) for mask in masks]

    for i in range(len(masks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not keep[j]:
                continue
            intersection = np.logical_and(masks[i] > 0.5, masks[j] > 0.5)
            if np.any(intersection):
                # Drop based on size comparison
                if keep_smaller:
                    if areas[i] < areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # i is dropped, skip further checks
                else:
                    if areas[i] > areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

    return np.array([mask for mask, k in zip(masks, keep) if k])
