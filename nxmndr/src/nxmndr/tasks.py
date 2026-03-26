# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from enum import Enum


class Task(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    EMBEDDING_VISUALIZATION = "embedding_visualization"
    FINE_TUNING = "fine_tuning"
