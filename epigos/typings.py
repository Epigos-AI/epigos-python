from __future__ import annotations

import enum


class ProjectType(str, enum.Enum):
    object_detection = "object_detection"
    classification = "classification"


class ModelType(str, enum.Enum):
    """
    Model type.

    Enums for types of models supported
    """

    classification = "classification"
    object_detection = "object_detection"


class AnnotationType:
    category = "category"
    bounding_box = "bounding_box"
