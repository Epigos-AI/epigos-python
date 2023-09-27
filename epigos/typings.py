from __future__ import annotations

import enum


class ModelType(str, enum.Enum):
    """
    Model type.

    Enums for types of models supported
    """

    classification = "classification"
    object_detection = "object_detection"
