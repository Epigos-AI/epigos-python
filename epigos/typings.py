from __future__ import annotations

import enum


class ProjectType(str, enum.Enum):
    """
    Project type

    Enums for types of projects
    """

    object_detection = "object_detection"
    classification = "classification"


class ModelType(str, enum.Enum):
    """
    Model type.

    Enums for types of models supported
    """

    classification = "classification"
    object_detection = "object_detection"


class AnnotationCategory(str, enum.Enum):
    """
    Annotation category

    Enums for types of annotations supported
    """

    category = "category"
    bounding_box = "bounding_box"


class BoxFormat(str, enum.Enum):
    """
    Bounding Box format

    Enums for types of annotation box format supported
    """

    pascal_voc = "pascal_voc"
    yolo = "yolo"
    coco = "coco"
