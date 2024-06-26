from __future__ import annotations

import enum
import typing


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


class DetectOptions(typing.TypedDict, total=False):
    """
    DetectOptions options used to customize the output.

    :param annotate: If True, it annotates the image with the predicted objects.
    :param stroke_width: Specify bounding boxes border size.
    :param show_prob: If True, detected objects will show detection confidence
        for each object in the image.
    """

    annotate: bool
    stroke_width: typing.Optional[int]
    show_prob: bool


class BoxFormat(str, enum.Enum):
    """
    Bounding Box format

    Enums for types of annotation box format supported
    """

    pascal_voc = "pascal_voc"
    yolo = "yolo"
    coco = "coco"
