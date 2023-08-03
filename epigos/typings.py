from __future__ import annotations

import enum
import typing

from typing_extensions import TypeAlias, TypedDict


class ModelType(str, enum.Enum):
    """
    Model type.

    Enums for types of models supported
    """

    classification = "classification"
    object_detection = "object_detection"


class PredictedClass(TypedDict):
    """
    Predicted Class.

    Represents classification prediction item
    """

    category: str
    confidence: float


class DetectedObject(TypedDict):
    """
    Detected Object.

    Represents object detection item
    """

    label: str
    confidence: float
    left: float
    top: float
    width: float
    height: float


class ObjectDetectionPrediction(TypedDict):
    """
    Object Detection Prediction.

    Represents object detection inference results.
    """

    predictions: typing.List[DetectedObject]


class ClassificationPrediction(PredictedClass):
    """
    Classification Prediction.

    Represents classification prediction results.
    """

    predictions: typing.List[PredictedClass]


Predictions: TypeAlias = typing.Union[
    ClassificationPrediction, ObjectDetectionPrediction
]
