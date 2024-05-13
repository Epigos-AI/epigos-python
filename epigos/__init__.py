from .client import Epigos
from .core import ClassificationModel, ObjectDetectionModel, Project
from .exceptions import EpigosException

__all__ = (
    "ClassificationModel",
    "Epigos",
    "EpigosException",
    "ObjectDetectionModel",
    "Project",
)
