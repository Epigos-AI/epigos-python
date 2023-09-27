from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from epigos.utils import image as image_utils

if TYPE_CHECKING:
    from epigos.client import Epigos


class PredictionModel(abc.ABC):
    """
    Prediction Model.

    A generic class to represent model inferences

    :param client: Client to use for interaction with the Epigos server
    :param model_id: Unique internal reference from the Epigos platform for the model
    """

    def __init__(self, client: "Epigos", model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    @abc.abstractmethod
    def _build_url(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def _prepare_image(image_path: str) -> str:
        if image_utils.is_path(image_path):
            image = image_utils.image_to_b64(image_path)
        elif image_utils.is_url(image_path):
            image = image_path
        else:
            raise ValueError(f"Image does not exist at {image_path}!")
        return image
