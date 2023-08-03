from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from epigos import typings

if TYPE_CHECKING:
    from epigos.client import Epigos


class PredictionModel(abc.ABC):
    """
    Prediction Model.

    A generic class to represent model inferences

    :param client: Client to use for interaction with the Epigos server
    :param model_id: Unique internal reference from the Epigos platform for the model
    """

    _model_type: typings.ModelType

    def __init__(self, client: "Epigos", model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    @abc.abstractmethod
    def _build_url(self) -> str:
        raise NotImplementedError()
