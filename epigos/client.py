import typing
from json import JSONDecodeError

import httpx

from .__version__ import __version__
from .core import ClassificationModel, ObjectDetectionModel

BASE_API = "https://api.epigos.ai"


class EpigosException(Exception):
    """
    Epigos Exception

    There was an error returned from making request to Epigos AI

    :param message: response status message received from api
    :param details: response error details.
    :param status_code: HTTP status code received from request.
    """

    def __init__(
        self,
        message: typing.Optional[typing.Any],
        details: typing.Optional[typing.Any],
        status_code: int,
    ) -> None:
        super().__init__(
            f"Error Reason: {message} \n Error Details: {details} "
            f"\n HTTP Status Code: {status_code}"
        )
        self.status_code = status_code
        self.details = details


class Epigos:
    """
    Epigos.

    API client for handling resource request to Epigos API.

    :param api_key: Your epigos.ai workspace api key
    :param base_url: Base url to the epigos api.
    """

    def __init__(self, api_key: str, base_url: str = BASE_API):
        self._api_key = api_key
        self.client = httpx.Client(
            base_url=httpx.URL(base_url),
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self._api_key,
                "X-Client-Sdk": f"Epigos-SDK/Python; Version: {__version__}",
            },
        )

    @staticmethod
    def _deserialize(response: httpx.Response) -> typing.Dict[str, typing.Any]:
        """
        Deserializes response into an object.

        :param response: Htttp response object to be deserialised
        :returns: deserialized response object
        """
        try:
            json_data = response.json()
        except JSONDecodeError:
            json_data = None

        data = json_data if isinstance(json_data, dict) else {"message": response.text}

        if not response.is_success:
            raise EpigosException(
                message=data.get("message"),
                details=data.get("details") or [],
                status_code=response.status_code,
            )
        return data

    def call_api(
        self,
        *,
        path: str,
        method: str,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: typing.Any,
    ) -> typing.Dict[str, typing.Any]:
        """
        Makes the HTTP request and returns deserialized data

        :param path: Path to method endpoint
        :param method: HTTP Method to call
        :param json: Request body
        :param params: Query parameters in the url
        :returns: Returns the response data from the api
        """
        response = self.client.request(
            method, path, params=httpx.QueryParams(params), json=json, **kwargs
        )
        return self._deserialize(response)

    def classification(self, model_id: str) -> ClassificationModel:
        """
        Creates an instance of classification model using the given model ID
        :param model_id: Model to load
        :return: ClassificationModel
        """
        if model_id is None:
            raise ValueError("model_id is required")
        return ClassificationModel(self, model_id)

    def object_detection(self, model_id: str) -> ObjectDetectionModel:
        """
        Creates an instance of object detection model using the given model ID
        :param model_id: Model to load
        :return: ObjectDetectionModel
        """
        if model_id is None:
            raise ValueError("model_id is required")
        return ObjectDetectionModel(self, model_id)
