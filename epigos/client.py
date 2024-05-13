import typing
from json import JSONDecodeError

import httpx
import tenacity

from .__version__ import __version__
from .core import ClassificationModel, ObjectDetectionModel, Project
from .exceptions import EpigosException
from .utils import logger

BASE_API = "https://api.epigos.ai"
RETRY_STATUS_CODES = [502, 503, 504]


class Epigos:
    """
    Epigos.

    API client for handling resource request to Epigos API.

    :param api_key: Your epigos.ai workspace api key
    :param base_url: Base url to the epigos api.
    :param timeout: HTTP request timeout in seconds. Defaults to 15 seconds.
    :param retries: Number of times to retry requests. Defaults to 3.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_API,
        timeout: float = 15.0,
        retries: int = 3,
    ):
        self._api_key = api_key
        self.client = httpx.Client(
            base_url=httpx.URL(base_url),
            timeout=httpx.Timeout(timeout=timeout),
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self._api_key,
                "X-Client-Sdk": f"Epigos-SDK/Python; Version: {__version__}",
            },
        )
        self.retry_max_attempts = retries
        self._retry: typing.Optional[tenacity.RetryCallState] = None

    @staticmethod
    def _deserialize(
        response: httpx.Response,
    ) -> typing.Any:
        """
        Deserializes response into an object.

        :param response: Htttp response object to be deserialised
        :returns: deserialized response object
        """
        try:
            json_data = response.json()
        except JSONDecodeError:
            json_data = {"message": response.text}

        if not response.is_success:
            raise EpigosException(
                message=json_data.get("message"),
                details=json_data.get("details") or [],
                status_code=response.status_code,
            )
        return json_data

    def make_request(
        self,
        *,
        path: str,
        method: str,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Makes the HTTP request and returns deserialized data

        :param path: Path to method endpoint
        :param method: HTTP Method to call
        :param json: Request body
        :param params: Query parameters in the url
        :returns: Returns the response data from the api
        """
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.retry_max_attempts),
            wait=tenacity.wait_random_exponential(multiplier=1, max=15),
            reraise=True,
            retry=tenacity.retry_if_exception(
                lambda exc: isinstance(exc, EpigosException)
                and exc.status_code in RETRY_STATUS_CODES
            ),
            before_sleep=tenacity.before_sleep_log(logger, logger.level),
        )
        for attempt in retryer:
            with attempt:
                attempt.retry_state.fn = self.make_request  # type: ignore[assignment]
                self._retry = attempt.retry_state
                response = self.client.request(
                    method,
                    path,
                    params=httpx.QueryParams(params),
                    json=json,
                    **kwargs,
                )
                return self._deserialize(response)

    def make_post(
        self,
        path: str,
        *,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Makes the HTTP POST request and returns deserialized data

        :param path: Path to method endpoint
        :param json: Request body
        :param params: Query parameters in the url
        :returns: Returns the response data from the api
        """
        return self.make_request(
            path=path, method="POST", json=json, params=params, **kwargs
        )

    def make_get(
        self,
        path: str,
        *,
        params: typing.Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Makes the HTTP GET request and returns deserialized data

        :param path: Path to method endpoint
        :param params: Query parameters in the url
        :returns: Returns the response data from the api
        """
        return self.make_request(path=path, method="GET", params=params, **kwargs)

    def project(self, project_id: str) -> Project:
        """
        Creates an instance of project using the given project_id
        :param project_id: ID of project to load
        :return: Project
        """
        if project_id is None:
            raise ValueError("project_id is required")
        return Project(self, project_id)

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
