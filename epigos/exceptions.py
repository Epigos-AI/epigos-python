import typing


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
