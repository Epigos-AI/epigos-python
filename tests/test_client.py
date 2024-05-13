import httpx
import pytest
import respx

from epigos import ClassificationModel, Epigos, EpigosException, ObjectDetectionModel
from epigos.__version__ import __version__
from epigos.client import RETRY_STATUS_CODES


def test_client_and_headers(client: Epigos):
    assert isinstance(client.client, httpx.Client)
    assert client.client.base_url == "http://test"

    expected_headers = {
        "Content-Type": "application/json",
        "X-Api-Key": "api_key",
        "X-Client-Sdk": f"Epigos-SDK/Python; Version: {__version__}",
    }
    for key, value in expected_headers.items():
        assert client.client.headers.get(key) == value


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_ok(
    client: Epigos, respx_mock: respx.MockRouter, path: str, method: str
):
    output = {"data": "ok"}
    respx_mock.request(method, path).mock(return_value=httpx.Response(200, json=output))
    resp = client.make_request(path=path, method=method)
    assert resp == output


def test_client_can_make_get(client: Epigos, respx_mock: respx.MockRouter):
    output = {"data": "ok"}
    respx_mock.get("/path").mock(return_value=httpx.Response(200, json=output))
    resp = client.make_get(path="/path")
    assert resp == output


def test_client_can_make_post(client: Epigos, respx_mock: respx.MockRouter):
    output = {"data": "ok"}
    respx_mock.post("/path").mock(return_value=httpx.Response(201, json=output))
    resp = client.make_post(path="/path", json={})
    assert resp == output


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_exception(
    client: Epigos, respx_mock: respx.MockRouter, path: str, method: str
):
    output = {"message": "validation errors", "details": [{"test": "error"}]}
    respx_mock.request(method, path).mock(return_value=httpx.Response(400, json=output))

    with pytest.raises(EpigosException) as exc:
        client.make_request(path=path, method=method)

    assert exc.value.details == output["details"]
    assert exc.value.status_code == 400
    assert client._retry.attempt_number == 1


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_exception_non_json_response(
    client: Epigos, respx_mock: respx.MockRouter, path: str, method: str
):
    output = "Validation error"
    respx_mock.request(method, path).mock(return_value=httpx.Response(400, text=output))

    with pytest.raises(EpigosException) as exc:
        client.make_request(path=path, method=method)
    assert exc.value.status_code == 400
    assert client._retry.attempt_number == 1


def test_client_resouce_methods(client: Epigos):
    assert isinstance(client.classification("model_id"), ClassificationModel)
    assert isinstance(client.object_detection("model_id"), ObjectDetectionModel)

    model_id = None
    with pytest.raises(ValueError):
        client.classification(model_id)

    with pytest.raises(ValueError):
        client.object_detection(model_id)


@pytest.mark.parametrize("status_code", RETRY_STATUS_CODES)
def test_client_can_call_api_retry_on_exception(
    client: Epigos, respx_mock: respx.MockRouter, status_code
):
    method = "post"
    path = "/foo"
    client.retry_max_attempts = 2
    output = {"message": "gateway errors", "details": [{"test": "error"}]}
    respx_mock.request(method, path).mock(
        return_value=httpx.Response(status_code, json=output)
    )

    with pytest.raises(EpigosException) as exc:
        client.make_request(path=path, method=method)

    assert exc.value.details == output["details"]
    assert exc.value.status_code == status_code
    assert client._retry.attempt_number == client.retry_max_attempts
