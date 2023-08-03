import httpx
import pytest
import respx

from epigos import ClassificationModel, Epigos, EpigosException, ObjectDetectionModel
from epigos.__version__ import __version__


def test_client_and_headers():
    api_key = "test"
    base_url = "http://localhost"
    epigos = Epigos(api_key, base_url=base_url)

    assert isinstance(epigos.client, httpx.Client)
    assert epigos.client.base_url == base_url

    expected_headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "user-agent": f"Epigos-SDK/Python; Version: {__version__}",
    }
    for key, value in expected_headers.items():
        assert epigos.client.headers.get(key) == value


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_ok(respx_mock: respx.MockRouter, path: str, method: str):
    api_key = "test"
    client = Epigos(api_key)

    output = {"data": "ok"}
    respx_mock.request(method, path).mock(return_value=httpx.Response(200, json=output))
    resp = client.call_api(path=path, method=method)
    assert resp == output


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_exception(
    respx_mock: respx.MockRouter, path: str, method: str
):
    api_key = "test"
    client = Epigos(api_key)

    output = {"message": "validation errors", "details": [{"test": "error"}]}
    respx_mock.request(method, path).mock(return_value=httpx.Response(400, json=output))

    with pytest.raises(EpigosException) as exc:
        client.call_api(path=path, method=method)

    assert exc.value.details == output["details"]
    assert exc.value.status_code == 400


@pytest.mark.parametrize("path,method", [("/foo", "post"), ("/foo", "get")])
def test_client_can_call_api_exception_non_json_response(
    respx_mock: respx.MockRouter, path: str, method: str
):
    api_key = "test"
    client = Epigos(api_key)

    output = "Validation error"
    respx_mock.request(method, path).mock(return_value=httpx.Response(400, text=output))

    with pytest.raises(EpigosException) as exc:
        client.call_api(path=path, method=method)
    assert exc.value.status_code == 400


def test_client_resouce_methods():
    api_key = "test"
    epigos = Epigos(api_key)

    assert isinstance(epigos.classification("model_id"), ClassificationModel)
    assert isinstance(epigos.object_detection("model_id"), ObjectDetectionModel)

    model_id = None
    with pytest.raises(ValueError):
        epigos.classification(model_id)

    with pytest.raises(ValueError):
        epigos.object_detection(model_id)
