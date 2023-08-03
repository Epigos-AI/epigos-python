import base64
import io
from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image

from epigos.utils import image

ASSETS_PATH = Path(__file__).parent.parent / "assets"


@pytest.mark.parametrize(
    "path, valid",
    [(str(ASSETS_PATH / "cat.jpg"), True), (str(ASSETS_PATH / "invalid.jpg"), False)],
    ids=["valid", "invalid"],
)
def test_is_path(path: str, valid: bool) -> None:
    assert image.is_path(path) == valid


@pytest.mark.parametrize(
    "url, valid",
    [("https://foo.bar/image.jpg", True), ("https://foo.bar/invalid.jpg", False)],
    ids=["valid", "invalid"],
)
def test_is_url(respx_mock: respx.MockRouter, url: str, valid: bool) -> None:
    respx_mock.head(url).mock(return_value=httpx.Response(200 if valid else 404))
    assert image.is_url(url) == valid


@pytest.mark.parametrize(
    "path, is_url, valid",
    [
        (str(ASSETS_PATH / "cat.jpg"), False, True),
        (str(ASSETS_PATH / "invalid.jpg"), False, False),
        ("https://foo.bar/image.jpg", True, True),
        ("https://foo.bar/invalid.jpg", True, False),
    ],
    ids=["valid_path", "invalid_path", "valid_url", "invalid_url"],
)
def test_check_image_path(
    respx_mock: respx.MockRouter, path: str, is_url: str, valid: bool
) -> None:
    if is_url:
        respx_mock.head(path).mock(return_value=httpx.Response(200 if valid else 404))
    assert image.check_image_path(path) == valid


def test_image_to_b64() -> None:
    path = str(ASSETS_PATH / "cat.jpg")

    img_str = image.image_to_b64(path)

    im2 = Image.open(io.BytesIO(base64.b64decode(img_str)))

    assert im2.width == 320
    assert im2.height == 267
