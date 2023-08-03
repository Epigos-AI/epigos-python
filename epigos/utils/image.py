import base64
import io
import os
import urllib.parse

import httpx
from PIL import Image


def is_path(file_path: str) -> bool:
    """
    Check whether file path is exists
    :param file_path: path of file to check
    :return: Boolean
    """
    return os.path.exists(file_path)


def is_url(url: str) -> bool:
    """
    Check whether a hosted image path is valid
    :param url: URL of image
    :returns: Boolean
    """
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        return False

    resp = httpx.head(url, follow_redirects=True)
    return resp.is_success


def check_image_path(image_path: str) -> bool:
    """
    Check whether a local OR remote image path is valid
    :param image_path: local path or URL of image
    :returns: Boolean
    """
    return is_path(image_path) or is_url(image_path)


def image_to_b64(image_path: str) -> str:
    """
    Convert local image file to base64 encoded string
    :param image_path: local path to image
    :return: base64 encoded string
    """
    # open Image in RGB Format
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buffer = io.BytesIO()
        im.save(buffer, quality=90, format="JPEG")
        # Base64 encode image
        img_bytes = base64.b64encode(buffer.getvalue())

    return img_bytes.decode("ascii")
