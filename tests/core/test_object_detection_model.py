from pathlib import Path

import httpx
import pytest
import respx

from epigos import Epigos

ASSETS_PATH = Path(__file__).parent.parent / "assets"


def test_detection_not_implemented(respx_mock: respx.MockRouter):
    client = Epigos("api_key")
    model = client.object_detection("model_id")

    url = model._build_url()
    respx_mock.post(url).mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(NotImplementedError):
        model.detect(str(ASSETS_PATH / "cat.jpg"))
