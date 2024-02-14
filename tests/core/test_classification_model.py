from pathlib import Path

import httpx
import pytest
import respx

from epigos import Epigos

ASSETS_PATH = Path(__file__).parent.parent / "assets"


def test_predict_invalid_image(client: Epigos, respx_mock: respx.MockRouter):
    with pytest.raises(ValueError):
        client.classification("model_id").predict("invalid.jpg")


@pytest.mark.parametrize(
    "image_path", [str(ASSETS_PATH / "cat.jpg"), "https://foo.bar/image.jpg"]
)
def test_predict_ok(
    client: Epigos,
    respx_mock: respx.MockRouter,
    classification_prediction,
    image_path: str,
):
    model = client.classification("model_id")

    url = model._build_url()
    respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=dict(classification_prediction))
    )
    if "https" in image_path:
        respx_mock.head(image_path).mock(return_value=httpx.Response(200))

    pred = model.predict(image_path)
    assert pred.dict() == classification_prediction
