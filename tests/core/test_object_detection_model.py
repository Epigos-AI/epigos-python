from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image

from epigos import Epigos
from epigos.core.object_detection import DetectOptions

ASSETS_PATH = Path(__file__).parent.parent / "assets"


def test_detect_invalid_image(client: Epigos, respx_mock: respx.MockRouter):
    with pytest.raises(ValueError):
        client.object_detection("model_id").detect("invalid.jpg")


@pytest.mark.parametrize("annotate", [True, False])
def test_detect_image_url(
    client: Epigos,
    respx_mock: respx.MockRouter,
    object_detection_prediction,
    annotate: bool,
):
    image_url = "https://foo.bar/image.jpg"
    model = client.object_detection("model_id")

    respx_mock.head(image_url).mock(return_value=httpx.Response(200))

    url = model._build_url()
    respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=dict(object_detection_prediction))
    )

    pred = model.detect(image_url, options=DetectOptions(annotate=annotate))
    assert pred.detections == object_detection_prediction["detections"]


@pytest.mark.parametrize("annotate", [True, False])
def test_detect_local_image(
    client: Epigos,
    respx_mock: respx.MockRouter,
    object_detection_prediction,
    annotate: bool,
):
    image_path = str(ASSETS_PATH / "cat.jpg")
    model = client.object_detection("model_id")

    if annotate:
        object_detection_prediction["image"] = model._prepare_image(image_path)

    url = model._build_url()
    respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=dict(object_detection_prediction))
    )

    pred = model.detect(image_path, options=DetectOptions(annotate=annotate))
    if annotate:
        assert isinstance(pred.get_image(), Image.Image)
    else:
        with pytest.raises(
            ValueError,
            match="No image returned for this prediction. Set `annotate=True` when making predictions",
        ):
            pred.get_image()

    assert pred.dict()["detections"] == object_detection_prediction["detections"]
