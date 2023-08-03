import typing

from epigos import typings
from epigos.core.base import PredictionModel
from epigos.utils import image as image_utils
from epigos.utils.prediction import Prediction


class ClassificationModel(PredictionModel):
    """
    Classification Model.

    Manages the model inferences for classification models trained in the platform
    """

    _model_type: typings.ModelType = typings.ModelType.classification

    def _build_url(self) -> str:
        return f"/predict/classify/{self._model_id}/"

    def predict(self, image_path: str, confidence: float = 0.7) -> Prediction:
        """
        Makes classifcation prediction for the given image.

        :param image_path: Path to image (can be local file or remote url).
        :param confidence: Prediction confidence
        :return: Prediction object
        """
        if image_utils.is_path(image_path):
            image = image_utils.image_to_b64(image_path)
        elif image_utils.is_url(image_path):
            image = image_path
        else:
            raise ValueError(f"Image does not exist at {image_path}!")

        data = {"image": image, "confidence": confidence}
        url = self._build_url()
        res = self._client.call_api(path=url, method="post", json=data)

        pred = typing.cast(typings.ClassificationPrediction, res)
        return Prediction(
            json_predictions=pred, image_path=image_path, model_type=self._model_type
        )
