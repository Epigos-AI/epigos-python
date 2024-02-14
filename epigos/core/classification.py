from epigos.core.base import PredictionModel
from epigos.data_classes.prediction import Classification


class ClassificationModel(PredictionModel):
    """
    Classification Model.

    Manages the model inferences for classification models trained in the platform
    """

    def _build_url(self) -> str:
        return f"/predict/classify/{self._model_id}/"

    def predict(self, image_path: str, confidence: float = 0.7) -> Classification:
        """
        Makes classifcation prediction for the given image.

        :param image_path: Path to image (can be local file or remote url).
        :param confidence: Prediction confidence
        :return: Prediction object
        """
        image = self._prepare_image(image_path)

        data = {"image": image, "confidence": confidence}
        url = self._build_url()
        res = self._client.make_post(path=url, json=data)

        return Classification(
            category=res["category"],
            confidence=res["confidence"],
            predictions=res["predictions"],
        )
