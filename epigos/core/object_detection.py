from epigos import typings
from epigos.core.base import PredictionModel
from epigos.utils.prediction import Prediction


class ObjectDetectionModel(PredictionModel):
    """
    Object Detection Model.

    Manages the model inferences for object detection models trained in the platform
    """

    _model_type: typings.ModelType = typings.ModelType.classification

    def _build_url(self) -> str:
        return f"/predict/detect/{self._model_id}/"

    def detect(self, image_path: str, confidence: float = 0.7) -> Prediction:
        """
        Infers detections based on image from specified model and image path.

        :param image_path: Path to image (can be local file or remote url).
        :param confidence: Prediction confidence
        :return: Prediction object
        """
        raise NotImplementedError()
