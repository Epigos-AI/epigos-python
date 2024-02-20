from typing_extensions import Unpack

from epigos import typings
from epigos.core.base import PredictionModel
from epigos.data_classes.prediction import ObjectDetection


class ObjectDetectionModel(PredictionModel):
    """
    Object Detection Model.

    Manages the model inferences for object detection models trained in the platform
    """

    def _build_url(self) -> str:
        return f"/predict/detect/{self._model_id}/"

    def detect(
        self,
        image_path: str,
        confidence: float = 0.7,
        **kwargs: Unpack[typings.DetectOptions],
    ) -> ObjectDetection:
        """
        Infers detections based on image from specified model and image path.

        :param image_path: Path to image (can be local file or remote url).
        :param confidence: Prediction confidence.
        :param kwargs: Annotation options for the prediction
        :return: ObjectDetection object
        """
        annotate = kwargs.get("annotate") or True
        stroke_width = kwargs.get("stroke_width")
        show_prob = kwargs.get("show_prob") or True

        image = self._prepare_image(image_path)

        data = {
            "image": image,
            "confidence": confidence,
            "annotate": annotate,
            "stroke_width": stroke_width,
            "show_prob": show_prob,
        }
        url = self._build_url()
        res = self._client.make_post(path=url, json=data)

        return ObjectDetection(
            detections=res["detections"], base64_image=res.get("image")
        )
