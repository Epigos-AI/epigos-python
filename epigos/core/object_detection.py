import dataclasses
import typing

from epigos.core.base import PredictionModel
from epigos.data_classes.prediction import ObjectDetection


@dataclasses.dataclass
class DetectOptions:
    """
    DetectOptions options used to customize the output.

    :param annotate: If True, it annotates the image with the predicted objects.
    :param stroke_width: Specify bounding boxes border size.
    :param show_prob: If True, detected objects will show detection confidence
        for each object in the image.
    """

    annotate: bool = dataclasses.field(default=True)
    stroke_width: typing.Optional[int] = dataclasses.field(default=None)
    show_prob: bool = dataclasses.field(default=True)


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
        options: typing.Optional[DetectOptions] = None,
    ) -> ObjectDetection:
        """
        Infers detections based on image from specified model and image path.

        :param image_path: Path to image (can be local file or remote url).
        :param confidence: Prediction confidence.
        :param options: Annotation options for the prediction
        :return: ObjectDetection object
        """
        if not options:
            options = DetectOptions()

        image = self._prepare_image(image_path)

        data = {
            "image": image,
            "confidence": confidence,
            "annotate": options.annotate,
            "stroke_width": options.stroke_width,
            "show_prob": options.show_prob,
        }
        url = self._build_url()
        res = self._client.make_post(path=url, json=data)

        return ObjectDetection(
            detections=res["detections"], base64_image=res.get("image")
        )
