import dataclasses
import typing

from PIL import Image

from epigos.utils.image import b64_to_image


@dataclasses.dataclass
class PredictedClass:
    """
    Predicted Class.

    Represents classification prediction item
    """

    category: str
    confidence: float


@dataclasses.dataclass
class Classification(PredictedClass):
    """
    Classification.

    Represents classification prediction results.
    """

    predictions: list[PredictedClass] = dataclasses.field(default_factory=list)

    def dict(self) -> typing.Dict[str, typing.Any]:
        """
        Return dict representation of the data
        :return: dicts
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DetectedObject:
    """
    Detected Object.

    Represents object detection item
    """

    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


@dataclasses.dataclass
class ObjectDetection:
    """
    Object Detection.

    Represents object detection inference results.
    """

    detections: typing.List[DetectedObject] = dataclasses.field(default_factory=list)
    base64_image: typing.Optional[str] = dataclasses.field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._image = b64_to_image(self.base64_image) if self.base64_image else None

    def get_image(self) -> Image.Image:
        """
        Get the decoded image from the API response
        :return: Pil.Image
        """
        if not self._image:
            raise ValueError(
                "No image returned for this prediction. "
                "Set `annotate=True` when making predictions "
                "to return the annotated image"
            )
        return self._image

    def show(self) -> None:
        """
        Displays the Pil.Image
        :return:
        """
        self.get_image().show()

    def dict(self) -> typing.Dict[str, typing.Any]:
        """
        Return dict representation of the data
        :return: dicts
        """
        return dataclasses.asdict(self)
