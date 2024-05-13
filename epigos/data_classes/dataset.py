import dataclasses
import typing


@dataclasses.dataclass
class Detection:
    """
    Dataclass containing information detection annotations
    """

    bbox: typing.Tuple[int, int, int, int]
    class_name: str
    class_id: int | None = None


@dataclasses.dataclass
class Classification:
    """
    Dataclass containing information image classification annotations
    """

    class_name: str
