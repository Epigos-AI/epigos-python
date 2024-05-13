from __future__ import annotations

import abc
import dataclasses
import typing
from pathlib import Path

from epigos import typings
from epigos.data_classes.dataset import Classification, Detection
from epigos.dataset import utils
from epigos.typings import BoxFormat


@dataclasses.dataclass
class BaseDataset(abc.ABC):
    """
    Base dataset class for all datasets
    """

    classes: typing.List[str]
    images: typing.Dict[str, Path]

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        :return:
            int: The number of images
        """
        return len(self.images)

    @abc.abstractmethod
    def __iter__(
        self,
    ) -> typing.Iterator[
        typing.Tuple[
            Path, typing.Union[typing.List[Classification], typing.List[Detection]]
        ]
    ]:
        raise NotImplementedError()


@dataclasses.dataclass
class DetectionDataset(BaseDataset):
    """
    Dataclass containing information about object detection dataset
    """

    annotations: typing.Dict[str, typing.List[Detection]]

    def __iter__(self) -> typing.Iterator[typing.Tuple[Path, typing.List[Detection]]]:
        """
        Iterate over the images and annotations in the dataset
        :return:
        """
        for image_name, image_path in self.images.items():
            yield image_path, self.annotations.get(image_name, [])

    @classmethod
    def from_pascal_voc(
        cls,
        images_directory_path: typing.Union[str, Path],
        annotations_directory_path: typing.Union[str, Path],
    ) -> DetectionDataset:
        """
        Read directory containing Pascal VOC dataset
        and return dictionary of image paths
        and their corresponding annotation path

        :param images_directory_path: Path to directory containing images
        :param annotations_directory_path: Path to directory containing annotations
        :return:
        """
        images_directory_path = Path(images_directory_path)
        annotations_directory_path = Path(annotations_directory_path)

        classes, images, pascal_annotations = utils.read_pascal_voc_directory(
            images_directory_path, annotations_directory_path
        )
        return cls(classes=classes, images=images, annotations=pascal_annotations)

    @classmethod
    def from_coco(
        cls,
        images_directory_path: typing.Union[str, Path],
        annotations_path: typing.Union[str, Path],
    ) -> DetectionDataset:
        """
        Read directory containing COCO dataset
        and return dictionary of image paths
        and their corresponding annotation path

        :param images_directory_path: Path to directory containing images
        :param annotations_path: Path to file containing annotations
        :return:
        """
        images_directory_path = Path(images_directory_path)
        annotations_path = Path(annotations_path)

        classes, images, coco_annotations = utils.read_coco_directory(
            images_directory_path, annotations_path
        )
        return cls(classes=classes, images=images, annotations=coco_annotations)

    @classmethod
    def from_yolo(
        cls,
        images_directory_path: typing.Union[str, Path],
        annotations_directory_path: typing.Union[str, Path],
        data_yaml_path: typing.Union[str, Path],
    ) -> DetectionDataset:
        """
        Read directory containing Pascal VOC dataset
        and return dictionary of image paths
        and their corresponding annotation path

        :param images_directory_path: Path to directory containing images
        :param annotations_directory_path: Path to directory containing annotations
        :param data_yaml_path: Path to file containing YOLO data configuration
        :return:
        """
        images_directory_path = Path(images_directory_path)
        annotations_directory_path = Path(annotations_directory_path)
        data_yaml_path = Path(data_yaml_path)

        classes, images, yolo_annotations = utils.read_yolo_directory(
            images_directory_path, annotations_directory_path, data_yaml_path
        )
        return cls(classes=classes, images=images, annotations=yolo_annotations)

    @classmethod
    def from_format(
        cls,
        box_format: typings.BoxFormat,
        images_directory_path: typing.Union[str, Path],
        annotations_directory_path: typing.Union[str, Path],
        data_yaml_path: typing.Union[str, Path],
    ) -> DetectionDataset:
        """
        Reads dataset directory with annotations for a given annotations format
        :param box_format:
        :param images_directory_path:
        :param annotations_directory_path:
        :param data_yaml_path:
        :return:
        """
        if box_format == BoxFormat.coco:
            return cls.from_coco(
                images_directory_path, annotations_path=annotations_directory_path
            )

        if box_format == BoxFormat.pascal_voc:
            return cls.from_pascal_voc(
                images_directory_path, annotations_directory_path
            )

        return cls.from_yolo(
            images_directory_path, annotations_directory_path, data_yaml_path
        )


@dataclasses.dataclass
class ClassificationDataset(BaseDataset):
    """
    Dataclass containing information about image classification dataset
    """

    annotations: typing.Dict[str, typing.List[Classification]]

    def __iter__(
        self,
    ) -> typing.Iterator[typing.Tuple[Path, typing.List[Classification]]]:
        """
        Iterate over the images and annotations in the dataset
        :return:
        """
        for image_name, image_path in self.images.items():
            yield image_path, self.annotations.get(image_name, [])

    @classmethod
    def from_folder(
        cls, images_directory_path: typing.Union[str, Path]
    ) -> ClassificationDataset:
        """
        Read directory image classification dataset and return
        dictionary of image paths and the class name.
        :param images_directory_path:
        :return:
        """
        images_directory_path = Path(images_directory_path)

        classes, images, annotations_ = utils.read_image_folder(images_directory_path)
        return cls(classes=classes, images=images, annotations=annotations_)
