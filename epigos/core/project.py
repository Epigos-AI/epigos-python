from __future__ import annotations

import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from tqdm import tqdm
from typing_extensions import deprecated

from epigos import typings
from epigos.data_classes import project as project_data_class
from epigos.dataset import BaseDataset, ClassificationDataset, DetectionDataset
from epigos.typings import BoxFormat
from epigos.utils import image as img_utils
from epigos.utils import logger

from .uploader import Uploader

if TYPE_CHECKING:
    from epigos.client import Epigos

ACCEPTED_IMAGE_FORMATS = ("JPEG", "PNG")
IMAGE_SIZE = (1024, 640)


class Project:
    """
    Project class represents a Project in the Epigos AI.
    """

    def __init__(self, client: "Epigos", project_id: str):
        self._client = client
        self.project_id = project_id

        project = self.get()
        self.name = project.name
        self.project_type = project.project_type
        self.workspace_id = project.workspace_id
        self._uploader = Uploader(self._client, self.project_id, self.project_type)

    @property
    def is_classification(self) -> bool:
        """
        Returns true of project type is classification
        :return: bool
        """
        return self.project_type == typings.ProjectType.classification

    @property
    def is_object_detection(self) -> bool:
        """
        Returns true of project type is object detection
        :return: bool
        """
        return self.project_type == typings.ProjectType.object_detection

    def get(self) -> project_data_class.Project:
        """
        Returns Project object from Epigos AI
        :return:
        """
        url = f"/projects/{self.project_id}/"
        res = self._client.make_get(path=url)
        return project_data_class.Project(
            id=res["id"],
            name=res["name"],
            workspace_id=res["workspaceId"],
            project_type=res["projectType"],
        )

    def upload(
        self,
        image_path: typing.Union[str, Path],
        *,
        annotation_path: typing.Optional[typing.Union[str, Path]] = None,
        box_format: BoxFormat = BoxFormat.pascal_voc,
        batch_name: str = "sdk-upload",
        batch_id: typing.Optional[str] = None,
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        use_folder_as_class_name: bool = False,
        yolo_labels_map: typing.Optional[typing.Dict[int, str]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Upload an image and with or without annotations to the Epigos API.
        :param image_path: Path or directory to images to upload.
        :param annotation_path: Path to annotation file to annotate the image
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc`.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param batch_id: ID of batch to upload to within project.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param use_folder_as_class_name: Use containing folder of image as class name.
        Only used for classification projects.
        :param yolo_labels_map: Class ID to label name mapping for YOLO annotation.
        :return:
        """
        is_file = img_utils.is_path(str(image_path))
        if not is_file:
            raise RuntimeError(f"Provided path does not exist at {image_path}!")

        if batch_id is None:
            batch_id = self._uploader.create_batch(batch_name)

        record = self._uploader.upload(
            batch_id,
            image_path,
            annotation_path=annotation_path,
            use_folder_as_class_name=use_folder_as_class_name,
            box_format=box_format,
            labels_map=labels_map,
            yolo_labels_map=yolo_labels_map,
        )
        return record

    def upload_classification_dataset(
        self,
        images_directory: typing.Union[str, Path],
        batch_name: str = "sdk-upload",
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload dataset containing image classification dataset.
        Image folder names will be used as class names for the images.
        :param images_directory: Path to folder containing images
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        return self._upload_dataset(
            images_directory,
            batch_name=batch_name,
            num_workers=num_workers,
        )

    def upload_coco_dataset(
        self,
        images_directory: typing.Union[str, Path],
        annotations_path: typing.Union[str, Path],
        batch_name: str = "sdk-upload",
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload dataset containing COCO annotations and images.
        :param images_directory: Path to folder containing images
        :param annotations_path: Path to the singel file containing the COCO annotations
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        return self._upload_dataset(
            images_directory,
            annotations_directory=annotations_path,
            box_format=BoxFormat.coco,
            batch_name=batch_name,
            labels_map=labels_map,
            num_workers=num_workers,
        )

    def upload_pascal_voc_dataset(
        self,
        images_directory: typing.Union[str, Path],
        annotations_directory: typing.Union[str, Path],
        batch_name: str = "sdk-upload",
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload dataset containing PASCAL VOC annotations and images.
        :param images_directory: Path to folder containing images.
        :param annotations_directory: Path to directory containing
        Pascal VOC annotations.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        return self._upload_dataset(
            images_directory,
            annotations_directory=annotations_directory,
            box_format=BoxFormat.pascal_voc,
            batch_name=batch_name,
            labels_map=labels_map,
            num_workers=num_workers,
        )

    def upload_yolo_dataset(
        self,
        images_directory: typing.Union[str, Path],
        annotations_directory: typing.Union[str, Path],
        data_yaml_path: typing.Optional[typing.Union[str, Path]] = None,
        batch_name: str = "sdk-upload",
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload dataset containing YOLO annotations and images.
        :param images_directory: Path to folder containing images
        :param annotations_directory: Path to directory containing YOLO annotations
        :param data_yaml_path: Path to YOLO data configuration file.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        return self._upload_dataset(
            images_directory,
            annotations_directory=annotations_directory,
            data_yaml_path=data_yaml_path,
            box_format=BoxFormat.yolo,
            batch_name=batch_name,
            labels_map=labels_map,
            num_workers=num_workers,
        )

    @deprecated(
        "Use `upload_coco_dataset`, `upload_yolo_dataset`, `upload_pascal_voc_dataset` "
        "or `upload_classification_dataset` instead."
    )
    def upload_dataset(  # pylint: disable=too-many-arguments
        self,
        data_dir: typing.Union[str, Path],
        *,
        annotations_directory: typing.Optional[typing.Union[str, Path]] = None,
        box_format: BoxFormat = BoxFormat.pascal_voc,
        data_yaml_path: typing.Optional[typing.Union[str, Path]] = None,
        batch_name: str = "sdk-upload",
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload an entire dataset to Epigos API.
        :param data_dir: Path to directory containing images and annotations to upload.
        :param annotations_directory: Path to directory containing
        annotations to upload.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc` and only used for object detection projects.
        :param data_yaml_path: Path to YOLO data configuration file.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        return self._upload_dataset(
            data_dir=data_dir,
            annotations_directory=annotations_directory,
            box_format=box_format,
            data_yaml_path=data_yaml_path,
            batch_name=batch_name,
            labels_map=labels_map,
            num_workers=num_workers,
        )

    def _upload_dataset(  # pylint: disable=too-many-arguments
        self,
        data_dir: typing.Union[str, Path],
        *,
        annotations_directory: typing.Optional[typing.Union[str, Path]] = None,
        box_format: BoxFormat = BoxFormat.pascal_voc,
        data_yaml_path: typing.Optional[typing.Union[str, Path]] = None,
        batch_name: str = "sdk-upload",
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        num_workers: int = 4,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload an entire dataset to Epigos API.
        :param data_dir: Path to directory containing images and annotations to upload.
        :param annotations_directory: Path to directory containing
        annotations to upload.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`.
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc` and only used for object detection projects.
        :param data_yaml_path: Path to YOLO data configuration file.
        :param labels_map: Class ID of label in Epigos AI to class name mapping.
        :param num_workers: Number of cpu workers to use for uploading.
        :return:
        """
        if not img_utils.is_path(str(data_dir)):
            raise RuntimeError(f"Provided path does not exist at {data_dir}!")

        data_dir = Path(data_dir)

        ds = self._read_dataset_directory(
            data_dir,
            box_format=box_format,
            annotations_directory_path=Path(
                annotations_directory or data_dir / "labels"
            ),
            data_yaml_path=Path(data_yaml_path or data_dir / "data.yaml"),
        )
        num_images = len(ds)

        if not num_images > 0:
            raise RuntimeError(
                "Could not read any images or annotations in the directory provided"
            )

        batch_id = self._uploader.create_batch(batch_name)

        if not labels_map and ds.classes:
            labels_map = self._uploader.create_labels(ds.classes)

        def _upload_file(
            img_path: Path, img_annotations: typing.List[typing.Any]
        ) -> typing.Dict[str, typing.Any]:

            record: typing.Dict[str, typing.Any] = {
                "img_path": img_path,
            }
            try:
                resp = self._uploader.upload(
                    batch_id,
                    img_path,
                    annotations=img_annotations,
                    box_format=box_format,
                    labels_map=labels_map,
                    label_names=ds.classes,
                )
                record["response"] = resp
            except httpx.HTTPError:
                logger.exception(
                    "Error occured while uploading file: %s", img_path, exc_info=True
                )
            return record

        with tqdm(total=num_images, desc="Uploading datasets", colour="green") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(
                    lambda p: _upload_file(*p),
                    ds,
                ):
                    pbar.update()
                    yield result

    def _read_dataset_directory(
        self,
        data_dir: Path,
        *,
        annotations_directory_path: Path,
        data_yaml_path: Path,
        box_format: BoxFormat,
    ) -> BaseDataset:
        """
        Reads dataset directory and returns a mapping for image files
        and its corresponding annotations file.
        """
        if self.is_object_detection:
            return DetectionDataset.from_format(
                box_format=box_format,
                images_directory_path=data_dir,
                annotations_directory_path=annotations_directory_path,
                data_yaml_path=data_yaml_path,
            )

        return ClassificationDataset.from_folder(data_dir)
