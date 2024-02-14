from __future__ import annotations

import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from tqdm import tqdm

from epigos import typings
from epigos.data_classes import project as project_data_class
from epigos.typings import BoxFormat
from epigos.utils import dataset as dataset_utils
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
        batch_name: str = "sdk-upload",
        box_format: BoxFormat = BoxFormat.pascal_voc,
        **kwargs: typing.Any,
    ) -> typing.Dict[str, typing.Any]:
        """
        Upload an image and with or without annotations to the Epigos API.
        :param image_path: Path or directory to images to upload
        :param annotation_path: Path to annotation file to annotate the image
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc`
        :param kwargs: Additional keyword arguments to pass to function
        :return:
        """
        is_file = img_utils.is_path(str(image_path))
        if not is_file:
            raise RuntimeError(f"Provided path does not exist at {image_path}!")

        labels_map = kwargs.get("labels_map")
        yolo_labels_map = kwargs.get("yolo_labels_map")
        batch_id = kwargs.get("batch_id")
        use_folder_as_class_name = kwargs.get("use_folder_as_class_name") or False

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

    def upload_dataset(
        self,
        data_dir: typing.Union[str, Path],
        *,
        batch_name: str = "sdk-upload",
        box_format: BoxFormat = BoxFormat.pascal_voc,
        num_workers: int = 4,
        **kwargs: typing.Any,
    ) -> typing.Iterator[dict[str, Any]]:
        """
        Upload an entire dataset to Epigos API.
        :param data_dir: Path to directory containing images and annotations to upload.
        :param batch_name: name of batch to upload to within project.
        Defaults to `sdk-upload`
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc` and only used for object detection projects
        :param num_workers: Number of cpu workers to use for uploading
        :return:
        """
        if not img_utils.is_path(str(data_dir)):
            raise RuntimeError(f"Provided path does not exist at {data_dir}!")

        data_dir = Path(data_dir)
        batch_id = kwargs.get("batch_id")
        labels_map = kwargs.get("labels_map")

        imgs, yolo_labels_map = self._read_dataset_directory(
            data_dir,
            box_format=box_format,
            annotation_dir_name=kwargs.get("annotation_dir_name") or "labels",
            config_file=kwargs.get("config_file") or "data.yaml",
        )

        if not imgs:
            raise RuntimeError(
                "Could not read any images or annotations in the directory provided"
            )

        if not batch_id:
            batch_id = self._uploader.create_batch(batch_name)

        if not labels_map and yolo_labels_map:
            labels_map = self._uploader.create_labels(list(yolo_labels_map.values()))

        def _upload_file(
            img_path: Path, annotation_path: typing.Union[str, Path]
        ) -> typing.Dict[str, typing.Any]:
            record: typing.Dict[str, typing.Any] = {
                "img_path": img_path,
                "annotation_path": annotation_path,
            }
            try:
                record["response"] = self.upload(
                    image_path=img_path,
                    annotation_path=annotation_path,
                    box_format=box_format,
                    yolo_labels_map=yolo_labels_map,
                    labels_map=labels_map,
                    batch_id=batch_id,
                )
            except httpx.HTTPError:
                logger.exception(
                    "Error occured while uploading file: %s", img_path, exc_info=True
                )
            return record

        with tqdm(total=len(imgs), desc="Uploading datasets", colour="green") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(_upload_file, imgs.keys(), imgs.values()):
                    pbar.update()
                    yield result

    def _read_dataset_directory(
        self,
        data_dir: Path,
        *,
        annotation_dir_name: str,
        config_file: str,
        box_format: BoxFormat,
    ) -> typing.Tuple[
        typing.Union[typing.Dict[Path, Path], typing.Dict[Path, str]],
        typing.Optional[typing.Dict[int, str]],
    ]:
        """
        Reads dataset directory and returns a mapping for image files
        and its corresponding annotations file.
        """
        yolo_labels_map = None
        imgs: typing.Union[typing.Dict[Path, Path], typing.Dict[Path, str]]
        if self.is_object_detection:
            if box_format == typings.BoxFormat.pascal_voc:
                imgs = dataset_utils.read_pascal_voc_directory(
                    data_dir, annotation_dir_name
                )
            else:
                imgs, yolo_labels_map = dataset_utils.read_yolo_directory(
                    data_dir, config_file, annotation_dir_name
                )
        else:
            imgs = dataset_utils.read_image_folder(data_dir)

        return imgs, yolo_labels_map
