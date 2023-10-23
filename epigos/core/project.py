from __future__ import annotations

import io
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from PIL import Image, ImageOps

from epigos import typings
from epigos.data_classes import project as project_data_class
from epigos.utils import image as img_utils
from epigos.utils.annotations import read_pascal_voc_to_coco

if TYPE_CHECKING:
    from epigos.client import Epigos

ACCEPTED_IMAGE_FORMATS = ("JPEG", "PNG")
IMAGE_SIZE = (1024, 640)


class Project:
    def __init__(self, client: "Epigos", project_id: str):
        self._client = client
        self.project_id = project_id
        project = self.get()
        self.name = project.name
        self.project_type = project.project_type
        self.workspace_id = project.workspace_id

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
        url = f"/projects/{self.project_id}/"
        res = self._client.call_api(path=url, method="get")
        return project_data_class.Project(
            id=res["id"],
            name=res["name"],
            workspace_id=res["workspaceId"],
            project_type=res["projectType"],
        )

    def upload(
        self,
        image_path: typing.Union[str, Path],
        annotation_path: typing.Optional[typing.Union[str, Path]] = None,
        batch_name: typing.Optional[str] = "sdk-upload",
        use_folder_as_class_name: bool = False,
    ) -> typing.Dict[str, typing.Any]:
        """
        Upload an image and with or without annotation to the Epigos API.
        :param image_path: Path or directory to images to upload
        :param annotation_path: Path to annotation file to annotate the image
        :param batch_name: name of batch to upload to within project. Defaults to `sdk-upload`
        :param use_folder_as_class_name: Use containing folder of image as class name.
            Only used for classification projects.
        :return:
        """
        is_file = img_utils.is_path(str(image_path))
        if not is_file:
            raise RuntimeError(f"Provided path does not exist at {image_path}!")

        image_path = Path(image_path)
        img = Image.open(image_path)
        img_format = img.format

        if img_format not in ACCEPTED_IMAGE_FORMATS:
            raise RuntimeError(f"Image format {img.format} not supported.")

        content_type = img.get_format_mimetype()  # type: ignore

        scale_boxes = False
        if img.width > IMAGE_SIZE[0] or img.height > IMAGE_SIZE[1]:
            img = ImageOps.contain(img, IMAGE_SIZE)
            scale_boxes = True

        record = self._upload_image(
            img,
            image_path,
            img_format=img_format,
            content_type=content_type,
            batch_name=batch_name,
        )

        if self.is_classification:
            annotation_path = (
                image_path.parent.name if use_folder_as_class_name else annotation_path
            )

        if annotation_path:
            annotations_resp = self._create_annotation(
                img, record["id"], str(annotation_path), scale_boxes=scale_boxes
            )
            if annotations_resp:
                record["annotations"] = annotations_resp

        img.close()
        return record

    def _upload_image(
        self,
        img: Image.Image,
        image_path: Path,
        img_format: str,
        content_type: str,
        batch_name: typing.Optional[str] = None,
    ) -> typing.Dict[str, typing.Any]:
        presigned = self._client.call_api(
            path="/upload/",
            method="post",
            json={"name": image_path.name, "content_type": content_type},
        )

        with io.BytesIO() as fp:
            img.save(fp, format=img_format)
            content = fp.getvalue()
            img_size = len(content)

            upload_response = httpx.put(
                presigned["uploadUrl"],
                content=content,
                headers={
                    "Content-Type": content_type,
                },
            )
            upload_response.raise_for_status()

        batch = self._client.call_api(
            path=f"/projects/{self.project_id}/batches/",
            method="post",
            json={"name": batch_name},
        )
        record_payload = {
            "name": image_path.name,
            "batchId": batch["id"],
            "height": img.height,
            "width": img.width,
            "contentType": content_type,
            "size": img_size,
            "source": presigned["uri"],
        }
        record = self._client.call_api(
            path=f"/projects/{self.project_id}/datasets/records/",
            method="post",
            json=record_payload,
        )
        return dict(record)

    def _create_annotation(
        self,
        img: Image.Image,
        record_id: str,
        annotation_path: str,
        scale_boxes: bool = False,
    ) -> typing.Optional[typing.List[typing.Any]]:
        payload: dict[str, typing.Any] = {
            "dataset_record_id": record_id,
            "annotations": [],
        }
        metadata = {"image": {"width": img.width, "height": img.height}}

        if self.project_type == typings.ProjectType.classification:
            # assumes annotation_path is the class name for the image
            labels_payload = [{"name": annotation_path}]
            payload["annotations"] = [
                {
                    "annotation": {
                        "category": typings.AnnotationType.category,
                        "metadata": metadata,
                    },
                    "label_id": annotation_path,
                }
            ]
        else:
            if not img_utils.is_path(annotation_path):
                print(f"No annotations file found: {annotation_path}!")
                return None

            raw_annotations = read_pascal_voc_to_coco(
                annotation_path,
                img_scale=(img.width, img.height) if scale_boxes else None,
            )
            labels_payload = [{"name": label} for label in raw_annotations["labels"]]
            payload["annotations"] = [
                {
                    "annotation": {
                        "category": typings.AnnotationType.bounding_box,
                        "left": box["x"],
                        "top": box["y"],
                        "width": box["width"],
                        "height": box["height"],
                        "metadata": metadata,
                    },
                    "label_id": box["label"],
                }
                for box in raw_annotations["boxes"]
            ]

        labels = self._client.call_api(
            path=f"/projects/{self.project_id}/annotations/labels/",
            method="post",
            json=labels_payload,
        )
        labels_id_map = {label["name"]: label["id"] for label in labels}
        for annotation in payload["annotations"]:
            annotation["label_id"] = labels_id_map[annotation["label_id"]]

        annotation_resp = self._client.call_api(
            path=f"/projects/{self.project_id}/annotations/",
            method="post",
            json=payload,
        )
        return list(annotation_resp)
