import io
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from PIL import Image, ImageOps

from epigos import typings
from epigos.typings import BoxFormat
from epigos.utils import image as img_utils
from epigos.utils import logger
from epigos.utils.dataset import read_pascal_voc_to_coco, read_yolo_to_coco

if TYPE_CHECKING:
    from epigos.client import Epigos

ACCEPTED_IMAGE_FORMATS = ("JPEG", "PNG")
DEFAULT_IMAGE_SIZE = (1024, 1024)


class Uploader:
    """
    Uploader is responsible for uploading images and annotations to Epigos AI
    """

    def __init__(
        self, client: "Epigos", project_id: str, project_type: typings.ProjectType
    ) -> None:
        self._client = client
        self._project_id = project_id
        self._project_type = project_type

    def upload(  # pylint: disable=too-many-arguments
        self,
        batch_id: str,
        image_path: typing.Union[str, Path],
        annotation_path: typing.Optional[typing.Union[str, Path]] = None,
        use_folder_as_class_name: bool = False,
        box_format: BoxFormat = BoxFormat.pascal_voc,
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
        yolo_labels_map: typing.Optional[typing.Dict[int, str]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Upload an image and with or without annotation to the Epigos API.
        :param image_path: Path or directory to images to upload
        :param annotation_path: Path to annotation file to annotate the image
        :param batch_id: ID of batch to upload to within project.
        :param use_folder_as_class_name: Use containing folder of image as class name.
        Only used for classification projects.
        :param box_format: Format of annotation to upload.
        Defaults to `pascal_voc`
        :param labels_map: Class ID to class name mapping for YOLO annotation.
        :param yolo_labels_map: Class ID to label name mapping for YOLO annotation
        :return:
        """

        image_path = Path(image_path)

        if not img_utils.is_path(str(image_path)):
            raise RuntimeError(f"Provided path does not exist at {image_path}!")

        if self._project_type == typings.ProjectType.classification:
            annotation_path = (
                image_path.parent.name if use_folder_as_class_name else annotation_path
            )

        with Image.open(image_path) as img:
            orig_image_size = img.size

            if img.format not in ACCEPTED_IMAGE_FORMATS:
                raise RuntimeError(f"Image format {img.format} not supported.")

            if img.width > DEFAULT_IMAGE_SIZE[0] or img.height > DEFAULT_IMAGE_SIZE[1]:
                img = ImageOps.contain(img, DEFAULT_IMAGE_SIZE)

            resize_img_size = img.size
            annotations = self._read_annotations(
                annotation_path=str(annotation_path),
                orig_image_size=orig_image_size,
                resize_img_size=resize_img_size,
                box_format=box_format,
                yolo_labels_map=yolo_labels_map,
            )
            record = self._upload_image(
                img,
                image_path,
                batch_id=batch_id,
            )

        created_annotations = None
        if annotations:
            created_annotations = self._create_annotation(
                record_id=record["id"],
                annotations=annotations,
                labels_map=labels_map,
            )

        record["annotations"] = created_annotations or []
        return record

    def create_batch(self, batch_name: str) -> str:
        """
        Create a batch for uploading images to Epigos AI
        :param batch_name:
        :return: batch ID
        """
        batch = self._client.make_post(
            path=f"/projects/{self._project_id}/batches/",
            json={"name": batch_name},
        )
        return str(batch["id"])

    def create_labels(self, names: typing.List[str]) -> typing.Dict[str, str]:
        """
        Creates annotation labels for the given names
        :param names: Label names to create
        :return:
        """
        labels_payload = [{"name": name} for name in names]

        labels = self._client.make_post(
            path=f"/projects/{self._project_id}/annotations/labels/",
            json=labels_payload,
        )
        return {label["name"]: label["id"] for label in labels}

    def _upload_image(
        self,
        img: Image.Image,
        image_path: Path,
        batch_id: str,
    ) -> typing.Dict[str, typing.Any]:
        content_type = img.get_format_mimetype()  # type: ignore

        presigned = self._client.make_post(
            path="/upload/",
            json={"name": image_path.name, "content_type": content_type},
        )

        with io.BytesIO() as fp:
            img.save(fp, format=img.format)
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

        record_payload = {
            "name": image_path.name,
            "batchId": batch_id,
            "height": img.height,
            "width": img.width,
            "contentType": content_type,
            "size": img_size,
            "source": presigned["uri"],
        }
        record = self._client.make_post(
            path=f"/projects/{self._project_id}/datasets/records/",
            json=record_payload,
        )
        return dict(record)

    def _create_annotation(
        self,
        *,
        record_id: str,
        annotations: typing.Dict[str, typing.Any],
        labels_map: typing.Optional[typing.Dict[str, str]] = None,
    ) -> typing.Optional[typing.List[typing.Any]]:

        payload: dict[str, typing.Any] = {
            "dataset_record_id": record_id,
            "annotations": annotations["annotations"],
        }

        label_names = annotations["labels"]

        if not labels_map:
            labels_map = self.create_labels(label_names)

        for annotation in payload["annotations"]:
            annotation["label_id"] = labels_map[annotation["label_id"]]

        annotation_resp = self._client.make_post(
            path=f"/projects/{self._project_id}/annotations/",
            json=payload,
        )
        return list(annotation_resp)

    def _read_annotations(
        self,
        annotation_path: str,
        orig_image_size: typing.Tuple[int, int],
        resize_img_size: typing.Tuple[int, int],
        **kwargs: typing.Any,
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        box_format = kwargs.get("box_format") or typings.BoxFormat.pascal_voc
        yolo_labels_map: typing.Optional[typing.Dict[int, str]] = kwargs.get(
            "yolo_labels_map"
        )
        scale_boxes = orig_image_size != resize_img_size

        img_wdith, img_height = resize_img_size
        metadata = {"image": {"width": img_wdith, "height": img_height}}

        if self._project_type == typings.ProjectType.classification:
            # assumes annotation_path is the class name for the image
            label_names = [annotation_path]
            annotations = [
                {
                    "annotation": {
                        "category": typings.AnnotationCategory.category,
                        "metadata": metadata,
                    },
                    "label_id": annotation_path,
                }
            ]
        else:
            if not img_utils.is_path(annotation_path):
                logger.warning("No annotations file found: %s!", annotation_path)
                return None

            img_scale = (img_wdith, img_height) if scale_boxes else None
            if box_format == typings.BoxFormat.yolo:
                raw_annotations = read_yolo_to_coco(
                    annotation_path,
                    orig_image_size=orig_image_size,
                    img_scale=img_scale,
                    yolo_labels_map=yolo_labels_map,
                )
            else:
                raw_annotations = read_pascal_voc_to_coco(
                    annotation_path,
                    img_scale=img_scale,
                )

            label_names = list(raw_annotations["labels"])
            annotations = [
                {
                    "annotation": {
                        "category": typings.AnnotationCategory.bounding_box,
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

        if not annotations or not label_names:
            logger.warning("No annotations found in file: %s!", annotation_path)
            return None

        return {"labels": label_names, "annotations": annotations}
