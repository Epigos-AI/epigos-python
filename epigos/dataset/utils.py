import collections
import json
import os
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import imagesize
import yaml
from pybboxes import BoundingBox

from epigos.data_classes.dataset import Classification, Detection

IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _extract_bbox_from_element(
    bb: ET.Element,
) -> typing.Optional[typing.Tuple[int, ...]]:
    attrs = ["xmin", "ymin", "xmax", "ymax"]

    bbox_els = (bb.find(attr) for attr in attrs)
    bbox_str = (x.text for x in bbox_els if x is not None)
    bbox = tuple(int(float(v)) for v in bbox_str if v is not None)
    if not len(bbox) == 4 or any((b is None for b in bbox)):
        return None
    return bbox


def _extract_size_from_element(
    size: ET.Element,
) -> typing.Optional[typing.Tuple[int, int]]:
    if size is None or len(size) < 1:
        return None

    attrs = ["width", "height"]
    size_els = (size.find(attr) for attr in attrs)
    size_str = (x.text for x in size_els if x is not None)
    imgs_size = tuple(int(x) for x in size_str if x)

    if not len(imgs_size) > 1:
        return None

    return imgs_size[0], imgs_size[1]


def _read_images_with_extensions(
    directory: typing.Union[str, Path], extensions: typing.List[str]
) -> typing.Dict[str, Path]:
    directory = Path(directory)
    imgs_paths = {p.name: p for ext in extensions for p in directory.glob(f"**/*{ext}")}
    return imgs_paths


def read_pascal_voc_to_coco(
    annotation_file: typing.Union[str, Path]
) -> typing.List[Detection]:
    """
    Reads Pascal VOC [xmin, ymin, xmax, ymax] annotations from file
    and converts it to COCO-style annotations.
    :param annotation_file: Path to annotation file to read.
    :return: List: Contains image metadata and COCO
    bounding boxes (x, y, width, height).
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    size_el = root.find("size")
    image_size = _extract_size_from_element(size_el) if size_el is not None else None

    annotations = []

    for obj in root.iter("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")

        label_name = name_el.text if name_el is not None else None
        bbox = _extract_bbox_from_element(bb) if bb is not None else None
        if not bbox or not label_name:
            continue

        coco_bbox = BoundingBox.from_voc(*bbox, image_size=image_size).to_coco(
            return_values=True
        )

        annotations.append(
            Detection(
                bbox=(
                    int(coco_bbox[0]),
                    int(coco_bbox[1]),
                    int(coco_bbox[2]),
                    int(coco_bbox[3]),
                ),
                class_name=label_name,
            )
        )

    return annotations


def _read_yolo_annotation_file(
    annotation_file: str,
) -> typing.List[typing.Tuple[int, float, float, float, float]]:
    """
    Read YOLO annotation file and parse bounding box annotations.

    Args:
        annotation_file (str or Path): Path to the YOLO annotation file.

    Returns:
        List of tuples: Each tuple contains
        (class_index, x_center, y_center, width, height)
        for each object.
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_index = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        annotations.append((class_index, x_center, y_center, width, height))

    return annotations


def read_yolo_to_coco(
    annotation_file: typing.Union[str, Path],
    image_size: typing.Tuple[int, int],
    idx_to_label: typing.Optional[typing.Dict[int, str]] = None,
) -> typing.List[Detection]:
    """
    Reads YOLO [x1, y1, x2, y2] annotations from file and
    converts it to COCO-style annotations.
    :param annotation_file: Path to annotation file to read.
    :param image_size: Original image size for YOLO annotations
    :param idx_to_label: Class ID to label name mapping for YOLO annotation
    :return: Dict: Contains image metadata and
    COCO bounding boxes (x, y, width, height).
    """
    annotations = []
    yolo_boxes = _read_yolo_annotation_file(str(annotation_file))
    idx_to_label = idx_to_label or {}

    for yolo_box in yolo_boxes:
        label_name = idx_to_label.get(yolo_box[0]) or str(yolo_box[0])
        coco_bbox = BoundingBox.from_yolo(
            *yolo_box[1:], image_size=image_size, strict=False
        ).to_coco(return_values=True)

        annotations.append(
            Detection(
                bbox=(
                    int(coco_bbox[0]),
                    int(coco_bbox[1]),
                    int(coco_bbox[2]),
                    int(coco_bbox[3]),
                ),
                class_name=label_name,
                class_id=yolo_box[0],
            )
        )

    return annotations


def read_yolo_config(file_name: Path) -> typing.Any:
    """Read YOLO configuration file"""
    with open(file_name, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return cfg


def read_coco_file(file_name: Path) -> typing.Any:
    """Read COCO annotations file"""
    with open(file_name, "r", encoding="utf-8") as fp:
        dataset = json.load(fp)

    assert isinstance(
        dataset, dict
    ), f"annotation file format {type(dataset)} not supported"
    return dataset


def read_image_folder(
    root_directory_path: Path,
) -> typing.Tuple[
    typing.List[str],
    typing.Dict[str, Path],
    typing.Dict[str, typing.List[Classification]],
]:
    """
    Read directory image classification dataset and return
    dictionary of image paths and the class name.
    :param root_directory_path:
    :return:
    """
    images = _read_images_with_extensions(
        root_directory_path, extensions=list(IMAGE_FILE_EXTENSIONS)
    )
    annotations = {
        p.name: [Classification(class_name=p.parent.name)] for p in images.values()
    }

    classes = sorted(set(os.listdir(root_directory_path)))

    return classes, images, annotations


def _get_image_size(img_path: Path) -> typing.Tuple[int, int]:
    width, height = imagesize.get(str(img_path))
    return width, height


def read_yolo_directory(
    images_directory_path: Path, annotations_directory_path: Path, data_yaml_path: Path
) -> typing.Tuple[
    typing.List[str], typing.Dict[str, Path], typing.Dict[str, typing.List[Detection]]
]:
    """
    Read directory containing YOLO dataset and return dictionary of image paths
    and their corresponding annotation path.
    :param images_directory_path:
    :param annotations_directory_path:
    :param data_yaml_path:
    :return:
    """
    yolo_config = read_yolo_config(data_yaml_path)
    yolo_names = yolo_config.get("names") or {}

    idx_to_label = {}
    if isinstance(yolo_names, list):
        idx_to_label = dict(enumerate(yolo_names))
    elif isinstance(yolo_names, dict):
        idx_to_label = yolo_names

    images = _read_images_with_extensions(
        images_directory_path, extensions=list(IMAGE_FILE_EXTENSIONS)
    )
    yolo_annotations = {
        p.name: read_yolo_to_coco(
            (annotations_directory_path / f"{p.stem}.txt").resolve(),
            _get_image_size(p),
            idx_to_label,
        )
        for p in images.values()
    }
    classes = list(sorted(list(idx_to_label.values())))
    return classes, images, yolo_annotations


def read_pascal_voc_directory(
    images_directory_path: Path,
    annotations_directory_path: Path,
) -> typing.Tuple[
    typing.List[str], typing.Dict[str, Path], typing.Dict[str, typing.List[Detection]]
]:
    """
    Read directory containing Pascal VOC dataset
    and return dictionary of image paths
    and their corresponding annotation path

    :param images_directory_path: Path to directory containing images
    :param annotations_directory_path: Path to directory containing annotations
    :return:
    """
    images = _read_images_with_extensions(
        images_directory_path, extensions=list(IMAGE_FILE_EXTENSIONS)
    )

    pascal_annotations = {
        p.name: read_pascal_voc_to_coco(
            (annotations_directory_path / f"{p.stem}.xml").resolve()
        )
        for p in images.values()
    }

    classes = {
        d.class_name for detections in pascal_annotations.values() for d in detections
    }
    return list(sorted(classes)), images, pascal_annotations


def read_single_coco_annotation(
    image_name: str, annotations_path: typing.Union[str, Path]
) -> typing.List[Detection]:
    """
    Read a COCO file contaning annotations for a single image.
    :param image_name:
    :param annotations_path:
    :type annotations_path:
    :return:
    :rtype:
    """
    dataset = read_coco_file(Path(annotations_path))
    label_to_idx = {
        label["name"]: label["id"] for label in dataset.get("categories") or []
    }
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    filename_to_img_id = {
        img["file_name"]: img["id"] for img in dataset.get("images") or []
    }

    output: typing.List[Detection] = []

    img_id = filename_to_img_id.get(image_name)
    if img_id is None:
        return output

    for ann in dataset.get("annotations") or []:
        if not ann["bbox"]:
            continue
        if ann["image_id"] != img_id:
            continue

        label_name = idx_to_label[ann["category_id"]]
        output.append(
            Detection(
                bbox=(
                    int(ann["bbox"][0]),
                    int(ann["bbox"][1]),
                    int(ann["bbox"][2]),
                    int(ann["bbox"][3]),
                ),
                class_name=label_name,
                class_id=ann["category_id"],
            )
        )
    return output


def read_coco_directory(
    images_directory_path: Path,
    annotations_path: Path,
) -> typing.Tuple[
    typing.List[str], typing.Dict[str, Path], typing.Dict[str, typing.List[Detection]]
]:
    """
    A COCO dataset helper for reading coco annotations and images in a directory.
    :param images_directory_path:
    :param annotations_path:
    :return:
    """
    dataset = read_coco_file(annotations_path)
    label_to_idx = {
        label["name"]: label["id"] for label in dataset.get("categories") or []
    }
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    imgs = {img["id"]: img for img in dataset.get("images") or []}

    imgs_paths = {
        img_id: (images_directory_path / imgs[img_id]["file_name"]).resolve()
        for img_id in imgs
        if Path(imgs[img_id]["file_name"]).suffix.lower() in IMAGE_FILE_EXTENSIONS
    }
    image_id_to_annotations = collections.defaultdict(list)

    for ann in dataset.get("annotations") or []:
        if not ann["bbox"]:
            continue
        label_name = idx_to_label[ann["category_id"]]
        image_id_to_annotations[ann["image_id"]].append(
            Detection(
                bbox=(
                    int(ann["bbox"][0]),
                    int(ann["bbox"][1]),
                    int(ann["bbox"][2]),
                    int(ann["bbox"][3]),
                ),
                class_name=label_name,
                class_id=ann["category_id"],
            )
        )

    images = {img_path.name: img_path for img_path in imgs_paths.values()}
    annotations = {
        img_path.name: image_id_to_annotations[img_id]
        for img_id, img_path in imgs_paths.items()
    }
    classes = list(sorted(list(label_to_idx.keys())))

    return classes, images, annotations


def resize_bounding_box(
    bbox: typing.Tuple[int, int, int, int],
    img_scale: typing.Tuple[int, int],
    orig_image_size: typing.Tuple[int, int],
) -> typing.Tuple[int, int, int, int]:
    """
    Resize bounding box according to img scale and original image size
    :param bbox:
    :param img_scale:
    :param orig_image_size:
    :return:
    """
    scale_x = img_scale[0] / orig_image_size[0]
    scale_y = img_scale[1] / orig_image_size[1]
    bbox = (
        int(bbox[0] * scale_x),
        int(bbox[1] * scale_y),
        int(bbox[2] * scale_x),
        int(bbox[3] * scale_y),
    )
    return bbox
