import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from pybboxes import BoundingBox

from epigos.utils import logger

FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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
    if not size and not len(size) > 1:
        return None

    attrs = ["width", "height"]
    size_els = (size.find(attr) for attr in attrs)
    size_str = (x.text for x in size_els if x is not None)
    imgs_size = tuple(int(x) for x in size_str if x)

    if not len(imgs_size) > 1:
        return None

    return imgs_size[0], imgs_size[1]


def read_pascal_voc_to_coco(
    annotation_file: str, img_scale: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Dict[str, typing.Any]:
    """
    Reads Pascal VOC [xmin, ymin, xmax, ymax] annotations from file
    and converts it to COCO-style annotations.
    :param annotation_file: Path to annotation file to read.
    :param img_scale: Image size to rescale annotations.
    :return: Dict: Contains image metadata and COCO
    bounding boxes (x, y, width, height).
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    size_el = root.find("size")
    orig_image_size = _extract_size_from_element(size_el) if size_el else None

    annotation: typing.Dict[str, typing.Any] = {
        "boxes": [],
        "labels": set(),
    }

    for obj in root.iter("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")

        label_name = name_el.text if name_el is not None else None
        bbox = _extract_bbox_from_element(bb) if bb is not None else None
        if not bbox or not label_name:
            continue

        coco_bbox = BoundingBox.from_voc(*bbox, image_size=orig_image_size).to_coco(
            return_values=True
        )

        if orig_image_size and img_scale:
            # scale annotations to fit image upload size
            scale_x = img_scale[0] / orig_image_size[0]
            scale_y = img_scale[1] / orig_image_size[1]

            coco_bbox = (
                coco_bbox[0] * scale_x,
                coco_bbox[1] * scale_y,
                coco_bbox[2] * scale_x,
                coco_bbox[3] * scale_y,
            )

        annotation["boxes"].append(
            {
                "label": label_name,
                "x": int(coco_bbox[0]),
                "y": int(coco_bbox[1]),
                "width": int(coco_bbox[2]),
                "height": int(coco_bbox[3]),
            }
        )
        annotation["labels"].add(label_name)

    annotation["labels"] = sorted(list(annotation["labels"]))
    return annotation


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
    annotation_file: str,
    orig_image_size: typing.Tuple[int, int],
    img_scale: typing.Optional[typing.Tuple[int, int]] = None,
    yolo_labels_map: typing.Optional[typing.Dict[int, str]] = None,
) -> typing.Dict[str, typing.Any]:
    """
    Reads YOLO [x1, y1, x2, y2] annotations from file and
    converts it to COCO-style annotations.
    :param annotation_file: Path to annotation file to read.
    :param orig_image_size: Original image size for YOLO annotations
    :param img_scale: Image size to rescale annotations.
    :param yolo_labels_map: Class ID to label name mapping for YOLO annotation
    :return: Dict: Contains image metadata and
    COCO bounding boxes (x, y, width, height).
    """
    annotation: typing.Dict[str, typing.Any] = {
        "boxes": [],
        "labels": set(),
    }
    yolo_boxes = _read_yolo_annotation_file(annotation_file)

    yolo_labels_map = yolo_labels_map or {}

    for bbox in yolo_boxes:
        label_name = yolo_labels_map.get(bbox[0]) or str(bbox[0])
        coco_bbox = BoundingBox.from_yolo(
            *bbox[1:], image_size=orig_image_size, strict=False
        ).to_coco(return_values=True)

        if img_scale:
            # scale annotations to fit image upload size
            scale_x = img_scale[0] / orig_image_size[0]
            scale_y = img_scale[1] / orig_image_size[1]

            coco_bbox = (
                coco_bbox[0] * scale_x,
                coco_bbox[1] * scale_y,
                coco_bbox[2] * scale_x,
                coco_bbox[3] * scale_y,
            )

        annotation["boxes"].append(
            {
                "label": label_name,
                "x": int(coco_bbox[0]),
                "y": int(coco_bbox[1]),
                "width": int(coco_bbox[2]),
                "height": int(coco_bbox[3]),
            }
        )
        annotation["labels"].add(label_name)

    annotation["labels"] = sorted(list(annotation["labels"]))
    return annotation


def read_yolo_config(file_name: Path) -> typing.Any:
    """Read YOLO configuration file"""
    try:
        with open(file_name, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
        return cfg
    except FileNotFoundError:
        logger.error("Could not find config file: %s", file_name)
        return {}


def read_image_folder(data_dir: Path) -> typing.Dict[Path, str]:
    """
    Read directory image classification dataset and return
    dictionary of image paths and the class name.
    :param data_dir:
    :return:
    """
    imgs_paths = (
        p.resolve()
        for p in data_dir.glob("**/*")
        if p.suffix.lower() in FILE_EXTENSIONS
    )
    return {p: p.parent.name for p in imgs_paths}


def read_yolo_directory(
    data_dir: Path, config_file: str, annotations_dir: str = "labels"
) -> typing.Tuple[typing.Dict[Path, Path], typing.Dict[int, str]]:
    """
    Read directory containing YOLO dataset and return dictionary of image paths
    and their corresponding annotation path.
    :param data_dir:
    :param config_file:
    :param annotations_dir:
    :return:
    """
    yolo_config = read_yolo_config(data_dir / config_file)
    yolo_names = yolo_config.get("names") or {}

    labels_map = {}
    if isinstance(yolo_names, list):
        labels_map = dict(enumerate(yolo_names))
    elif isinstance(yolo_names, dict):
        labels_map = yolo_names

    imgs_paths = (
        p.resolve()
        for p in data_dir.glob("**/*")
        if p.suffix.lower() in FILE_EXTENSIONS
    )
    paths = {
        p: (data_dir / annotations_dir / p.parent.name / f"{p.stem}.txt").resolve()
        for p in imgs_paths
    }
    return paths, labels_map


def read_pascal_voc_directory(
    data_dir: Path, annotations_dir: str = "Annotations"
) -> typing.Dict[Path, Path]:
    """
    Read directory containing Pascal VOC dataset
    and return dictionary of image paths
    and their corresponding annotation path.
    :param data_dir:
    :param annotations_dir:
    :return:
    """
    imgs_paths = (
        p.resolve()
        for p in data_dir.glob("**/*")
        if p.suffix.lower() in FILE_EXTENSIONS
    )
    return {
        p: (data_dir / annotations_dir / p.parent.name / f"{p.stem}.xml").resolve()
        for p in imgs_paths
    }
