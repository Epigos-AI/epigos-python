from pathlib import Path

from pybboxes import BoundingBox

from epigos.utils.dataset import (
    read_image_folder,
    read_pascal_voc_directory,
    read_pascal_voc_to_coco,
    read_yolo_config,
    read_yolo_directory,
    read_yolo_to_coco,
)


def test_read_pascal_voc_to_coco(pascal_voc_annotation):
    img_scale = (200, 200)
    coco_annotation = read_pascal_voc_to_coco(str(pascal_voc_annotation), img_scale)

    expected_labels = ["car", "person"]
    assert len(coco_annotation["boxes"]) == 2
    assert coco_annotation["labels"] == expected_labels

    for idx, label in enumerate(expected_labels):
        assert coco_annotation["boxes"][idx]["label"] == label
        assert coco_annotation["boxes"][idx]["x"]
        assert coco_annotation["boxes"][idx]["y"]
        assert coco_annotation["boxes"][idx]["width"]
        assert coco_annotation["boxes"][idx]["height"]


def test_read_pascal_voc_to_coco_no_resize(pascal_voc_annotation):
    img_scale = (500, 375)
    coco_annotation = read_pascal_voc_to_coco(str(pascal_voc_annotation), img_scale)

    expected_labels = ["car", "person"]
    assert len(coco_annotation["boxes"]) == 2
    assert coco_annotation["labels"] == expected_labels

    expected_bboxes = [[179, 85, 231, 144], [112, 145, 135, 175]]
    for idx, (label, voc_bbox) in enumerate(zip(expected_labels, expected_bboxes)):
        bbox = BoundingBox.from_voc(*voc_bbox).to_coco(return_values=True)

        assert coco_annotation["boxes"][idx]["label"] == label
        assert coco_annotation["boxes"][idx]["x"] == bbox[0]
        assert coco_annotation["boxes"][idx]["y"] == bbox[1]
        assert coco_annotation["boxes"][idx]["width"] == bbox[2]
        assert coco_annotation["boxes"][idx]["height"] == bbox[3]


def test_read_yolo_to_coco(yolo_annotation):
    orig_image_size = (100, 100)
    img_scale = (200, 200)
    yolo_labels_map = {0: "car", 1: "person"}
    coco_annotation = read_yolo_to_coco(
        str(yolo_annotation), orig_image_size, img_scale, yolo_labels_map
    )

    expected_labels = ["car", "person"]
    assert len(coco_annotation["boxes"]) == 2
    assert coco_annotation["labels"] == expected_labels

    for idx, label in enumerate(expected_labels):
        assert coco_annotation["boxes"][idx]["label"] == label
        assert coco_annotation["boxes"][idx]["x"]
        assert coco_annotation["boxes"][idx]["y"]
        assert coco_annotation["boxes"][idx]["width"]
        assert coco_annotation["boxes"][idx]["height"]


def test_read_yolo_config(yolo_config_file):
    config = read_yolo_config(yolo_config_file)
    assert isinstance(config, dict)
    assert config["names"] == ["car", "person"]


def test_read_invalid_yolo_config(yolo_config_file):
    config = read_yolo_config(Path("invalid/data.yaml"))
    assert isinstance(config, dict)
    assert config == {}


def test_read_image_folder(image_dataset_folder):
    image_mapping = read_image_folder(image_dataset_folder)
    assert len(image_mapping) == 10

    assert all(isinstance(path, Path) for path in image_mapping.keys())
    assert all(isinstance(label, str) for label in image_mapping.values())
    assert all(path.exists() for path in image_mapping.keys())
    assert all(path.parent.name == label for path, label in image_mapping.items())


def test_read_yolo_directory(yolo_directory):
    config_file = "data.yaml"

    img_paths, labels_map = read_yolo_directory(yolo_directory, config_file)
    assert len(img_paths) == 10
    assert all(isinstance(path, Path) for path in img_paths.keys())
    assert all(path.exists() for path in img_paths.keys())
    assert all(
        isinstance(annotation_path, Path) for annotation_path in img_paths.values()
    )
    assert all(annotation_path.exists() for annotation_path in img_paths.values())
    assert labels_map == {0: "birds", 1: "cats", 2: "dogs"}


def test_pascal_voc_directory(pascal_voc_directory):
    img_paths = read_pascal_voc_directory(pascal_voc_directory)

    assert len(img_paths) == 8
    assert all(isinstance(path, Path) for path in img_paths.keys())
    assert all(path.exists() for path in img_paths.keys())
    assert all(
        isinstance(annotation_path, Path) for annotation_path in img_paths.values()
    )
    assert all(annotation_path.exists() for annotation_path in img_paths.values())
