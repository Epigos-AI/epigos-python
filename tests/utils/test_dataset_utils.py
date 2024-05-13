import pytest

from epigos.dataset.utils import (
    read_pascal_voc_to_coco,
    read_single_coco_annotation,
    read_yolo_config,
    read_yolo_to_coco,
)


def test_read_pascal_voc_to_coco(pascal_voc_annotation):
    detections = read_pascal_voc_to_coco(pascal_voc_annotation)

    expected_labels = ["car", "person"]
    assert len(detections) == 2

    for annot in detections:
        assert annot.class_name in expected_labels
        assert annot.bbox


def test_read_yolo_to_coco(yolo_annotation):
    image_size = (100, 100)
    idx_to_label = {0: "car", 1: "person"}
    detections = read_yolo_to_coco(str(yolo_annotation), image_size, idx_to_label)

    expected_labels = ["car", "person"]
    assert len(detections) == 2

    for annot in detections:
        assert annot.class_name in expected_labels
        assert annot.bbox


def test_read_yolo_config(yolo_config_file):
    config = read_yolo_config(yolo_config_file)
    assert isinstance(config, dict)
    assert config["names"] == ["car", "person"]


@pytest.mark.parametrize("image_name", ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"])
def test_read_single_coco_annotation(coco_directory, image_name):
    annotation_path = coco_directory / "coco.json"
    ds = read_single_coco_annotation(image_name, annotation_path)

    assert len(ds) == 2
