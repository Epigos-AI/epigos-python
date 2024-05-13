import os
from pathlib import Path

import pytest

from epigos.dataset import ClassificationDataset, DetectionDataset


@pytest.mark.parametrize("split", ["train", "val"])
def test_pascal_voc_directory(pascal_voc_directory: Path, split: str) -> None:
    ds = DetectionDataset.from_pascal_voc(
        images_directory_path=pascal_voc_directory / split / "images",
        annotations_directory_path=pascal_voc_directory / split / "labels",
    )

    assert len(ds) == 4
    expected_labels = ["car", "person"]
    assert ds.classes == expected_labels

    assert all(isinstance(path, Path) for path in ds.images.values())
    assert all(path.exists() for path in ds.images.values())

    assert len(ds.images) == len(ds.annotations)


def test_coco_directory(coco_directory):
    ds = DetectionDataset.from_coco(
        images_directory_path=coco_directory,
        annotations_path=coco_directory / "coco.json",
    )

    assert len(ds) == 4
    expected_labels = ["cat", "dog"]
    assert ds.classes == expected_labels

    assert all(isinstance(path, Path) for path in ds.images.values())
    assert all(path.exists() for path in ds.images.values())

    assert len(ds.images) == len(ds.annotations)


@pytest.mark.parametrize("split", ["train", "val"])
def test_read_yolo_directory(yolo_directory, split: str):
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_directory / split / "images",
        annotations_directory_path=yolo_directory / split / "labels",
        data_yaml_path=yolo_directory / "data.yaml",
    )

    assert len(ds) == 5
    expected_labels = ["birds", "cats", "dogs"]
    assert ds.classes == expected_labels

    assert all(isinstance(path, Path) for path in ds.images.values())
    assert all(path.exists() for path in ds.images.values())

    assert len(ds.images) == len(ds.annotations)


@pytest.mark.parametrize("split", ["train", "val"])
def test_read_image_folder(image_dataset_folder, split):
    images_directory_path = image_dataset_folder / split
    ds = ClassificationDataset.from_folder(
        images_directory_path=images_directory_path,
    )

    assert len(ds) == 5
    expected_labels = sorted(set(os.listdir(images_directory_path)))
    assert ds.classes == expected_labels

    assert all(isinstance(path, Path) for path in ds.images.values())
    assert all(path.exists() for path in ds.images.values())

    assert len(ds.images) == len(ds.annotations)
