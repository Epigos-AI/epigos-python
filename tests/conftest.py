import json
import tempfile
import typing
from pathlib import Path

import httpx
import pytest
import respx
import yaml
from PIL import Image

from epigos import Epigos


@pytest.fixture
def client():
    return Epigos("api_key", base_url="http://test", retries=0)


@pytest.fixture
def classification_prediction() -> typing.Dict[str, typing.Any]:
    return dict(
        category="foo",
        confidence=0.7,
        predictions=[
            dict(
                category="foo",
                confidence=0.7,
            ),
            dict(
                category="bar",
                confidence=0.3,
            ),
        ],
    )


@pytest.fixture
def object_detection_prediction() -> typing.Dict[str, typing.Any]:
    return dict(
        detections=[
            dict(label="foo", confidence=0.7, x=1, y=2, width=10, height=10),
        ],
    )


@pytest.fixture
def pascal_voc_annotation_content() -> str:
    xml_content = """
            <annotation>
                <folder>path/to/image</folder>
                <filename>000001.jpg</filename>
                <path>000001.jpg</path>
                <size>
                    <width>500</width>
                    <height>375</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>car</name>
                    <bndbox>
                        <xmin>179</xmin>
                        <xmax>231</xmax>
                        <ymin>85</ymin>
                        <ymax>144</ymax>
                    </bndbox>
                </object>
                <object>
                    <name>person</name>
                    <bndbox>
                        <xmin>112</xmin>
                        <xmax>135</xmax>
                        <ymin>145</ymin>
                        <ymax>175</ymax>
                    </bndbox>
                </object>
            </annotation>
        """
    return xml_content


@pytest.fixture
def pascal_voc_annotation(tmp_path, pascal_voc_annotation_content) -> Path:
    temp_file = tmp_path / "test.xml"
    with open(temp_file, "w") as fp:
        fp.write(pascal_voc_annotation_content)
    return temp_file


@pytest.fixture
def yolo_annotation(tmp_path) -> Path:
    annotation_content = (
        "0 0.606687 0.341381 0.544156 0.51\n1 0.206687 0.341381 0.544156 0.51\n"
    )
    temp_file = tmp_path / "test.txt"
    with open(temp_file, "w") as fp:
        fp.write(annotation_content)
    return temp_file


@pytest.fixture
def yolo_config_file(tmp_path) -> Path:
    cfg = {"train": "images/train", "val": "images/val", "names": ["car", "person"]}
    temp_file = tmp_path / "data.yaml"
    with open(temp_file, "w") as fp:
        yaml.safe_dump(cfg, fp)
    return temp_file


@pytest.fixture
def image_dataset_folder():
    class_image_files = {
        "cats": ["cat1.jpg", "cat2.jpg"],
        "dogs": ["dog1.jpg", "dog2.jpg"],
        "birds": ["bird1.jpg"],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        for stage in ["train", "val"]:
            for class_name, image_files in class_image_files.items():
                class_dir = base_dir / stage / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                for image_file in image_files:
                    img_path = class_dir / image_file

                    img = Image.new("RGB", (100, 100))
                    with open(img_path, "w") as fp:
                        img.save(fp)

        yield base_dir


@pytest.fixture
def yolo_directory():
    class_image_files = {
        "cats": ["cat1.jpg", "cat2.jpg"],
        "dogs": ["dog1.jpg", "dog2.jpg"],
        "birds": ["bird1.jpg"],
    }
    labels = {n: idx for idx, n in enumerate(sorted(class_image_files.keys()))}

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        cfg = {
            "train": "images/train",
            "val": "images/val",
            "names": ["birds", "cats", "dogs"],
        }
        with open(base_dir / "data.yaml", "w") as cfp:
            yaml.safe_dump(cfg, cfp)

        for stage in ["train", "val"]:
            for class_name, image_files in class_image_files.items():
                img_dir = base_dir / stage / "images"
                label_dir = base_dir / stage / "labels"

                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)

                for image_file in image_files:
                    img_path = img_dir / image_file
                    img = Image.new("RGB", (100, 100))
                    with open(img_path, "w") as fp:
                        img.save(fp)

                    label_path = label_dir / f"{img_path.stem}.txt"
                    with open(label_path, "w") as fp:
                        fp.write(
                            f"{labels[class_name]} 0.606687 0.341381 0.544156 0.51\n"
                        )

        yield base_dir


@pytest.fixture
def pascal_voc_directory(pascal_voc_annotation_content):
    image_files = ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"]

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        for stage in ["train", "val"]:
            for image_file in image_files:
                img_dir = base_dir / stage / "images"
                label_dir = base_dir / stage / "labels"

                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)

                img_path = img_dir / image_file
                img = Image.new("RGB", (100, 100))
                with open(img_path, "w") as fp:
                    img.save(fp)

                label_path = label_dir / f"{img_path.stem}.xml"
                with open(label_path, "w") as fp:
                    fp.write(pascal_voc_annotation_content)

        yield base_dir


@pytest.fixture
def coco_directory():
    image_files = ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"]
    labels = ["cat", "dog"]

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        categories = [{"id": idx, "name": name} for idx, name in enumerate(labels)]
        images = []
        annotations = []

        annotations_idx = 0
        for idx, image_file in enumerate(image_files):
            img_path = base_dir / image_file
            img = Image.new("RGB", (100, 100))
            with open(img_path, "w") as fp:
                img.save(fp)

            images.append(
                {
                    "id": idx,
                    "file_name": image_file,
                    "height": 275,
                    "width": 490,
                }
            )
            for category in categories:
                ann = {
                    "id": annotations_idx,
                    "image_id": idx,
                    "category_id": category["id"],
                    "bbox": [45, 2, 85, 85],
                }
                annotations.append(ann)
                annotations_idx += 1

        dataset = {
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        with open(base_dir / "coco.json", "w") as fp:
            json.dump(dataset, fp)

        yield base_dir


@pytest.fixture
def mock_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (200, 200), color="white")
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_large_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (2048, 1024), color="white")
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_upload_api_calls(respx_mock: respx.MockRouter):
    def wrapper(labels: list[str]):
        labels_payload = [
            {"id": f"label-{idx}", "name": name} for idx, name in enumerate(labels)
        ]

        respx_mock.post("/projects/project_id/upload/").mock(
            return_value=httpx.Response(
                201, json={"uploadUrl": "http://upload", "uri": "s3://bucket/path"}
            )
        )
        respx_mock.put("http://upload").mock(return_value=httpx.Response(204, json={}))
        respx_mock.post("/projects/project_id/datasets/records/").mock(
            return_value=httpx.Response(201, json={"id": "record-id"})
        )
        respx_mock.post("/projects/project_id/annotations/labels/").mock(
            return_value=httpx.Response(201, json=labels_payload)
        )
        respx_mock.post("/projects/project_id/annotations/").mock(
            return_value=httpx.Response(201, json=[{"id": "annotation-id"}])
        )
        return respx_mock

    return wrapper
