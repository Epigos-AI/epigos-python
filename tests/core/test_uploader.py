import logging
from pathlib import Path
from unittest import mock

import httpx
import pytest
import respx
from PIL import Image

from epigos import Epigos, typings
from epigos.core.uploader import Uploader
from epigos.dataset import utils
from epigos.utils import logger


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_can_create_batch(client: Epigos, respx_mock: respx.MockRouter):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    url = "/projects/project_id/batches/"
    respx_mock.post(url).mock(return_value=httpx.Response(201, json={"id": "test-id"}))
    batch_id = uploader.create_batch("batch")
    assert batch_id == "test-id"


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_can_create_labels(client: Epigos, respx_mock: respx.MockRouter):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    names = ["cats", "dogs"]
    url = "/projects/project_id/annotations/labels/"
    resp_payload = [{"id": f"{idx}", "name": name} for idx, name in enumerate(names)]
    respx_mock.post(url).mock(return_value=httpx.Response(201, json=resp_payload))
    labels_map = uploader.create_labels(names)
    assert len(labels_map) == len(names)
    assert all((name in names for name in labels_map.keys()))


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_invalid_path(client: Epigos, respx_mock: respx.MockRouter):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    with pytest.raises(
        RuntimeError, match=f"Provided path does not exist at invalid.jpg!"
    ):
        uploader.upload(
            batch_id="batch-id",
            image_path=Path("invalid.jpg"),
            use_folder_as_class_name=True,
        )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_invalid_image_format(
    client: Epigos, respx_mock: respx.MockRouter, tmp_path
):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    img_path = tmp_path / "test.bmp"
    img = Image.new("RGB", (200, 200), color="white")
    img.save(img_path, format="BMP")

    with pytest.raises(RuntimeError, match=f"Image format BMP not supported."):
        uploader.upload(
            batch_id="batch-id", image_path=img_path, use_folder_as_class_name=True
        )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_classification_image_use_folder_as_class_name(
    client: Epigos, mock_image, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    mock_upload_api_calls(labels=[mock_image.parent.name])

    rec = uploader.upload(
        batch_id="batch-id", image_path=mock_image, use_folder_as_class_name=True
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_classification_image_with_annotation_path(
    client: Epigos, mock_image, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.classification)

    annotation_path = "test"
    mock_upload_api_calls(labels=[annotation_path])

    rec = uploader.upload(
        batch_id="batch-id", image_path=mock_image, annotation_path=annotation_path
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_pascal_voc_annotations(
    client: Epigos, mock_image: Path, pascal_voc_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=pascal_voc_annotation,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_pascal_voc_annotations_with_folder_annotations(
    client: Epigos, mock_image: Path, pascal_voc_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    annotations = utils.read_pascal_voc_to_coco(annotation_file=pascal_voc_annotation)

    mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotations=annotations,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=False)
def test_upload_pascal_voc_annotations_empty_file(
    client: Epigos, mock_image: Path, pascal_voc_annotation: Path, mock_upload_api_calls
):
    with open(pascal_voc_annotation, "w") as fp:
        fp.write("<annotation></annotation>")

    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    http_mock = mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=pascal_voc_annotation,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == []

    label_call = http_mock.routes[3]
    assert label_call.called is False

    annotation_call = http_mock.routes[4]
    assert annotation_call.called is False


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_yolo_annotations_with_yolo_class_map(
    client: Epigos, mock_image: Path, yolo_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)
    yolo_labels_map = {0: "car", 1: "person"}

    mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=yolo_annotation,
        box_format=typings.BoxFormat.yolo,
        yolo_labels_map=yolo_labels_map,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_yolo_annotations_with_folder_annotations(
    client: Epigos, mock_image: Path, yolo_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    idx_to_label = {0: "car", 1: "person"}

    image_size = utils._get_image_size(mock_image)
    annotations = utils.read_yolo_to_coco(
        annotation_file=yolo_annotation,
        image_size=image_size,
        idx_to_label=idx_to_label,
    )

    mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotations=annotations,
        box_format=typings.BoxFormat.yolo,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=False)
def test_upload_yolo_annotations_with_label_ids_map(
    client: Epigos, mock_image: Path, yolo_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)
    yolo_labels_map = {0: "car", 1: "person"}
    labels_map = {"car": "label-1", "person": "label-2"}

    http_mock = mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=yolo_annotation,
        box_format=typings.BoxFormat.yolo,
        yolo_labels_map=yolo_labels_map,
        labels_map=labels_map,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]

    label_call = http_mock.routes[3]
    assert label_call.called is False


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=False)
def test_upload_yolo_empty_annotations(
    client: Epigos, mock_image: Path, yolo_annotation: Path, mock_upload_api_calls
):
    with open(yolo_annotation, "w") as fp:
        fp.write("\n")

    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)
    yolo_labels_map = {0: "car", 1: "person"}

    http_mock = mock_upload_api_calls(labels=["car", "person"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=yolo_annotation,
        box_format=typings.BoxFormat.yolo,
        yolo_labels_map=yolo_labels_map,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == []

    label_call = http_mock.routes[3]
    assert label_call.called is False

    annotation_call = http_mock.routes[4]
    assert annotation_call.called is False


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=True)
def test_upload_yolo_annotations_with_no_labels_mapping(
    client: Epigos, mock_image: Path, yolo_annotation: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    mock_upload_api_calls(labels=["0", "1"])

    rec = uploader.upload(
        batch_id="batch-id",
        image_path=mock_image,
        annotation_path=yolo_annotation,
        box_format=typings.BoxFormat.yolo,
        yolo_labels_map=None,
        labels_map=None,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.parametrize("image_name", ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"])
@pytest.mark.respx(assert_all_mocked=True, assert_all_called=True)
def test_upload_coco_annotations(
    client: Epigos, image_name: str, coco_directory: Path, mock_upload_api_calls
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    mock_upload_api_calls(labels=["cat", "dog"])

    image_path = coco_directory / image_name
    annotation_path = coco_directory / "coco.json"
    rec = uploader.upload(
        batch_id="batch-id",
        image_path=image_path,
        annotation_path=annotation_path,
        box_format=typings.BoxFormat.coco,
    )
    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=False)
def test_upload_coco_annotations_no_annotations(
    client: Epigos,
    mock_image: Path,
    coco_directory: Path,
    mock_upload_api_calls,
    caplog,
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    mock_upload_api_calls(labels=["cat", "dog"])

    annotation_path = coco_directory / "coco.json"
    with caplog.at_level(logging.WARNING, logger=logger.name):
        rec = uploader.upload(
            batch_id="batch-id",
            image_path=mock_image,
            annotation_path=annotation_path,
            box_format=typings.BoxFormat.coco,
        )
        assert rec["id"] == "record-id"
        assert rec["annotations"] == []

        assert (
            "No annotations available for %s in file: %s!"
            % (mock_image.name, annotation_path)
            in caplog.text
        )


@pytest.mark.respx(assert_all_mocked=True, assert_all_called=False)
def test_upload_invalid_annotation_file(
    client: Epigos,
    mock_image: Path,
    coco_directory: Path,
    mock_upload_api_calls,
    caplog,
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    mock_upload_api_calls(labels=["cat", "dog"])

    annotation_path = coco_directory / "invalid.json"
    with caplog.at_level(logging.WARNING, logger=logger.name):
        rec = uploader.upload(
            batch_id="batch-id",
            image_path=mock_image,
            annotation_path=annotation_path,
            box_format=typings.BoxFormat.coco,
        )
        assert rec["id"] == "record-id"
        assert rec["annotations"] == []

        assert "No annotations file found: %s!" % annotation_path in caplog.text


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_upload_with_image_scaling(
    client: Epigos,
    mock_large_image: Path,
    pascal_voc_annotation: Path,
    mock_upload_api_calls,
):
    uploader = Uploader(client, "project_id", typings.ProjectType.object_detection)

    annotations = utils.read_pascal_voc_to_coco(annotation_file=pascal_voc_annotation)

    mock_upload_api_calls(labels=["car", "person"])

    with mock.patch.object(
        utils, "resize_bounding_box", return_value=(3, 1, 2, 10)
    ) as mock_resize:
        rec = uploader.upload(
            batch_id="batch-id",
            image_path=mock_large_image,
            annotations=annotations,
        )
        mock_resize.assert_called()
        assert mock_resize.call_count == len(annotations)

    assert rec["id"] == "record-id"
    assert rec["annotations"] == [{"id": "annotation-id"}]
