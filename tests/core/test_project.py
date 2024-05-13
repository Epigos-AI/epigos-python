from unittest.mock import MagicMock, call

import httpx
import pytest
import respx
import yaml

from epigos import Epigos, typings
from epigos.core.uploader import Uploader
from epigos.dataset import ClassificationDataset, DetectionDataset


@pytest.fixture
def mock_project(respx_mock: respx.MockRouter):
    def _mocker(project_type: typings.ProjectType):
        respx_mock.get("/projects/project_id/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "test-id",
                    "name": "test-name",
                    "workspaceId": "workspace-id",
                    "projectType": project_type,
                },
            )
        )
        return respx_mock

    return _mocker


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
@pytest.mark.parametrize(
    "project_type",
    [typings.ProjectType.classification, typings.ProjectType.object_detection],
)
def test_get_project(client: Epigos, mock_project, project_type: typings.ProjectType):
    mock_project(project_type)
    project = client.project("project_id")

    assert project.project_id == "project_id"
    assert project.project_type == project_type
    assert project.name == "test-name"
    assert project.workspace_id == "workspace-id"
    if project_type == typings.ProjectType.object_detection:
        assert project.is_object_detection is True
        assert project.is_classification is False
    else:
        assert project.is_classification is True
        assert project.is_object_detection is False


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
@pytest.mark.parametrize(
    "project_type",
    [typings.ProjectType.classification, typings.ProjectType.object_detection],
)
def test_project_upload_invalid_file(
    client: Epigos,
    mock_project,
    project_type: typings.ProjectType,
):
    mock_project(project_type)
    project = client.project("project_id")

    project._uploader = MagicMock(spec=Uploader)
    with pytest.raises(
        RuntimeError, match="Provided path does not exist at invalid.jpg!"
    ):
        project.upload("invalid.jpg")


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_classification(
    client: Epigos,
    mock_project,
    mock_image,
):
    mock_project(typings.ProjectType.classification)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    project.upload(mock_image, use_folder_as_class_name=True)

    uploader.create_batch.assert_called_once_with("sdk-upload")
    uploader.upload.assert_called_once_with(
        uploader.create_batch.return_value,
        mock_image,
        annotation_path=None,
        use_folder_as_class_name=True,
        box_format=typings.BoxFormat.pascal_voc,
        labels_map=None,
        yolo_labels_map=None,
    )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_classification_with_batch_id(
    client: Epigos,
    mock_project,
    mock_image,
):
    mock_project(typings.ProjectType.classification)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    batch_id = "batch_id"
    project.upload(mock_image, batch_id=batch_id, use_folder_as_class_name=True)

    uploader.create_batch.assert_not_called()
    uploader.upload.assert_called_once_with(
        batch_id,
        mock_image,
        annotation_path=None,
        use_folder_as_class_name=True,
        box_format=typings.BoxFormat.pascal_voc,
        labels_map=None,
        yolo_labels_map=None,
    )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_pascal_annotation(
    client: Epigos,
    mock_project,
    mock_image,
    pascal_voc_annotation,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    project.upload(mock_image, annotation_path=pascal_voc_annotation)

    uploader.create_batch.assert_called_once_with("sdk-upload")
    uploader.upload.assert_called_once_with(
        uploader.create_batch.return_value,
        mock_image,
        annotation_path=pascal_voc_annotation,
        use_folder_as_class_name=False,
        box_format=typings.BoxFormat.pascal_voc,
        labels_map=None,
        yolo_labels_map=None,
    )


@pytest.mark.parametrize("image_name", ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"])
@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_coco_annotation(
    client: Epigos, mock_project, coco_directory, image_name
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    image_path = coco_directory / image_name
    annotation_path = coco_directory / "coco.json"
    project.upload(
        image_path, annotation_path=annotation_path, box_format=typings.BoxFormat.coco
    )

    uploader.create_batch.assert_called_once_with("sdk-upload")
    uploader.upload.assert_called_once_with(
        uploader.create_batch.return_value,
        image_path,
        annotation_path=annotation_path,
        use_folder_as_class_name=False,
        box_format=typings.BoxFormat.coco,
        labels_map=None,
        yolo_labels_map=None,
    )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation(
    client: Epigos,
    mock_project,
    mock_image,
    yolo_annotation,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader
    yolo_labels_map = {0: "car", 1: "person"}

    project.upload(
        mock_image, annotation_path=yolo_annotation, yolo_labels_map=yolo_labels_map
    )

    uploader.create_batch.assert_called_once_with("sdk-upload")
    uploader.upload.assert_called_once_with(
        uploader.create_batch.return_value,
        mock_image,
        annotation_path=yolo_annotation,
        use_folder_as_class_name=False,
        box_format=typings.BoxFormat.pascal_voc,
        labels_map=None,
        yolo_labels_map=yolo_labels_map,
    )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation_with_labels_map(
    client: Epigos,
    mock_project,
    mock_image,
    yolo_annotation,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader
    yolo_labels_map = {0: "car", 1: "person"}
    labels_map = {"car": "label-1", "person": "label-2"}

    project.upload(
        mock_image,
        annotation_path=yolo_annotation,
        yolo_labels_map=yolo_labels_map,
        labels_map=labels_map,
    )

    uploader.create_batch.assert_called_once_with("sdk-upload")
    uploader.upload.assert_called_once_with(
        uploader.create_batch.return_value,
        mock_image,
        annotation_path=yolo_annotation,
        use_folder_as_class_name=False,
        box_format=typings.BoxFormat.pascal_voc,
        labels_map=labels_map,
        yolo_labels_map=yolo_labels_map,
    )


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
@pytest.mark.parametrize(
    "project_type",
    [typings.ProjectType.classification, typings.ProjectType.object_detection],
)
def test_project_upload_dataset_invalid_directory(
    client: Epigos,
    mock_project,
    project_type: typings.ProjectType,
):
    mock_project(project_type)
    project = client.project("project_id")

    project._uploader = MagicMock(spec=Uploader)
    with pytest.raises(
        RuntimeError, match="Provided path does not exist at invalid/folder!"
    ):
        tuple(project._upload_dataset("invalid/folder"))


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
@pytest.mark.parametrize(
    "project_type",
    [typings.ProjectType.classification, typings.ProjectType.object_detection],
)
@pytest.mark.parametrize(
    "box_format",
    [typings.BoxFormat.pascal_voc, typings.BoxFormat.yolo],
)
def test_project_upload_dataset_empty_directory(
    client: Epigos,
    mock_project,
    project_type: typings.ProjectType,
    box_format: typings.BoxFormat,
    tmp_path,
):
    with open(tmp_path / "data.yaml", "w") as fp:
        yaml.safe_dump({}, fp)

    mock_project(project_type)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader
    with pytest.raises(
        RuntimeError,
        match="Could not read any images or annotations in the directory provided",
    ):
        tuple(project._upload_dataset(tmp_path, box_format=box_format))

    uploader.create_batch.assert_not_called()
    uploader.create_labels.assert_not_called()
    uploader.upload.assert_not_called()


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_classification_dataset(
    client: Epigos, mock_project, image_dataset_folder, split
):
    mock_project(typings.ProjectType.classification)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    data_dir = image_dataset_folder / split
    recs = tuple(project.upload_classification_dataset(data_dir))

    assert len(recs) == 5
    uploader.create_batch.assert_called_once_with("sdk-upload")

    ds = ClassificationDataset.from_folder(data_dir)

    uploader.create_labels.assert_called_once_with(ds.classes)

    assert uploader.upload.call_count == len(ds)

    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotations=annots,
            box_format=typings.BoxFormat.pascal_voc,
            labels_map=uploader.create_labels.return_value,
            label_names=ds.classes,
        )
        for img, annots in ds
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_pascal_annotation_dataset(
    client: Epigos, mock_project, pascal_voc_directory, split
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    images_dir = pascal_voc_directory / split / "images"
    annotations_directory_path = pascal_voc_directory / split / "labels"

    recs = tuple(
        project.upload_pascal_voc_dataset(
            images_dir, annotations_directory=annotations_directory_path
        )
    )

    assert len(recs) == 4
    uploader.create_batch.assert_called_once_with("sdk-upload")

    ds = DetectionDataset.from_pascal_voc(
        images_directory_path=images_dir,
        annotations_directory_path=annotations_directory_path,
    )
    uploader.create_labels.assert_called_once_with(ds.classes)

    assert uploader.upload.call_count == len(ds)
    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotations=annot,
            box_format=typings.BoxFormat.pascal_voc,
            labels_map=uploader.create_labels.return_value,
            label_names=ds.classes,
        )
        for img, annot in ds
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation_dataset(
    client: Epigos, mock_project, yolo_directory, split
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    data_yaml_path = yolo_directory / "data.yaml"
    images_dir = yolo_directory / split / "images"
    annotations_directory_path = yolo_directory / split / "labels"

    recs = tuple(
        project.upload_yolo_dataset(
            images_dir,
            annotations_directory=annotations_directory_path,
            data_yaml_path=data_yaml_path,
        )
    )

    assert len(recs) == 5
    uploader.create_batch.assert_called_once_with("sdk-upload")

    ds = DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=annotations_directory_path,
        data_yaml_path=data_yaml_path,
    )
    uploader.create_labels.assert_called_once_with(ds.classes)

    assert uploader.upload.call_count == len(ds)

    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotations=annot,
            box_format=typings.BoxFormat.yolo,
            labels_map=uploader.create_labels.return_value,
            label_names=ds.classes,
        )
        for img, annot in ds
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation_dataset_with_labels_map(
    client: Epigos, mock_project, yolo_directory, split
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    data_yaml_path = yolo_directory / "data.yaml"
    images_dir = yolo_directory / split / "images"
    annotations_directory_path = yolo_directory / split / "labels"

    ds = DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=annotations_directory_path,
        data_yaml_path=data_yaml_path,
    )
    labels_map = {n: f"label-{n}" for n in ds.classes}
    recs = tuple(
        project.upload_yolo_dataset(
            images_dir,
            annotations_directory=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            labels_map=labels_map,
        )
    )

    assert len(recs) == 5
    uploader.create_batch.assert_called_once_with("sdk-upload")

    uploader.create_labels.assert_not_called()

    assert uploader.upload.call_count == len(ds)

    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotations=annot,
            box_format=typings.BoxFormat.yolo,
            labels_map=labels_map,
            label_names=ds.classes,
        )
        for img, annot in ds
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_coco_annotation_dataset(
    client: Epigos, mock_project, coco_directory
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    annotations_path = coco_directory / "coco.json"

    recs = tuple(
        project.upload_coco_dataset(
            coco_directory,
            annotations_path=annotations_path,
        )
    )

    assert len(recs) == 4
    uploader.create_batch.assert_called_once_with("sdk-upload")

    ds = DetectionDataset.from_coco(
        images_directory_path=coco_directory,
        annotations_path=annotations_path,
    )
    uploader.create_labels.assert_called_once_with(ds.classes)

    assert uploader.upload.call_count == len(ds)

    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotations=annot,
            box_format=typings.BoxFormat.coco,
            labels_map=uploader.create_labels.return_value,
            label_names=ds.classes,
        )
        for img, annot in ds
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)
