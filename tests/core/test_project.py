from unittest.mock import MagicMock, call

import httpx
import pytest
import respx

from epigos import Epigos, typings
from epigos.core.uploader import Uploader
from epigos.utils.dataset import (
    read_image_folder,
    read_pascal_voc_directory,
    read_yolo_directory,
)


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
        tuple(project.upload_dataset("invalid/folder"))


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
    mock_project(project_type)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader
    with pytest.raises(
        RuntimeError,
        match="Could not read any images or annotations in the directory provided",
    ):
        tuple(project.upload_dataset(tmp_path, box_format=box_format))

    uploader.create_batch.assert_not_called()
    uploader.create_labels.assert_not_called()
    uploader.upload.assert_not_called()


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_classification_dataset(
    client: Epigos,
    mock_project,
    image_dataset_folder,
):
    mock_project(typings.ProjectType.classification)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    recs = tuple(project.upload_dataset(image_dataset_folder))

    assert len(recs) == 10
    uploader.create_batch.assert_called_once_with("sdk-upload")

    image_mapping = read_image_folder(image_dataset_folder)
    assert uploader.upload.call_count == len(image_mapping)
    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotation_path=annot,
            use_folder_as_class_name=False,
            box_format=typings.BoxFormat.pascal_voc,
            labels_map=None,
            yolo_labels_map=None,
        )
        for img, annot in image_mapping.items()
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_classification_dataset_with_batch_id(
    client: Epigos,
    mock_project,
    image_dataset_folder,
):
    mock_project(typings.ProjectType.classification)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    batch_id = "batch-id"
    recs = tuple(project.upload_dataset(image_dataset_folder, batch_id=batch_id))

    assert len(recs) == 10
    uploader.create_batch.assert_not_called()

    image_mapping = read_image_folder(image_dataset_folder)
    assert uploader.upload.call_count == len(image_mapping)
    calls = [
        call(
            batch_id,
            img,
            annotation_path=annot,
            use_folder_as_class_name=False,
            box_format=typings.BoxFormat.pascal_voc,
            labels_map=None,
            yolo_labels_map=None,
        )
        for img, annot in image_mapping.items()
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_pascal_annotation_dataset(
    client: Epigos,
    mock_project,
    pascal_voc_directory,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    recs = tuple(
        project.upload_dataset(pascal_voc_directory, annotation_dir_name="Annotations")
    )

    assert len(recs) == 8
    uploader.create_batch.assert_called_once_with("sdk-upload")

    image_mapping = read_pascal_voc_directory(pascal_voc_directory)

    assert uploader.upload.call_count == len(image_mapping)
    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotation_path=annot,
            use_folder_as_class_name=False,
            box_format=typings.BoxFormat.pascal_voc,
            labels_map=None,
            yolo_labels_map=None,
        )
        for img, annot in image_mapping.items()
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation_dataset(
    client: Epigos,
    mock_project,
    yolo_directory,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    config_file = "data.yaml"
    recs = tuple(
        project.upload_dataset(
            yolo_directory, box_format=typings.BoxFormat.yolo, config_file=config_file
        )
    )

    assert len(recs) == 10
    uploader.create_batch.assert_called_once_with("sdk-upload")

    img_paths, yolo_labels_map = read_yolo_directory(yolo_directory, config_file)

    uploader.create_labels.assert_called_once_with(list(yolo_labels_map.values()))

    assert uploader.upload.call_count == len(img_paths)
    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotation_path=annot,
            use_folder_as_class_name=False,
            box_format=typings.BoxFormat.yolo,
            labels_map=uploader.create_labels.return_value,
            yolo_labels_map=yolo_labels_map,
        )
        for img, annot in img_paths.items()
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)


@pytest.mark.respx(assert_all_called=True, assert_all_mocked=True)
def test_project_upload_object_detection_yolo_annotation_dataset_with_labels_map(
    client: Epigos,
    mock_project,
    yolo_directory,
):
    mock_project(typings.ProjectType.object_detection)
    project = client.project("project_id")

    uploader = MagicMock(spec=Uploader)
    project._uploader = uploader

    config_file = "data.yaml"
    img_paths, yolo_labels_map = read_yolo_directory(yolo_directory, config_file)
    labels_map = {n: f"label-{n}" for n in yolo_labels_map.values()}

    recs = tuple(
        project.upload_dataset(
            yolo_directory,
            box_format=typings.BoxFormat.yolo,
            config_file=config_file,
            labels_map=labels_map,
        )
    )

    assert len(recs) == 10
    uploader.create_batch.assert_called_once_with("sdk-upload")

    uploader.create_labels.assert_not_called()

    assert uploader.upload.call_count == len(img_paths)
    calls = [
        call(
            uploader.create_batch.return_value,
            img,
            annotation_path=annot,
            use_folder_as_class_name=False,
            box_format=typings.BoxFormat.yolo,
            labels_map=labels_map,
            yolo_labels_map=yolo_labels_map,
        )
        for img, annot in img_paths.items()
    ]
    uploader.upload.assert_has_calls(calls, any_order=True)
