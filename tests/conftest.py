import typing

import pytest


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
