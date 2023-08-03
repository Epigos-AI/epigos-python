import pytest

from epigos import typings


@pytest.fixture
def classification_prediction() -> typings.ClassificationPrediction:
    return typings.ClassificationPrediction(
        category="foo",
        confidence=0.7,
        predictions=[
            typings.PredictedClass(
                category="foo",
                confidence=0.7,
            ),
            typings.PredictedClass(
                category="bar",
                confidence=0.3,
            ),
        ],
    )
