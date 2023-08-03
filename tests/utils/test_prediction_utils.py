from epigos import typings
from epigos.utils.prediction import Prediction


def test_can_get_classification_json(classification_prediction):
    prediction = Prediction(
        json_predictions=classification_prediction,
        image_path="test.jpg",
        model_type=typings.ModelType.classification,
    )
    actual_json = prediction.json()
    assert classification_prediction == actual_json
