from epigos.data_classes.prediction import Classification, ObjectDetection


def test_can_get_classification_dict(classification_prediction):
    prediction = Classification(
        category=classification_prediction["category"],
        confidence=classification_prediction["confidence"],
        predictions=classification_prediction["predictions"],
    )
    expected_dict = prediction.dict()
    assert classification_prediction == expected_dict


def test_can_get_object_detection_dict(object_detection_prediction):
    prediction = ObjectDetection(detections=object_detection_prediction["detections"])
    expected_dict = prediction.dict()
    object_detection_prediction["base64_image"] = None
    assert object_detection_prediction == expected_dict
