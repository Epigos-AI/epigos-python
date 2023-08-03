from epigos import typings


class Prediction:
    """
    Prediction.

    Generalized object for both Object Detection and Classification Models.

    :param json_predictions: JSON response from the API
    :param image_path: Image path used for the prediction
    :param model_type: Model type
    """

    def __init__(
        self,
        json_predictions: typings.Predictions,
        image_path: str,
        model_type: typings.ModelType,
    ) -> None:
        self._json_predictions = json_predictions
        self._image_path = image_path
        self._model_type = model_type

    def json(self) -> typings.Predictions:
        """
        Gets the json response of the prediction
        :return: typings.Predictions
        """
        return self._json_predictions
