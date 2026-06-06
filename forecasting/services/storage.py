from forecasting.models import ForecastResult
from datetime import datetime


def save_prediction(model_type, pred, actual=None):

    error = None

    if actual is not None:
        error = abs(actual - pred)

    ForecastResult.objects.create(
        model_type=model_type,
        input_timestamp=datetime.now(),
        predicted_value=pred,
        actual_value=actual,
        error=error
    )