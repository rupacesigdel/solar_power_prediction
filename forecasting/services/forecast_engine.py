import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from .preprocessing import create_features, scale_data, create_sequences


# Load models once (important for Django performance)
ann_model = load_model("models/ann_model.keras")
lstm_model = load_model("models/lstm_model.keras")

scaler = joblib.load("models/scaler.pkl")


# ---------------------------
# 1. Load models
# ---------------------------
def get_model(model_type="lstm"):
    if model_type == "ann":
        return ann_model
    return lstm_model


# ---------------------------
# 2. Predict on test data
# ---------------------------
def predict(model_type, X_test, sequence=False):

    model = get_model(model_type)

    if sequence:
        X_test = create_sequences(X_test)

    y_pred = model.predict(X_test)

    return y_pred


# ---------------------------
# 3. Forecast next N steps
# ---------------------------
def forecast_future(model_type, last_sequence, steps=24):

    model = get_model(model_type)

    predictions = []

    current_input = last_sequence.copy()

    for _ in range(steps):

        pred = model.predict(current_input[np.newaxis, ...])[0][0]
        predictions.append(pred)

        # shift window (LSTM style)
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = pred

    return np.array(predictions)


# ---------------------------
# 4. Compare models
# ---------------------------
def compare_models(X_test_lstm, X_test_ann):

    lstm_pred = predict("lstm", X_test_lstm, sequence=True)
    ann_pred = predict("ann", X_test_ann, sequence=False)

    return {
        "lstm": lstm_pred,
        "ann": ann_pred
    }