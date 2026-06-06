import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load once
lstm_model = load_model("models/lstm_model.h5", compile=False)
ann_model = load_model("models/ann_model.h5", compile=False)
scaler = joblib.load("models/feature_scaler.pkl")


# -------------------------
# ANN Prediction
# -------------------------
from .storage import save_prediction

def predict_ann(features, actual=None):

    scaled = scaler.transform([features])

    pred = ann_model.predict(scaled, verbose=0)

    value = float(pred[0][0])

    save_prediction("ANN", value, actual)

    return value


# -------------------------
# LSTM Prediction (FIXED)
# -------------------------
def predict_lstm(sequence, actual=None):

    scaled = scaler.transform(sequence)

    scaled = scaled.reshape(1, scaled.shape[0], scaled.shape[1])

    pred = lstm_model.predict(scaled, verbose=0)

    value = float(pred[0][0])

    save_prediction("LSTM", value, actual)

    return value

def predict_energy(model_type, input_data):

    if model_type == "lstm":
        return predict_lstm(input_data)

    return predict_ann(input_data)