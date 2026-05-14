import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from metrics import evaluate_model

from feature_engineering import (
    load_dataset,
    create_time_features,
    prepare_features
)

# LOAD DATA
df = load_dataset(
    "data/processed/super_master_dataset.csv"
)

df = create_time_features(df)

X_train, X_test, y_train, y_test = prepare_features(df)

# LOAD ANN
ann_model = load_model(
    "models/ann_model.h5"
)

# ANN PREDICTION
ann_pred = ann_model.predict(X_test)

# ANN EVALUATION
ann_results = evaluate_model(
    y_test,
    ann_pred
)

print("\nANN RESULTS")
print(ann_results)

# LSTM PREP
TIME_STEPS = 24

def create_sequences(X, y):

    Xs = []
    ys = []

    for i in range(len(X) - TIME_STEPS):

        Xs.append(X[i:i+TIME_STEPS])

        ys.append(y.iloc[i+TIME_STEPS])

    return np.array(Xs), np.array(ys)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test
)

# LOAD LSTM
lstm_model = load_model(
    "models/lstm_model.h5"
)

# PREDICT
lstm_pred = lstm_model.predict(
    X_test_seq
)

# EVALUATE
lstm_results = evaluate_model(
    y_test_seq,
    lstm_pred
)

print("\nLSTM RESULTS")
print(lstm_results)