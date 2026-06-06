import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURES = [
    "avg_air_temp",
    "avg_global_rad",
    "avg_cell_rad",
    "avg_surface_temp",
    "avg_wind_speed",
    "irradiance_avg",
    "hour_sin",
    "hour_cos",
    "day",
    "month"
]

TARGET = "ITS_Energy (kWh)"


# ---------------------------
# Feature engineering
# ---------------------------
def create_features(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df.dropna(inplace=True)

    return df


# ---------------------------
# Scaling
# ---------------------------
def scale_data(train, val, test):

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[FEATURES])
    X_val = scaler_X.transform(val[FEATURES])
    X_test = scaler_X.transform(test[FEATURES])

    y_train = scaler_y.fit_transform(train[[TARGET]])
    y_val = scaler_y.transform(val[[TARGET]])
    y_test = scaler_y.transform(test[[TARGET]])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y


# ---------------------------
# Sequence creation (LSTM only)
# ---------------------------
def create_sequences(X, y, time_steps=24):

    Xs, ys = [], []

    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])

    return np.array(Xs), np.array(ys)