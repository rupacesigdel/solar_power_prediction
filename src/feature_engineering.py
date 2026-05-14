import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import joblib


def load_dataset(path):

    df = pd.read_csv(path)

    df['timestamp'] = pd.to_datetime(
        df['timestamp'],
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna()

    return df


def create_time_features(df):

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    return df


def prepare_features(df):

    target = 'ITS_Energy (kWh)'

    feature_cols = [

        col for col in df.columns

        if col not in ['timestamp', target]

    ]

    X = df[feature_cols]

    y = df[target]

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test