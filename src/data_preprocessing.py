import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# LOAD DATASET

df = pd.read_csv(
    "data/processed/super_master_dataset.csv"
)

# REMOVE NULLS

df = df.dropna()

# REMOVE DUPLICATES

df = df.drop_duplicates()

# FEATURE SELECTION

X = df[
    [
        'avg_air_temp',
        'avg_global_rad',
        'avg_cell_rad',
        'avg_surface_temp',
        'avg_wind_speed',
        'irradiance_avg',
        'hour',
        'day',
        'month',
        'year'
    ]
]

y = df['ITS_Energy (kWh)']

# NORMALIZATION

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X)

y_scaled = target_scaler.fit_transform(
    y.values.reshape(-1, 1)
)

# Save scalers
joblib.dump(
    feature_scaler,
    "models/feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    "models/target_scaler.pkl"
)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    shuffle=False
)

# SAVE ARRAYS

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print("Data preprocessing completed!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)