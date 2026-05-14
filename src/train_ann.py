import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib
import math

# LOAD DATA

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# BUILD ANN MODEL

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1))

# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# EARLY STOPPING

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# PREDICTION

y_pred = model.predict(X_test)

# EVALUATION

mae = mean_absolute_error(y_test, y_pred)

rmse = math.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nANN MODEL RESULTS")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# SAVE MODEL

model.save("models/ann_model.h5")

print("\nANN model saved successfully!")