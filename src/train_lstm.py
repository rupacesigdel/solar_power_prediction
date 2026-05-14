import numpy as np
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD DATA

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# CREATE SEQUENCES

TIME_STEPS = 24

def create_sequences(X, y, time_steps):

    Xs = []
    ys = []

    for i in range(len(X) - time_steps):

        Xs.append(X[i:(i + time_steps)])

        ys.append(y[i + time_steps])

    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(
    X_train,
    y_train,
    TIME_STEPS
)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test,
    TIME_STEPS
)

# BUILD LSTM MODEL

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            X_train_seq.shape[1],
            X_train_seq.shape[2]
        )
    )
)

model.add(Dropout(0.2))

model.add(LSTM(64))

model.add(Dropout(0.2))

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
    X_train_seq,
    y_train_seq,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# PREDICTION

y_pred = model.predict(X_test_seq)

# EVALUATION

mae = mean_absolute_error(
    y_test_seq,
    y_pred
)

rmse = math.sqrt(
    mean_squared_error(
        y_test_seq,
        y_pred
    )
)

r2 = r2_score(
    y_test_seq,
    y_pred
)

print("\nLSTM MODEL RESULTS")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# SAVE MODEL

model.save("models/lstm_model.h5")

print("\nLSTM model saved successfully!")