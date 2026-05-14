import os
import pandas as pd
import matplotlib.pyplot as plt


# CREATE FOLDER
def create_directory(path):

    if not os.path.exists(path):

        os.makedirs(path)


# SAVE FORECAST CSV
def save_forecast(df, path):

    df.to_csv(path, index=False)

    print(f"Forecast saved to {path}")


# PLOT PREDICTIONS
def plot_predictions(
    y_true,
    y_pred,
    title="Prediction"
):

    plt.figure(figsize=(12,5))

    plt.plot(y_true, label='Actual')

    plt.plot(y_pred, label='Predicted')

    plt.title(title)

    plt.legend()

    plt.show()


# SAVE FIGURE
def save_plot(path):

    plt.savefig(path)

    print(f"Plot saved to {path}")