import matplotlib.pyplot as plt


def plot_forecast(actual, predicted):

    plt.figure()
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.legend()
    plt.title("Solar Power Forecast")
    return plt