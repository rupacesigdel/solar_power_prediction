# [Solar power Prediction](https://github.com/rupacesigdel/solar_power_prediction): A Comparative Study of ANN and LSTM Architectures

![Project Status](https://img.shields.io/badge/Status-Research_Complete-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange)

## 📌 Project Overview
This research focuses on predicting solar energy output (kWh) based on meteorological features such as solar radiation, air temperature, and cell temperature. By leveraging **Artificial Neural Networks (ANN)** and **Long Short-Term Memory (LSTM)** networks, the project explores the efficacy of deep learning in handling the stochastic and seasonal nature of solar energy data.

The model is trained on historical data and provides a granular, hourly forecast for the **2026–2027** period, successfully overcoming common recursive prediction pitfalls.

---

## 📊 Methodology & Features
The project utilizes a "Super Master Dataset" containing hourly environmental readings. To ensure model convergence and accuracy, the following preprocessing steps were implemented:

*   **Feature Engineering:** Extraction of temporal features (`hour`, `day`, `month`, `year`) to capture diurnal and seasonal cycles.
*   **Data Scaling:** Robust normalization using `MinMaxScaler` for both input features and the target variable (`ITS_Energy`).
*   **Architectures:**
    *   **ANN:** A dense multilayer perceptron optimized for feature-based regression.
    *   **LSTM:** A recurrent neural network designed to capture time-series dependencies and long-term patterns.

### **Core Features Used:**
*   Average Global Radiation
*   Average Cell/Surface Temperature
*   Wind Speed
*   Irradiance
*   Temporal Markers (Hourly/Monthly)

---

## 📈 Performance Metrics
The models were evaluated using Root Mean Square Error (RMSE) and the Coefficient of Determination ($R^2$). 

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **ANN** | 11,507 | 18,588 | **0.8217** |
| **LSTM** | 10,871 | 23,551 | **0.7137** |

> **Research Note:** The ANN outperformed the LSTM in overall variance explanation ($R^2$), while the LSTM showed strong capability in identifying specific peak-hour trends.

---
## Installation

### Setting Up Your Environment

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/rupacesigdel/solar_power_prediction.git
    cd solar_power_prediction
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv myenv
    source .venv/Scripts/activate  # On windows
    ```

3. **Upgrade pip (if needed):**

    ```bash
    pip install --upgrade pip
    ```

4. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 🔮 Future Forecasts (2026–2027)
To avoid the "recursive drift" often found in long-term AI forecasts (where predictions flatten over time), this project utilizes **Historical Hourly Profiling**. By feeding the model average meteorological "fingerprints" from past years alongside future timestamps, we generate a realistic energy production wave.

**Forecast Highlights:**
*   Successful capturing of the daily peak/off-peak solar cycle.
*   Automated generation of `1_year_forecast_data.csv` for downstream energy grid planning.

---

## 🛠 Author
**[Rupesh Sigdel](https://github.com/rupacesigdel)**
- *Research Area: Deep Learning in Renewable Energy*
- *LinkedIn:* [rupesh-sigdel](https://www.linkedin.com/in/rupesh-sigdel-63252425b/) | *Email:* rupeshcgdl2060@gmail.com

---

## 🤝 Acknowledgments
*   This research was conducted as part of an independent study into **Renewable Energy Forecasting**.
*   Data provided by Nuwakot devighat Solar Power Plant 25MW .
*   Special thanks to mentors and peers who provided feedback on the deep learning architectures used in this study.

## 📜 License & Copyright
Copyright (c) 2026 **Rupesh Sigdel**

This project is licensed under the **MIT License** - see the details below:

*   **Permission** is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.
*   **Attribution** is required: Any publication or distribution of this work must include a citation to this repository and the author.

---
