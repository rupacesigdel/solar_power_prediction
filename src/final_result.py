import pandas as pd
import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(base_dir, "..", "result", "graphs")

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Created directory: {graph_dir}")


# 2. DEFINE ALL PATHS DYNAMICALLY
data_path = os.path.join(base_dir, "..", "data", "processed", "super_master_dataset.csv")
feature_scaler_path = os.path.join(base_dir, "..", "models", "feature_scaler.pkl")
target_scaler_path = os.path.join(base_dir, "..", "models", "target_scaler.pkl")
ann_model_path = os.path.join(base_dir, "..", "models", "ann_model.h5")
lstm_model_path = os.path.join(base_dir, "..", "models", "lstm_model.h5")

# 3. LOAD FILES
df = pd.read_csv(data_path)
print("Data loaded successfully!")

feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("Scalers loaded successfully!")

ann_model = load_model(ann_model_path, compile=False)
lstm_model = load_model(lstm_model_path, compile=False)
print("Models loaded successfully!")

# 2. PREPARE TEST DATA (Original Units)
X = df.drop(columns=['ITS_Energy (kWh)', 'timestamp'])
y_actual = df['ITS_Energy (kWh)'].values.reshape(-1, 1)

X_scaled = feature_scaler.transform(X)

# 3. PREDICTIONS & INVERSE SCALING
# ANN Prediction
ann_pred_scaled = ann_model.predict(X_scaled)
ann_pred = target_scaler.inverse_transform(ann_pred_scaled)

# LSTM Prediction (Requires sequence reshaping)
TIME_STEPS = 24
X_seq = []
y_true_lstm = []

for i in range(len(X_scaled) - TIME_STEPS):
    X_seq.append(X_scaled[i:i+TIME_STEPS])
    y_true_lstm.append(y_actual[i+TIME_STEPS])

X_seq = np.array(X_seq)
lstm_pred_scaled = lstm_model.predict(X_seq)
lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled)
y_true_lstm = np.array(y_true_lstm)

# 4. ACCURATE EVALUATION (In kWh)
def get_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = [
    get_metrics(y_actual, ann_pred, "ANN"),
    get_metrics(y_true_lstm, lstm_pred, "LSTM")
]
results_df = pd.DataFrame(results)
print(results_df)

# Create the result folder for CSVs
result_dir = os.path.join(base_dir, "..", "result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Save your results
results_df.to_csv(os.path.join(result_dir, "evaluation_results.csv"), index=False)



plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'].iloc[-500:], y_actual[-500:], label="Actual (Past)", alpha=0.7)
plt.plot(df['timestamp'].iloc[-500:], lstm_pred[-500:], label="LSTM Predicted", linestyle="--")
plt.title("Comparison: Past Actual Data vs Model Prediction")
plt.ylabel("Energy (kWh)")
plt.legend()
save_path = os.path.join(graph_dir, "past_vs_predicted.png")
plt.savefig(save_path)
print(f"Graph saved successfully at: {save_path}")
plt.show()

# 1. Calculate average weather patterns per hour
# Ensure only numbers are averaged
hourly_avg_features = df.groupby('hour').mean(numeric_only=True)

# 2. Keep only the features used during training
# Make sure this matches the exact columns your ANN expects
features_to_keep = [
    'avg_air_temp', 'avg_global_rad', 'avg_cell_rad', 
    'avg_surface_temp', 'avg_wind_speed', 'irradiance_avg'
]

# 3. Generate Future Predictions (1 year example)
future_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=24*30*12, freq='h')
future_preds = []

for date in future_dates:
    # Get the historical average weather for this hour
    avg_feat = hourly_avg_features.loc[date.hour][features_to_keep].values
    
    # Append the time features for the specific future date
    # Order must match: features + hour, day, month, year
    current_row = np.append(avg_feat, [date.hour, date.day, date.month, date.year])
    
    # Scale and predict
    scaled_row = feature_scaler.transform(current_row.reshape(1, -1))
    pred_scaled = ann_model.predict(scaled_row, verbose=0)
    
    # Inverse scale back to kWh
    final_val = target_scaler.inverse_transform(pred_scaled)[0][0]
    future_preds.append(max(0, final_val)) # Energy cannot be negative

forecast_2026 = pd.DataFrame({
    'timestamp': future_dates,
    'predicted_energy': future_preds
})
forecast_path = os.path.join(result_dir, "1_year_forecast_data.csv")
forecast_2026.to_csv(forecast_path, index=False)

print(f"✅ 1-Year 2026 forecast data saved successfully at: {forecast_path}")
print(f"Total records saved: {len(forecast_2026):,} rows")


# Plotting the forecast
plt.figure(figsize=(15, 6))
plt.plot(future_dates, future_preds, color='orange')
plt.title("1-Year Forecast based on Historical Hourly Weather Profiles")
plt.ylabel("Expected Energy (kWh)")
save_path = os.path.join(graph_dir, "year_2026_forecast.png")
plt.savefig(save_path)
print(f"Graph saved successfully at: {save_path}")
plt.show()