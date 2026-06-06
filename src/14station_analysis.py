import pandas as pd

df = pd.read_csv(
    "data/processed/master_weather_dataset.csv",
    header=[0,1]
)

# Flatten column names
df.columns = [
    f"{a}_{b}" if b != "" else a
    for a,b in df.columns
]

# Timestamp
timestamp_col = df.columns[0]

# Air Temperature columns
air_cols = [
    c for c in df.columns
    if "Air Temperature" in c
]

# Wind Speed columns
wind_cols = [
    c for c in df.columns
    if "Wind Speec" in c
]

# Global Radiation columns
global_cols = [
    c for c in df.columns
    if "Global Radiation" in c
]

# Cell Radiation columns
cell_cols = [
    c for c in df.columns
    if "Cell Radiation" in c
]

# Surface Temperature columns
surface_cols = [
    c for c in df.columns
    if "Surface Temperature" in c
]

weather_agg = pd.DataFrame()

weather_agg["timestamp"] = df[timestamp_col]

weather_agg["avg_air_temp"] = (
    df[air_cols].mean(axis=1)
)

weather_agg["avg_wind_speed"] = (
    df[wind_cols].mean(axis=1)
)

weather_agg["avg_global_rad"] = (
    df[global_cols].mean(axis=1)
)

weather_agg["avg_cell_rad"] = (
    df[cell_cols].mean(axis=1)
)

weather_agg["avg_surface_temp"] = (
    df[surface_cols].mean(axis=1)
)

weather_agg.to_csv(
    "weather_aggregated.csv",
    index=False
)

print(weather_agg.head())