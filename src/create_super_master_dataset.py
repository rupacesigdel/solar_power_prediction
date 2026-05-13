import pandas as pd
# LOAD WEATHER DATA
weather_df = pd.read_csv(
    "data/processed/master_weather_dataset.csv",
    header=[0, 1],
    low_memory=False
)
# FLATTEN MULTI-LEVEL COLUMNS
weather_df.columns = [
    f"{col1}_{col2}" if "Unnamed" not in str(col2) else str(col1)
    for col1, col2 in weather_df.columns
]

# Rename timestamp column
weather_df.rename(columns={weather_df.columns[0]: 'timestamp'}, inplace=True)
# FIX DATE FORMAT
weather_df['timestamp'] = pd.to_datetime(
    weather_df['timestamp'],
    dayfirst=True,
    errors='coerce'
)

# Remove invalid timestamps
weather_df = weather_df.dropna(subset=['timestamp'])
# LOAD IRRADIANCE DATA
irradiance_df = pd.read_csv(
    "data/processed/master_weather_irradiance.csv",
    header=[0, 1],
    low_memory=False
)

irradiance_df.columns = [
    f"{col1}_{col2}" if "Unnamed" not in str(col2) else str(col1)
    for col1, col2 in irradiance_df.columns
]

irradiance_df.rename(
    columns={irradiance_df.columns[0]: 'timestamp'},
    inplace=True
)

irradiance_df['timestamp'] = pd.to_datetime(
    irradiance_df['timestamp'],
    dayfirst=True,
    errors='coerce'
)

irradiance_df = irradiance_df.dropna(subset=['timestamp'])

# LOAD ENERGY DATA
energy_df = pd.read_csv(
    "data/processed/master_weather.csv"
)

energy_df.rename(columns={'Date:': 'timestamp'}, inplace=True)

energy_df['timestamp'] = pd.to_datetime(
    energy_df['timestamp'],
    dayfirst=True,
    errors='coerce'
)

energy_df = energy_df.dropna(subset=['timestamp'])

# SELECT WEATHER FEATURES
temp_cols = [
    col for col in weather_df.columns
    if 'Air Temperature' in col
]

global_rad_cols = [
    col for col in weather_df.columns
    if 'Global Radiation 1' in col
]

cell_rad_cols = [
    col for col in weather_df.columns
    if 'Cell Radiation 1' in col
]

surface_temp_cols = [
    col for col in weather_df.columns
    if 'Surface Temperature' in col
]

wind_cols = [
    col for col in weather_df.columns
    if 'Wind Speec' in col
]

# Convert numeric
for cols in [
    temp_cols,
    global_rad_cols,
    cell_rad_cols,
    surface_temp_cols,
    wind_cols
]:
    weather_df[cols] = weather_df[cols].apply(
        pd.to_numeric,
        errors='coerce'
    )

# CREATE AVERAGE FEATURES
weather_df['avg_air_temp'] = weather_df[temp_cols].mean(axis=1)

weather_df['avg_global_rad'] = weather_df[global_rad_cols].mean(axis=1)

weather_df['avg_cell_rad'] = weather_df[cell_rad_cols].mean(axis=1)

weather_df['avg_surface_temp'] = weather_df[surface_temp_cols].mean(axis=1)

weather_df['avg_wind_speed'] = weather_df[wind_cols].mean(axis=1)

weather_selected = weather_df[
    [
        'timestamp',
        'avg_air_temp',
        'avg_global_rad',
        'avg_cell_rad',
        'avg_surface_temp',
        'avg_wind_speed'
    ]
]

# IRRADIANCE FEATURES
irr_cols = irradiance_df.columns[1:]

irradiance_df[irr_cols] = irradiance_df[irr_cols].apply(
    pd.to_numeric,
    errors='coerce'
)

irradiance_df['irradiance_avg'] = irradiance_df[
    irr_cols
].mean(axis=1)

irradiance_selected = irradiance_df[
    ['timestamp', 'irradiance_avg']
]

# ENERGY FEATURES

energy_df['ITS_Energy (kWh)'] = pd.to_numeric(
    energy_df['ITS_Energy (kWh)'],
    errors='coerce'
)

energy_selected = energy_df[
    ['timestamp', 'ITS_Energy (kWh)']
]

# MERGE DATASETS
merged_df = pd.merge(
    weather_selected,
    irradiance_selected,
    on='timestamp',
    how='inner'
)

merged_df = pd.merge(
    merged_df,
    energy_selected,
    on='timestamp',
    how='inner'
)

# FEATURE ENGINEERING
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['day'] = merged_df['timestamp'].dt.day
merged_df['month'] = merged_df['timestamp'].dt.month
merged_df['year'] = merged_df['timestamp'].dt.year

# REMOVE NULLS
merged_df = merged_df.dropna()


merged_df.to_csv(
    "data/processed/super_master_dataset.csv",
    index=False
)

print("Super master dataset created successfully!")
print(merged_df.head())
print(merged_df.shape)