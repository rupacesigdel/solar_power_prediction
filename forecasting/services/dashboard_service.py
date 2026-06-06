import pandas as pd

DATA_PATH = "data/processed/super_master_dataset.csv"


def get_dashboard_stats():

    df = pd.read_csv(DATA_PATH)

    return {
        "total_records": len(df),
        "avg_energy": round(df["ITS_Energy (kWh)"].mean(), 2),
        "max_energy": round(df["ITS_Energy (kWh)"].max(), 2),
        "min_energy": round(df["ITS_Energy (kWh)"].min(), 2),
    }