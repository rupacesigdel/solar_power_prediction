import pandas as pd
import numpy as np
import glob
import os

INPUT_FOLDER = "data/raw/weather"
OUTPUT_FILE = "master_weather_irradiance.csv"

all_files = glob.glob(os.path.join(INPUT_FOLDER, "**/*.xls"), recursive=True)

all_data = []


def extract_date(df):
    for i in range(3):  # top rows
        for j in range(df.shape[1]):
            val = str(df.iloc[i, j])
            if "202" in val:
                try:
                    return val.split(":")[-1].strip()
                except:
                    pass
    return None


for file in all_files:
    print(f"Processing: {file}")

    df = pd.read_excel(file, header=None)

    try:
        date = extract_date(df)

        if date is None:
            print(f"Skipping file (no date found): {file}")
            continue

        time = df.iloc[4:, 0].astype(str).reset_index(drop=True)

        time = time.replace("nan", "00:00")


        timestamp = pd.to_datetime(
            date + " " + time,
            errors="coerce"
        )

        valid = timestamp.notna()
        timestamp = timestamp[valid].reset_index(drop=True)


        data = df.iloc[4:, 1:].reset_index(drop=True)
        data = data.loc[valid].reset_index(drop=True)

        headers = df.iloc[1:4, 1:]

        cols = []
        for i in range(headers.shape[1]):
            parts = []
            for r in range(headers.shape[0]):
                val = str(headers.iloc[r, i]).strip()

                if val != "nan":
                    parts.append(val.replace(" ", "_"))

            col_name = "_".join(parts)
            cols.append(col_name)

        seen = {}
        unique_cols = []

        for c in cols:
            if c in seen:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                unique_cols.append(c)

        data.columns = unique_cols

        data = data.replace(-32000, np.nan)

        df_final = pd.concat([timestamp.rename("timestamp"), data], axis=1)

        all_data.append(df_final)

    except Exception as e:
        print(f"Error in file {file}: {e}")


master = pd.concat(all_data, ignore_index=True)

master = master.sort_values("timestamp")

master.to_csv(OUTPUT_FILE, index=False)

print("DONE ✔ Master file created:", OUTPUT_FILE)