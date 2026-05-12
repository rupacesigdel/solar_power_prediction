import pandas as pd
import glob

files = glob.glob("data/raw/weather/**/*.xls", recursive=True)

all_data = []

for file in files:
    print("\nProcessing:", file.split("\\")[-1])

    try:

        raw = pd.read_excel(file, sheet_name="Sheet1", header=None)

        header_row = None

        for i in range(min(30, len(raw))):
            row = raw.iloc[i].fillna("").astype(str).str.lower()

            if any(("wind" in str(x) or "temperature" in str(x)) for x in row):
                header_row = i
                break

        if header_row is None:
            print("Skipping file -> no header found")
            continue


        df = pd.read_excel(file, sheet_name="Sheet1", header=header_row)

        df.columns = df.columns.astype(str)


        df.columns = [f"col_{i}" if "unnamed" in c.lower() else c for i, c in enumerate(df.columns)]


        # -------------------------
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        df = df.dropna(subset=["timestamp"])

        for col in df.columns:
            if col != "timestamp":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(how="all")

        all_data.append(df)

        print("✔ Loaded:", df.shape)

    except Exception as e:
        print("Error:", e)


if len(all_data) == 0:
    print("\n❌ No valid data found. Check Excel structure.")
else:
    master = pd.concat(all_data, ignore_index=True)

    master = master.sort_values("timestamp")

    master = master.ffill().bfill()

    master.to_csv("master_weather_dataset.csv", index=False)

    print("\n✅ DONE!")
    print("Final Shape:", master.shape)
    print(master.head())