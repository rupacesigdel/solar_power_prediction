import os
import glob
import pandas as pd

BASE_PATH = "data/raw/weather"

def read_excel_safe(file_path):
    """
    Try reading Excel in multiple ways because files are inconsistent.
    """
    try:
        df = pd.read_excel(file_path)
        if df is not None and not df.empty:
            return df
    except:
        pass

    try:

        df = pd.read_excel(file_path, header=None)
        return df
    except:
        return None


def detect_and_standardize(df, file_name):
    """
    Convert messy formats into structured rows.
    """

    if df is None or df.empty:
        return None

    df = df.dropna(how="all")

    cols = [str(c) for c in df.columns]


    if len(df.columns) == 2 and "Date" in str(df.iloc[:, 0].values):
        try:
            key = str(df.iloc[0, 0]).replace(":", "").strip()
            value = df.iloc[0, 1]

            return pd.DataFrame([{
                "key": key,
                "value": value,
                "source_file": file_name
            }])
        except:
            pass

    if all("Unnamed" in str(c) for c in cols):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]


    df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed")]
    df = df.dropna(how="all")

    df["source_file"] = file_name

    return df



def process_file(file_path):
    print(f"\nProcessing: {file_path}")

    df = read_excel_safe(file_path)

    df = detect_and_standardize(df, os.path.basename(file_path))

    if df is None or df.empty:
        print("Skipped (no usable data)")
        return None

    print("Final columns:", df.columns.tolist())

    return df


def load_all_data():
    all_files = glob.glob(os.path.join(BASE_PATH, "**", "*.xls"), recursive=True)

    dataset = []

    for file in all_files:
        df = process_file(file)
        if df is not None:
            dataset.append(df)

    if len(dataset) == 0:
        raise ValueError("No valid data found in any file!")

    return pd.concat(dataset, ignore_index=True)


if __name__ == "__main__":
    master_df = load_all_data()

    print("\n========================")
    print("FINAL DATASET INFO")
    print("========================")
    print(master_df.info())

    output_file = "master_energy.csv"
    master_df.to_csv(output_file, index=False)

    print(f"\nSaved successfully → {output_file}")