import os
import pandas as pd

RAW_PATH = "data/raw/transactions.csv"
OUT_V0 = "data/v0/transactions_2022.csv"
OUT_V1 = "data/v1/transactions_2023.csv"


def main():
    # Check file existence
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Input file not found: {RAW_PATH}")

    print(f"Loading dataset from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    if "Time" not in df.columns:
        raise ValueError("Expected column 'Time' not found in dataset.")

    print("Sorting by Time column...")
    df = df.sort_values("Time").reset_index(drop=True)

    # Split by half based on time
    mid_index = len(df) // 2

    df_v0 = df.iloc[:mid_index]
    df_v1 = df.iloc[mid_index:]

    # Ensure directories exist
    os.makedirs("data/v0", exist_ok=True)
    os.makedirs("data/v1", exist_ok=True)

    # Save splits
    print(f"Saving 2022 data → {OUT_V0}")
    df_v0.to_csv(OUT_V0, index=False)

    print(f"Saving 2023 data → {OUT_V1}")
    df_v1.to_csv(OUT_V1, index=False)

    print("\n✔ Split complete!")
    print(f"v0 size: {len(df_v0)} rows")
    print(f"v1 size: {len(df_v1)} rows")


if __name__ == "__main__":
    main()

