import pandas as pd
import numpy as np
import os

def add_sensitive_feature(data_path: str):
    """
    Adds a synthetic 'location' column to the dataset at data_path.
    The file is modified in-place.

    Args:
        data_path (str): Path to the CSV dataset.
    """
    print(f"--- Adding sensitive feature to {data_path} ---")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Skip if already present
    if "location" in df.columns:
        print("✔ 'location' column already exists. Skipping.")
        return

    # Create synthetic Location_A / Location_B
    np.random.seed(42)
    df["location"] = np.random.choice(["Location_A", "Location_B"], size=len(df))

    # Save back to the same file
    df.to_csv(data_path, index=False)

    print(f"✔ Successfully added 'location' to {data_path}")
    print("---------------------------------------------\n")


if __name__ == "__main__":
    # Process both splits
    add_sensitive_feature("data/v0/transactions_2022.csv")
    add_sensitive_feature("data/v1/transactions_2023.csv")

