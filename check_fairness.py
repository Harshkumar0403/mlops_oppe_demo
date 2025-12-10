import os
import json

import joblib
import pandas as pd
from fairlearn.metrics import demographic_parity_difference


DATA_PATH = "data/v0/transactions_2022.csv"
MODEL_PATH = "artifacts/model.pkl"
ARTIFACTS_DIR = "artifacts"


def check_model_fairness(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
):
    """
    Loads the trained fraud detection model and assesses its fairness based
    on the 'location' sensitive feature using demographic parity difference.
    """
    print("--- Checking Model Fairness ---")

    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # Load data
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        if "location" not in df.columns:
            print("❌ Error: 'location' column not found. Please run add_location.py first.")
            return

        if "Class" not in df.columns:
            print("❌ Error: 'Class' column not found in data.")
            return

        # Get features the model was trained on (preferred)
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            X = df[expected_features]
        else:
            # Fallback: drop target and non-feature columns
            X = df.drop(columns=["Class", "Time"], errors="ignore")

        y_true = df["Class"]
        sensitive_feature = df["location"]

        print("✅ Model and data with 'location' feature loaded.")
    except (FileNotFoundError, AttributeError, KeyError, ValueError) as e:
        print(f"❌ Error during data/model loading: {e}")
        return

    # Predict with the model
    print("Running model predictions...")
    y_pred = model.predict(X)

    # --- Calculate Demographic Parity Difference for the fraud class (Class=1) ---
    print("\n📏 Calculating Demographic Parity Difference for the fraud class...")

    # Binary labels: 1 = fraud, 0 = non-fraud
    y_true_binary = (y_true == 1)
    y_pred_binary = (y_pred == 1)

    dpd = demographic_parity_difference(
        y_true_binary,
        y_pred_binary,
        sensitive_features=sensitive_feature,
    )

    fairness_report = {
        "demographic_parity_difference_fraud": dpd,
    }

    print("\n✅ Overall Fairness Report:")
    print(json.dumps(fairness_report, indent=2))

    # Save report to JSON file
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "fairness_report.json")
    with open(report_path, "w") as f:
        json.dump(fairness_report, f, indent=4)

    print(f"\n💾 Fairness report saved to: {report_path}")
    print("-----------------------------\n")


if __name__ == "__main__":
    check_model_fairness()

