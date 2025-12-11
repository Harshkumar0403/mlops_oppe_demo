# locust_test.py
from locust import HttpUser, task, between
import pandas as pd
import random
import os
import sys

# Try to use dvc to read tracked file; fall back to local file
try:
    from dvc.api import open as dvc_open
    DVC_AVAILABLE = True
except Exception:
    DVC_AVAILABLE = False

# The exact feature order your model expects
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

class FraudDetectionUser(HttpUser):
    """
    Locust user that posts real transaction feature rows (no 'location') to /predict.
    """
    wait_time = between(1, 2)
    transaction_rows = []  # cached list of dicts ready to POST

    def on_start(self):
        if FraudDetectionUser.transaction_rows:
            return

        print("Locust: loading dataset...", file=sys.stderr)
        df = None

        # 1) Try to load via DVC (recommended if dataset tracked)
        if DVC_AVAILABLE:
            try:
                print("Locust: trying dvc.api.open()", file=sys.stderr)
                with dvc_open(path="data/v0/transactions_2022.csv", mode="r") as fd:
                    df = pd.read_csv(fd)
                print("Locust: loaded dataset via DVC", file=sys.stderr)
            except Exception as e:
                print(f"Locust: dvc load failed: {e}", file=sys.stderr)
                df = None

        # 2) Fallback to local CSV if DVC not available or failed
        if df is None:
            local_path = "data/v0/transactions_2022.csv"
            if os.path.exists(local_path):
                try:
                    df = pd.read_csv(local_path)
                    print(f"Locust: loaded dataset from local path {local_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Locust: failed to read local CSV: {e}", file=sys.stderr)
                    df = None
            else:
                print(f"Locust: data file not found at {local_path}", file=sys.stderr)

        # 3) Prepare rows: drop unwanted cols and keep only FEATURE_COLUMNS
        if df is None or df.empty:
            print("Locust: no data available; tasks will be no-ops", file=sys.stderr)
            return

        # drop metadata columns if they exist
        for col in ["Time", "Class", "location"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure all FEATURE_COLUMNS exist; if missing, add with zeros
        for feat in FEATURE_COLUMNS:
            if feat not in df.columns:
                print(f"Locust: warning - missing feature column '{feat}', filling with 0", file=sys.stderr)
                df[feat] = 0.0

        # Select only the columns in the REQUIRED order
        df = df[FEATURE_COLUMNS]

        # Sample a reasonable number of rows into memory to avoid huge memory usage
        sample_n = min(len(df), 5000)
        sampled = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

        # Convert sampled rows to list of plain dicts (convert numpy types)
        def to_py(d):
            out = {}
            for k, v in d.items():
                # convert numpy types to python native
                try:
                    if hasattr(v, "item"):
                        out[k] = v.item()
                    else:
                        out[k] = v
                except Exception:
                    out[k] = v
            return out

        FraudDetectionUser.transaction_rows = [to_py(row) for _, row in sampled.iterrows()]
        print(f"Locust: prepared {len(FraudDetectionUser.transaction_rows)} rows for testing", file=sys.stderr)

    @task
    def predict_endpoint(self):
        if not FraudDetectionUser.transaction_rows:
            return

        payload = random.choice(FraudDetectionUser.transaction_rows)

        # POST to /predict and mark failures in Locust UI
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"{response.status_code}: {response.text}")
            else:
                response.success()

