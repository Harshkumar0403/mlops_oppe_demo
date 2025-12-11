# locust_test.py
from locust import HttpUser, task, between
import pandas as pd
import random
import os
import sys

# Try to use DVC if available
try:
    from dvc.api import open as dvc_open
    DVC_AVAILABLE = True
except Exception:
    DVC_AVAILABLE = False


class FraudDetectionUser(HttpUser):
    """
    Locust user that posts only transaction_id (Time column)
    to the /predict endpoint.
    """
    wait_time = between(1, 2)
    transaction_ids = []   # cached list of valid IDs

    def on_start(self):
        if FraudDetectionUser.transaction_ids:
            return

        print("Locust: Loading dataset...", file=sys.stderr)
        df = None

        # --- 1) Try to load via DVC ---
        if DVC_AVAILABLE:
            try:
                print("Locust: trying dvc.api.open()", file=sys.stderr)
                with dvc_open("data/v0/transactions_2022.csv", mode="r") as fd:
                    df = pd.read_csv(fd)
                print("Loaded dataset via DVC", file=sys.stderr)
            except Exception as e:
                print(f"DVC load error: {e}", file=sys.stderr)
                df = None

        # --- 2) Local fallback ---
        if df is None:
            local_path = "data/v0/transactions_2022.csv"
            if os.path.exists(local_path):
                try:
                    df = pd.read_csv(local_path)
                    print(f"Loaded local CSV: {local_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Failed local load: {e}", file=sys.stderr)
                    df = None
            else:
                print("Dataset not found", file=sys.stderr)

        # --- 3) Ensure data available ---
        if df is None or df.empty:
            print("Locust: NO DATA AVAILABLE. Tests will be no-op.", file=sys.stderr)
            return

        # Ensure Time column exists
        if "Time" not in df.columns:
            print("ERROR: 'Time' column missing — cannot generate transaction_id list.", file=sys.stderr)
            return

        # Extract transaction IDs
        FraudDetectionUser.transaction_ids = df["Time"].astype(int).tolist()

        print(f"Loaded {len(FraudDetectionUser.transaction_ids)} transaction IDs.", file=sys.stderr)

    @task
    def predict_endpoint(self):
        """
        Send only transaction_id = random Time value
        """
        if not FraudDetectionUser.transaction_ids:
            return

        txid = random.choice(FraudDetectionUser.transaction_ids)

        payload = {"transaction_id": txid}

        # POST request
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"{response.status_code}: {response.text}")
            else:
                response.success()

