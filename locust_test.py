# locust_test.py
from locust import HttpUser, task, between

# Fixed payload (exactly the JSON you provided)
FIXED_PAYLOAD = {
    "V1": -1.2, "V2": 0.5, "V3": 1.1, "V4": -0.3, "V5": 0.4,
    "V6": 0.0, "V7": 1.8, "V8": -0.4, "V9": 0.2, "V10": 0.1,
    "V11": 0.6, "V12": -0.1, "V13": 0.2, "V14": -0.5, "V15": 1.0,
    "V16": -0.1, "V17": 0.4, "V18": -1.1, "V19": 2.1, "V20": -0.3,
    "V21": 0.1, "V22": 0.2, "V23": -0.4, "V24": 0.9, "V25": 0.2,
    "V26": 0.1, "V27": -0.2, "V28": 0.3, "Amount": 123.45
}

class FraudDetectionUser(HttpUser):
    """
    Locust user that always posts the same well-formed feature payload
    to /predict to avoid feature-name mismatch errors.
    """
    wait_time = between(1, 2)

    @task
    def predict_fixed(self):
        with self.client.post("/predict", json=FIXED_PAYLOAD, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"{resp.status_code}: {resp.text}")
            else:
                resp.success()

