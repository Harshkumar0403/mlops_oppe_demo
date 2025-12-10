import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from google.cloud import storage
import mlflow
import mlflow.sklearn


# --------- CONFIG ---------
DEFAULT_DATA_PATH = "data/v0/transactions_2022.csv"  # update if needed
MLFLOW_TRACKING_URI = "http://34.16.34.114:8100"
MLFLOW_EXPERIMENT_NAME = "Fraud_Detection_Training"
GCS_BUCKET_NAME = "data_mlops_oppe2prep"
GCS_MODEL_DESTINATION = "production_models/model.pkl"
LOCAL_MODEL_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")


def upload_to_gcs(bucket_name: str, source_path: str, destination_path: str):
    """Uploads a local file to GCS."""
    print(f"Uploading {source_path} to gs://{bucket_name}/{destination_path} ...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(source_path)
    print("✔ Upload complete.")


def train_model(data_path: str = DEFAULT_DATA_PATH):
    """
    Train Decision Tree model and log to MLflow.
    EXCLUDES sensitive attribute 'location' from training.
    """

    print("--- Training Model (with MLflow logging) ---")

    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:

        # -------------------------------
        # Load Data
        # -------------------------------
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        if "Class" not in df.columns:
            raise ValueError("Dataset missing 'Class' column.")

        # -----------------------------------------------------
        # EXCLUDE sensitive + non-feature columns from training
        # -----------------------------------------------------
        X = df.drop(
            columns=["Class", "Time", "location"],  # location intentionally dropped
            errors="ignore",
        )
        y = df["Class"]

        # Safety check
        if X.empty:
            raise ValueError("Feature matrix X is empty after dropping columns.")

        # -------------------------------
        # Train/Val Split
        # -------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # -------------------------------
        # Train Model
        # -------------------------------
        params = {
            "class_weight": {0:1,1:2},
            "max_depth": 5,
            "random_state": 42,
        }

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        # -------------------------------
        # Evaluate
        # -------------------------------
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        print(f"Validation F1 Score: {f1:.4f}")

        # -------------------------------
        # MLflow Logging
        # -------------------------------
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud-detection-dt",
        )

        print(f"Model logged to MLflow (Run ID: {run.info.run_id})")

        # -------------------------------
        # Save local Pkl
        # -------------------------------
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        joblib.dump(model, LOCAL_MODEL_PATH)
        print(f"Local model saved → {LOCAL_MODEL_PATH}")

        # -------------------------------
        # Upload final model to GCS
        # -------------------------------
        upload_to_gcs(GCS_BUCKET_NAME, LOCAL_MODEL_PATH, GCS_MODEL_DESTINATION)


if __name__ == "__main__":
    train_model()

