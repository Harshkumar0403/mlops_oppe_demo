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
DEFAULT_DATA_PATH = "data/v0/transactions_2022.csv"  # you can override when calling train_model
MLFLOW_TRACKING_URI = "http://34.46.228.116:8100"    # <-- your MLflow server
MLFLOW_EXPERIMENT_NAME = "Fraud_Detection_Training"
GCS_BUCKET_NAME = "data_mlops_oppe2prep"
GCS_MODEL_DESTINATION = "production_models/model.pkl"
LOCAL_MODEL_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")


def upload_to_gcs(bucket_name: str, source_path: str, destination_path: str) -> None:
    """
    Uploads a local file to a GCS bucket.
    """
    print(f"Uploading {source_path} to gs://{bucket_name}/{destination_path} ...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(source_path)
    print("✔ Upload complete.")


def train_model(data_path: str = DEFAULT_DATA_PATH):
    """
    Trains a Decision Tree model for fraud detection, logs the experiment
    to MLflow, and saves the final model artifact locally and to GCS.
    """
    print("--- Training Model & Logging with MLflow ---")

    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # ---- Load data ----
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Define features (X) and target (y)
        if "Class" not in df.columns:
            raise ValueError("Column 'Class' not found in the dataset.")
        if "Time" not in df.columns:
            raise ValueError("Column 'Time' not found in the dataset.")

        X = df.drop(columns=["Class", "Time"])
        y = df["Class"]

        # ---- Train/val split ----
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # ---- Train model ----
        params = {
            "class_weight": {0: 1, 1: 2},
            "max_depth": 3,
            "random_state": 42,
        }
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        # ---- Evaluate model ----
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"Validation F1-Score: {f1:.4f}")

        # ---- Log to MLflow ----
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)

        # Log the model as an MLflow artifact (for registry / UI)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud-detection-dt",
        )

        print(f"Model logged to MLflow. Run ID: {run.info.run_id}")

        # ---- Save model locally ----
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        joblib.dump(model, LOCAL_MODEL_PATH)
        print(f"Model artifact saved locally to: {LOCAL_MODEL_PATH}")

        # ---- Upload model to GCS ----
        upload_to_gcs(
            bucket_name=GCS_BUCKET_NAME,
            source_path=LOCAL_MODEL_PATH,
            destination_path=GCS_MODEL_DESTINATION,
        )


if __name__ == "__main__":
    # Make sure your MLflow server is running and accessible
    # You can also override via env: MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Default: train on v0 / 2022 data
    train_model()

