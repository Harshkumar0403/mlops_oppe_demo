#!/usr/bin/env python3
"""
train_model.py

- Loads the v0 dataset (or path specified via TRAINING_DATA env var).
- Drops sensitive columns (location, Time) and trains a DecisionTreeClassifier.
- Uses MLflow for tracking. If env var MLFLOW_TRACKING_URI is set it will be used.
  Otherwise it defaults to a local sqlite DB at ./mlflow.db to avoid file:// permission issues.
- Saves the trained model to artifacts/model.pkl and optionally uploads to GCS when GCS_BUCKET is set.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from google.cloud import storage

# ---------------------------
# Configuration (environment)
# ---------------------------
PROJECT_ID = os.environ.get("GCP_PROJECT", os.environ.get("PROJECT_ID", None))
GCS_BUCKET = os.environ.get("GCS_BUCKET", "data_mlops_oppe2prep")
GCS_MODEL_DEST = os.environ.get("GCS_MODEL_DEST", "production_models/model.pkl")
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "artifacts")
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
DATA_PATH = os.environ.get("TRAINING_DATA", "data/v0/transactions_2022.csv")
MLFLOW_TRACKING_URI_ENV = os.environ.get("MLFLOW_TRACKING_URI", "").strip()

# ---------------------------
# Utilities
# ---------------------------
def choose_mlflow_tracking_uri():
    """
    Decide MLflow tracking URI:
    - If MLFLOW_TRACKING_URI env is set -> use it
    - Else default to sqlite:///absolute/path/mlflow.db (safe, no root perms)
    """
    if MLFLOW_TRACKING_URI_ENV:
        uri = MLFLOW_TRACKING_URI_ENV
        print(f"[mlflow] Using MLFLOW_TRACKING_URI from env: {uri}")
        return uri

    # Default to a repo-local sqlite DB to avoid permission issues with file://
    sqlite_path = os.path.abspath(os.path.join(os.getcwd(), "mlflow.db"))
    uri = f"sqlite:///{sqlite_path}"
    print(f"[mlflow] No MLFLOW_TRACKING_URI provided. Using local sqlite backend: {uri}")
    return uri


def upload_to_gcs(bucket_name: str, local_path: str, dest_path: str):
    """
    Upload a local file to GCS. Requires application default credentials
    or a service account available to the environment.
    """
    if not os.path.exists(local_path):
        print(f"[gcs] Local file not found: {local_path}. Skipping upload.")
        return

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(local_path)
        print(f"[gcs] Uploaded {local_path} -> gs://{bucket_name}/{dest_path}")
    except Exception as e:
        print(f"[gcs] Warning: failed to upload to GCS: {e}")


# ---------------------------
# Main training function
# ---------------------------
def train_model(data_path: str = DATA_PATH):
    print("--- Training Model & Logging with MLflow ---")

    # 1) MLflow tracking setup
    tracking_uri = choose_mlflow_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as e:
        print(f"[mlflow] Error setting tracking URI '{tracking_uri}': {e}")
        raise

    mlflow.set_experiment("Fraud_Detection_Training")

    # Ensure artifacts dir exists
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # 2) Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    print(f"[data] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[data] Loaded {len(df)} rows, columns: {list(df.columns)[:10]}...")

    # 3) Prepare X, y; drop sensitive/metadata
    if "Class" not in df.columns:
        raise KeyError("Dataset must contain 'Class' target column.")

    # Drop Time and location if present (sensitive / metadata)
    drop_cols = [c for c in ["Time", "location"] if c in df.columns]
    if drop_cols:
        print(f"[data] Dropping columns before training: {drop_cols}")

    X = df.drop(columns=["Class"] + drop_cols, errors="ignore")
    y = df["Class"]

    # 4) Train/validation split (handle edge cases)
    # If the class is extremely imbalanced or only one class present, handle gracefully
    stratify_arg = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
    except Exception as e:
        print(f"[data] Warning: train_test_split with stratify failed: {e}. Falling back to simple split.")
        X_train = X.iloc[: int(0.8 * len(X))]
        X_val   = X.iloc[int(0.8 * len(X)) :]
        y_train = y.iloc[: int(0.8 * len(y))]
        y_val   = y.iloc[int(0.8 * len(y)) :]

    print(f"[train] Train size: {len(X_train)}, Val size: {len(X_val)}")

    # 5) Model training
    params = {"class_weight": {0:1,1:2}, "max_depth": 3, "random_state": 42}
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_train, y_train)

    # 6) Evaluation (F1 if possible)
    f1 = None
    try:
        y_pred = clf.predict(X_val)
        f1 = float(f1_score(y_val, y_pred))
        print(f"[eval] Validation F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"[eval] Warning: evaluation failed: {e}")

    # 7) MLflow logging + model save + upload
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        if f1 is not None:
            mlflow.log_metric("f1_score", f1)

        # log the model artifact to MLflow registry/artifacts (best-effort)
        try:
            mlflow.sklearn.log_model(sk_model=clf, artifact_path="model", registered_model_name="fraud-detection-dt")
            print("[mlflow] Model logged to MLflow (artifact + registry attempt).")
        except Exception as e:
            # continue even if registry/registering fails (ex: remote server missing or permissions)
            print(f"[mlflow] Warning: mlflow.sklearn.log_model failed: {e}. Will still save local artifact.")

        # Save local model for serving
        try:
            joblib.dump(clf, LOCAL_MODEL_PATH)
            print(f"[artifact] Saved local model to: {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"[artifact] Error saving local model to {LOCAL_MODEL_PATH}: {e}")
            raise

        # Upload model to GCS (if GCS_BUCKET is set)
        try:
            if GCS_BUCKET:
                upload_to_gcs(GCS_BUCKET, LOCAL_MODEL_PATH, GCS_MODEL_DEST)
        except Exception as e:
            print(f"[gcs] Warning: upload_to_gcs raised an exception: {e}")

        print(f"[mlflow] Finished run id: {run.info.run_id}")

    print("Training complete.")


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    try:
        train_model()
    except Exception as exc:
        print(f"[fatal] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)

