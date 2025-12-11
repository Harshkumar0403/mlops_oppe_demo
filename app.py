# app.py
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import logging
import json
import sys

# Optional OpenTelemetry (only enabled if OTEL_EXPORTER_OTLP_ENDPOINT set)
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# -------------------------
# Config
# -------------------------
BUCKET_NAME = os.environ.get("GCS_BUCKET", "data_mlops_oppe2prep")
MODEL_BLOB = os.environ.get("GCS_MODEL_BLOB", "production_models/model.pkl")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "artifacts/model.pkl")
# Local (DVC) paths we expect to exist after 'dvc pull'
LOCAL_V0 = "data/v0/transactions_2022.csv"
LOCAL_V1 = "data/v1/transactions_2023.csv"
FALLBACK_RAW = "data/raw/transactions.csv"
TOP_LEVEL = "data/transactions.csv"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "data_cached")

# OTEL
OTEL_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

# -------------------------
# Logging
# -------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"severity": record.levelname, "message": record.getMessage()})

logger = logging.getLogger("fraud-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.handlers.clear()
logger.addHandler(handler)

# Silence opentelemetry noisy loggers unless debugging
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
logging.getLogger("opentelemetry.sdk").setLevel(logging.WARNING)
logging.getLogger("opentelemetry.instrumentation").setLevel(logging.WARNING)

# -------------------------
# OpenTelemetry setup (only if OTLP endpoint provided)
# -------------------------
tracer = None
otel_enabled = bool(OTEL_OTLP_ENDPOINT)
if otel_enabled:
    try:
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=OTEL_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry OTLP exporter enabled.")
    except Exception as e:
        logger.warning(f"Failed to initialize OTLP exporter: {e}. Tracing disabled.")
        otel_enabled = False

if not otel_enabled:
    tracer = trace.get_tracer(__name__)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Fraud Detection API")
if otel_enabled:
    try:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    except Exception as e:
        logger.warning(f"FastAPI instrumentation failed: {e}")

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def download_blob(bucket_name, blob_name, local_path):
    """Download a blob from GCS to local_path (will overwrite if exists)."""
    logger.info(f"Downloading gs://{bucket_name}/{blob_name} -> {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    ensure_dir(os.path.dirname(local_path) or ".")
    blob.download_to_filename(local_path)
    logger.info("Download complete.")

def load_data_local_first():
    """
    Load data preferring local DVC-pulled files.
    Order:
      1) data/v0 & data/v1 (local)
      2) data/raw/transactions.csv (local)
      3) data/transactions.csv (top-level local)
      4) fallback to GCS: data/v0 & data/v1 objects (only if local not found)
    Returns combined dataframe or empty df.
    """
    ensure_dir(LOCAL_DATA_DIR)
    local_v0_cached = os.path.join(LOCAL_DATA_DIR, "transactions_2022.csv")
    local_v1_cached = os.path.join(LOCAL_DATA_DIR, "transactions_2023.csv")

    dfs = []

    # 1) Prefer the DVC-pulled locations first (recommended)
    if os.path.exists(LOCAL_V0):
        try:
            dfs.append(pd.read_csv(LOCAL_V0))
            logger.info(f"Loaded DVC local v0: {LOCAL_V0}")
        except Exception as e:
            logger.error(f"Failed reading {LOCAL_V0}: {e}")
    if os.path.exists(LOCAL_V1):
        try:
            dfs.append(pd.read_csv(LOCAL_V1))
            logger.info(f"Loaded DVC local v1: {LOCAL_V1}")
        except Exception as e:
            logger.error(f"Failed reading {LOCAL_V1}: {e}")

    # 2) fallback to raw single file
    if not dfs and os.path.exists(FALLBACK_RAW):
        try:
            dfs.append(pd.read_csv(FALLBACK_RAW))
            logger.info(f"Loaded fallback raw CSV: {FALLBACK_RAW}")
        except Exception as e:
            logger.error(f"Failed to read {FALLBACK_RAW}: {e}")

    # 3) fallback to top-level data/transactions.csv
    if not dfs and os.path.exists(TOP_LEVEL):
        try:
            dfs.append(pd.read_csv(TOP_LEVEL))
            logger.info(f"Loaded top-level CSV: {TOP_LEVEL}")
        except Exception as e:
            logger.error(f"Failed to read {TOP_LEVEL}: {e}")

    # 4) If nothing local, try to download from GCS (only then)
    if not dfs:
        try:
            # try download to cache files (do not fail hard)
            try:
                download_blob(BUCKET_NAME, "data/v0/transactions_2022.csv", local_v0_cached)
                dfs.append(pd.read_csv(local_v0_cached))
                logger.info(f"Downloaded and loaded v0 from GCS into {local_v0_cached}")
            except Exception:
                logger.info("GCS v0 not found or download failed; skipping v0")
            try:
                download_blob(BUCKET_NAME, "data/v1/transactions_2023.csv", local_v1_cached)
                dfs.append(pd.read_csv(local_v1_cached))
                logger.info(f"Downloaded and loaded v1 from GCS into {local_v1_cached}")
            except Exception:
                logger.info("GCS v1 not found or download failed; skipping v1")
        except Exception as e:
            logger.error(f"GCS attempts failed: {e}")

    if not dfs:
        logger.error("No dataset files available locally or in GCS.")
        return pd.DataFrame()

    try:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded combined dataset with {len(combined)} rows.")
        return combined
    except Exception as e:
        logger.error(f"Failed to concat dataframes: {e}")
        return pd.DataFrame()

# -------------------------
# Startup: load model + data
# -------------------------
model = None
FEATURES = None
data_cache = None

@app.on_event("startup")
def startup():
    global model, FEATURES, data_cache
    logger.info("Startup: loading model and dataset...")

    # load model (local preferred, then GCS)
    try:
        ensure_dir(os.path.dirname(LOCAL_MODEL_PATH) or ".")
        if not os.path.exists(LOCAL_MODEL_PATH):
            try:
                # attempt to download model from GCS blob path into local model path
                download_blob(BUCKET_NAME, MODEL_BLOB, LOCAL_MODEL_PATH)
            except Exception as e:
                logger.warning(f"Model not found locally and GCS download failed (ignored): {e}")
        model = joblib.load(LOCAL_MODEL_PATH)
        FEATURES = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        logger.info(f"Loaded model; expects features: {FEATURES}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        FEATURES = None

    # load data preferring local DVC-pulled files
    try:
        data_cache = load_data_local_first()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        data_cache = pd.DataFrame()

# -------------------------
# Schemas, health, predict
# -------------------------
class TransactionIDRequest(BaseModel):
    transaction_id: int

class PredictionResponse(BaseModel):
    is_fraud: int
    probability_fraud: float

@app.get("/live_check")
def live():
    return {"status": "alive"}

@app.get("/ready_check")
def ready():
    ready_state = model is not None and data_cache is not None and not data_cache.empty
    return {"status": "ready" if ready_state else "not ready"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionIDRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if data_cache is None or data_cache.empty:
        raise HTTPException(status_code=503, detail="Data not available for lookup")

    # Use Time column first, else index fallback
    txn_rows = pd.DataFrame()
    try:
        txn_rows = data_cache.loc[data_cache["Time"] == req.transaction_id]
    except Exception:
        txn_rows = pd.DataFrame()

    if not txn_rows.empty:
        txn_row = txn_rows.iloc[0]
    else:
        if 0 <= req.transaction_id < len(data_cache):
            txn_row = data_cache.iloc[req.transaction_id]
        else:
            raise HTTPException(status_code=404, detail="Transaction not found (by Time or index)")

    # drop label & sensitive cols
    row = txn_row.drop(labels=[c for c in ["Class", "location", "Time"] if c in txn_row.index], errors="ignore")

    # align features and fill missing with 0
    if FEATURES:
        aligned = []
        missing = []
        for feat in FEATURES:
            if feat in row.index:
                try:
                    aligned.append(float(row[feat]))
                except Exception:
                    aligned.append(0.0)
                    missing.append(feat)
            else:
                aligned.append(0.0)
                missing.append(feat)
        if missing:
            logger.info(f"Missing features filled with 0: {missing}")
        X = np.array(aligned).reshape(1, -1)
    else:
        try:
            vals = row.values.astype(float)
            X = np.array(vals).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to prepare input array: {e}")
            raise HTTPException(status_code=500, detail="Failed to prepare features for prediction")

    # inference
    with tracer.start_as_current_span("model_inference"):
        try:
            proba = float(model.predict_proba(X)[0][1])
            is_fraud = 1 if proba > 0.5 else 0
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise HTTPException(status_code=500, detail="Model inference error")

    logger.info(f"prediction: tx={req.transaction_id} pred={is_fraud} p={proba:.5f}")
    return PredictionResponse(is_fraud=is_fraud, probability_fraud=proba)

# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

