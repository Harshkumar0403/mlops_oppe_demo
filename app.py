import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import logging
import json
import sys

# -------------------------------
# CONFIG
# -------------------------------
BUCKET_NAME = "data_mlops_oppe2prep"
MODEL_BLOB = "production_models/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

# -------------------------------
# OpenTelemetry Setup
# -------------------------------
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())   # sends to OTLP endpoint
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# -------------------------------
# Logging
# -------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"severity": record.levelname, "message": record.getMessage()})


logger = logging.getLogger("fraud-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# -------------------------------
# FastAPI
# -------------------------------
app = FastAPI(title="Fraud Detection API")
FastAPIInstrumentor.instrument_app(app)

# -------------------------------
# Request Schema
# -------------------------------
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# -------------------------------
# MODEL LOADING
# -------------------------------
def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Model already exists locally. Skipping download.")
        return

    logger.info("Downloading model from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    logger.info("Model downloaded successfully.")


def load_model():
    download_model()
    logger.info("Loading model.pkl locally...")
    model = joblib.load(LOCAL_MODEL_PATH)
    logger.info("Model loaded.")
    return model


model = load_model()


# -------------------------------
# HEALTH CHECKS
# -------------------------------
@app.get("/live_check")
def live():
    return {"status": "alive"}


@app.get("/ready_check")
def ready():
    if model is None:
        return {"status": "not ready"}
    return {"status": "ready"}


# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = np.array([list(transaction.dict().values())]).reshape(1, -1)

        # --- OTEL custom tracing span for prediction ---
        with tracer.start_as_current_span("model_prediction"):
            pred_proba = model.predict_proba(data)[0][1]
            pred_class = 1 if pred_proba > 0.5 else 0

        return {
            "prediction": pred_class,
            "fraud_probability": float(pred_proba)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Model inference error")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
