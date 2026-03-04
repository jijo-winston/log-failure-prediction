from pathlib import Path
import os

import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import PATHS
from src.features.text_cleaning import normalize_log_text
from src.serving.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="HDFS Log Anomaly Detection API",
    version="1.0.0",
    description="Inference API for TF-IDF + Logistic Regression anomaly detector.",
)

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer

    model_path = PATHS.models_dir / "baseline_lr.joblib"
    vec_path = PATHS.models_dir / "tfidf_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts.\n"
            f"Expected:\n - {model_path}\n - {vec_path}\n"
            f"Run: python -m src.modeling.train_baseline"
        )

    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vec_path)


@app.on_event("startup")
def startup_event():
    _load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _vectorizer is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    cleaned = normalize_log_text(req.text)
    X = _vectorizer.transform([cleaned])

    prob = float(_model.predict_proba(X)[0, 1])
    pred = int(prob >= req.threshold)

    return PredictResponse(prob_anomaly=prob, pred_label=pred, threshold=req.threshold)