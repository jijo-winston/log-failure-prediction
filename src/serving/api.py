import json
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import PATHS
from src.features.text_cleaning import normalize_log_text
from src.serving.schemas import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    ModelInfoResponse,
)

app = FastAPI(
    title="HDFS Log Anomaly Detection API",
    version="1.0.0",
    description="Inference API for TF-IDF + Logistic Regression anomaly detector.",
)

_model = None
_vectorizer = None
_params: Optional[dict] = None


def _load_artifacts():
    global _model, _vectorizer, _params

    model_path = PATHS.models_dir / "baseline_lr.joblib"
    vec_path = PATHS.models_dir / "tfidf_vectorizer.joblib"
    params_path = PATHS.models_dir / "params.json"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts.\n"
            f"Expected:\n - {model_path}\n - {vec_path}\n"
            f"Run: python -m src.modeling.train_baseline"
        )

    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vec_path)

    # params.json is optional but recommended
    if params_path.exists():
        try:
            _params = json.loads(params_path.read_text())
        except Exception:
            _params = None
    else:
        _params = None


@app.on_event("startup")
def startup_event():
    _load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if _model is None or _vectorizer is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    # vectorizer settings (sklearn stores these)
    vec_cfg = {
        "max_features": getattr(_vectorizer, "max_features", None),
        "ngram_range": getattr(_vectorizer, "ngram_range", None),
        "min_df": getattr(_vectorizer, "min_df", None),
        "max_df": getattr(_vectorizer, "max_df", None),
        "sublinear_tf": getattr(_vectorizer, "sublinear_tf", None),
    }

    recommended_threshold = None
    # If you later save a recommended threshold, you can store it in params.json
    if _params and isinstance(_params, dict):
        recommended_threshold = _params.get("recommended_threshold")

    return ModelInfoResponse(
        model_type=type(_model).__name__,
        vectorizer_type=type(_vectorizer).__name__,
        vectorizer_config=vec_cfg,
        threshold_recommended=recommended_threshold,
        artifacts={
            "model_path": str(PATHS.models_dir / "baseline_lr.joblib"),
            "vectorizer_path": str(PATHS.models_dir / "tfidf_vectorizer.joblib"),
            "params_path": str(PATHS.models_dir / "params.json"),
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _vectorizer is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    cleaned = normalize_log_text(req.text)
    X = _vectorizer.transform([cleaned])

    prob = float(_model.predict_proba(X)[0, 1])
    pred = int(prob >= req.threshold)

    return PredictResponse(prob_anomaly=prob, pred_label=pred, threshold=req.threshold)


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    if _model is None or _vectorizer is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    cleaned_texts = [normalize_log_text(t) for t in req.texts]
    X = _vectorizer.transform(cleaned_texts)

    probs = _model.predict_proba(X)[:, 1]
    preds = (probs >= req.threshold).astype(int)

    results = [
        PredictResponse(prob_anomaly=float(p), pred_label=int(y), threshold=req.threshold)
        for p, y in zip(probs, preds)
    ]

    return PredictBatchResponse(threshold=req.threshold, results=results)