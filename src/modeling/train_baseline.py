import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

from src.config import (
    PATHS,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
)


def ensure_dirs():
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)


def load_train_data() -> pd.DataFrame:
    path = PATHS.processed_dir / "dataset_train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipelines.build_dataset"
        )
    df = pd.read_parquet(path)
    if not {"block_id", "text", "label"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=0.95,
        sublinear_tf=True,
    )


def train():
    ensure_dirs()

    df = load_train_data()
    X_text = df["text"]
    y = df["label"].to_numpy()

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",  # important for ~3% anomalies
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    # quick train metrics (sanity only; real metrics come from evaluate.py on test set)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_rows": int(len(df)),
        "anomaly_rate_train": float(np.mean(y)),
        "f1_train": float(f1_score(y, preds)),
        "precision_train": float(precision_score(y, preds, zero_division=0)),
        "recall_train": float(recall_score(y, preds, zero_division=0)),
        "pr_auc_train": float(average_precision_score(y, probs)),
        "threshold": 0.5,
    }

    params = {
        "vectorizer": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
        "model": {
            "type": "LogisticRegression",
            "max_iter": 300,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        },
    }

    model_path = PATHS.models_dir / "baseline_lr.joblib"
    vec_path = PATHS.models_dir / "tfidf_vectorizer.joblib"
    metrics_path = PATHS.models_dir / "metrics.json"
    params_path = PATHS.models_dir / "params.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    params_path.write_text(json.dumps(params, indent=2))

    print("Saved:")
    print(" -", model_path)
    print(" -", vec_path)
    print(" -", metrics_path)
    print(" -", params_path)
    print("\nTrain metrics (sanity):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()