import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)

from src.config import PATHS


def ensure_dirs():
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)


def load_test_data() -> pd.DataFrame:
    path = PATHS.processed_dir / "dataset_test.parquet"
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


def load_model_and_vectorizer():
    model_path = PATHS.models_dir / "baseline_lr.joblib"
    vec_path = PATHS.models_dir / "tfidf_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Model/vectorizer not found. Run: python -m src.modeling.train_baseline"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def save_confusion_matrix(cm: np.ndarray, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])

    # annotate
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve (Test)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate(threshold: float = 0.5):
    ensure_dirs()

    df = load_test_data()
    y_true = df["label"].to_numpy()

    model, vectorizer = load_model_and_vectorizer()
    X = vectorizer.transform(df["text"])

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_rows": int(len(df)),
        "anomaly_rate_test": float(np.mean(y_true)),
        "threshold": float(threshold),
        "f1_test": float(f1_score(y_true, y_pred)),
        "precision_test": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_test": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc_test": float(average_precision_score(y_true, y_prob)),
    }

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Confusion Matrix (Test) ===")
    print(cm)

    # Save artifacts
    metrics_path = PATHS.reports_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    cm_path = PATHS.figures_dir / "confusion_matrix.png"
    pr_path = PATHS.figures_dir / "pr_curve.png"
    save_confusion_matrix(cm, cm_path)
    save_pr_curve(y_true, y_prob, pr_path)

    print("\nSaved:")
    print(" -", metrics_path)
    print(" -", cm_path)
    print(" -", pr_path)

    # Optional: show a few top-scoring anomalies
    out = df[["block_id", "label"]].copy()
    out["prob_anomaly"] = y_prob
    top = out.sort_values("prob_anomaly", ascending=False).head(10)
    print("\nTop 10 highest anomaly probabilities (test):")
    print(top.to_string(index=False))


if __name__ == "__main__":
    evaluate(threshold=0.5)