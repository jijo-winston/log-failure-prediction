import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

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
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_probability_distribution,
)


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


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Find threshold that maximizes F1 using the Precision-Recall curve.
    Returns:
        best_threshold, best_f1
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision/recall have length = len(thresholds)+1
    # compute f1 for each point; ignore last point (no threshold)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)

    # thresholds correspond to f1_scores[0:len(thresholds)]
    f1_scores_for_thresholds = f1_scores[:-1]
    best_idx = int(np.argmax(f1_scores_for_thresholds))

    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores_for_thresholds[best_idx])

    return best_threshold, best_f1


def evaluate(use_best_threshold: bool = True, threshold: float = 0.5):
    """
    Evaluate on test set. If use_best_threshold=True, pick threshold that maximizes F1.
    Otherwise use provided threshold.
    """
    ensure_dirs()

    df = load_test_data()
    y_true = df["label"].to_numpy()

    model, vectorizer = load_model_and_vectorizer()
    X = vectorizer.transform(df["text"])

    y_prob = model.predict_proba(X)[:, 1]
    pr_auc = float(average_precision_score(y_true, y_prob))

    if use_best_threshold:
        threshold, best_f1 = find_best_threshold(y_true, y_prob)
        print(f"\nOptimal threshold (F1-max): {threshold:.6f}")
        print(f"Best achievable F1 at this threshold: {best_f1:.6f}")

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_rows": int(len(df)),
        "anomaly_rate_test": float(np.mean(y_true)),
        "threshold": float(threshold),
        "threshold_strategy": "f1_max" if use_best_threshold else "fixed_0.5",
        "f1_test": float(f1_score(y_true, y_pred)),
        "precision_test": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_test": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc_test": float(pr_auc),
    }

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Confusion Matrix (Test) ===")
    print(cm)

    # Save metrics JSON
    metrics_path = PATHS.reports_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Curves / plots
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    cm_path = PATHS.figures_dir / "confusion_matrix.png"
    pr_path = PATHS.figures_dir / "pr_curve.png"
    dist_path = PATHS.figures_dir / "probability_distribution.png"

    plot_confusion_matrix(cm, cm_path)
    plot_pr_curve(precision_curve, recall_curve, pr_auc, pr_path)
    plot_probability_distribution(y_prob, y_true, dist_path)

    print("\nSaved:")
    print(" -", metrics_path)
    print(" -", cm_path)
    print(" -", pr_path)
    print(" -", dist_path)

    # Optional: show a few top-scoring anomalies
    out = df[["block_id", "label"]].copy()
    out["prob_anomaly"] = y_prob
    top = out.sort_values("prob_anomaly", ascending=False).head(10)
    print("\nTop 10 highest anomaly probabilities (test):")
    print(top.to_string(index=False))


if __name__ == "__main__":
    # default: imbalance-aware threshold tuning enabled
    evaluate(use_best_threshold=True, threshold=0.5)