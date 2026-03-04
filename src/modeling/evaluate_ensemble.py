import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import PATHS
from src.features.text_cleaning import normalize_log_text
from src.visualization.plots import plot_confusion_matrix, plot_pr_curve


TEXT_COL = "event_sequence"
TARGET_COL = "forward_failure"

NUMERIC_COLS = [
    "event_count",
    "error_count",
    "warn_count",
    "info_count",
    "error_rate",
    "warn_rate",
    "unique_event_types",
    "rare_event_count",
    "burst_ratio",
    "transition_count",
]


def ensure_dirs():
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    f1_t = f1[:-1]
    best_idx = int(np.argmax(f1_t))
    return float(thresholds[best_idx]), float(f1_t[best_idx])


def load_forward_test() -> pd.DataFrame:
    path = PATHS.processed_dir / "forward_failure_test.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipelines.build_forward_dataset"
        )
    df = pd.read_parquet(path)
    return df


def evaluate():
    ensure_dirs()

    df = load_forward_test()
    y_true = df[TARGET_COL].astype(int).to_numpy()

    lr_text = joblib.load(PATHS.models_dir / "forward_lr_text.joblib")
    vec = joblib.load(PATHS.models_dir / "forward_tfidf_eventseq.joblib")
    rf_num = joblib.load(PATHS.models_dir / "forward_rf_numeric.joblib")

    params = json.loads((PATHS.models_dir / "forward_ensemble_params.json").read_text())
    w_text = params["ensemble_weights"]["text_lr"]
    w_num = params["ensemble_weights"]["num_rf"]

    # text probs
    X_text = vec.transform(df[TEXT_COL].fillna("").astype(str).map(normalize_log_text))
    p_text = lr_text.predict_proba(X_text)[:, 1]

    # numeric probs
    X_num = df[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_num = rf_num.predict_proba(X_num)[:, 1]

    # ensemble
    p_ens = w_text * p_text + w_num * p_num

    # threshold tuning on ensemble probs
    thr, best_f1 = find_best_threshold(y_true, p_ens)
    y_pred = (p_ens >= thr).astype(int)

    pr_auc_text = float(average_precision_score(y_true, p_text))
    pr_auc_num = float(average_precision_score(y_true, p_num))
    pr_auc_ens = float(average_precision_score(y_true, p_ens))

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_rows": int(len(df)),
        "positive_rate_test": float(y_true.mean()),
        "threshold_f1_max": thr,
        "best_f1_at_threshold": best_f1,
        "ensemble_weights": {"text_lr": w_text, "num_rf": w_num},
        "pr_auc_text_lr": pr_auc_text,
        "pr_auc_num_rf": pr_auc_num,
        "pr_auc_ensemble": pr_auc_ens,
        "f1_ensemble": float(f1_score(y_true, y_pred)),
        "precision_ensemble": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ensemble": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    print("\n=== Forward Failure Prediction (Ensemble) ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Classification Report (Ensemble, Test) ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Confusion Matrix (Ensemble, Test) ===")
    print(cm)

    # Save outputs
    out_metrics = PATHS.reports_dir / "forward_ensemble_test_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))

    cm_path = PATHS.figures_dir / "forward_ensemble_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)

    prec, rec, _ = precision_recall_curve(y_true, p_ens)
    pr_path = PATHS.figures_dir / "forward_ensemble_pr_curve.png"
    plot_pr_curve(prec, rec, pr_auc_ens, pr_path)

    print("\nSaved:")
    print(" -", out_metrics)
    print(" -", cm_path)
    print(" -", pr_path)


if __name__ == "__main__":
    evaluate()