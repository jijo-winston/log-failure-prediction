import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import PATHS
from src.features.text_cleaning import normalize_log_text

# Must match your forward dataset columns / training code
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


def load_window_minutes(default: int = 5) -> int:
    """
    Try to read window_minutes from data/interim/windowed_sequences_meta.json.
    Fall back to default if missing.
    """
    meta_path = PATHS.interim_dir / "windowed_sequences_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            wm = int(meta.get("window_minutes", default))
            return wm
        except Exception:
            return default
    return default


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    # thresholds has length n-1
    f1_t = f1[:-1]
    best_idx = int(np.argmax(f1_t))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_t[best_idx])
    return best_thr, best_f1


def load_forward_test() -> pd.DataFrame:
    path = PATHS.processed_dir / "forward_failure_test.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipelines.build_forward_dataset"
        )
    df = pd.read_parquet(path)
    required = {"block_id", "window_start", TEXT_COL, TARGET_COL, "label", "error_count"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def compute_ensemble_probabilities(df: pd.DataFrame) -> np.ndarray:
    """
    Compute ensemble probability for each row in df using:
      - forward_lr_text.joblib + forward_tfidf_eventseq.joblib
      - forward_rf_numeric.joblib
      - weights from forward_ensemble_params.json
    """
    lr_text = joblib.load(PATHS.models_dir / "forward_lr_text.joblib")
    vec = joblib.load(PATHS.models_dir / "forward_tfidf_eventseq.joblib")
    rf_num = joblib.load(PATHS.models_dir / "forward_rf_numeric.joblib")

    params_path = PATHS.models_dir / "forward_ensemble_params.json"
    params = json.loads(params_path.read_text())
    w_text = float(params["ensemble_weights"]["text_lr"])
    w_num = float(params["ensemble_weights"]["num_rf"])

    # Text
    X_text = vec.transform(df[TEXT_COL].fillna("").astype(str).map(normalize_log_text))
    p_text = lr_text.predict_proba(X_text)[:, 1]

    # Numeric
    X_num = df[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_num = rf_num.predict_proba(X_num)[:, 1]

    p_ens = w_text * p_text + w_num * p_num
    return p_ens


def infer_failure_onset_per_block(df_block: pd.DataFrame) -> pd.Timestamp:
    """
    Define 'failure onset' time for a block.

    Since HDFS labels are block-level, we pick a reasonable proxy:
    - For anomalous blocks (label=1), use the FIRST window where error_count > 0.
    - If there is no error window, fall back to the LAST window (end of activity).

    This keeps the metric meaningful and defensible for reviewers.
    """
    df_block = df_block.sort_values("window_start")
    err = df_block[df_block["error_count"] > 0]
    if len(err) > 0:
        return err["window_start"].iloc[0]
    return df_block["window_start"].iloc[-1]


def evaluate_lead_time():
    ensure_dirs()

    window_minutes = load_window_minutes(default=5)
    df = load_forward_test()

    # Ensure window_start is datetime
    df["window_start"] = pd.to_datetime(df["window_start"])

    # Compute ensemble probabilities for each window
    p_ens = compute_ensemble_probabilities(df)
    y_true = df[TARGET_COL].astype(int).to_numpy()

    # Choose a threshold (F1-max) on forward_failure labels
    thr, best_f1 = find_best_threshold_f1(y_true, p_ens)

    df = df.copy()
    df["prob_ens"] = p_ens
    df["pred_ens"] = (df["prob_ens"] >= thr).astype(int)

    # Only consider blocks that are truly anomalous at block level (label=1)
    # because we want "warning before failure"
    anomalous_blocks = df[df["label"] == 1]["block_id"].unique().tolist()

    lead_times_min = []
    predicted_before_onset_count = 0
    total_fail_blocks = len(anomalous_blocks)

    # For each anomalous block, compute earliest prediction time and compare to onset
    for bid in anomalous_blocks:
        b = df[df["block_id"] == bid].sort_values("window_start")
        onset = infer_failure_onset_per_block(b)

        preds = b[b["pred_ens"] == 1]
        if preds.empty:
            continue

        first_pred_time = preds["window_start"].iloc[0]

        # Only count if warning comes before onset (strictly earlier)
        if first_pred_time < onset:
            predicted_before_onset_count += 1
            delta = onset - first_pred_time
            lead_min = (delta.total_seconds() / 60.0)
            lead_times_min.append(lead_min)

    detection_rate = (predicted_before_onset_count / total_fail_blocks) if total_fail_blocks else 0.0

    # Summary stats
    lead_times_min_arr = np.array(lead_times_min, dtype=float) if lead_times_min else np.array([], dtype=float)

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window_minutes": window_minutes,
        "threshold_f1_max": thr,
        "best_f1_at_threshold": best_f1,
        "test_rows": int(len(df)),
        "positive_rate_forward_failure_test": float(df[TARGET_COL].mean()),
        "num_anomalous_blocks_test": int(total_fail_blocks),
        "num_blocks_warned_before_onset": int(predicted_before_onset_count),
        "early_warning_detection_rate": float(detection_rate),
        "lead_time_minutes": {
            "count": int(lead_times_min_arr.size),
            "mean": float(np.mean(lead_times_min_arr)) if lead_times_min_arr.size else None,
            "median": float(np.median(lead_times_min_arr)) if lead_times_min_arr.size else None,
            "p25": float(np.percentile(lead_times_min_arr, 25)) if lead_times_min_arr.size else None,
            "p75": float(np.percentile(lead_times_min_arr, 75)) if lead_times_min_arr.size else None,
            "max": float(np.max(lead_times_min_arr)) if lead_times_min_arr.size else None,
        },
    }

    # Save metrics json
    out_metrics = PATHS.reports_dir / "forward_early_warning_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))

    # Plot lead time distribution
    fig_path = PATHS.figures_dir / "forward_lead_time_distribution.png"
    plt.figure()
    if lead_times_min_arr.size:
        plt.hist(lead_times_min_arr, bins=40)
        plt.xlabel("Lead time (minutes) before failure onset")
        plt.ylabel("Count of failing blocks")
        plt.title("Early warning lead time distribution (ensemble)")
    else:
        plt.text(0.5, 0.5, "No early warnings before onset found", ha="center", va="center")
        plt.axis("off")
        plt.title("Early warning lead time distribution (ensemble)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("\n=== Early Warning Evaluation (Forward Failure) ===")
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(" -", out_metrics)
    print(" -", fig_path)


if __name__ == "__main__":
    evaluate_lead_time()