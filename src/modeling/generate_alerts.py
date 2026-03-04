import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.config import PATHS
from src.features.text_cleaning import normalize_log_text

TEXT_COL = "event_sequence"

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


def load_forward_test() -> pd.DataFrame:
    path = PATHS.processed_dir / "forward_failure_test.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipelines.build_forward_dataset"
        )
    df = pd.read_parquet(path)
    required = {"block_id", "window_start", TEXT_COL}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def compute_ensemble_probabilities(df: pd.DataFrame) -> np.ndarray:
    lr_text = joblib.load(PATHS.models_dir / "forward_lr_text.joblib")
    vec = joblib.load(PATHS.models_dir / "forward_tfidf_eventseq.joblib")
    rf_num = joblib.load(PATHS.models_dir / "forward_rf_numeric.joblib")

    params = json.loads((PATHS.models_dir / "forward_ensemble_params.json").read_text())
    w_text = float(params["ensemble_weights"]["text_lr"])
    w_num = float(params["ensemble_weights"]["num_rf"])

    X_text = vec.transform(df[TEXT_COL].fillna("").astype(str).map(normalize_log_text))
    p_text = lr_text.predict_proba(X_text)[:, 1]

    X_num = df[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_num = rf_num.predict_proba(X_num)[:, 1]

    return w_text * p_text + w_num * p_num


def generate_alerts(top_k: int = 50):
    ensure_dirs()
    df = load_forward_test()

    df["window_start"] = pd.to_datetime(df["window_start"])

    probs = compute_ensemble_probabilities(df)
    df = df.copy()
    df["failure_probability"] = probs

    # Alert per entity (block): take max probability across its windows
    per_block = (
        df.groupby("block_id")
        .agg(
            max_failure_probability=("failure_probability", "max"),
            first_seen=("window_start", "min"),
            last_seen=("window_start", "max"),
            total_windows=("window_start", "size"),
        )
        .reset_index()
        .sort_values("max_failure_probability", ascending=False)
    )

    top = per_block.head(top_k).copy()

    # ✅ Make JSON serializable
    top["first_seen"] = pd.to_datetime(top["first_seen"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    top["last_seen"] = pd.to_datetime(top["last_seen"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    top["max_failure_probability"] = top["max_failure_probability"].astype(float)
    top["total_windows"] = top["total_windows"].astype(int)

    csv_path = PATHS.reports_dir / "alerts_topk.csv"
    json_path = PATHS.reports_dir / "alerts_topk.json"

    top.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "top_k": int(top_k),
        "alerts": top.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print("Saved alerts:")
    print(" -", csv_path)
    print(" -", json_path)
    print("\nTop 10 alerts:")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    generate_alerts(top_k=50)