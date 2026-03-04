import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from src.config import PATHS, RANDOM_SEED
from src.features.text_cleaning import normalize_log_text


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
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)


def load_forward_train() -> pd.DataFrame:
    path = PATHS.processed_dir / "forward_failure_train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipelines.build_forward_dataset"
        )
    df = pd.read_parquet(path)
    for c in ["block_id", "window_start", TEXT_COL, TARGET_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    return df


def build_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def train():
    ensure_dirs()
    df = load_forward_train()

    y = df[TARGET_COL].astype(int).to_numpy()

    # -------- Text model (TF-IDF over event_sequence) --------
    text_series = df[TEXT_COL].fillna("").astype(str).map(normalize_log_text)

    vectorizer = build_text_vectorizer()
    X_text = vectorizer.fit_transform(text_series)

    lr_text = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    lr_text.fit(X_text, y)

    # -------- Numeric model (temporal/stat features) --------
    X_num = df[NUMERIC_COLS].copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rf_num = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    rf_num.fit(X_num, y)

    # Simple sanity metric on train (PR-AUC)
    p_text = lr_text.predict_proba(X_text)[:, 1]
    p_num = rf_num.predict_proba(X_num)[:, 1]
    p_ens = 0.6 * p_text + 0.4 * p_num

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_rows": int(len(df)),
        "positive_rate_train": float(y.mean()),
        "pr_auc_train_text_lr": float(average_precision_score(y, p_text)),
        "pr_auc_train_num_rf": float(average_precision_score(y, p_num)),
        "pr_auc_train_ensemble": float(average_precision_score(y, p_ens)),
        "ensemble_weights": {"text_lr": 0.6, "num_rf": 0.4},
        "target": TARGET_COL,
        "text_col": TEXT_COL,
        "numeric_cols": NUMERIC_COLS,
    }

    # Save artifacts
    joblib.dump(lr_text, PATHS.models_dir / "forward_lr_text.joblib")
    joblib.dump(vectorizer, PATHS.models_dir / "forward_tfidf_eventseq.joblib")
    joblib.dump(rf_num, PATHS.models_dir / "forward_rf_numeric.joblib")

    (PATHS.models_dir / "forward_ensemble_params.json").write_text(
        json.dumps(
            {
                "ensemble_weights": metrics["ensemble_weights"],
                "target": TARGET_COL,
                "text_col": TEXT_COL,
                "numeric_cols": NUMERIC_COLS,
            },
            indent=2,
        )
    )
    (PATHS.models_dir / "forward_ensemble_train_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("Saved:")
    print(" -", PATHS.models_dir / "forward_lr_text.joblib")
    print(" -", PATHS.models_dir / "forward_tfidf_eventseq.joblib")
    print(" -", PATHS.models_dir / "forward_rf_numeric.joblib")
    print(" -", PATHS.models_dir / "forward_ensemble_params.json")
    print(" -", PATHS.models_dir / "forward_ensemble_train_metrics.json")
    print("\nTrain metrics (sanity):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()