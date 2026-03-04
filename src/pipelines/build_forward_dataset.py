import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS, RANDOM_SEED
from src.features.forward_labeling import apply_forward_failure_labeling


def build_forward_dataset(horizon: int = 3, test_size: float = 0.2):
    """
    Builds forward-horizon early warning dataset from windowed temporal features.

    Inputs:
      - data/interim/windowed_features.parquet
      - data/raw/anomaly_label.csv (supports Normal/Anomaly strings)

    Outputs (saved under data/processed/):
      - forward_failure_full.parquet
      - forward_failure_train.parquet
      - forward_failure_test.parquet
      - forward_failure_meta.json

    Labels:
      - label: anomaly at block level (0/1)
      - forward_failure: 1 if an anomaly occurs within the next `horizon` windows for the same block
    """
    features_path = PATHS.interim_dir / "windowed_features.parquet"
    labels_path = PATHS.raw_dir / "anomaly_label.csv"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} missing. Run: python -m src.pipelines.build_temporal_features"
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} missing. Ensure anomaly_label.csv exists in data/raw/"
        )

    print("Loading windowed features...")
    df_feat = pd.read_parquet(features_path)
    if "block_id" not in df_feat.columns:
        raise ValueError(f"windowed_features missing block_id. Columns: {df_feat.columns.tolist()}")
    df_feat["block_id"] = df_feat["block_id"].astype(str)

    print("Loading anomaly labels...")
    df_labels = pd.read_csv(labels_path)

    # Normalize label file schema to: block_id, label
    rename_map = {}
    if "BlockId" in df_labels.columns:
        rename_map["BlockId"] = "block_id"
    elif "block_id" in df_labels.columns:
        rename_map["block_id"] = "block_id"

    if "Label" in df_labels.columns:
        rename_map["Label"] = "label"
    elif "label" in df_labels.columns:
        rename_map["label"] = "label"

    df_labels = df_labels.rename(columns=rename_map)

    if "block_id" not in df_labels.columns:
        raise ValueError(
            f"Expected a block id column in anomaly_label.csv. Found: {df_labels.columns.tolist()}"
        )
    if "label" not in df_labels.columns:
        raise ValueError(
            f"Expected a label column in anomaly_label.csv. Found: {df_labels.columns.tolist()}"
        )

    df_labels["block_id"] = df_labels["block_id"].astype(str)

    # Normalize label values to {0,1}
    if df_labels["label"].dtype == object:
        df_labels["label"] = (
            df_labels["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "normal": 0,
                "anomaly": 1,
                "abnormal": 1,
                "anomalous": 1,
                "failure": 1,
                "fail": 1,
                "ok": 0,
            })
        )

    df_labels["label"] = pd.to_numeric(df_labels["label"], errors="coerce")
    df_labels["label"] = df_labels["label"].replace({-1: 1})

    if df_labels["label"].isna().any():
        bad_vals = df_labels[df_labels["label"].isna()].head(10)
        raise ValueError(
            "Found non-convertible label values in anomaly_label.csv. "
            f"Sample bad rows:\n{bad_vals}"
        )

    df_labels["label"] = df_labels["label"].astype(int)
    df_labels = df_labels[["block_id", "label"]]

    # Merge block-level label into window-level features
    df = df_feat.merge(df_labels, on="block_id", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    if "window_start" not in df.columns:
        raise ValueError("window_start missing from features. Ensure Commit 14 + 15 ran correctly.")

    print("Applying forward failure labeling...")
    df = apply_forward_failure_labeling(df, horizon=horizon)

    # Split by block_id to avoid leakage across windows
    blocks = df[["block_id", "label"]].drop_duplicates()
    y_blocks = blocks["label"].to_numpy()

    train_blocks, test_blocks = train_test_split(
        blocks["block_id"].to_numpy(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_blocks,
    )

    train_set = df[df["block_id"].isin(set(train_blocks))].copy()
    test_set = df[df["block_id"].isin(set(test_blocks))].copy()

    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)

    full_path = PATHS.processed_dir / "forward_failure_full.parquet"
    train_path = PATHS.processed_dir / "forward_failure_train.parquet"
    test_path = PATHS.processed_dir / "forward_failure_test.parquet"
    meta_path = PATHS.processed_dir / "forward_failure_meta.json"

    df.to_parquet(full_path, index=False)
    train_set.to_parquet(train_path, index=False)
    test_set.to_parquet(test_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows_full": int(len(df)),
        "rows_train": int(len(train_set)),
        "rows_test": int(len(test_set)),
        "horizon_windows": horizon,
        "test_size": test_size,
        "positive_rate_anomaly_full": float(df["label"].mean()),
        "positive_rate_forward_failure_full": float(df["forward_failure"].mean()),
        "positive_rate_forward_failure_train": float(train_set["forward_failure"].mean()),
        "positive_rate_forward_failure_test": float(test_set["forward_failure"].mean()),
        "label_source": str(labels_path),
        "features_source": str(features_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved:")
    print(" -", full_path)
    print(" -", train_path)
    print(" -", test_path)
    print("Meta →", meta_path)
    print("Stats:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    build_forward_dataset(horizon=3, test_size=0.2)