import json
from datetime import datetime, timezone

import pandas as pd

from src.config import PATHS
from src.features.forward_labeling import apply_forward_failure_labeling


def build_forward_dataset(horizon: int = 3):
    """
    Build a forward-horizon failure prediction dataset from windowed features.

    - Loads windowed_features.parquet (Commit 15 output)
    - Loads anomaly_label.csv (HDFS labels; may contain Normal/Anomaly strings)
    - Merges labels by block_id
    - Creates forward_failure label:
        forward_failure(t)=1 if anomaly occurs within next `horizon` windows for same block.
    - Saves data/processed/forward_failure_dataset.parquet + meta json
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
    # Common formats: "Normal"/"Anomaly", "Normal"/"Abnormal", 0/1, -1/1, etc.
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

    # Coerce anything else to numeric if possible
    df_labels["label"] = pd.to_numeric(df_labels["label"], errors="coerce")

    # Some datasets use -1 for anomaly
    df_labels["label"] = df_labels["label"].replace({-1: 1})

    # Validate label conversion
    if df_labels["label"].isna().any():
        bad_vals = df_labels[df_labels["label"].isna()].head(10)
        raise ValueError(
            "Found non-convertible label values in anomaly_label.csv. "
            f"Sample bad rows:\n{bad_vals}"
        )

    df_labels["label"] = df_labels["label"].astype(int)

    # Keep only needed columns for merge
    df_labels = df_labels[["block_id", "label"]]

    df = df_feat.merge(df_labels, on="block_id", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    if "window_start" not in df.columns:
        raise ValueError(
            "window_start missing from features. Ensure you built time windows (Commit 14) "
            "and temporal features (Commit 15)."
        )

    print("Applying forward failure labeling...")
    df = apply_forward_failure_labeling(df, horizon=horizon)

    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = PATHS.processed_dir / "forward_failure_dataset.parquet"
    df.to_parquet(output_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": int(len(df)),
        "horizon_windows": horizon,
        "positive_rate_anomaly": float(df["label"].mean()),
        "positive_rate_forward_failure": float(df["forward_failure"].mean()),
        "label_source": str(labels_path),
        "features_source": str(features_path),
    }

    meta_path = PATHS.processed_dir / "forward_failure_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved forward dataset →", output_path)
    print("Meta →", meta_path)
    print("Stats:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    build_forward_dataset(horizon=3)