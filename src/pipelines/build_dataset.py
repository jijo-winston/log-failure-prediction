import json
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS, RANDOM_SEED


def ensure_dirs():
    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)


def load_blocks_text() -> pd.DataFrame:
    path = PATHS.interim_dir / "blocks_text.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.parse_hdfs"
        )
    df = pd.read_parquet(path)
    if not {"block_id", "text"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    return df


def load_labels() -> pd.DataFrame:
    path = PATHS.anomaly_labels
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Expected anomaly_label.csv in data/raw/")
    labels = pd.read_csv(path)

    # LogHub HDFS labels typically: BlockId, Label (Normal/Anomaly)
    if not {"BlockId", "Label"}.issubset(labels.columns):
        raise ValueError(f"Expected columns ['BlockId','Label'] in {path}, got {labels.columns.tolist()}")

    labels = labels.rename(columns={"BlockId": "block_id", "Label": "label_str"})
    labels["block_id"] = labels["block_id"].astype(str)

    # Normalize to 0/1
    labels["label"] = labels["label_str"].map({"Normal": 0, "Anomaly": 1})
    if labels["label"].isna().any():
        bad = labels[labels["label"].isna()]["label_str"].value_counts().head(5)
        raise ValueError(f"Unexpected label values found: {bad.to_dict()}")

    return labels[["block_id", "label"]]


def build_full_dataset() -> pd.DataFrame:
    blocks = load_blocks_text()
    labels = load_labels()

    df = blocks.merge(labels, on="block_id", how="inner")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    # Basic sanity checks
    if df.empty:
        raise RuntimeError("Merged dataset is empty. Check block_id formats and label file.")
    if df["label"].nunique() < 2:
        raise RuntimeError("Only one class present after merge. Something is wrong with labels/merge.")

    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2):
    y = df["label"].to_numpy()

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    full_path = PATHS.processed_dir / "dataset_full.parquet"
    train_path = PATHS.processed_dir / "dataset_train.parquet"
    test_path = PATHS.processed_dir / "dataset_test.parquet"

    df.to_parquet(full_path, index=False)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows_full": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "anomaly_rate_full": float(df["label"].mean()),
        "anomaly_rate_train": float(train_df["label"].mean()),
        "anomaly_rate_test": float(test_df["label"].mean()),
    }

    meta_path = PATHS.processed_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(stats, indent=2))

    print("Saved:")
    print(" -", full_path)
    print(" -", train_path)
    print(" -", test_path)
    print(" -", meta_path)
    print("\nDataset stats:", json.dumps(stats, indent=2))


def main():
    ensure_dirs()
    df = build_full_dataset()
    split_and_save(df, test_size=0.2)


if __name__ == "__main__":
    main()