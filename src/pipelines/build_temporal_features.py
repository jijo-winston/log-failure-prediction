import json
from datetime import datetime, timezone

import pandas as pd

from src.config import PATHS
from src.features.temporal_features import compute_temporal_features


def build_temporal_features():

    input_path = PATHS.interim_dir / "windowed_sequences.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run: python -m src.pipelines.build_time_windows"
        )

    print("Loading windowed sequences...")

    df = pd.read_parquet(input_path)

    print("Computing temporal features...")

    df_feat = compute_temporal_features(df)

    output_path = PATHS.interim_dir / "windowed_features.parquet"
    df_feat.to_parquet(output_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_rows": int(len(df)),
        "output_rows": int(len(df_feat)),
        "features": [
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
        ],
    }

    meta_path = PATHS.interim_dir / "windowed_features_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved features →", output_path)
    print("Meta →", meta_path)
    print("Stats:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    build_temporal_features()