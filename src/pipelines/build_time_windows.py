import json
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.config import PATHS
from src.features.time_windows import parse_hdfs_line


def ensure_dirs():
    PATHS.interim_dir.mkdir(parents=True, exist_ok=True)


def build_windowed_sequences(window_minutes: int = 5, debug_limit: int | None = None):
    """
    Builds time-windowed behavioral sequences per (block_id, window_start).

    Output: data/interim/windowed_sequences.parquet
    Columns:
      - block_id
      - window_start
      - first_ts, last_ts
      - event_count, error_count, warn_count, info_count
      - event_sequence  (behavioral sequence string)
    """
    ensure_dirs()

    raw_log_path = PATHS.raw_dir / "HDFS.log"
    if not raw_log_path.exists():
        raise FileNotFoundError(f"Missing raw log file: {raw_log_path}")

    out_path = PATHS.interim_dir / "windowed_sequences.parquet"
    meta_path = PATHS.interim_dir / "windowed_sequences_meta.json"

    print(f"Reading raw log: {raw_log_path}")
    rows = []

    with raw_log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(tqdm(f, desc="Parsing windowed events")):
            parsed = parse_hdfs_line(line)
            if parsed is None:
                continue

            ts, block_id, level, component, msg = parsed

            # Event token (behavioral event type)
            # Keep it simple and stable: LEVEL + component
            event_type = f"{level}:{component}"

            rows.append((ts, block_id, level, event_type))

            if debug_limit is not None and i >= debug_limit:
                break

    if not rows:
        raise RuntimeError("No parsable lines with block_id found. Check log format and dataset.")

    df = pd.DataFrame(rows, columns=["ts", "block_id", "level", "event_type"])

    # floor timestamps to window
    df["window_start"] = df["ts"].dt.floor(f"{window_minutes}min")

    # aggregate into behavioral sequences per (block_id, window_start)
    # keep ordering by time for realistic "sequence"
    df = df.sort_values(["block_id", "ts"])

    grouped = df.groupby(["block_id", "window_start"])

    agg = grouped.agg(
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        event_count=("event_type", "size"),
        error_count=("level", lambda s: int((s == "ERROR").sum())),
        warn_count=("level", lambda s: int((s == "WARN").sum())),
        info_count=("level", lambda s: int((s == "INFO").sum())),
        event_sequence=("event_type", lambda s: " ".join(s.tolist())),
    ).reset_index()

    agg.to_parquet(out_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_file": str(raw_log_path),
        "window_minutes": window_minutes,
        "rows_events": int(len(df)),
        "rows_windows": int(len(agg)),
        "unique_blocks": int(df["block_id"].nunique()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved windowed sequences → {out_path}")
    print("Meta →", meta_path)
    print("Stats:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    # Default: 5-minute windows
    build_windowed_sequences(window_minutes=5, debug_limit=None)