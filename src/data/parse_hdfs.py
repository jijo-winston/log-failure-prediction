import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.config import PATHS, BLOCK_ID_REGEX


def ensure_dirs():
    PATHS.interim_dir.mkdir(parents=True, exist_ok=True)


def parse_raw_hdfs_log(debug_limit=None):
    """
    Reads HDFS.log and extracts:
        block_id
        log_line

    Saves:
        data/interim/parsed_lines.parquet
    """

    log_path = PATHS.hdfs_log

    if not log_path.exists():
        raise FileNotFoundError(f"HDFS.log not found at {log_path}")

    block_pattern = re.compile(BLOCK_ID_REGEX)

    parsed_rows = []

    print("Parsing HDFS.log...")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(tqdm(f)):
            match = block_pattern.search(line)
            if match:
                block_id = match.group(1)
                parsed_rows.append((block_id, line.strip()))

            if debug_limit and i >= debug_limit:
                break

    df = pd.DataFrame(parsed_rows, columns=["block_id", "log_line"])

    output_path = PATHS.interim_dir / "parsed_lines.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved parsed lines → {output_path}")
    print(f"Total parsed rows: {len(df):,}")

    return df


def aggregate_blocks():
    """
    Aggregates parsed lines into block-level documents.

    Output:
        data/interim/blocks_text.parquet

    Columns:
        block_id
        text
    """

    parsed_path = PATHS.interim_dir / "parsed_lines.parquet"

    if not parsed_path.exists():
        raise FileNotFoundError("Run parse_raw_hdfs_log() first.")

    df = pd.read_parquet(parsed_path)

    print("Aggregating logs per block_id...")

    df_agg = (
        df.groupby("block_id")["log_line"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"log_line": "text"})
    )

    output_path = PATHS.interim_dir / "blocks_text.parquet"
    df_agg.to_parquet(output_path, index=False)

    print(f"Saved block-level dataset → {output_path}")
    print(f"Total unique blocks: {len(df_agg):,}")

    return df_agg


if __name__ == "__main__":
    ensure_dirs()

    # Set debug_limit to e.g. 1_000_000 if you want partial run
    parse_raw_hdfs_log(debug_limit=None)
    aggregate_blocks()