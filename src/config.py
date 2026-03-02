from dataclasses import dataclass
from pathlib import Path

# Project root = folder that contains data/, src/, models/, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    # Data folders
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"

    # Key input files
    hdfs_log: Path = raw_dir / "HDFS.log"
    anomaly_labels: Path = raw_dir / "anomaly_label.csv"

    # Outputs
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = reports_dir / "figures"
    run_logs_dir: Path = reports_dir / "run_logs"

PATHS = Paths()

# Reproducibility
RANDOM_SEED = 42

# Dataset build settings
BLOCK_ID_REGEX = r"(blk_-?\d+)"
MIN_LINES_PER_BLOCK = 1          # keep all blocks by default
MAX_BLOCKS_DEBUG = None          # set to e.g. 5000 for quick dev runs

# Modeling defaults
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2