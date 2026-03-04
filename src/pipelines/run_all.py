"""
End-to-End Training Pipeline

Runs the full anomaly detection workflow:

1. Parse raw HDFS logs
2. Build dataset
3. Train baseline model
4. Evaluate model
5. Generate explainability output

Usage:
python -m src.pipelines.run_all
"""

import subprocess
import sys


def run_step(module_name: str):
    """Run a python module as a subprocess."""
    print(f"\n========== Running: {module_name} ==========\n")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {module_name}")


def main():

    steps = [
        "src.data.parse_hdfs",
        "src.pipelines.build_time_windows",
        "src.pipelines.build_dataset",
        "src.modeling.train_baseline",
        "src.modeling.evaluate",
    ]

    for step in steps:
        run_step(step)

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()