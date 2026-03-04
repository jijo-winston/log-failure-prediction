import subprocess
import sys


def run_step(module_name: str):
    print(f"\n========== Running: {module_name} ==========\n")
    result = subprocess.run([sys.executable, "-m", module_name], text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {module_name}")


def main():
    steps = [
        "src.data.parse_hdfs",
        "src.pipelines.build_time_windows",
        "src.pipelines.build_temporal_features",
        "src.pipelines.build_forward_dataset",
        "src.modeling.train_ensemble",
        "src.modeling.evaluate_ensemble",
    ]
    for s in steps:
        run_step(s)

    print("\n✅ Forward prediction pipeline completed successfully.")


if __name__ == "__main__":
    main()