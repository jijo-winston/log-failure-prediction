import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import PATHS


def load_model_and_vectorizer():
    model_path = PATHS.models_dir / "baseline_lr.joblib"
    vec_path = PATHS.models_dir / "tfidf_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Model/vectorizer not found. Run: python -m src.modeling.train_baseline"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def load_dataset(split: str) -> pd.DataFrame:
    if split not in {"train", "test", "full"}:
        raise ValueError("split must be one of: train, test, full")

    name = {
        "train": "dataset_train.parquet",
        "test": "dataset_test.parquet",
        "full": "dataset_full.parquet",
    }[split]

    path = PATHS.processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.pipelines.build_dataset")

    df = pd.read_parquet(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["block_id"] = df["block_id"].astype(str)
    return df


def top_global_features(model, vectorizer, top_n: int = 30):
    """
    Prints the tokens with the largest positive weights (most indicative of anomaly)
    and largest negative weights (most indicative of normal).
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_.ravel()

    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]

    print("\n=== Top Anomaly-Indicating Features (Global) ===")
    for i in top_pos_idx:
        print(f"{feature_names[i]:40s}  weight={coef[i]:.4f}")

    print("\n=== Top Normal-Indicating Features (Global) ===")
    for i in top_neg_idx:
        print(f"{feature_names[i]:40s}  weight={coef[i]:.4f}")


def explain_block(block_id: str, split: str = "test", top_n: int = 20, save_csv: bool = False):
    model, vectorizer = load_model_and_vectorizer()
    df = load_dataset(split)

    row = df[df["block_id"] == str(block_id)]
    if row.empty:
        raise ValueError(f"block_id {block_id} not found in {split} dataset")

    text = row.iloc[0]["text"]
    label = int(row.iloc[0]["label"])

    X = vectorizer.transform([text])  # 1 x V sparse

    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)

    print("\n=== Block Explanation ===")
    print(f"block_id      : {block_id}")
    print(f"true_label    : {label} (1=Anomaly,0=Normal)")
    print(f"pred_label    : {pred}")
    print(f"prob_anomaly  : {prob:.6f}")

    # contribution per token for linear model = tfidf_value * coefficient
    coef = model.coef_.ravel()
    feature_names = np.array(vectorizer.get_feature_names_out())

    # X is sparse; get non-zero entries only
    x_csr: csr_matrix = X.tocsr()
    idx = x_csr.indices
    vals = x_csr.data

    contrib = vals * coef[idx]
    order = np.argsort(contrib)[::-1]  # descending

    top_idx = idx[order][:top_n]
    top_contrib = contrib[order][:top_n]
    top_vals = vals[order][:top_n]

    print("\nTop contributing features (towards anomaly):")
    rows = []
    for f_idx, c, v in zip(top_idx, top_contrib, top_vals):
        rows.append((feature_names[f_idx], float(v), float(coef[f_idx]), float(c)))
        print(f"{feature_names[f_idx]:40s}  tfidf={v:.5f}  w={coef[f_idx]:.4f}  contrib={c:.5f}")

    if save_csv:
        PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = PATHS.reports_dir / f"explain_{split}_{block_id}.csv"
        pd.DataFrame(rows, columns=["feature", "tfidf", "weight", "contribution"]).to_csv(out_path, index=False)
        print("\nSaved explanation CSV:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="show_global", action="store_true", help="Print global top features")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--split", choices=["train", "test", "full"], default="test")
    ap.add_argument("--block-id", type=str, default=None, help="Explain a specific block_id")
    ap.add_argument("--save-csv", action="store_true", help="Save per-block explanation as CSV to reports/")
    args = ap.parse_args()

    model, vectorizer = load_model_and_vectorizer()

    if args.show_global:
        top_global_features(model, vectorizer, top_n=args.top_n)

    if args.block_id:
        explain_block(args.block_id, split=args.split, top_n=min(args.top_n, 200), save_csv=args.save_csv)

    if not args.show_global and not args.block_id:
        print("Nothing to do. Use --global and/or --block-id <id>.")


if __name__ == "__main__":
    main()