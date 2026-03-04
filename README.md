# HDFS Log Anomaly Detection

## Overview

This project builds a machine learning pipeline to detect anomalous HDFS blocks using raw system logs.

The objective is to:
- Parse raw log lines
- Aggregate logs at block level
- Merge anomaly labels
- Train a baseline classifier
- Evaluate performance on unseen data
- Provide interpretability for predictions

The dataset used is the publicly available HDFS log dataset.

---

## Dataset

- ~11M raw log lines
- 575,061 unique block IDs
- ~2.93% anomaly rate
- Stratified train/test split

Block-level aggregation is performed before modeling.

---

## Pipeline

1. **Raw Parsing**
   - Extract `block_id` from each log line
   - Aggregate log messages per block

2. **Dataset Building**
   - Merge anomaly labels
   - Create stratified train/test split
   - Save parquet artifacts

3. **Feature Extraction**
   - TF-IDF (unigram + bigram)
   - max_features = configurable

4. **Model**
   - Logistic Regression
   - `class_weight="balanced"`

5. **Evaluation**
   - F1 Score
   - Precision / Recall
   - PR-AUC
   - Confusion Matrix
   - Precision-Recall curve

---

## Baseline Results (Test Set)

- F1 Score: **0.9495**
- Precision: **0.9043**
- Recall: **0.9994**
- PR-AUC: **0.9840**
- Accuracy: 99.69%

Confusion Matrix:

|            | Pred Normal | Pred Anomaly |
|------------|------------|-------------|
| Normal     | 111,289    | 356         |
| Anomaly    | 2          | 3,366       |

---

## Explainability

The linear model allows:

- Global feature importance (top anomaly-indicating tokens)
- Local per-block explanation (feature contributions)

Example anomaly indicators:
- `unexpected error`
- `not found`
- `redundant`
- `addstoredblock`

---

## How to Run

### 1. Build dataset
python -m src.pipelines.build_dataset

### 2. Train baseline
python -m src.modeling.train_baseline

### 3. Evaluate
python -m src.modeling.evaluate

### 4. Explain predictions
python -m src.modeling.explain --global

### Future Improvements

Log normalization improvements
Template-based log parsing (Drain)
Gradient Boosting / XGBoost comparison
Threshold tuning for optimal F1
Cross-validation
Drift monitoring

### Status

Baseline model implemented and evaluated.
Further feature engineering and model refinement ongoing.