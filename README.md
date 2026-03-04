# Distributed System Failure Prediction from Operational Logs

Machine Learning pipeline for predicting distributed system failures
from operational log streams using temporal feature engineering and
ensemble learning.

This project converts semi-structured system logs into behavioral
sequences and predicts failures **before they occur**, enabling early
warning for system reliability engineering.

------------------------------------------------------------------------

# Overview

Modern distributed systems generate massive volumes of operational
logs.\
Failures are often preceded by subtle behavioral changes such as:

-   bursty error activity
-   unusual event transitions
-   rare system events
-   degraded operational patterns

This project builds a machine learning pipeline that learns these
patterns and predicts failures **ahead of time**.

The system converts raw logs into time-windowed behavioral features and
trains an ensemble model to predict failures within a forward time
horizon.

------------------------------------------------------------------------

# Key Features

• Log parsing for large-scale distributed system logs\
• Time-windowed behavioral sequence modeling\
• Temporal degradation feature engineering\
• Forward-horizon failure prediction\
• Ensemble learning (text + statistical features)\
• Precision--Recall evaluation for imbalanced data\
• Early-warning lead-time analysis\
• Fully reproducible ML pipeline

------------------------------------------------------------------------

# Dataset

Dataset used: **HDFS Log Dataset**

Contains:

-   11,175,629 log events\
-   575,061 unique blocks\
-   Anomaly labels for failing blocks

Source: https://zenodo.org/record/3227177

Files used:

-   `HDFS.log`
-   `anomaly_label.csv`

------------------------------------------------------------------------

# System Architecture

Raw Logs → Log Parsing → Time Window Aggregation → Temporal Feature
Engineering → Forward Failure Labeling → Ensemble Model → Early Warning
Prediction

------------------------------------------------------------------------

# Machine Learning Pipeline

## 1. Log Parsing

Semi‑structured logs are parsed into structured events containing:

-   timestamp
-   block id
-   log message
-   event type

Example:

    081109 203518 INFO dfs.DataNode$BlockReceiver: Receiving block blk_38865049064139660

------------------------------------------------------------------------

## 2. Time Window Behavioral Sequences

Logs are aggregated into **5‑minute windows per block** to capture
behavioral patterns over time.

Example window statistics:

-   event_count
-   error_count
-   warn_count

------------------------------------------------------------------------

## 3. Temporal Feature Engineering

Key degradation indicators:

-   event_count
-   error_rate
-   rare_event_count
-   burst_ratio
-   transition_count
-   unique_event_types

These features capture abnormal system behavior.

------------------------------------------------------------------------

## 4. Forward Failure Prediction

Instead of detecting failures after they occur, the model predicts:

    features at time t → failure within next H windows

Default prediction horizon:

**3 windows (15 minutes)**

------------------------------------------------------------------------

## 5. Ensemble Learning

Two complementary models are trained.

### Text Model

Log message sequences → TF‑IDF → Logistic Regression

### Temporal Model

Statistical behavioral features → Random Forest

### Ensemble

Soft voting combination:

    Ensemble = 0.6 * Text Model + 0.4 * Temporal Model

------------------------------------------------------------------------

# Evaluation Metrics

Because failures are rare, we use:

-   Precision
-   Recall
-   F1 Score
-   PR‑AUC

instead of raw accuracy.

------------------------------------------------------------------------

# Results

Test dataset:

-   390,517 time windows
-   Failure rate: **2.24%**

Model performance:

-   PR‑AUC: **0.265**
-   F1 Score: **0.36**
-   Precision: **0.51**
-   Recall: **0.28**

------------------------------------------------------------------------

# Early Warning Capability

Failing blocks in test set: **3368**\
Failures predicted early: **1107**

Early warning detection rate:

**32.9% of failures predicted before onset**

Lead time statistics:

-   Median warning time: **130 minutes**
-   Mean warning time: **238 minutes**
-   Maximum warning time: **885 minutes**

This demonstrates the system can provide **hours of advance warning
before failures occur**.

------------------------------------------------------------------------

# Project Structure

    log-failure-prediction
    │
    ├── data
    │   ├── raw
    │   ├── interim
    │   └── processed
    │
    ├── models
    ├── reports
    │   └── figures
    │
    ├── src
    │   ├── config.py
    │   ├── data
    │   │   └── parse_hdfs.py
    │   ├── features
    │   │   ├── text_cleaning.py
    │   │   └── forward_labeling.py
    │   ├── modeling
    │   │   ├── train_ensemble.py
    │   │   ├── evaluate_ensemble.py
    │   │   └── evaluate_lead_time.py
    │   ├── pipelines
    │   │   ├── build_time_windows.py
    │   │   ├── build_temporal_features.py
    │   │   ├── build_forward_dataset.py
    │   │   └── run_forward.py
    │   └── visualization
    │       └── plots.py

------------------------------------------------------------------------

# Running the Project

Clone repository:

    git clone https://github.com/jijo-winston/log-failure-prediction.git
    cd log-failure-prediction

Create environment:

    python -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Run full pipeline:

    python -m src.pipelines.run_forward

------------------------------------------------------------------------

# Applications

Potential real‑world uses:

-   Distributed system reliability monitoring
-   Cloud infrastructure failure prediction
-   Operational anomaly detection
-   SRE early warning systems

------------------------------------------------------------------------

# Author

Jijo Winston
GitHub: https://github.com/jijo-winston
