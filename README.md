# HDFS Log Anomaly Detection

## Overview

This project aims to build a machine learning pipeline to detect anomalous blocks in the HDFS (Hadoop Distributed File System) log dataset.

The dataset consists of raw system logs and block-level anomaly labels. The objective is to aggregate log messages per block and train a classifier to identify anomalous behavior.

## Planned Approach

- Parse raw HDFS logs
- Aggregate logs per BlockId
- Merge anomaly labels
- Extract text features
- Train a baseline classification model
- Evaluate using F1 score and PR-AUC

## Project Structure
data/
src/
models/
reports/

## Status

Initial project setup.