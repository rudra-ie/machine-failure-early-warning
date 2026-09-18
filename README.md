# Machine Failure Early-Warning System

An end-to-end predictive maintenance system that analyzes machine sensor data and predicts potential failures before they occur.

The project combines classical machine learning, a from-scratch NumPy LSTM, feature engineering, risk scoring, explainability, and an interactive web dashboard.

The objective of this project is to predict whether a machine is likely to fail before the actual failure occurs. The system analyzes sensor and operational data such as temperature, rotational speed, torque, and other machine parameters and generates an early warning so maintenance can be performed proactively.

---

## Project Overview

Industrial machines continuously generate sensor measurements such as:

- Vibration
- Temperature
- Current
- Pressure

Changes in these measurements can indicate gradual equipment degradation.

This project simulates that environment and builds a complete machine-learning pipeline that:

1. Generates realistic synthetic machine sensor data
2. Creates rolling-window and trend-based features
3. Predicts whether a failure may occur within the next 48 hours
4. Compares a Random Forest model with a NumPy LSTM
5. Generates machine-level risk scores
6. Detects early warnings before failures
7. Measures warning lead time
8. Provides feature importance for explainability
9. Displays results through an interactive dashboard

---

# System Architecture

```text
                    ┌──────────────────────┐
                    │   Synthetic Data     │
                    │      Generator       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Sensor Dataset    │
                    │    Vibration         │
                    │    Temperature       │
                    │    Current           │
                    │    Pressure          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Rolling statistics   │
                    │ Differences          │
                    │ Rate of change       │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │  Random Forest   │       │    NumPy LSTM    │
       │     Baseline     │       │ From Scratch     │
       └────────┬─────────┘       └────────┬─────────┘
                │                          │
                └────────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    │ Precision            │
                    │ Recall               │
                    │ F1                   │
                    │ ROC-AUC              │
                    │ False Alarm Rate     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Risk Engine      │
                    │ LSTM probability     │
                    │ Sensor abnormality   │
                    │ Trend score          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Machine Risk Ranking │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Interactive Dashboard│
                    └──────────────────────┘

```
## Dataset

The project uses a synthetically generated predictive-maintenance dataset.

### Dataset Configuration

- 40 simulated machines
- Up to 720 hourly readings per machine
- Approximately 30 days of monitoring
- 4 sensor measurements
- Gradual degradation before failure

### Sensors

| Sensor | Description |
|---|---|
| Vibration | Mechanical vibration level |
| Temperature | Machine operating temperature |
| Current | Electrical load/current |
| Pressure | System pressure |

Some machines remain healthy while others undergo gradual degradation followed by a failure event.

## Results

The models were evaluated using a machine-level train/test split.

| Metric | Random Forest | NumPy LSTM |
|---|---:|---:|
| Precision | 0.8623 | 0.6995 |
| Recall | 0.8264 | 0.7760 |
| F1 Score | 0.8440 | 0.7358 |
| ROC-AUC | 0.9946 | 0.9949 |
| False Alarm Rate | 0.0078 | 0.0116 |

The Random Forest achieved higher precision, recall and F1 score in this experiment, while the NumPy LSTM achieved a slightly higher ROC-AUC.

## Early Warning Analysis

The LSTM predictions were also analyzed as an operational early-warning signal.

- Failures analyzed: 4
- Warnings detected: 4
- Warning detection rate: 100%
- Average warning lead time: 37.25 hours
- Minimum lead time: 26 hours
- Maximum lead time: 51 hours

The lead-time analysis uses the held-out test predictions and a warning threshold of 0.50.

## Limitations

- The dataset is synthetically generated and does not represent a real industrial environment.
- Sensor ranges and risk thresholds are designed for this project demonstration.
- The NumPy LSTM is implemented from scratch for educational and experimental purposes.
- Model performance may differ significantly on real industrial data.
- Further validation would be required before production deployment.