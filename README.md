# Machine Failure Early-Warning System

An end-to-end predictive maintenance system that analyzes machine sensor data and predicts potential failures before they occur.

The project combines classical machine learning, a from-scratch NumPy LSTM, feature engineering, risk scoring, explainability, and an interactive web dashboard.

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