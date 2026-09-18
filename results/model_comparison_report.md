# Machine Failure Early-Warning Model Comparison

This report compares the Random Forest baseline with the NumPy LSTM model for the synthetic machine failure early-warning dataset.

## Dataset and Evaluation Setup

- Random Forest training machines: 30
- Random Forest testing machines: 10
- LSTM training machines: 30
- LSTM testing machines: 10
- LSTM sequence length: 24 hours

## Model Metrics

| Metric | Random Forest | NumPy LSTM |
|---|---:|---:|
| Precision | 0.8623 | 0.6995 |
| Recall | 0.8264 | 0.7760 |
| F1 | 0.8440 | 0.7358 |
| Roc Auc | 0.9946 | 0.9949 |
| False Alarm Rate | 0.0078 | 0.0116 |

## Confusion Matrices

### Random Forest

- True Negative: 4847
- False Positive: 38
- False Negative: 50
- True Positive: 238

### NumPy LSTM

- True Negative: 5461
- False Positive: 64
- False Negative: 43
- True Positive: 149

## Interpretation

The Random Forest provides a non-sequential baseline using engineered sensor features, while the NumPy LSTM processes sequences of sensor observations and can capture temporal patterns in machine degradation.

The reported metrics should be interpreted in the context of the synthetic dataset and its controlled degradation patterns. They should not be treated as real-world predictive-maintenance performance.
