import json
import pandas as pd


BASELINE_PATH = "results/baseline_results.json"
LSTM_PATH = "results/lstm_results.json"
OUTPUT_PATH = "results/model_comparison_report.md"


def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def main():

    print("Loading Random Forest results...")
    baseline = load_results(BASELINE_PATH)

    print("Loading NumPy LSTM results...")
    lstm = load_results(LSTM_PATH)

    # ---------------------------------------------------------
    # Extract metrics
    # ---------------------------------------------------------

    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_alarm_rate"
    ]

    rows = []

    for metric in metrics:

        rows.append({
            "Metric": metric.replace("_", " ").title(),
            "Random Forest": baseline.get(metric),
            "NumPy LSTM": lstm.get(metric)
        })

    comparison = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Create Markdown report
    # ---------------------------------------------------------

    report = []

    report.append("# Machine Failure Early-Warning Model Comparison")
    report.append("")

    report.append(
        "This report compares the Random Forest baseline "
        "with the NumPy LSTM model for the synthetic "
        "machine failure early-warning dataset."
    )

    report.append("")

    report.append("## Dataset and Evaluation Setup")
    report.append("")

    report.append(
        f"- Random Forest training machines: "
        f"{baseline.get('train_machines', 'N/A')}"
    )

    report.append(
        f"- Random Forest testing machines: "
        f"{baseline.get('test_machines', 'N/A')}"
    )

    report.append(
        f"- LSTM training machines: "
        f"{lstm.get('train_machines', 'N/A')}"
    )

    report.append(
        f"- LSTM testing machines: "
        f"{lstm.get('test_machines', 'N/A')}"
    )

    report.append(
        f"- LSTM sequence length: "
        f"{lstm.get('sequence_length', 'N/A')} hours"
    )

    report.append("")

    report.append("## Model Metrics")
    report.append("")

    report.append(
        "| Metric | Random Forest | NumPy LSTM |"
    )
    report.append(
        "|---|---:|---:|"
    )

    for _, row in comparison.iterrows():

        report.append(
            f"| {row['Metric']} | "
            f"{format_metric(row['Random Forest'])} | "
            f"{format_metric(row['NumPy LSTM'])} |"
        )

    report.append("")

    report.append("## Confusion Matrices")
    report.append("")

    rf_cm = baseline.get("confusion_matrix", {})
    lstm_cm = lstm.get("confusion_matrix", {})

    report.append("### Random Forest")
    report.append("")

    report.append(
        f"- True Negative: {rf_cm.get('true_negative', 'N/A')}"
    )

    report.append(
        f"- False Positive: {rf_cm.get('false_positive', 'N/A')}"
    )

    report.append(
        f"- False Negative: {rf_cm.get('false_negative', 'N/A')}"
    )

    report.append(
        f"- True Positive: {rf_cm.get('true_positive', 'N/A')}"
    )

    report.append("")

    report.append("### NumPy LSTM")
    report.append("")

    report.append(
        f"- True Negative: {lstm_cm.get('true_negative', 'N/A')}"
    )

    report.append(
        f"- False Positive: {lstm_cm.get('false_positive', 'N/A')}"
    )

    report.append(
        f"- False Negative: {lstm_cm.get('false_negative', 'N/A')}"
    )

    report.append(
        f"- True Positive: {lstm_cm.get('true_positive', 'N/A')}"
    )

    report.append("")

    report.append("## Interpretation")
    report.append("")

    report.append(
        "The Random Forest provides a non-sequential baseline "
        "using engineered sensor features, while the NumPy LSTM "
        "processes sequences of sensor observations and can "
        "capture temporal patterns in machine degradation."
    )

    report.append("")

    report.append(
        "The reported metrics should be interpreted in the "
        "context of the synthetic dataset and its controlled "
        "degradation patterns. They should not be treated as "
        "real-world predictive-maintenance performance."
    )

    report.append("")

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(report))

    print("\n===== MODEL COMPARISON =====")
    print(comparison.to_string(index=False))

    print(
        f"\nReport saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()