import pandas as pd
import numpy as np


SENSOR_DATA = "data/sensor_data.csv"
PREDICTIONS = "results/final_lstm_predictions.csv"

OUTPUT_PATH = "results/deployment_validation.csv"


def main():

    print("Loading data...")

    sensors = pd.read_csv(SENSOR_DATA)
    predictions = pd.read_csv(PREDICTIONS)

    sensors["timestamp"] = pd.to_datetime(sensors["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    # ---------------------------------------------------------
    # 1. Machine-level ground truth
    # ---------------------------------------------------------

    machine_status = (
        sensors.groupby("machine_id")
        .agg(
            actual_failure=("failure_event", "max"),
            total_readings=("timestamp", "count"),
            max_temperature=("temperature", "max"),
            max_vibration=("vibration", "max"),
            max_current=("current", "max"),
            min_pressure=("pressure", "min")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 2. Merge predictions with actual machine status
    # ---------------------------------------------------------

    result = predictions.merge(
        machine_status,
        on="machine_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 3. Risk levels
    # ---------------------------------------------------------

    def risk_level(probability):

        if probability >= 0.85:
            return "CRITICAL"

        elif probability >= 0.60:
            return "HIGH"

        elif probability >= 0.30:
            return "MEDIUM"

        else:
            return "LOW"

    result["risk_level"] = result[
        "lstm_probability"
    ].apply(risk_level)

    # ---------------------------------------------------------
    # 4. Whether prediction agrees with actual failure
    # ---------------------------------------------------------

    result["prediction_correct"] = (
        result["lstm_prediction"]
        == result["actual_failure"]
    )

    # ---------------------------------------------------------
    # 5. Sort by risk
    # ---------------------------------------------------------

    result = result.sort_values(
        "lstm_probability",
        ascending=False
    ).reset_index(drop=True)

    result["rank"] = (
        result.index + 1
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Print summary
    # ---------------------------------------------------------

    print("\n===== DEPLOYMENT VALIDATION =====")

    print(
        f"Total machines: "
        f"{len(result)}"
    )

    print(
        f"Actual failing machines: "
        f"{result['actual_failure'].sum()}"
    )

    print(
        f"Predicted high-risk machines: "
        f"{(result['lstm_probability'] >= 0.60).sum()}"
    )

    print("\nRisk distribution:")

    print(
        result["risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nPrediction distribution:")

    print(
        result["lstm_prediction"]
        .value_counts()
        .rename({
            0: "LOW / NO WARNING",
            1: "FAILURE WARNING"
        })
        .to_string()
    )

    print("\n===== TOP 20 MACHINES =====")

    print(
        result[
            [
                "rank",
                "machine_id",
                "lstm_probability",
                "risk_level",
                "actual_failure",
                "prediction_correct"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n===== PROBABILITY STATISTICS =====")

    print(
        result["lstm_probability"]
        .describe()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()