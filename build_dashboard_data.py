import pandas as pd


PROCESSED_DATA_PATH = "data/processed_features.csv"
LSTM_PREDICTIONS_PATH = "results/lstm_test_predictions.csv"
OUTPUT_PATH = "results/machine_risk_ranking.csv"


def get_risk_level(probability):
    if probability >= 0.85:
        return "CRITICAL"
    elif probability >= 0.60:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


def main():

    print("Loading processed sensor data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Loading LSTM predictions...")
    lstm = pd.read_csv(LSTM_PREDICTIONS_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    lstm["timestamp"] = pd.to_datetime(lstm["timestamp"])

    # ---------------------------------------------------------
    # Get the latest sensor reading for every machine
    # ---------------------------------------------------------

    latest_sensor = (
        df.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .copy()
    )

    # ---------------------------------------------------------
    # Get the latest LSTM prediction for every test machine
    # ---------------------------------------------------------

    latest_lstm = (
        lstm.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .copy()
    )

    latest_lstm = latest_lstm[
        [
            "machine_id",
            "timestamp",
            "lstm_probability",
            "lstm_prediction"
        ]
    ]

    # ---------------------------------------------------------
    # Combine sensor information with LSTM predictions
    # ---------------------------------------------------------

    ranking = latest_lstm.merge(
        latest_sensor[
            [
                "machine_id",
                "vibration",
                "temperature",
                "current",
                "pressure"
            ]
        ],
        on="machine_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Convert LSTM probability into risk level
    # ---------------------------------------------------------

    ranking["risk_probability"] = ranking[
        "lstm_probability"
    ]

    ranking["risk_level"] = ranking[
        "risk_probability"
    ].apply(get_risk_level)

    # ---------------------------------------------------------
    # Sort highest risk first
    # ---------------------------------------------------------

    ranking = ranking.sort_values(
        "risk_probability",
        ascending=False
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1)
    )

    # ---------------------------------------------------------
    # Save final ranking
    # ---------------------------------------------------------

    ranking.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Display summary
    # ---------------------------------------------------------

    print("\n===== MACHINE RISK SUMMARY =====")

    print(
        ranking[
            [
                "rank",
                "machine_id",
                "risk_probability",
                "risk_level"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nMachines with LSTM predictions: "
        f"{len(ranking)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()