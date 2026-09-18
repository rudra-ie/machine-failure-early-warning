import pandas as pd


SENSOR_DATA_PATH = "data/sensor_data.csv"
LSTM_PREDICTIONS_PATH = "results/lstm_test_predictions.csv"
OUTPUT_PATH = "results/failure_lead_time.csv"

WARNING_THRESHOLD = 0.50


def main():

    print("Loading sensor data...")
    sensor_data = pd.read_csv(SENSOR_DATA_PATH)

    print("Loading LSTM predictions...")
    predictions = pd.read_csv(LSTM_PREDICTIONS_PATH)

    sensor_data["timestamp"] = pd.to_datetime(
        sensor_data["timestamp"]
    )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

    results = []

    # ---------------------------------------------------------
    # Analyze each machine with LSTM predictions
    # ---------------------------------------------------------

    for machine_id in predictions["machine_id"].unique():

        machine_predictions = predictions[
            predictions["machine_id"] == machine_id
        ].sort_values("timestamp")

        machine_sensor_data = sensor_data[
            sensor_data["machine_id"] == machine_id
        ].sort_values("timestamp")

        # Find the actual failure event
        failures = machine_sensor_data[
            machine_sensor_data["failure_event"] == 1
        ]

        # Skip machines that did not actually fail
        if failures.empty:
            continue

        failure_timestamp = failures.iloc[0]["timestamp"]

        # Only predictions before the actual failure
        before_failure = machine_predictions[
            machine_predictions["timestamp"] < failure_timestamp
        ]

        if before_failure.empty:
            results.append({
                "machine_id": machine_id,
                "failure_timestamp": failure_timestamp,
                "first_warning_timestamp": None,
                "lead_time_hours": None,
                "warning_threshold": WARNING_THRESHOLD,
                "warning_detected": False
            })

            continue

        # Find first prediction crossing warning threshold
        warnings = before_failure[
            before_failure["lstm_probability"]
            >= WARNING_THRESHOLD
        ]

        if warnings.empty:

            results.append({
                "machine_id": machine_id,
                "failure_timestamp": failure_timestamp,
                "first_warning_timestamp": None,
                "lead_time_hours": None,
                "warning_threshold": WARNING_THRESHOLD,
                "warning_detected": False
            })

        else:

            first_warning = warnings.iloc[0]

            warning_timestamp = first_warning["timestamp"]

            lead_time = (
                failure_timestamp
                - warning_timestamp
            ).total_seconds() / 3600

            results.append({
                "machine_id": machine_id,
                "failure_timestamp": failure_timestamp,
                "first_warning_timestamp": warning_timestamp,
                "lead_time_hours": round(
                    lead_time,
                    2
                ),
                "warning_threshold": WARNING_THRESHOLD,
                "warning_detected": True
            })

    # ---------------------------------------------------------
    # Create result dataframe
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n===== FAILURE LEAD-TIME ANALYSIS =====")

    if results_df.empty:

        print(
            "No failing test machines were found."
        )

        return

    print(
        results_df.to_string(index=False)
    )

    detected = results_df[
        results_df["warning_detected"] == True
    ]

    print(
        f"\nFailing machines analyzed: "
        f"{len(results_df)}"
    )

    print(
        f"Warnings detected before failure: "
        f"{len(detected)}"
    )

    if not detected.empty:

        print(
            f"Average lead time: "
            f"{detected['lead_time_hours'].mean():.2f} hours"
        )

        print(
            f"Minimum lead time: "
            f"{detected['lead_time_hours'].min():.2f} hours"
        )

        print(
            f"Maximum lead time: "
            f"{detected['lead_time_hours'].max():.2f} hours"
        )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()