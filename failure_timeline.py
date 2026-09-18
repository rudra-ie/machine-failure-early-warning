import pandas as pd
import json


SENSOR_DATA = "data/sensor_data.csv"
LSTM_HISTORY = "results/lstm_test_predictions.csv"

OUTPUT_PATH = "dashboard/failure_timeline.json"

WARNING_THRESHOLD = 0.50
WARNING_WINDOW_HOURS = 48


def main():

    print("Loading failure timeline data...")

    sensors = pd.read_csv(SENSOR_DATA)

    predictions = pd.read_csv(LSTM_HISTORY)

    sensors["timestamp"] = pd.to_datetime(
        sensors["timestamp"]
    )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )


    # ---------------------------------------------------------
    # Find actual failure timestamp for each machine
    # ---------------------------------------------------------

    failure_events = (
        sensors[
            sensors["failure_event"] == 1
        ]
        .groupby("machine_id")["timestamp"]
        .min()
        .reset_index()
    )

    failure_events.rename(
        columns={
            "timestamp": "failure_timestamp"
        },
        inplace=True
    )


    # ---------------------------------------------------------
    # Build timeline for every machine with historical
    # LSTM predictions
    # ---------------------------------------------------------

    timeline = {}


    for machine_id, machine_predictions in predictions.groupby(
        "machine_id"
    ):

        machine_predictions = (
            machine_predictions
            .sort_values("timestamp")
            .copy()
        )


        failure_row = failure_events[
            failure_events["machine_id"] == machine_id
        ]


        # -----------------------------------------------------
        # No actual failure
        # -----------------------------------------------------

        if failure_row.empty:

            timeline[machine_id] = {

                "has_failure": False,

                "failure_timestamp": None,

                "warning_timestamp": None,

                "lead_time_hours": None,

                "warning_threshold": WARNING_THRESHOLD,

                "status": "No failure recorded",

                "prediction_history": [

                    {
                        "timestamp":
                            row["timestamp"].strftime(
                                "%Y-%m-%d %H:%M"
                            ),

                        "probability":
                            round(
                                float(
                                    row["lstm_probability"]
                                ),
                                4
                            )

                    }

                    for _, row
                    in machine_predictions.iterrows()

                ]

            }

            continue


        failure_timestamp = failure_row.iloc[0][
            "failure_timestamp"
        ]


        # -----------------------------------------------------
        # Only predictions BEFORE the failure
        # -----------------------------------------------------

        before_failure = machine_predictions[
            machine_predictions["timestamp"]
            < failure_timestamp
        ].copy()


        # -----------------------------------------------------
        # First time probability crosses warning threshold
        # -----------------------------------------------------

        warnings = before_failure[
            before_failure["lstm_probability"]
            >= WARNING_THRESHOLD
        ]


        if warnings.empty:

            warning_timestamp = None

            lead_time_hours = None

            status = "No early warning detected"

        else:

            warning_timestamp = warnings.iloc[0][
                "timestamp"
            ]

            lead_time_hours = (
                failure_timestamp -
                warning_timestamp
            ).total_seconds() / 3600

            status = "Early warning detected"


        timeline[machine_id] = {

            "has_failure": True,

            "failure_timestamp":
                failure_timestamp.strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "warning_timestamp":
                (
                    warning_timestamp.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if warning_timestamp is not None
                    else None
                ),

            "lead_time_hours":
                (
                    round(
                        lead_time_hours,
                        2
                    )
                    if lead_time_hours is not None
                    else None
                ),

            "warning_threshold":
                WARNING_THRESHOLD,

            "status":
                status,

            "prediction_history": [

                {

                    "timestamp":
                        row["timestamp"].strftime(
                            "%Y-%m-%d %H:%M"
                        ),

                    "probability":
                        round(
                            float(
                                row["lstm_probability"]
                            ),
                            4
                        )

                }

                for _, row
                in machine_predictions.iterrows()

            ]

        }


    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    detected_lead_times = [

        item["lead_time_hours"]

        for item in timeline.values()

        if item["lead_time_hours"] is not None

    ]


    summary = {

        "machines_with_failure": sum(
            item["has_failure"]
            for item in timeline.values()
        ),

        "warnings_detected": len(
            detected_lead_times
        ),

        "average_lead_time_hours":
            round(
                sum(detected_lead_times)
                / len(detected_lead_times),
                2
            )
            if detected_lead_times
            else None,

        "minimum_lead_time_hours":
            round(
                min(detected_lead_times),
                2
            )
            if detected_lead_times
            else None,

        "maximum_lead_time_hours":
            round(
                max(detected_lead_times),
                2
            )
            if detected_lead_times
            else None

    }


    output = {

        "summary": summary,

        "machines": timeline

    }


    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print(
        "===== EARLY WARNING TIMELINE ====="
    )

    print(
        f"Machines with failure: "
        f"{summary['machines_with_failure']}"
    )

    print(
        f"Warnings detected: "
        f"{summary['warnings_detected']}"
    )

    if detected_lead_times:

        print(
            f"Average lead time: "
            f"{summary['average_lead_time_hours']:.2f} hours"
        )

        print(
            f"Minimum lead time: "
            f"{summary['minimum_lead_time_hours']:.2f} hours"
        )

        print(
            f"Maximum lead time: "
            f"{summary['maximum_lead_time_hours']:.2f} hours"
        )


    print()
    print(
        "Saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()