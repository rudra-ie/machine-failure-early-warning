import pandas as pd
import json


INPUT_FILE = "results/failure_lead_time.csv"
OUTPUT_FILE = "dashboard/early_warning_summary.json"


def main():

    print("Loading early-warning data...")

    df = pd.read_csv(INPUT_FILE)

    total_failures = len(df)

    warnings_detected = int(
        df["warning_detected"].sum()
    )

    warnings_not_detected = (
        total_failures - warnings_detected
    )

    detected = df[
        df["warning_detected"] == True
    ]

    if len(detected) > 0:

        average_lead_time = round(
            detected["lead_time_hours"].mean(),
            2
        )

        minimum_lead_time = round(
            detected["lead_time_hours"].min(),
            2
        )

        maximum_lead_time = round(
            detected["lead_time_hours"].max(),
            2
        )

    else:

        average_lead_time = None
        minimum_lead_time = None
        maximum_lead_time = None


    warning_detection_rate = (

        warnings_detected / total_failures * 100

        if total_failures > 0

        else 0

    )


    summary = {

        "total_failures_analyzed":
            total_failures,

        "warnings_detected":
            warnings_detected,

        "warnings_not_detected":
            warnings_not_detected,

        "warning_detection_rate":
            round(
                warning_detection_rate,
                2
            ),

        "average_lead_time_hours":
            average_lead_time,

        "minimum_lead_time_hours":
            minimum_lead_time,

        "maximum_lead_time_hours":
            maximum_lead_time
    }


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )


    print("\n===== EARLY WARNING SUMMARY =====")

    print(
        "Failures analyzed:",
        total_failures
    )

    print(
        "Warnings detected:",
        warnings_detected
    )

    print(
        "Warnings not detected:",
        warnings_not_detected
    )

    print(
        "Warning detection rate:",
        f"{warning_detection_rate:.2f}%"
    )

    print(
        "Average lead time:",
        average_lead_time,
        "hours"
    )

    print(
        "Minimum lead time:",
        minimum_lead_time,
        "hours"
    )

    print(
        "Maximum lead time:",
        maximum_lead_time,
        "hours"
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()