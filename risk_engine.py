import pandas as pd
import numpy as np


SENSOR_DATA = "data/sensor_data.csv"
PREDICTIONS = "results/final_lstm_predictions.csv"

OUTPUT_PATH = "results/final_machine_risk.csv"


# ---------------------------------------------------------
# Sensor abnormality scores
# ---------------------------------------------------------

def calculate_sensor_scores(df):

    scores = []

    # Normal operating ranges used by our synthetic system
    normal_ranges = {
        "vibration": (0.25, 0.60),
        "temperature": (50.0, 70.0),
        "current": (10.0, 15.0),
        "pressure": (90.0, 110.0)
    }

    for _, row in df.iterrows():

        sensor_scores = []

        for sensor, (low, high) in normal_ranges.items():

            value = row[sensor]

            if low <= value <= high:
                score = 0.0

            elif value < low:
                score = min(
                    1.0,
                    (low - value) / (low * 0.5)
                )

            else:
                score = min(
                    1.0,
                    (value - high) / (high * 0.5)
                )

            # Pressure has degradation in the downward direction
            if sensor == "pressure":

                if value >= 90:
                    score = 0.0

                else:
                    score = min(
                        1.0,
                        (90 - value) / 40
                    )

            sensor_scores.append(score)

        scores.append({
            "machine_id": row["machine_id"],
            "timestamp": row["timestamp"],
            "sensor_abnormality": np.mean(sensor_scores)
        })

    return pd.DataFrame(scores)


# ---------------------------------------------------------
# Trend score
# ---------------------------------------------------------

def calculate_trend_scores(df):

    results = []

    sensors = [
        "vibration",
        "temperature",
        "current",
        "pressure"
    ]

    for machine_id, machine_df in df.groupby("machine_id"):

        machine_df = machine_df.sort_values(
            "timestamp"
        ).copy()

        trend_scores = []

        for sensor in sensors:

            values = machine_df[sensor].values

            if len(values) < 24:
                trend_scores.append(0.0)
                continue

            recent = values[-24:]

            x = np.arange(
                len(recent)
            )

            slope = np.polyfit(
                x,
                recent,
                1
            )[0]

            # Normalize slope into a useful score
            normalized = abs(slope)

            normalized = min(
                1.0,
                normalized * 10
            )

            trend_scores.append(
                normalized
            )

        results.append({
            "machine_id": machine_id,
            "trend_score": np.mean(
                trend_scores
            )
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main risk engine
# ---------------------------------------------------------

def main():

    print("Loading machine data...")

    sensors = pd.read_csv(
        SENSOR_DATA
    )

    predictions = pd.read_csv(
        PREDICTIONS
    )

    sensors["timestamp"] = pd.to_datetime(
        sensors["timestamp"]
    )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

    # -----------------------------------------------------
    # Latest reading for every machine
    # -----------------------------------------------------

    latest = (
        sensors.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Sensor abnormality
    # -----------------------------------------------------

    sensor_scores = calculate_sensor_scores(
        latest
    )

    # -----------------------------------------------------
    # Sensor trends
    # -----------------------------------------------------

    trend_scores = calculate_trend_scores(
        sensors
    )

    # -----------------------------------------------------
    # Merge everything
    # -----------------------------------------------------

    result = predictions.merge(
        sensor_scores[
            [
                "machine_id",
                "sensor_abnormality"
            ]
        ],
        on="machine_id",
        how="left"
    )

    result = result.merge(
        trend_scores,
        on="machine_id",
        how="left"
    )

    result = result.merge(
        latest[
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

    # -----------------------------------------------------
    # Combined risk score
    # -----------------------------------------------------

    result["risk_score"] = (
        0.60 * result["lstm_probability"]
        +
        0.25 * result["sensor_abnormality"]
        +
        0.15 * result["trend_score"]
    )

    result["risk_score"] = result[
        "risk_score"
    ].clip(0, 1)

    # -----------------------------------------------------
    # Risk level
    # -----------------------------------------------------

    def get_risk_level(score):

        if score >= 0.85:
            return "CRITICAL"

        elif score >= 0.60:
            return "HIGH"

        elif score >= 0.30:
            return "MEDIUM"

        else:
            return "LOW"

    result["risk_level"] = result[
        "risk_score"
    ].apply(get_risk_level)

    # -----------------------------------------------------
    # Rank machines
    # -----------------------------------------------------

    result = result.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)

    result["risk_rank"] = (
        result.index + 1
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print(
        "\n===== FINAL FLEET RISK ANALYSIS ====="
    )

    print(
        result[
            [
                "risk_rank",
                "machine_id",
                "risk_score",
                "risk_level",
                "lstm_probability",
                "sensor_abnormality",
                "trend_score"
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\n===== RISK DISTRIBUTION ====="
    )

    print(
        result["risk_level"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()