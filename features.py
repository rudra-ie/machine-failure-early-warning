import pandas as pd
import numpy as np


SENSORS = [
    "vibration",
    "temperature",
    "current",
    "pressure"
]


def create_features(df):
    """
    Create time-series features for machine failure prediction.
    """

    df = df.copy()

    # Make sure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by machine and time
    df = df.sort_values(
        ["machine_id", "timestamp"]
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Rolling features
    # -------------------------------------------------

    for sensor in SENSORS:

        for window in [6, 24, 72]:

            grouped = df.groupby("machine_id")[sensor]

            df[f"{sensor}_rolling_mean_{window}h"] = (
                grouped
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )
            )

            df[f"{sensor}_rolling_std_{window}h"] = (
                grouped
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=1
                    ).std()
                )
            )

    # -------------------------------------------------
    # Rate of change
    # -------------------------------------------------

    for sensor in SENSORS:

        df[f"{sensor}_diff"] = (
            df.groupby("machine_id")[sensor]
            .diff()
        )

        df[f"{sensor}_rate"] = (
            df.groupby("machine_id")[sensor]
            .pct_change()
        )

    # -------------------------------------------------
    # Create 48-hour failure target
    # -------------------------------------------------

    # For every machine, find the next failure event.
    df["future_failure"] = (
        df.groupby("machine_id")["failure_event"]
        .transform(
            lambda x:
            x.iloc[::-1]
            .rolling(
                window=48,
                min_periods=1
            )
            .max()
            .iloc[::-1]
        )
    )

    df["failure_next_48h"] = (
        df["future_failure"] > 0
    ).astype(int)

    # Remove helper column
    df.drop(
        columns=["future_failure"],
        inplace=True
    )

    # -------------------------------------------------
    # Clean generated values
    # -------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(0, inplace=True)

    return df


def main():

    input_path = "data/sensor_data.csv"
    output_path = "data/processed_features.csv"

    print("Loading sensor data...")

    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    df = create_features(df)

    print("Feature engineering complete!")
    print("New shape:", df.shape)

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved processed data to: {output_path}"
    )


if __name__ == "__main__":
    main()