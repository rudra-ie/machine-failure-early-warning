"""
Synthetic predictive-maintenance dataset generator.

Generates 40 machines with four sensors:
- vibration
- temperature
- current
- pressure

Some machines experience a gradual degradation period
followed by a failure event.
"""

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

RNG_SEED = 42

N_MACHINES = 40
STEPS_PER_MACHINE = 720

FAILURE_FRACTION = 0.55

DEGRADATION_MIN = 60
DEGRADATION_MAX = 180

START_TIME = pd.Timestamp("2026-06-01 00:00:00")
FREQ = "h"


# Healthy sensor baseline
BASELINE = {
    "vibration": {
        "mean": 0.35,
        "std": 0.03
    },
    "temperature": {
        "mean": 55.0,
        "std": 1.2
    },
    "current": {
        "mean": 12.0,
        "std": 0.4
    },
    "pressure": {
        "mean": 100.0,
        "std": 2.0
    }
}


def simulate_machine(machine_id, rng):
    """
    Simulate one machine's sensor history.
    """

    steps = STEPS_PER_MACHINE

    # Decide whether this machine will fail
    will_fail = rng.random() < FAILURE_FRACTION

    # -----------------------------
    # Healthy sensor behavior
    # -----------------------------

    vibration = rng.normal(
        BASELINE["vibration"]["mean"],
        BASELINE["vibration"]["std"],
        steps
    )

    temperature = rng.normal(
        BASELINE["temperature"]["mean"],
        BASELINE["temperature"]["std"],
        steps
    )

    current = rng.normal(
        BASELINE["current"]["mean"],
        BASELINE["current"]["std"],
        steps
    )

    pressure = rng.normal(
        BASELINE["pressure"]["mean"],
        BASELINE["pressure"]["std"],
        steps
    )

    # Natural sensor drift
    for sensor, drift in [
        (vibration, 0.004),
        (temperature, 0.05),
        (current, 0.02),
        (pressure, 0.08)
    ]:

        sensor += np.cumsum(
            rng.normal(0, drift, steps)
        )

    failure_step = None

    # -----------------------------
    # Failure degradation
    # -----------------------------

    if will_fail:

        degradation_length = int(
            rng.integers(
                DEGRADATION_MIN,
                DEGRADATION_MAX
            )
        )

        failure_step = int(
            rng.integers(
                degradation_length + 10,
                steps - 5
            )
        )

        onset = failure_step - degradation_length

        t = np.arange(degradation_length)

        # Accelerating degradation curve
        ramp = (
            t / degradation_length
        ) ** 1.6

        # Increasing noise near failure
        noise_growth = 1.0 + 3.0 * ramp

        # Vibration increases
        vibration[onset:failure_step] += (
            ramp
            * rng.uniform(0.9, 1.4)
            * 1.1
        )

        vibration[onset:failure_step] += (
            rng.normal(
                0,
                BASELINE["vibration"]["std"],
                degradation_length
            )
            * noise_growth
        )

        # Temperature increases
        temperature[onset:failure_step] += (
            ramp * rng.uniform(18, 28)
        )

        # Current increases
        current[onset:failure_step] += (
            ramp * rng.uniform(4, 7)
        )

        # Pressure decreases
        pressure[onset:failure_step] -= (
            ramp * rng.uniform(15, 25)
        )

        # Sharp failure spike
        vibration[failure_step:] += rng.uniform(
            0.8, 1.3
        )

        temperature[failure_step:] += rng.uniform(
            15, 25
        )

        current[failure_step:] += rng.uniform(
            3, 6
        )

        pressure[failure_step:] -= rng.uniform(
            10, 20
        )

    # Keep values within reasonable ranges
    pressure = np.clip(pressure, 5, None)
    vibration = np.clip(vibration, 0.01, None)
    current = np.clip(current, 0.5, None)

    # -----------------------------
    # Build dataframe
    # -----------------------------

    timestamps = pd.date_range(
        START_TIME,
        periods=steps,
        freq=FREQ
    )

    df = pd.DataFrame({
        "machine_id": f"M{machine_id:03d}",
        "timestamp": timestamps,
        "step": np.arange(steps),
        "vibration": vibration,
        "temperature": temperature,
        "current": current,
        "pressure": pressure,
        "failure_event": 0
    })

    # Mark failure
    if will_fail:

        df.loc[
            df["step"] == failure_step,
            "failure_event"
        ] = 1

        # Stop shortly after failure
        cutoff = min(
            failure_step + 15,
            steps - 1
        )

        df = df.iloc[:cutoff + 1].copy()

    return df, will_fail, failure_step


def main():

    rng = np.random.default_rng(RNG_SEED)

    frames = []
    summary = []

    # Generate all machines
    for machine_id in range(N_MACHINES):

        df, will_fail, failure_step = simulate_machine(
            machine_id,
            rng
        )

        frames.append(df)

        summary.append({
            "machine_id": f"M{machine_id:03d}",
            "will_fail": will_fail,
            "failure_step": failure_step,
            "n_rows": len(df)
        })

    # Combine all machines
    sensor_data = pd.concat(
        frames,
        ignore_index=True
    )

    machine_summary = pd.DataFrame(
        summary
    )

    # Save inside our data directory
    sensor_data.to_csv(
        "data/sensor_data.csv",
        index=False
    )

    machine_summary.to_csv(
        "data/machine_summary.csv",
        index=False
    )

    print(
        f"Machines: {N_MACHINES}"
    )

    print(
        f"Rows: {len(sensor_data)}"
    )

    print(
        f"Failing machines: "
        f"{machine_summary['will_fail'].sum()}"
    )

    print("\nFiles created:")
    print("data/sensor_data.csv")
    print("data/machine_summary.csv")


if __name__ == "__main__":
    main()