import pandas as pd
import json


# ============================================================
# FILE PATHS
# ============================================================

RISK_FILE = "results/final_machine_risk.csv"
FEATURE_FILE = "results/feature_importance.csv"
SENSOR_FILE = "data/sensor_data.csv"
LSTM_HISTORY_FILE = "results/final_lstm_prediction_history.csv"

RF_RESULTS_FILE = "results/baseline_results.json"
LSTM_RESULTS_FILE = "results/lstm_results.json"

OUTPUT_FILE = "dashboard/dashboard_data.json"


# ============================================================
# SETTINGS
# ============================================================

SENSOR_HISTORY_HOURS = 168   # 7 days


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading dashboard data...")

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    risk_df = pd.read_csv(RISK_FILE)

    feature_df = pd.read_csv(FEATURE_FILE)

    sensor_df = pd.read_csv(SENSOR_FILE)

    lstm_df = pd.read_csv(LSTM_HISTORY_FILE)

    # Model evaluation results

    with open(
        RF_RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        rf_results = json.load(file)

    with open(
        LSTM_RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        lstm_results = json.load(file)

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    sensor_df["timestamp"] = pd.to_datetime(
        sensor_df["timestamp"]
    )

    lstm_df["timestamp"] = pd.to_datetime(
        lstm_df["timestamp"]
    )

    # --------------------------------------------------------
    # Fleet summary
    # --------------------------------------------------------

    fleet_summary = {

        "total_machines":
            int(len(risk_df)),

        "critical":
            int(
                (risk_df["risk_level"] == "CRITICAL").sum()
            ),

        "high":
            int(
                (risk_df["risk_level"] == "HIGH").sum()
            ),

        "medium":
            int(
                (risk_df["risk_level"] == "MEDIUM").sum()
            ),

        "low":
            int(
                (risk_df["risk_level"] == "LOW").sum()
            )
    }

    # --------------------------------------------------------
    # Top 10 machines
    # --------------------------------------------------------

    top_machines = (

        risk_df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # Complete machine ranking
    # --------------------------------------------------------

    machine_ranking = (

        risk_df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = (

        feature_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
        [
            [
                "rank",
                "feature",
                "importance"
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    model_comparison = {

        "random_forest": {

            "precision":
                float(
                    rf_results["precision"]
                ),

            "recall":
                float(
                    rf_results["recall"]
                ),

            "f1":
                float(
                    rf_results["f1"]
                ),

            "roc_auc":
                float(
                    rf_results["roc_auc"]
                ),

            "false_alarm_rate":
                float(
                    rf_results["false_alarm_rate"]
                )
        },

        "numpy_lstm": {

            "precision":
                float(
                    lstm_results["precision"]
                ),

            "recall":
                float(
                    lstm_results["recall"]
                ),

            "f1":
                float(
                    lstm_results["f1"]
                ),

            "roc_auc":
                float(
                    lstm_results["roc_auc"]
                ),

            "false_alarm_rate":
                float(
                    lstm_results["false_alarm_rate"]
                )
        }
    }

    # --------------------------------------------------------
    # Machine details
    # --------------------------------------------------------

    machine_details = {}

    for _, row in risk_df.iterrows():

        machine_id = row["machine_id"]

        # ====================================================
        # Sensor history
        # ====================================================

        machine_sensor_data = (

            sensor_df[
                sensor_df["machine_id"] == machine_id
            ]
            .sort_values("timestamp")
            .tail(SENSOR_HISTORY_HOURS)
        )

        sensor_history = []

        for _, sensor_row in machine_sensor_data.iterrows():

            sensor_history.append({

                "timestamp":
                    sensor_row["timestamp"]
                    .strftime("%Y-%m-%d %H:%M"),

                "vibration":
                    round(
                        float(
                            sensor_row["vibration"]
                        ),
                        4
                    ),

                "temperature":
                    round(
                        float(
                            sensor_row["temperature"]
                        ),
                        4
                    ),

                "current":
                    round(
                        float(
                            sensor_row["current"]
                        ),
                        4
                    ),

                "pressure":
                    round(
                        float(
                            sensor_row["pressure"]
                        ),
                        4
                    )
            })

        # ====================================================
        # LSTM prediction history
        # ====================================================

        machine_predictions = (

            lstm_df[
                lstm_df["machine_id"] == machine_id
            ]
            .sort_values("timestamp")
        )

        prediction_history = []

        for _, prediction_row in machine_predictions.iterrows():

            prediction_history.append({

                "timestamp":
                    prediction_row["timestamp"]
                    .strftime("%Y-%m-%d %H:%M"),

                "probability":
                    round(
                        float(
                            prediction_row[
                                "lstm_probability"
                            ]
                        ),
                        4
                    )
            })

        # ====================================================
        # Store machine details
        # ====================================================

        machine_details[machine_id] = {

            "risk_level":
                row["risk_level"],

            "risk_score":
                float(
                    row["risk_score"]
                ),

            "lstm_probability":
                float(
                    row["lstm_probability"]
                ),

            "sensor_abnormality":
                float(
                    row["sensor_abnormality"]
                ),

            "trend_score":
                float(
                    row["trend_score"]
                ),

            "sensors": {

                "vibration":
                    float(
                        row["vibration"]
                    ),

                "temperature":
                    float(
                        row["temperature"]
                    ),

                "current":
                    float(
                        row["current"]
                    ),

                "pressure":
                    float(
                        row["pressure"]
                    )
            },

            "sensor_history":
                sensor_history,

            "prediction_history":
                prediction_history
        }

    # --------------------------------------------------------
    # Final dashboard object
    # --------------------------------------------------------

    dashboard_data = {

        "fleet_summary":
            fleet_summary,

        "top_machines":
            top_machines,

        "machine_ranking":
            machine_ranking,

        "machine_details":
            machine_details,

        "feature_importance":
            feature_importance,

        "model_comparison":
            model_comparison
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dashboard_data,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n===== DASHBOARD DATA =====")

    print(
        f"Total machines: "
        f"{fleet_summary['total_machines']}"
    )

    print(
        f"Critical: "
        f"{fleet_summary['critical']}"
    )

    print(
        f"High: "
        f"{fleet_summary['high']}"
    )

    print(
        f"Medium: "
        f"{fleet_summary['medium']}"
    )

    print(
        f"Low: "
        f"{fleet_summary['low']}"
    )

    print("\nTop features:")

    for feature in feature_importance[:5]:

        print(
            f"{feature['rank']}. "
            f"{feature['feature']} "
            f"({feature['importance']:.4f})"
        )

    # --------------------------------------------------------
    # Model comparison output
    # --------------------------------------------------------

    print("\n===== MODEL COMPARISON =====")

    print(
        f"Random Forest F1: "
        f"{model_comparison['random_forest']['f1']:.4f}"
    )

    print(
        f"NumPy LSTM F1: "
        f"{model_comparison['numpy_lstm']['f1']:.4f}"
    )

    print(
        f"Random Forest ROC-AUC: "
        f"{model_comparison['random_forest']['roc_auc']:.4f}"
    )

    print(
        f"NumPy LSTM ROC-AUC: "
        f"{model_comparison['numpy_lstm']['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # History verification
    # --------------------------------------------------------

    print("\n===== HISTORY VERIFICATION =====")

    print(
        "Machines with LSTM history:",
        lstm_df["machine_id"].nunique()
    )

    print(
        "Total LSTM prediction points:",
        len(lstm_df)
    )

    for machine_id in [
        "M003",
        "M014",
        "M015"
    ]:

        machine_history = lstm_df[
            lstm_df["machine_id"] == machine_id
        ]

        print(
            f"{machine_id}: "
            f"{len(machine_history)} LSTM predictions"
        )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()