import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed_features.csv"
OUTPUT_PATH = "results/baseline_results.json"


def main():

    print("Loading processed features...")

    df = pd.read_csv(INPUT_PATH)

    # -------------------------------------------------
    # Select features
    # -------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "machine_id",
            "timestamp",
            "failure_event",
            "failure_next_48h"
        ]
    ]

    X = df[feature_columns]
    y = df["failure_next_48h"]

    # -------------------------------------------------
    # Split by machine
    # -------------------------------------------------

    machines = np.array(
        df["machine_id"].unique()
    )

    train_machines, test_machines = train_test_split(
        machines,
        test_size=0.25,
        random_state=42
    )

    train_df = df[
        df["machine_id"].isin(train_machines)
    ]

    test_df = df[
        df["machine_id"].isin(test_machines)
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["failure_next_48h"]

    X_test = test_df[feature_columns]
    y_test = test_df["failure_next_48h"]

    print(f"Training machines: {len(train_machines)}")
    print(f"Testing machines: {len(test_machines)}")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # -------------------------------------------------
    # Train Random Forest
    # -------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )
    except ValueError:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    results = {
        "model": "Random Forest",
        "train_machines": len(train_machines),
        "test_machines": len(test_machines),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": feature_columns,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("\n===== RANDOM FOREST RESULTS =====")

    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 Score:         {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:          {roc_auc:.4f}")

    print(
        f"False Alarm Rate: {false_alarm_rate:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        f"TN={tn}  FP={fp}"
    )
    print(
        f"FN={fn}  TP={tp}"
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()