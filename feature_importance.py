import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


INPUT_PATH = "data/processed_features.csv"
OUTPUT_PATH = "results/feature_importance.csv"


def main():

    print("Loading processed features...")

    df = pd.read_csv(INPUT_PATH)

    # ---------------------------------------------------------
    # Select features
    # ---------------------------------------------------------

    excluded_columns = [
        "machine_id",
        "timestamp",
        "failure_event",
        "failure_next_48h"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns]
    y = df["failure_next_48h"]

    print(f"Number of features: {len(feature_columns)}")
    print(f"Training rows: {len(X)}")

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X, y)

    # ---------------------------------------------------------
    # Calculate feature importance
    # ---------------------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1)
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Display top features
    # ---------------------------------------------------------

    print("\n===== TOP 15 FEATURE IMPORTANCES =====")

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()