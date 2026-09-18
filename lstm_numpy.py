import json
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


INPUT_PATH = "data/processed_features.csv"
OUTPUT_PATH = "results/lstm_results.json"
PREDICTIONS_PATH = "results/lstm_test_predictions.csv"

SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 32

LEARNING_RATE = 0.001
EPOCHS = 5


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def create_sequences(df, feature_columns):

    X = []
    y = []
    machine_ids = []
    timestamps = []

    for machine_id, machine_df in df.groupby(
        "machine_id"
    ):

        machine_df = machine_df.sort_values(
            "timestamp"
        )

        values = machine_df[
            feature_columns
        ].values

        targets = machine_df[
            "failure_next_48h"
        ].values

        time_values = machine_df[
            "timestamp"
        ].values

        for i in range(
            SEQUENCE_LENGTH,
            len(machine_df)
        ):

            X.append(
                values[
                    i - SEQUENCE_LENGTH:i
                ]
            )

            y.append(
                targets[i]
            )

            machine_ids.append(
                machine_id
            )

            timestamps.append(
                time_values[i]
            )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(machine_ids),
        np.array(timestamps)
    )


class NumPyLSTM:

    def __init__(
        self,
        input_size,
        hidden_size,
        seed=42
    ):

        rng = np.random.default_rng(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = 1.0 / np.sqrt(
            input_size + hidden_size
        )

        self.W = rng.normal(
            0,
            scale,
            (
                4 * hidden_size,
                input_size + hidden_size
            )
        )

        self.b = np.zeros(
            4 * hidden_size
        )

        self.Wy = rng.normal(
            0,
            scale,
            hidden_size
        )

        self.by = 0.0

    def forward(self, sequence):

        h = np.zeros(
            self.hidden_size
        )

        c = np.zeros(
            self.hidden_size
        )

        cache = []

        for x in sequence:

            combined = np.concatenate(
                [h, x]
            )

            gates = (
                self.W @ combined
                + self.b
            )

            i = sigmoid(
                gates[
                    :self.hidden_size
                ]
            )

            f = sigmoid(
                gates[
                    self.hidden_size:
                    2 * self.hidden_size
                ]
            )

            o = sigmoid(
                gates[
                    2 * self.hidden_size:
                    3 * self.hidden_size
                ]
            )

            g = np.tanh(
                gates[
                    3 * self.hidden_size:
                ]
            )

            previous_c = c.copy()

            c = (
                f * c
                + i * g
            )

            h = (
                o * np.tanh(c)
            )

            cache.append(
                (
                    x,
                    previous_c,
                    i,
                    f,
                    o,
                    g,
                    c.copy(),
                    h.copy(),
                    combined
                )
            )

        output = (
            self.Wy @ h
            + self.by
        )

        probability = sigmoid(
            output
        )

        return probability, cache

    def train_one(
        self,
        sequence,
        target
    ):

        probability, cache = self.forward(
            sequence
        )

        eps = 1e-8

        loss = -(
            target * np.log(
                probability + eps
            )
            +
            (1 - target)
            * np.log(
                1 - probability + eps
            )
        )

        # -------------------------------------------------
        # Output gradients
        # -------------------------------------------------

        d_output = (
            probability - target
        )

        final_h = cache[-1][7]

        dWy = (
            d_output * final_h
        )

        dby = d_output

        dW = np.zeros_like(
            self.W
        )

        db = np.zeros_like(
            self.b
        )

        d_h_next = (
            d_output * self.Wy
        )

        d_c_next = np.zeros(
            self.hidden_size
        )

        # -------------------------------------------------
        # Backpropagation Through Time
        # -------------------------------------------------

        for t in reversed(
            range(len(cache))
        ):

            (
                x,
                previous_c,
                i,
                f,
                o,
                g,
                c,
                h,
                combined
            ) = cache[t]

            tanh_c = np.tanh(c)

            d_o = (
                d_h_next
                * tanh_c
            )

            d_c = (
                d_c_next
                +
                d_h_next
                * o
                * (1 - tanh_c ** 2)
            )

            d_f = (
                d_c
                * previous_c
            )

            d_i = (
                d_c
                * g
            )

            d_g = (
                d_c
                * i
            )

            d_c_next = (
                d_c * f
            )

            d_i_raw = (
                d_i
                * i
                * (1 - i)
            )

            d_f_raw = (
                d_f
                * f
                * (1 - f)
            )

            d_o_raw = (
                d_o
                * o
                * (1 - o)
            )

            d_g_raw = (
                d_g
                * (1 - g ** 2)
            )

            dgates = np.concatenate(
                [
                    d_i_raw,
                    d_f_raw,
                    d_o_raw,
                    d_g_raw
                ]
            )

            dW += np.outer(
                dgates,
                combined
            )

            db += dgates

            d_combined = (
                self.W.T @ dgates
            )

            d_h_next = (
                d_combined[
                    :self.hidden_size
                ]
            )

        # -------------------------------------------------
        # Gradient clipping
        # -------------------------------------------------

        np.clip(
            dW,
            -1.0,
            1.0,
            out=dW
        )

        np.clip(
            db,
            -1.0,
            1.0,
            out=db
        )

        np.clip(
            dWy,
            -1.0,
            1.0,
            out=dWy
        )

        dby = np.clip(
            dby,
            -1.0,
            1.0
        )

        # -------------------------------------------------
        # Parameter updates
        # -------------------------------------------------

        self.W -= (
            LEARNING_RATE * dW
        )

        self.b -= (
            LEARNING_RATE * db
        )

        self.Wy -= (
            LEARNING_RATE * dWy
        )

        self.by -= (
            LEARNING_RATE * dby
        )

        return float(loss)

    def predict_probability(
        self,
        sequence
    ):

        probability, _ = self.forward(
            sequence
        )

        return float(
            probability
        )


def evaluate(
    model,
    X,
    y
):

    probabilities = []

    for sequence in X:

        probabilities.append(
            model.predict_probability(
                sequence
            )
        )

    probabilities = np.array(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

    except ValueError:

        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {
        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

        "false_alarm_rate": float(
            false_alarm_rate
        ),

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },

        "probabilities": probabilities
    }


def main():

    print(
        "Loading processed data..."
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # -------------------------------------------------
    # Features
    # -------------------------------------------------

    feature_columns = [

        "vibration",
        "temperature",
        "current",
        "pressure",

        "vibration_rolling_mean_6h",
        "temperature_rolling_mean_6h",
        "current_rolling_mean_6h",
        "pressure_rolling_mean_6h",

        "vibration_rolling_std_24h",
        "temperature_rolling_std_24h",
        "current_rolling_std_24h",
        "pressure_rolling_std_24h",

        "vibration_rate",
        "temperature_rate",
        "current_rate",
        "pressure_rate"
    ]

    # -------------------------------------------------
    # Machine-level split
    # -------------------------------------------------

    machines = np.array(
        df["machine_id"].unique()
    )

    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        machines
    )

    split_index = int(
        len(machines) * 0.75
    )

    train_machines = machines[
        :split_index
    ]

    test_machines = machines[
        split_index:
    ]

    train_df = df[
        df["machine_id"].isin(
            train_machines
        )
    ]

    test_df = df[
        df["machine_id"].isin(
            test_machines
        )
    ]

    print(
        f"Training machines: "
        f"{len(train_machines)}"
    )

    print(
        f"Testing machines: "
        f"{len(test_machines)}"
    )

    # -------------------------------------------------
    # Create sequences
    # -------------------------------------------------

    print(
        "\nCreating sequences..."
    )

    (
        X_train,
        y_train,
        _,
        _
    ) = create_sequences(
        train_df,
        feature_columns
    )

    (
        X_test,
        y_test,
        test_machine_ids,
        test_timestamps
    ) = create_sequences(
        test_df,
        feature_columns
    )

    print(
        "Training:",
        X_train.shape
    )

    print(
        "Testing:",
        X_test.shape
    )

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = NumPyLSTM(
        input_size=len(
            feature_columns
        ),
        hidden_size=HIDDEN_SIZE
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    print(
        "\nStarting LSTM training..."
    )

    for epoch in range(
        EPOCHS
    ):

        indices = np.arange(
            len(X_train)
        )

        rng.shuffle(
            indices
        )

        total_loss = 0.0

        for index in indices:

            loss = model.train_one(
                X_train[index],
                y_train[index]
            )

            total_loss += loss

        average_loss = (
            total_loss
            / len(X_train)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: "
            f"{average_loss:.6f}"
        )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    print(
        "\nEvaluating LSTM..."
    )

    metrics = evaluate(
        model,
        X_test,
        y_test
    )

    probabilities = metrics.pop(
        "probabilities"
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # -------------------------------------------------
    # Save individual predictions
    # -------------------------------------------------

    prediction_df = pd.DataFrame({

        "machine_id":
            test_machine_ids,

        "timestamp":
            pd.to_datetime(
                test_timestamps
            ),

        "actual_failure_next_48h":
            y_test.astype(int),

        "lstm_probability":
            probabilities,

        "lstm_prediction":
            predictions
    })

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print(
        "\nSaved individual LSTM "
        "predictions to:"
    )

    print(
        PREDICTIONS_PATH
    )

    # -------------------------------------------------
    # Print metrics
    # -------------------------------------------------

    print(
        "\n===== NUMPY LSTM RESULTS ====="
    )

    print(
        f"Precision:        "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:           "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:         "
        f"{metrics['f1']:.4f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC:          "
            f"{metrics['roc_auc']:.4f}"
        )

    print(
        f"False Alarm Rate: "
        f"{metrics['false_alarm_rate']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )

    # -------------------------------------------------
    # Save model results
    # -------------------------------------------------

    results = {

        "model":
            "NumPy LSTM",

        "sequence_length":
            SEQUENCE_LENGTH,

        "hidden_size":
            HIDDEN_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "train_machines":
            len(train_machines),

        "test_machines":
            len(test_machines),

        "train_sequences":
            len(X_train),

        "test_sequences":
            len(X_test),

        **metrics
    }

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

    print(
        f"\nResults saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()