import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed_features.csv"

LATEST_OUTPUT = "results/final_lstm_predictions.csv"
HISTORY_OUTPUT = "results/final_lstm_prediction_history.csv"

SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

RANDOM_SEED = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "vibration",
    "temperature",
    "current",
    "pressure",

    "vibration_rolling_mean_6h",
    "vibration_rolling_std_6h",
    "vibration_rolling_mean_24h",
    "vibration_rolling_std_24h",

    "temperature_rolling_mean_6h",
    "temperature_rolling_std_6h",
    "temperature_rolling_mean_24h",
    "temperature_rolling_std_24h",

    "current_rolling_mean_6h",
    "current_rolling_std_6h",
    "current_rolling_mean_24h",
    "current_rolling_std_24h"
]


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def sigmoid(x):

    x = np.clip(x, -50, 50)

    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# NUMPY LSTM
# ============================================================

class NumPyLSTM:

    def __init__(
        self,
        input_size,
        hidden_size,
        learning_rate=0.001,
        seed=42
    ):

        rng = np.random.default_rng(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # ----------------------------------------------------
        # Combined input + hidden weights
        # ----------------------------------------------------

        scale = np.sqrt(
            1.0 / (input_size + hidden_size)
        )

        self.W = (
            rng.normal(
                0,
                scale,
                (
                    4 * hidden_size,
                    input_size + hidden_size
                )
            )
        )

        self.b = np.zeros(
            4 * hidden_size
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.Wy = (
            rng.normal(
                0,
                scale,
                hidden_size
            )
        )

        self.by = 0.0

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, X):

        sequence_length = X.shape[0]

        h = np.zeros(
            self.hidden_size
        )

        c = np.zeros(
            self.hidden_size
        )

        cache = []

        for t in range(sequence_length):

            x = X[t]

            combined = np.concatenate(
                [h, x]
            )

            gates = (
                self.W @ combined
                + self.b
            )

            i = sigmoid(
                gates[
                    0:self.hidden_size
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
                    4 * self.hidden_size
                ]
            )

            c_previous = c.copy()

            h_previous = h.copy()

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
                    h_previous,
                    c_previous,
                    i,
                    f,
                    o,
                    g,
                    c.copy(),
                    h.copy()
                )
            )

        logit = (
            self.Wy @ h
            + self.by
        )

        probability = sigmoid(
            logit
        )

        return probability, cache

    # ========================================================
    # BACKPROPAGATION THROUGH TIME
    # ========================================================

    def backward(
        self,
        X,
        y,
        probability,
        cache
    ):

        dW = np.zeros_like(
            self.W
        )

        db = np.zeros_like(
            self.b
        )

        dWy = np.zeros_like(
            self.Wy
        )

        dby = 0.0

        # ----------------------------------------------------
        # Output gradient
        # ----------------------------------------------------

        dlogit = probability - y

        final_h = cache[-1][8]

        dWy = (
            dlogit * final_h
        )

        dby = dlogit

        dh = (
            dlogit * self.Wy
        )

        dc = np.zeros(
            self.hidden_size
        )

        # ----------------------------------------------------
        # Backpropagate through time
        # ----------------------------------------------------

        for t in reversed(
            range(len(cache))
        ):

            (
                x,
                h_previous,
                c_previous,
                i,
                f,
                o,
                g,
                c,
                h
            ) = cache[t]

            tanh_c = np.tanh(c)

            # ------------------------------------------------
            # Cell state gradient
            # ------------------------------------------------

            dc_total = (
                dc
                + dh * o * (
                    1 - tanh_c ** 2
                )
            )

            # ------------------------------------------------
            # Gate gradients
            # ------------------------------------------------

            do = (
                dh * tanh_c
            )

            di = (
                dc_total * g
            )

            df = (
                dc_total * c_previous
            )

            dg = (
                dc_total * i
            )

            # ------------------------------------------------
            # Activation derivatives
            # ------------------------------------------------

            do_pre = (
                do * o * (1 - o)
            )

            di_pre = (
                di * i * (1 - i)
            )

            df_pre = (
                df * f * (1 - f)
            )

            dg_pre = (
                dg * (1 - g ** 2)
            )

            # ------------------------------------------------
            # Combine gate gradients
            # ------------------------------------------------

            dgates = np.concatenate(
                [
                    di_pre,
                    df_pre,
                    do_pre,
                    dg_pre
                ]
            )

            combined = np.concatenate(
                [
                    h_previous,
                    x
                ]
            )

            dW += np.outer(
                dgates,
                combined
            )

            db += dgates

            # ------------------------------------------------
            # Gradient to previous hidden state
            # ------------------------------------------------

            dcombined = (
                self.W.T @ dgates
            )

            dh = dcombined[
                :self.hidden_size
            ]

            dc = (
                dc_total * f
            )

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        np.clip(
            dW,
            -5,
            5,
            out=dW
        )

        np.clip(
            db,
            -5,
            5,
            out=db
        )

        np.clip(
            dWy,
            -5,
            5,
            out=dWy
        )

        dby = np.clip(
            dby,
            -5,
            5
        )

        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

        lr = self.learning_rate

        self.W -= lr * dW

        self.b -= lr * db

        self.Wy -= lr * dWy

        self.by -= lr * dby

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        X_train,
        y_train,
        epochs=5
    ):

        n_samples = len(
            X_train
        )

        for epoch in range(epochs):

            indices = np.arange(
                n_samples
            )

            np.random.shuffle(
                indices
            )

            total_loss = 0.0

            for count, idx in enumerate(
                indices
            ):

                X = X_train[idx]

                y = y_train[idx]

                probability, cache = (
                    self.forward(X)
                )

                # Binary cross entropy

                p = np.clip(
                    probability,
                    1e-7,
                    1 - 1e-7
                )

                loss = -(
                    y * np.log(p)
                    +
                    (1 - y) * np.log(1 - p)
                )

                total_loss += loss

                self.backward(
                    X,
                    y,
                    probability,
                    cache
                )

            average_loss = (
                total_loss / n_samples
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.6f}"
            )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(df):

    X_sequences = []

    y_sequences = []

    metadata = []

    machines = np.array(
        df["machine_id"].unique()
    )

    for machine_id in machines:

        machine_df = (
            df[
                df["machine_id"] == machine_id
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        values = machine_df[
            FEATURES
        ].values.astype(float)

        targets = machine_df[
            "failure_next_48h"
        ].values.astype(float)

        timestamps = machine_df[
            "timestamp"
        ].values

        # ----------------------------------------------------
        # Create rolling sequences
        # ----------------------------------------------------

        for i in range(
            SEQUENCE_LENGTH,
            len(machine_df)
        ):

            X = values[
                i - SEQUENCE_LENGTH:i
            ]

            y = targets[i]

            prediction_timestamp = (
                timestamps[i]
            )

            X_sequences.append(X)

            y_sequences.append(y)

            metadata.append({

                "machine_id":
                    machine_id,

                "timestamp":
                    prediction_timestamp

            })

    return (
        np.array(X_sequences),
        np.array(y_sequences),
        metadata
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    print(
        "Loading processed data..."
    )

    df = pd.read_csv(
        DATA_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Machines: "
        f"{df['machine_id'].nunique()}"
    )

    print(
        f"Rows: "
        f"{len(df)}"
    )

    # ========================================================
    # CREATE SEQUENCES
    # ========================================================

    print(
        "\nCreating sequences..."
    )

    (
        X,
        y,
        metadata
    ) = create_sequences(df)

    print(
        f"Sequences: {X.shape}"
    )

    print(
        f"Positive sequences: "
        f"{int(y.sum())}"
    )

    print(
        f"Negative sequences: "
        f"{int((y == 0).sum())}"
    )

    # ========================================================
    # TRAIN FINAL MODEL
    # ========================================================

    print(
        "\nStarting final LSTM training..."
    )

    model = NumPyLSTM(

        input_size=len(FEATURES),

        hidden_size=HIDDEN_SIZE,

        learning_rate=LEARNING_RATE,

        seed=RANDOM_SEED
    )

    model.train(
        X,
        y,
        epochs=EPOCHS
    )

    # ========================================================
    # GENERATE ALL MACHINE PREDICTION HISTORY
    # ========================================================

    print(
        "\nGenerating prediction history..."
    )

    history_rows = []

    for index in range(
        len(X)
    ):

        probability, _ = (
            model.forward(
                X[index]
            )
        )

        metadata_row = (
            metadata[index]
        )

        history_rows.append({

            "machine_id":
                metadata_row[
                    "machine_id"
                ],

            "timestamp":
                metadata_row[
                    "timestamp"
                ],

            "lstm_probability":
                float(probability),

            "lstm_prediction":
                int(
                    probability >= 0.50
                )
        })

    history_df = pd.DataFrame(
        history_rows
    )

    history_df["timestamp"] = (
        pd.to_datetime(
            history_df["timestamp"]
        )
    )

    history_df = (
        history_df
        .sort_values(
            [
                "machine_id",
                "timestamp"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SAVE FULL HISTORY
    # ========================================================

    history_df.to_csv(
        HISTORY_OUTPUT,
        index=False
    )

    print(
        "\nSaved full prediction history to:"
    )

    print(
        HISTORY_OUTPUT
    )

    # ========================================================
    # LATEST PREDICTION PER MACHINE
    # ========================================================

    latest_predictions = (

        history_df
        .sort_values("timestamp")
        .groupby(
            "machine_id",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    latest_predictions = (
        latest_predictions
        .sort_values(
            "lstm_probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    latest_predictions.to_csv(
        LATEST_OUTPUT,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n===== FINAL MACHINE PREDICTIONS ====="
    )

    print(

        latest_predictions[
            [
                "machine_id",
                "timestamp",
                "lstm_probability",
                "lstm_prediction"
            ]
        ]
        .to_string(index=False)

    )

    # ========================================================
    # HISTORY SUMMARY
    # ========================================================

    print(
        "\n===== PREDICTION HISTORY SUMMARY ====="
    )

    print(
        f"Machines with prediction history: "
        f"{history_df['machine_id'].nunique()}"
    )

    print(
        f"Total prediction points: "
        f"{len(history_df)}"
    )

    for machine_id in [
        "M000",
        "M003",
        "M014",
        "M015"
    ]:

        machine_history = history_df[
            history_df["machine_id"] == machine_id
        ]

        if not machine_history.empty:

            print(
                f"{machine_id}: "
                f"{len(machine_history)} predictions"
            )

    print(
        "\nFiles created:"
    )

    print(
        f"- {LATEST_OUTPUT}"
    )

    print(
        f"- {HISTORY_OUTPUT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()