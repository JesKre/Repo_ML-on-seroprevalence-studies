from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MUSPAD_DATA_DIR", REPO_ROOT / "Datasets")).resolve()


###works with tensorflow==2.13.0 (and lower)
from typing import List

### works with tensorflow==2.13.0 (and lower)
from typing import List
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, TimeDistributed, Dense
import os
import shutil

# Portable paths relative to this checkout (override with MUSPAD_DATA_DIR).
# train_csv_path = str(DATA_DIR / "MuSPAD_traindat_LSTM.csv")
# test_csv_path  = str(DATA_DIR / "MuSPAD_testdat_LSTM.csv")
save_dir_hard  = str(REPO_ROOT / "LSTM/Python_results/Python_Original/Different_seeds/keras_tuner")

# Function to run one horizon
def run_prediction_horizon_onlyInc_seed(prediction, dataset_variant, seeds):
    """
    prediction: horizon length (7 or 14)
    dataset_variant: 'full' (default), '14', 'threefourthData'
    """
    # Pick correct dataset depending on horizon and variant

    base_path = str(DATA_DIR / "LSTM_data_subsets")

    if dataset_variant == "full" and prediction == 7:
        train_csv_path = f'{base_path}/MuSPAD_traindat_LSTM.csv'
        test_csv_path = f'{base_path}/MuSPAD_testdat_LSTM.csv'

    elif dataset_variant == "full" and prediction == 14:
        train_csv_path = f'{base_path}/MuSPAD_traindat_LSTM_14.csv'
        test_csv_path = f'{base_path}/MuSPAD_testdat_LSTM_14.csv'

    elif prediction == 7:
        train_csv_path = (
            f'{base_path}/MuSPAD_traindat_LSTM_{dataset_variant}.csv'
        )
        test_csv_path = (
            f'{base_path}/MuSPAD_testdat_LSTM_{dataset_variant}.csv'
        )

    else:
        raise ValueError(
            f"Unknown dataset_variant/prediction combination: "
            f"{dataset_variant}, {prediction}"
        )

    # Define results folders based on horizon + variant
    if dataset_variant == "full":
        local_result_dir = f"results_h{prediction}"
        hard_result_dir  = os.path.join(save_dir_hard, f"results_h{prediction}")
    else:
        local_result_dir = f"results_h{prediction}_{dataset_variant}"
        hard_result_dir  = os.path.join(save_dir_hard, f"results_h{prediction}_{dataset_variant}")

    os.makedirs(local_result_dir, exist_ok=True)
    os.makedirs(hard_result_dir, exist_ok=True)

    print(f"\n{'='*50}\nRunning LSTM for horizon {prediction} days, variant: {dataset_variant}\n{'='*50}")

    traindat = pd.read_csv(train_csv_path)
    traindat = traindat.iloc[:, 1]

    testdat = pd.read_csv(test_csv_path)
    testdat = testdat.iloc[:, 1]

    lag = prediction  # Match R's variable naming
    scaled_train = traindat.values

    n_samples = scaled_train.shape[0] - lag - prediction + 1

    # Create input sequences
    train_x = []
    for i in range(n_samples):
        sequence = scaled_train[i:i + lag]  # No need for [:, 0]
        train_x.append(sequence)

    train_x = np.array(train_x)
    train_x = train_x.reshape((n_samples, lag, 1))  # Reshape to 3D for LSTM

    # Create target sequences (Y)
    train_y = []
    for i in range(n_samples):
        start_idx = i + lag
        end_idx = start_idx + prediction
        target_sequence = scaled_train[start_idx:end_idx]
        train_y.append(target_sequence)

    train_y = np.array(train_y)
    train_y = train_y.reshape((n_samples, prediction, 1))  # Also 3D

    # Create x_pred_arr (test input for next prediction)
    test_x = scaled_train[-lag:].reshape((1, lag, 1))  # Last input sequence

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Build model
        n_features = 1
        n_timesteps = lag
        batch_size = 1

        inputs = Input(batch_shape=(batch_size, n_timesteps, n_features))
        x = LSTM(50, return_sequences=True, stateful=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(50, return_sequences=True, stateful=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(50, return_sequences=True, stateful=True)(x)
        x = Dropout(0.2)(x)
        outputs = TimeDistributed(Dense(1))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

        def build_model(hp):

            model = keras.Sequential()

            n_layers = hp.Int("n_layers", min_value=1, max_value=4)

            for i in range(n_layers):

                units = hp.Int(
                    f"units_{i}",
                    min_value=32,
                    max_value=128,
                    step=32
                )

                dropout = hp.Float(
                    f"dropout_{i}",
                    min_value=0.0,
                    max_value=0.5,
                    step=0.1
                )

                # all but last LSTM return sequences
                return_sequences = i < n_layers - 1

                if i == 0:
                    model.add(
                        keras.layers.LSTM(
                            units,
                            return_sequences=return_sequences,
                            input_shape=(timesteps, n_features)
                        )
                    )
                else:
                    model.add(
                        keras.layers.LSTM(
                            units,
                            return_sequences=return_sequences
                        )
                    )

                model.add(keras.layers.Dropout(dropout))

            model.add(keras.layers.Dense(1))

            lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

            model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss="mse"
            )

            return model

        # Train model
        model.fit(train_x, train_y, batch_size=1, shuffle=False, epochs=20)

        # Save model
        model_filename = f"final_model_only_incidence_tplus{prediction}_seed{seed}_testset{dataset_variant}.keras"
        model.save(os.path.join(local_result_dir, model_filename))
        model.save(os.path.join(hard_result_dir, model_filename))

        # Predictions for train data
        train_predictions = model.predict(train_x, batch_size=1)

        predicted_series_full = []

        predicted_series_full.extend(train_predictions[0])  # first forecast

        for i in range(1, len(train_predictions)):
            predicted_series_full.append(train_predictions[i][prediction - 1])  # last step in horizon

        # Predictions for test data
        test_predictions = model.predict(test_x, batch_size=1)

        # Create DataFrames
        train_true_labels = traindat.values[prediction:]
        test_true_labels = testdat.values

        train_df = pd.DataFrame({'real': train_true_labels, 'predicted': predicted_series_full})
        test_df = pd.DataFrame({'real': test_true_labels, 'predicted': test_predictions.flatten()})

        # Save predictions
        train_csv_name = f"train_predictions_only_incidence_tplus{prediction}_seed{seed}.csv"
        test_csv_name = f"test_predictions_only_incidence_tplus{prediction}_seed{seed}.csv"

        train_df.to_csv(os.path.join(local_result_dir, train_csv_name), index=False)
        test_df.to_csv(os.path.join(local_result_dir, test_csv_name), index=False)
        train_df.to_csv(os.path.join(hard_result_dir, train_csv_name), index=False)
        test_df.to_csv(os.path.join(hard_result_dir, test_csv_name), index=False)

        print(f"Finished horizon {prediction} days → results saved in {local_result_dir} and {hard_result_dir}")


# Main loop
#if __name__ == "__main__":
 #   for pred in [7, 14]:
  #      run_prediction_horizon_onlyInc(pred)

if __name__ == "__main__":

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Full datasets
    run_prediction_horizon_onlyInc_seed(
        7, "full", seeds
    )

    # Different temporal splits
    run_prediction_horizon_onlyInc_seed(
        7, "65percentData", seeds
    )

    run_prediction_horizon_onlyInc_seed(
        7, "675percentData", seeds
    )

    run_prediction_horizon_onlyInc_seed(
        7, "threefourthData", seeds
    )

    run_prediction_horizon_onlyInc_seed(
        7, "80percentData", seeds
    )

