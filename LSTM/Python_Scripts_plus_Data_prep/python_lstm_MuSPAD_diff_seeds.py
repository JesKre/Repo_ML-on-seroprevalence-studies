
###works with tensorflow==2.13.0 (and lower)
from typing import List

### works with tensorflow==2.13.0 (and lower)
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import keras
import keras_tuner as kt
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, TimeDistributed, Dense
import os
import lime
from lime import lime_tabular
import shap

import shutil

# Hard-coded source paths (unchanged from original)
# train_csv_path = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_traindat_LSTM.csv'
# test_csv_path  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_testdat_LSTM.csv'
save_dir_hard  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results\\Python_Original\\Different_seeds\\keras_tuner'

# Function to run one horizon
def train_and_predict_seed(prediction, dataset_variant, seeds):
    """
    prediction: horizon length (7 or 14)
    dataset_variant: 'full' (default), '14', 'threefourthData'
    """

    # Pick correct dataset depending on horizon and variant

    base_path = (
        r'S:\PROJACTIVE\MUSPAD-LABORERGEBNISSE\LOKI_ML'
        r'\Datasets\LSTM_data_subsets'
    )

    if dataset_variant == "full" and prediction == 7:
        train_csv_path = fr'{base_path}\MuSPAD_traindat_LSTM.csv'
        test_csv_path = fr'{base_path}\MuSPAD_testdat_LSTM.csv'

    elif dataset_variant == "full" and prediction == 14:
        train_csv_path = fr'{base_path}\MuSPAD_traindat_LSTM_14.csv'
        test_csv_path = fr'{base_path}\MuSPAD_testdat_LSTM_14.csv'

    elif prediction == 7:
        train_csv_path = (
            fr'{base_path}\MuSPAD_traindat_LSTM_{dataset_variant}.csv'
        )
        test_csv_path = (
            fr'{base_path}\MuSPAD_testdat_LSTM_{dataset_variant}.csv'
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
    traindat = traindat.drop(traindat.columns[[0]], axis=1)
    n_features = 122
    data_columns = traindat.columns
    print("Training data columns:", data_columns)

    testdat = pd.read_csv(test_csv_path)
    testdat = testdat.drop(testdat.columns[[0]], axis=1)

    n_features = traindat.shape[1]
    data_columns = traindat.columns

    lag = prediction  # 7 days

    # =========================
    # Split validation data
    # =========================

    valdat = traindat.iloc[-prediction:].copy()
    train_data = traindat.iloc[:-prediction].copy()

    # =========================
    # Function for sequence creation
    # =========================

    def create_sequences(data, lag, prediction):

        scaled_data = data.values

        n_samples = scaled_data.shape[0] - lag - prediction + 1
        n_features = scaled_data.shape[1]

        # Create input sequences
        x_train_list = []

        for i in range(n_features):
            feature_sequences = []

            for j in range(n_samples):
                sequence = scaled_data[j:j + lag, i]
                feature_sequences.append(sequence)

            x_train_list.append(np.array(feature_sequences))

        # Reshape to 3D array
        x = np.zeros((n_samples, lag, n_features))

        for i in range(n_features):
            x[:, :, i] = x_train_list[i]

        # Create target sequences
        scaled_y = scaled_data[:, 0]

        y = []

        for i in range(n_samples):
            start_idx = i + lag
            end_idx = start_idx + prediction

            target_sequence = scaled_y[start_idx:end_idx]
            y.append(target_sequence)

        y = np.array(y).reshape((n_samples, prediction, 1))

        return x, y

    # =========================
    # Train data
    # =========================

    train_x, train_y = create_sequences(
        train_data,
        lag,
        prediction
    )

    # =========================
    # Validation input
    # =========================

    val_x = np.zeros((1, lag, n_features))

    for i in range(n_features):
        last_sequence = train_data.values[-lag:, i]
        val_x[0, :, i] = last_sequence

    val_y = valdat.iloc[:, 0].values.reshape(
        (1, prediction, 1)
    )


    # =========================
    # Train final model on train + validation
    # =========================

    train_val = traindat.copy()

    train_val_x, train_val_y = create_sequences(
        train_val,
        lag,
        prediction
    )

    # =========================
    # Test input
    # =========================

    test_x = np.zeros((1, lag, n_features))

    for i in range(n_features):
        last_sequence = train_val.values[-lag:, i]
        test_x[0, :, i] = last_sequence

    def build_model(hp):

        model = keras.Sequential()

        model.add(
            keras.Input(
                batch_shape=(1, lag, n_features)
            )
        )

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

            return_sequences = True

            model.add(
                LSTM(
                    units,
                    return_sequences=True,
                    stateful=True
                )
            )

            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(1))

        learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.RMSprop(
                learning_rate=learning_rate
            ),
            loss="mse"
        )

        return model


    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        tuner = kt.BayesianOptimization(
            build_model,
            objective="val_loss",
            max_trials=30,
            directory=local_result_dir,
            project_name="lstm_validation"
        )

        tuner.search(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            batch_size=1,
            shuffle=False,
            epochs=20
        )

        # =========================
        # Get best hyperparameters
        # =========================

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(best_hp.values)
        print("Best validation MSE:")
        print(tuner.oracle.get_best_trials(num_trials=1)[0].score)

        # =========================
        # Train final model on train + validation
        # =========================

        final_model = tuner.hypermodel.build(best_hp)

        final_model.fit(
            train_val_x,
            train_val_y,
            batch_size=1,
            shuffle=False,
            epochs=20
        )

        # Save model
        model_filename = f"final_model_tplus{prediction}_seed{seed}_testset{dataset_variant}.keras"
        final_model.save(os.path.join(local_result_dir, model_filename))
        final_model.save(os.path.join(hard_result_dir, model_filename))

        # Predictions for train data
        train_predictions = final_model.predict(train_val_x, batch_size=1)
        predicted_series_full = []
        predicted_series_full.extend(train_predictions[0])  # first forecast
        for i in range(1, len(train_predictions)):
            predicted_series_full.append(train_predictions[i][prediction-1])  # last step in horizon

        # Predictions for test data
        test_predictions = final_model.predict(test_x, batch_size=1)

        # Create DataFrames
        train_true_labels = traindat['Inzidenz'].values[prediction:]
        test_true_labels = testdat['Inzidenz'].values

        train_df = pd.DataFrame({'real': train_true_labels, 'predicted': predicted_series_full})
        test_df = pd.DataFrame({'real': test_true_labels, 'predicted': test_predictions.flatten()})

        # Save predictions
        train_csv_name = f"train_predictions_tplus{prediction}_seed{seed}.csv"
        test_csv_name = f"test_predictions_tplus{prediction}_seed{seed}.csv"

        train_df.to_csv(os.path.join(local_result_dir, train_csv_name), index=False)
        test_df.to_csv(os.path.join(local_result_dir, test_csv_name), index=False)
        train_df.to_csv(os.path.join(hard_result_dir, train_csv_name), index=False)
        test_df.to_csv(os.path.join(hard_result_dir, test_csv_name), index=False)

        print(f"Finished horizon {prediction} days, seed {seed} → results saved in {local_result_dir} and {hard_result_dir}")




# Main loop
if __name__ == "__main__":

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Full dataset
    train_and_predict_seed(7, "full", seeds)

    # Different temporal splits
    train_and_predict_seed(7, "65percentData", seeds)
    train_and_predict_seed(7, "675percentData", seeds)
    train_and_predict_seed(7, "threefourthData", seeds)
    train_and_predict_seed(7, "80percentData", seeds)

# --------------------------------------------------------

