
###works with tensorflow==2.13.0 (and lower)
from typing import List

### works with tensorflow==2.13.0 (and lower)
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, TimeDistributed, Dense
import os
import lime
from lime import lime_tabular
import shap

import shutil

# Hard-coded source paths (unchanged from original)
# train_csv_path = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_traindat_LSTM.csv'
# test_csv_path  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_testdat_LSTM.csv'
save_dir_hard  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results\\Python_Original\\'

# Function to run one horizon
def train_and_predict(prediction, dataset_variant="full"):
    """
    prediction: horizon length (7 or 14)
    dataset_variant: 'full' (default), '14', 'threefourthData'
    """

    # Pick correct dataset depending on horizon and variant
    if dataset_variant == "full" and prediction == 7:
        train_csv_path = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_traindat_LSTM.csv'
        test_csv_path  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_testdat_LSTM.csv'

    elif dataset_variant == "full" and prediction == 14:
        train_csv_path = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_traindat_LSTM_14.csv'
        test_csv_path  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_testdat_LSTM_14.csv'

    elif dataset_variant == "threefourthData" and prediction == 7:
        train_csv_path = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_traindat_LSTM_threefourthData.csv'
        test_csv_path  = 'S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_testdat_LSTM_threefourthData.csv'

    else:
        raise ValueError(f"No dataset defined for prediction={prediction}, variant={dataset_variant}")

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

    lag = prediction  # Match R's variable naming
    scaled_train = traindat.values

    n_samples = scaled_train.shape[0] - lag - prediction + 1
    n_features = scaled_train.shape[1]

    # Create input sequences
    x_train_list = []
    for i in range(n_features):
        feature_sequences = []
        for j in range(n_samples):
            sequence = scaled_train[j:j+lag, i]
            feature_sequences.append(sequence)
        x_train_list.append(np.array(feature_sequences))

    # Reshape to 3D array
    train_x = np.zeros((n_samples, lag, n_features))
    for i in range(n_features):
        train_x[:, :, i] = x_train_list[i]

    # Create target sequences
    scaled_train_y = scaled_train[:, 0]  # First column as target
    train_y = []
    for i in range(n_samples):
        start_idx = i + lag
        end_idx = start_idx + prediction
        target_sequence = scaled_train_y[start_idx:end_idx]
        train_y.append(target_sequence)
    train_y = np.array(train_y).reshape((n_samples, prediction, 1))

    # Test input
    test_x = np.zeros((1, lag, n_features))
    for i in range(n_features):
        last_sequence = scaled_train[-lag:, i]
        test_x[0, :, i] = last_sequence

    # Build model
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
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    # Train model
    model.fit(train_x, train_y, batch_size=1, shuffle=False, epochs=20)

    # Save model
    model_filename = "final_model.keras"
    model.save(os.path.join(local_result_dir, model_filename))
    model.save(os.path.join(hard_result_dir, model_filename))

    # Predictions for train data
    train_predictions = model.predict(train_x, batch_size=1)
    predicted_series_full = []
    predicted_series_full.extend(train_predictions[0])  # first forecast
    for i in range(1, len(train_predictions)):
        predicted_series_full.append(train_predictions[i][prediction-1])  # last step in horizon

    # Predictions for test data
    test_predictions = model.predict(test_x, batch_size=1)

    # Create DataFrames
    train_true_labels = traindat['Inzidenz'].values[prediction:]
    test_true_labels = testdat['Inzidenz'].values

    train_df = pd.DataFrame({'real': train_true_labels, 'predicted': predicted_series_full})
    test_df = pd.DataFrame({'real': test_true_labels, 'predicted': test_predictions.flatten()})

    # Save predictions
    train_csv_name = f"train_predictions_0p2_AB_noreset_tplus{prediction}.csv"
    test_csv_name  = f"test_predictions_0p2_AB_noreset_tplus{prediction}.csv"

    train_df.to_csv(os.path.join(local_result_dir, train_csv_name), index=False)
    test_df.to_csv(os.path.join(local_result_dir, test_csv_name), index=False)
    train_df.to_csv(os.path.join(hard_result_dir, train_csv_name), index=False)
    test_df.to_csv(os.path.join(hard_result_dir, test_csv_name), index=False)

    print(f"Finished horizon {prediction} days → results saved in {local_result_dir} and {hard_result_dir}")

    return model, train_x, train_y, test_x, data_columns


# Main loop
if __name__ == "__main__":
    # Full datasets
    train_and_predict(7, "full")
    train_and_predict(14, "full")

    # Three-fourths dataset, 7-day horizon
    train_and_predict(7, "threefourthData")

# --------------------------------------------------------

# Explainability function:
# Can only be used when using the whole data so far (dataset_variant = "full")

def explain_model(model, train_x, train_y, test_x, data_columns, prediction, dataset_variant):
    # Calculate LIME and Shap per sample in incidence clusters

    all_dat = pd.read_csv('S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\MuSPAD_alldat_LSTM.csv')
    all_dat = all_dat.drop(all_dat.columns[[0]], axis=1)

    clus_low = pd.read_csv("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\cluster_low_LSTM.csv", sep=";")
    clus_high = pd.read_csv("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\cluster_high_LSTM.csv", sep=";")

    # Define results folders based on horizon + variant
    if dataset_variant == "full":
        local_result_dir = f"results_h{prediction}"
        hard_result_dir  = os.path.join(save_dir_hard, f"results_h{prediction}")
    else:
        local_result_dir = f"results_h{prediction}_{dataset_variant}"
        hard_result_dir  = os.path.join(save_dir_hard, f"results_h{prediction}_{dataset_variant}")

    os.makedirs(local_result_dir, exist_ok=True)
    os.makedirs(hard_result_dir, exist_ok=True)

    # --------------------------------------------------------

    n_features = all_dat.shape[1]

    # Calculate LIME and Shap for Low-Cluster-Incidence clus_low

    # Create a list to store the test_x sequences
    test_low_x_list = []

    # Only keep indices where i >= (2*prediction - 1)
    valid_indices_low = clus_low['x'][clus_low['x'] >= (2*prediction - 1)]

    # Loop over each index in clus_low['x']
    for i in valid_indices_low:
        # Get the rows i-13 to i-6 (exclusive) → 7 rows
        window = all_dat.iloc[i - (2*prediction - 1): i - (prediction - 1)].copy()

        # Optionally, check shape or drop labels if included
        # window = window.drop(columns=['target'])

        # Convert to numpy and store
        test_low_x_list.append(window.to_numpy())

    # Convert to 3D array: (samples, time_steps, features)
    test_low_x_array = np.stack(test_low_x_list)  # shape: (n_samples, prediction, n_features)

    # ------------------ LIME ------------------

    explainer_LIME = lime.lime_tabular.RecurrentTabularExplainer(training_data=train_x, mode="regression",
                                                                 training_labels=train_y,
                                                                 feature_names=data_columns,
                                                                 discretize_continuous=True,
                                                                 discretizer='decile')

    def regressor_pred(x):
        # Reshape flat input back to (n_samples, 7, n_features)
        reshaped = x.reshape((x.shape[0], prediction, n_features))  # (5000, 7, 122)
        preds = model.predict(reshaped, batch_size=1, verbose=0)  # (5000, 7, 1)
        return preds[:, (prediction-1), 0]

    all_lime_dfs_low = []

    for i, x in enumerate(test_low_x_array):
        x_flat = x.flatten()
        exp = explainer_LIME.explain_instance(x_flat, regressor_pred, num_features=train_x.shape[2])

        from pathlib import Path

        # subfolder = Path("lime_html")  # subfolder in current directory
        # subfolder.mkdir(exist_ok=True)  # create it if it doesn't exist
        # filepath = subfolder / f"lime_explanation_low_{i}.html"

        # exp.save_to_file(str(filepath))
        # exp.save_to_file(f"lime_explanation_low_{i}.html")

        # Extract and convert to DataFrame
        lime_values = exp.as_list(label=(prediction-1))
        df = pd.DataFrame(lime_values, columns=['Feature', 'LIME Value'])
        df['Sample'] = clus_low.iloc[i]['x']
        df['Label'] = (prediction-1)   # to indicate this is t+7

        all_lime_dfs_low.append(df)

    print(exp.as_list())

    # Combine all into a single DataFrame
    lime_results_df_low = pd.concat(all_lime_dfs_low, ignore_index=True)

    # Save lime values
    lime_csv_name = f"lime_tplus{prediction}_all_samples_low.csv"

    lime_results_df_low.to_csv(os.path.join(local_result_dir, lime_csv_name), index=False)
    lime_results_df_low.to_csv(os.path.join(hard_result_dir, lime_csv_name), index=False)

    # ------------------ SHAP ------------------

    print('Computing shap values...')

    # DeepExplainer: meant to approximate SHAP values for deep learning models.
    # GradientExplainer: explains a model using expected gradients (an extension
    # of integrated gradients).

    # Copy weights into stateless model for SHAP

    # ------------------------------------------------------------------------------

    # Optional: use stateless model

    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.models import Model

    # Build stateless version for SHAP

    inputs_stateless = Input(shape=(prediction, n_features))
    x = LSTM(50, return_sequences=True)(inputs_stateless)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(1))(x)

    stateless_model = Model(inputs=inputs_stateless, outputs=outputs)

    # Transfer weights from stateful model
    stateless_model.set_weights(model.get_weights())


    # Slice last timestep from output
    output_t7 = Lambda(lambda x: tf.squeeze(x[:, (prediction-1), :], axis=-1))(stateless_model.output)

    # Create new model with sliced output
    model_t7 = Model(inputs=stateless_model.input, outputs=output_t7)

    # ----------------------------------------------------------------------------------------------

    # Or use stateful model

    # Slice timestep 6 (t+7) from output
    # output_t7 = Lambda(lambda x: tf.squeeze(x[:, 6, :], axis=-1))(model.output)

    # Create new model with sliced output
    # model_t7 = Model(inputs=model.input, outputs=output_t7)

    # ------------------------------------------------------------------------------------------

    # Now pass model_t7 to GradientExplainer
    # explainer_Shap = shap.GradientExplainer(model_t7, train_x)

    # Or: use only data subset

    samp = np.random.choice(train_x.shape[0], size=100, replace=False)

    background = train_x[samp]
    background.shape

    # Create the KernelExplainer
    explainer_Shap = shap.GradientExplainer(model_t7, background)

    model_preds = model_t7.predict(background)  # shape: (n_samples)

    # Compute baseline (expected value) for time step 7
    shap_baseline = np.mean(model_preds)

    print(shap_baseline)

    # Create SHAP model and run explainer

    all_shap_dfs_low = []

    for i, x in enumerate(test_low_x_array):
        x_reshaped = x[np.newaxis, :, :]  # shape: (1, 7, n_features=122)

        # Get shap values: shape (7, 122, 1)
        shap_values = explainer_Shap.shap_values(x_reshaped)

        if i == 0:
            print(f"Shape of shap_values[0]: {np.array(shap_values[0]).shape}")

        # Squeeze to remove singleton dims: shape (7, 122)
        shap_values = shap_values.squeeze()  # now (7, 122)

        # Prepare lists
        data_records = []
        time_steps = [f"t-{(prediction-1) - t}" for t in range(prediction)]  # from t-6 to t-0

        for t_idx, t_label in enumerate(time_steps):
            for f_idx, feature_name in enumerate(all_dat.columns):
                data_records.append({
                    'Feature': feature_name,
                    'Time_Step': t_label,
                    'SHAP_Value': shap_values[t_idx, f_idx],
                    'Sample': clus_low.iloc[i]['x']
                })

        # Convert to DataFrame
        df = pd.DataFrame(data_records)

        all_shap_dfs_low.append(df)



    # Combine all
    shap_results_df_low = pd.concat(all_shap_dfs_low, ignore_index=True)
    #shap_results_df_low.head()
    #shap_results_df_low.tail()
    #shap_results_df_low.shape

    # Count positive SHAP values
    num_positive = (shap_results_df_low['SHAP_Value'] > 0).sum()

    # Count negative SHAP values
    num_negative = (shap_results_df_low['SHAP_Value'] < 0).sum()

    # Count zero SHAP values
    num_zero = (shap_results_df_low['SHAP_Value'] == 0).sum()

    print(f"Positive SHAP values: {num_positive}")
    print(f"Negative SHAP values: {num_negative}")
    print(f"Zero SHAP values: {num_zero}")

    # Save lime values
    shap_csv_name = f"shap_tplus{prediction}_all_samples_low.csv"

    shap_results_df_low.to_csv(os.path.join(local_result_dir, shap_csv_name), index=False)
    shap_results_df_low.to_csv(os.path.join(hard_result_dir, shap_csv_name), index=False)

    # --------------------------------------------------------

    # Calculate LIME and Shap for High-Cluster-Incidence clus_high

    # Create a list to store your test_x sequences
    test_high_x_list = []

    # Only keep indices where i >= (2*prediction-1)
    valid_indices_high = clus_high['x'][clus_high['x'] >= (2*prediction-1)]

    # Loop over each index in clus_high['x']
    for i in valid_indices_high:
        window = all_dat.iloc[i - (2*prediction-1):i - (prediction-1)].copy()
        test_high_x_list.append(window.to_numpy())

    # Convert to 3D array: (samples, time_steps, features)
    test_high_x_array = np.stack(test_high_x_list)

    # ------------------ LIME ------------------

    all_lime_dfs_high = []

    for i, x in enumerate(test_high_x_array):
        x_flat = x.flatten()
        exp = explainer_LIME.explain_instance(x_flat, regressor_pred, num_features=train_x.shape[2])

        from pathlib import Path

        # subfolder = Path("lime_html")  # subfolder in current directory
        # subfolder.mkdir(exist_ok=True)  # create it if it doesn't exist
        # filepath = subfolder / f"lime_explanation_high_{i}.html"

        # exp.save_to_file(str(filepath))

        lime_values = exp.as_list(label=(prediction-1))
        df = pd.DataFrame(lime_values, columns=['Feature', 'LIME Value'])
        df['Sample'] = clus_high.iloc[i]['x']
        df['Label'] = (prediction-1)
        all_lime_dfs_high.append(df)

    lime_results_df_high = pd.concat(all_lime_dfs_high, ignore_index=True)

    # Save lime values
    lime_csv_name = f"lime_tplus{prediction}_all_samples_high.csv"

    lime_results_df_high.to_csv(os.path.join(local_result_dir, lime_csv_name), index=False)
    lime_results_df_high.to_csv(os.path.join(hard_result_dir, lime_csv_name), index=False)


    # ------------------ SHAP ------------------

    # ------------------- Now loop over Low Cluster -------------------


    all_shap_dfs_high = []

    for i, x in enumerate(test_high_x_array):
        x_reshaped = x[np.newaxis, :, :]  # shape: (1, 7, n_features=122)

        # Get shap values: shape (7, 122, 1)
        shap_values = explainer_Shap.shap_values(x_reshaped)

        if i == 0:
            print(f"Shape of shap_values[0]: {np.array(shap_values[0]).shape}")

        # Squeeze to remove singleton dims: shape (7, 122)
        shap_values = shap_values.squeeze()  # now (7, 122)

        # Prepare lists
        data_records = []
        time_steps = [f"t-{(prediction-1) - t}" for t in range(prediction)]  # from t-6 to t-0

        for t_idx, t_label in enumerate(time_steps):
            for f_idx, feature_name in enumerate(all_dat.columns):
                data_records.append({
                    'Feature': feature_name,
                    'Time_Step': t_label,
                    'SHAP_Value': shap_values[t_idx, f_idx],
                    'Sample': clus_high.iloc[i]['x']
                })

        # Convert to DataFrame
        df = pd.DataFrame(data_records)

        all_shap_dfs_high.append(df)


    # Combine all
    shap_results_df_high = pd.concat(all_shap_dfs_high, ignore_index=True)
    #shap_results_df_high.head()
    #shap_results_df_high.tail()
    #shap_results_df_high.shape

    # Count positive SHAP values
    num_positive = (shap_results_df_high['SHAP_Value'] > 0).sum()

    # Count negative SHAP values
    num_negative = (shap_results_df_high['SHAP_Value'] < 0).sum()

    # Count zero SHAP values
    num_zero = (shap_results_df_high['SHAP_Value'] == 0).sum()

    print(f"Positive SHAP values: {num_positive}")
    print(f"Negative SHAP values: {num_negative}")
    print(f"Zero SHAP values: {num_zero}")

    # Save lime values
    shap_csv_name = f"shap_tplus{prediction}_all_samples_high.csv"

    shap_results_df_high.to_csv(os.path.join(local_result_dir, shap_csv_name), index=False)
    shap_results_df_high.to_csv(os.path.join(hard_result_dir, shap_csv_name), index=False)



if __name__ == "__main__":
    model, train_x, train_y, test_x, cols = train_and_predict(14, "full")
    explain_model(model, train_x, train_y, test_x, cols, 14, "full")



if __name__ == "__main__":
    model, train_x, train_y, test_x, cols = train_and_predict(7, "full")
    explain_model(model, train_x, train_y, test_x, cols, 7, "full")

