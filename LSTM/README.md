# LSTM models

This directory contains the scripts used to train, evaluate, and interpret the Long Short-Term Memory (LSTM) models for predicting COVID-19 incidence.

Two models were evaluated:

- **MuSPAD-based LSTM:** Uses 122 predictors derived from the MuSPAD study.
- **Incidence-only LSTM:** Uses previous incidence observations as the baseline.

Both models use seven-day input sequences to predict incidence over the following seven days. Model performance was evaluated at five rolling forecast origins (65%, 67.5%, 75%, 80%, and 100% of the original training period), using 10 random seeds.

## Scripts

### Data preparation and model training

The directory `Python_Scripts_plus_Data_prep/` contains the following scripts:

| Script | Description |
|---|---|
| `Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R` | Prepares the full time-series dataset for LSTM training and evaluation, including imputation, scaling, and the generation of training and test inputs. Also contains code for assembling saved predictions into RData objects. |
| `Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R` | Prepares the datasets for the four additional rolling forecast origins (65%, 67.5%, 75%, and 80%). Also contains post-processing code for assembling the corresponding prediction RData files. |
| `Data_prep_threefourthData_diff_seeds.R` | Earlier data-preparation script for the 75% forecast origin. Superseded by the preceding script and retained for reference. |
| `python_lstm_MuSPAD_diff_seeds.py` | Trains the MuSPAD-based LSTM using 122 predictors and hyperparameter optimization. Runs the model with different random seeds and saves fitted models and predictions. |
| `python_lstm_onlyIncidence_diff_seeds.py` | Trains the incidence-only LSTM baseline using the same forecasting horizon and multiple random seeds. Saves fitted models and predictions. |
| `python_lstm_lime_shap_different_clusters.py` | Computes LIME and SHAP feature attributions for the LSTM model across different incidence clusters. |

The data-preparation scripts require the original study data, which are subject to access restrictions. The saved predictions needed to reproduce the evaluation results are provided separately in the repository.

## Saved model predictions

The LSTM predictions are available in:

`Compare_models/Graphics/Predicted_values/`

The files are organized by forecast origin:

- `65percentData/`
- `675percentData/`
- `threefourthData/`
- `80percentData/`
- `fullData/`

Each directory contains two `.RData` files, corresponding to the MuSPAD-based and incidence-only LSTM models. These files contain the predictions from 10 random seeds and the corresponding observed incidence values.

The saved predictions allow the model evaluation to be reproduced without retraining the LSTMs or accessing the original individual-level MuSPAD data.

## Reproducing the manuscript results

The script

[`Metrics/lstm_seedavg_batch_2026-10-03.R`](Metrics/lstm_seedavg_batch_2026-10-03.R)

calculates performance metrics, confidence intervals, calibration statistics, and leave-one-forecast-origin-out sensitivity analyses from the saved LSTM predictions.

It reproduces the **LSTM results** in the following manuscript and supplementary tables:

| Table | Results |
|---|---|
| **Table 1** (main manuscript) | RMSE and sMAPE for training and test predictions |
| **Table S6** | Additional performance metrics: MAE, MASE, and R² |
| **Table S7** | Performance variability across 10 random seeds |
| **Table S8** | Calibration intercept and slope |
| **Table S9** | Leave-one-forecast-origin-out sensitivity analysis: RMSE and sMAPE |
| **Table S10** | Leave-one-forecast-origin-out sensitivity analysis: MAE, MASE, and R² |

The script also calculates prediction interval coverage as an additional evaluation.

Only the LSTM results are reproduced by this script; the corresponding results for the other model classes are generated separately.

For details of the evaluation methods, required inputs, random seeds, and generated outputs, see:

[`Metrics/README_lstm_seedavg_2026-10-03.md`](Metrics/README.md)

