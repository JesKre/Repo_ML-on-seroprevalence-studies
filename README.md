# Predicting COVID-19 incidence from MuSPAD seroprevalence data

Code accompanying the manuscript **“Predicting COVID-19 incidence from seroprevalence and population-based cohort data using interpretable machine learning with differential privacy analysis.”**

This study investigates whether aggregated data from the German Multilocal SeroPrevalence (MuSPAD) study can predict local seven-day COVID-19 incidence and identify interpretable predictors of transmission dynamics. Four model families are compared:

- least absolute shrinkage and selection operator regression (LASSO);
- multilayer perceptrons (MLPs);
- sparse vector autoregression (VAR); and
- long short-term memory networks (LSTMs).

Feature importance is obtained from model coefficients for LASSO and VAR and from Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) for the neural networks. The MLP analysis also includes a custom differentially private stochastic gradient descent (DP-SGD) implementation.

## Study design at a glance

MuSPAD comprised two cross-sectional survey rounds conducted in 2020–2021, with more than 32,000 participants across eight German regions. The analysis combines questionnaire and serological measurements with county-level seven-day incidence from the Robert Koch Institute (RKI).

The preprocessing workflow selects 77 source variables and aggregates individual records by participation day. Numeric variables are represented by daily means; binary and categorical variables are expanded into daily category counts. This produces 122 MuSPAD-derived features.

| Analysis | Observations and split | Target | Models |
|---|---|---|---|
| Time-agnostic | 279 sampled days; first 90% for training/validation and final 10% for testing | Regional seven-day incidence at `t + 7` | LASSO and MLP |
| Time-aware | 408 consecutive days; final 7 days for testing, preceding 7 for validation | Seven-day incidence sequence | VAR and LSTM |
| Robustness | First three-quarters of the time series for training, followed by a 7-day test window | Seven-day incidence sequence | VAR and LSTM |
| Secondary horizon | Full time series with a 14-day input/output horizon | Fourteen-day incidence sequence | LSTM |

Missing values are imputed separately by split using `missForest`. Predictor scaling is learned from the relevant training subset and reused for validation and test data.

The baselines use only time for LASSO and MLP and only previous incidence for VAR and LSTM.

## Repository layout

```text
.
├── Artificial_datasets/          # Committed synthetic MuSPAD datasets (time-agnostic and time-aware)
├── Clustering_sensitivity_MLP_LSTM/  # Sensitivity checks for the MLP/LSTM explanation tables
├── Compare_models/               # Cross-model metrics and manuscript graphics
├── Data_Preprocessing/           # Daily aggregation and construction of modeling datasets
├── Datasets/                     # Committed modeling dataset inputs (weekly time-series RData)
├── LASSO/                        # MuSPAD and time-only LASSO models and reproduction results
├── LSTM/                         # R/Python LSTM workflow, metrics batch, and provenance record
├── MLP/                          # MLP, baseline, optimizer comparison, and DP-SGD
├── VAR/                          # Sparse VAR models and incidence-only baselines
├── Variable_Importance/          # Cross-model SHAP/LIME/coefficient tables and topic summaries
└── Variable_selection/           # Variable-selection codebooks (selected and excluded variables)
```

The script catalogue below covers every script currently present in the repository. Historical `Archive` or `Archiv` directories, if present in another branch or checkout, are intentionally excluded.

## Data access

Individual-level MuSPAD data are not distributed in this repository because they contain sensitive health information. Access can be requested through the [MuSPAD data access process at SeroHub](https://serohub.net/) and is subject to approval and applicable data-protection requirements.

The following external inputs are required:

- the cleaned individual-level MuSPAD object used as `df_merge_clean`;
- the selected-variable file corresponding to the included [`Variable_selection/Variable_selection_codebook.csv`](Variable_selection/Variable_selection_codebook.csv);
- RKI county-level incidence data, referred to in the scripts as `COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv`; and
- intermediate `.RData` and `.csv` files produced by earlier workflow stages.

`Variable_selection/Variable_selection_codebook.csv` is semicolon-delimited and documents the 77 selected source variables, including source, title, data type, description, unit, and response options. The same folder contains `Codebook_variables_excluded.xlsx` with the variables that were considered but excluded.

## Software requirements

The active scripts use the following packages:

```r
install.packages(c(
  "abind", "bigtime", "doParallel", "dplyr", "foreach", "furrr",
  "ggplot2", "glmnet", "iml", "keras", "knitr", "lubridate",
  "magrittr", "Metrics", "missForest", "purrr", "R6", "readxl",
  "reshape2", "stringr", "tensorflow", "tibble", "tidyr", "tidyverse"
))
```

The LSTM implementation is in Python and states compatibility with TensorFlow 2.13.0 or earlier:

```bash
python -m pip install "tensorflow==2.13.0" numpy pandas lime shap
```

The R `keras`/`tensorflow` scripts also require a compatible Python TensorFlow environment. GPU support is optional; several scripts use parallel CPU workers and may be resource intensive.

The MLP refit scripts (`MLP/repro_mlp_k2.R` and `MLP/repro_mlp_k2_trainfull_rmse.R`) need a separate Python environment with the Keras 2 stack (TensorFlow 2.15.1, Keras 2.15.1; R `keras` 2.16.1 was used), selected via the `MLP_PYTHON` environment variable — they cannot share the TensorFlow 2.13.0 environment used for the LSTM scripts.

## Before running the code

The scripts retain the original project’s absolute Windows paths (`S:\\...`) and are analysis scripts rather than a portable command-line pipeline. Before running them:

1. replace each `setwd()` and absolute input/output path with paths on your system;
2. create the expected output directories;
3. obtain the restricted MuSPAD data and the RKI incidence input;
4. run sections in order, because several scripts depend on objects established earlier in the same interactive R session; and
5. set and record software versions and random seeds for a reproducible rerun.

A practical execution order is:

```text
Data_Preprocessing
  -> LASSO / MLP
  -> LSTM R data export -> LSTM Python model -> LSTM R post-processing
  -> LSTM metrics batch (LSTM/Metrics/lstm_seedavg_batch_2026-10-03.R)
  -> VAR
  -> Compare_models
  -> Variable_Importance
```

The module READMEs (`LASSO/README.md`, `MLP/README.md`, `VAR/README.md` and `LSTM/README.md`) document the current per-module scripts, shared inputs, and table outputs; the LSTM metrics batch has its own protocol record in `LSTM/Metrics/README_lstm_seedavg_2026-10-03.txt`.

## Script reference

### Data preprocessing

| Script | Purpose |
|---|---|
| [`Data_Preprocessing/01_Data_Aggregation.R`](Data_Preprocessing/01_Data_Aggregation.R) | Loads cleaned participant-level MuSPAD data, keeps the selected variables, corrects known values, converts dates and an ordinal health-status field, identifies numeric/binary/categorical fields, expands category levels, and aggregates participant records by day. It also identifies variables that are completely missing on at least 20% of sampling days and creates city-presence and daily time-series outputs. |
| [`Data_Preprocessing/02_Prepare_for_Analysis.R`](Data_Preprocessing/02_Prepare_for_Analysis.R) | Removes high-missingness and near-zero-information features, joins county-level RKI incidence, and creates the two modeling datasets. The time-agnostic dataset aligns MuSPAD values at `t` with incidence at `t + 7`; the time-aware dataset fills the complete daily calendar, carries sampling location forward, and retains current incidence for sequential models. |
| [`Data_Preprocessing/Create_aggregated_artificial_dataset.R`](Data_Preprocessing/Create_aggregated_artificial_dataset.R) | Builds the committed synthetic datasets with `synthpop` (`syn()`, seed 2024) on `missForest`-imputed daily values: the time-agnostic variant drops `time_day == 168`, the time-aware variant keeps the full daily calendar; time dependencies are not preserved. It writes `Artificial_data_MuSPAD_time_agnostic.csv` and `Artificial_data_MuSPAD_time_aware.csv` (committed, together with their RData counterparts, under `Artificial_datasets/`) and prints short `xtable` examples for the supplement. Retains stale `S:` share paths. |

The committed `Data_Preprocessing/Participants_per_day.csv` and `Data_Preprocessing/participant_table.tex` are supplementary participant-count outputs of the daily aggregation workflow (scripts 01/02), not of a single script.

### LASSO

| Script | Purpose |
|---|---|
| [`LASSO/LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R`](LASSO/LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R) | Fits the main MuSPAD LASSO model. It applies a chronological 90/10 train-test split, separate `missForest` imputation, training-derived z-scaling, a log transform of incidence, and 10-fold `cv.glmnet` selection of `lambda.min`. It back-transforms predictions, calculates errors, ranks non-zero coefficients for variable importance, and saves predictions for cross-model plots. |
| [`LASSO/LASSO_model_only_time_log_all_cities.R`](LASSO/LASSO_model_only_time_log_all_cities.R) | Fits the time-only LASSO baseline with the same chronological split, log-incidence target, 10-fold lambda selection, back-transformed predictions, metrics, and plotting outputs. |
| [`LASSO/lasso_fullmetrics_2026-10-02.R`](LASSO/lasso_fullmetrics_2026-10-02.R) | Computes the Table 1 and Table S3 metrics (RMSE, sMAPE on the 0–2 scale, MAE, MASE, R², MAPE) for both models from the two saved prediction objects in `Compare_models/Graphics/Predicted_values/` (251 training and 27 test observations per model). It derives the 95% CIs itself by 10,000 moving-block bootstrap resamples (block length 7, seed 42) with IID intervals exported for comparison; MASE uses the training-series persistence MAE. No re-fitting and no prediction-interval coverage (not needed for these tables); base R, writes to `repro_results/`. |
| [`LASSO/lasso_pointcalibration_2026-10-02.R`](LASSO/lasso_pointcalibration_2026-10-02.R) | Computes the Table S5 calibration of the 27 held-out test predictions per model by OLS: intercept and slope with t-based 95% CIs and the joint test of the ideal values (α = 0, β = 1); base R, reads the same saved predictions, writes to `repro_results/`. |

See `LASSO/README.md` for the exact workflow (required inputs and environment variables); both metric scripts can consume the original saved predictions without re-fitting, so the earlier `LASSO_testset_moving_block_bootstrap.R` (and `LASSO_reproduce_paper_results.R`) is no longer committed.

### Multilayer perceptron and differential privacy

| Script | Purpose |
|---|---|
| [`MLP/Keras_MuSPAD_Expl_cluster_incidence.R`](MLP/Keras_MuSPAD_Expl_cluster_incidence.R) | Fits the primary MuSPAD MLP with four hidden layers of 10 ReLU units, a linear output, and MSE loss. It uses a 90/10 chronological train-test split and one randomly assigned fold for validation, then retrains on the training-validation data. The latter half clusters incidence values and computes/aggregates `iml` Shapley and LIME explanations for the lowest- and highest-incidence clusters. It also exports predictions and model-specific MSE values. |
| [`MLP/Keras_only_time.R`](MLP/Keras_only_time.R) | Fits the time-only MLP baseline using the same four-by-ten architecture, validation design, 30 epochs, prediction export, and plotting workflow. |
| [`MLP/DP_Analysis/Keras_MuSPAD_compare_DP_10foldcv_Expl.R`](MLP/DP_Analysis/Keras_MuSPAD_compare_DP_10foldcv_Expl.R) | Runs the non-private 10-fold MLP comparison between Adam and RMSprop. It saves weights at epochs 1, 15, and 30, calculates fold-level training/validation MSE, aggregates Shapley and LIME importance, retrains a final model, joins its performance with the DP results, and produces the privacy-comparison plot. |
| [`MLP/DP_Analysis/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R`](MLP/DP_Analysis/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R) | Implements custom DP-SGD for the four-by-ten MLP. It computes gradients per example, clips each gradient tensor to an L2 norm of `C = 1`, adds Gaussian noise with standard deviation `sigma * C`, and averages by a nominal batch size of 32. Validation compares Adam and RMSprop over 10 folds for noise multipliers 1.627197265625, 2.7294921875, and 9.0625; final testing uses Adam, learning rate 0.01, 30 epochs, and eight nominal batches per epoch. The script exports model weights, predictions, MAPE/SMAPE/RMSLE/RMSE, and fold-level SHAP/LIME summaries. |
| [`MLP/Keras_hyperparameters_imputation_layer_epochs.R`](MLP/Keras_hyperparameters_imputation_layer_epochs.R) | Hyperparameter and imputation comparison on the time-agnostic seven-day data (`time_day == 168` dropped): a 10-fold, 10-repeat grid over 1–4 hidden layers and 30/45/60 epochs comparing mean imputation, `missForest` imputation, and time-only input, recording training/validation/test MSE per configuration. Retains stale `S:` share paths; its result summary is committed as `MLP/MLP_Results_meanImp_vs_missForest_vs_onlyTime.csv`. |
| [`MLP/repro_mlp_k2.R`](MLP/repro_mlp_k2.R) | Ten-seed fitting grid (Keras 2): three input treatments, 1–4 hidden layers and 30/45/60 epochs; saves the test predictions and observed series for the selected MuSPAD and baseline configuration. Needs the separate Keras-2 Python environment (see Software requirements). |
| [`MLP/repro_mlp_k2_trainfull_rmse.R`](MLP/repro_mlp_k2_trainfull_rmse.R) | Ten-seed rerun of the selected four-layer, 30-epoch models; saves the full-training-block predictions that `MLP_generate_tables.R` uses for the training-metric and stability summaries. |
| [`MLP/MLP_generate_tables.R`](MLP/MLP_generate_tables.R) | Base-R table generator: loads the two ten-seed prediction objects (`MLP_7day_MuSPAD_seeds.RData` and `MLP_7day_baseline_seeds.RData` under `Compare_models/Graphics/Predicted_values/`) and writes the MLP rows of Tables 1, S3, S4 and S5 to `MLP_table1.csv`, `MLP_tableS3.csv`, `MLP_tableS4.csv` and `MLP_tableS5.csv`; test CIs use 10,000 moving-block resamples (block 7, seed 42); no prediction-interval coverage. With `--from-runs` it first assembles the prediction objects from the two refit scripts' outputs. |

See `MLP/README.md` for the exact workflows and inputs. The DP-SGD reproduction of Tables 7 and S8 has a dedicated workflow in `MLP/Table_reproduction_DP/MLP_with_DP/` (see its README).

### Vector autoregression

| Script | Purpose |
|---|---|
| [`VAR/bigtime_MuSPAD_allData.R`](VAR/bigtime_MuSPAD_allData.R) | Fits sparse VAR models to the full time-aware MuSPAD dataset. Validation searches `HLag` versus `L1`, `cv`/`bic`/`aic` selection, and `cvcut` 0.8/0.9 at a 7-day horizon. The test block fits the selected `L1` plus time-series-CV specification, evaluates forecasts and fitted values, and exports predictions for the manuscript comparisons. Set `p` to 7, 14, or 21 for the three reported lag variants. |
| [`VAR/bigtime_MuSPAD_threefourthData.R`](VAR/bigtime_MuSPAD_threefourthData.R) | Repeats sparse-VAR tuning and evaluation using the first-three-quarters robustness split generated by the LSTM data-preparation script. It writes validation results, saved models, MSE values, predictions, and plots. |
| [`VAR/bigtime_noMuSPAD_only_incidence_allData.R`](VAR/bigtime_noMuSPAD_only_incidence_allData.R) | Fits the full-data VAR baseline using incidence as the only series, recursively forecasts seven days, evaluates training/test MSE, and exports prediction objects and plots. Set `p` to 7, 14, or 21 to match the MuSPAD models. |
| [`VAR/bigtime_noMuSPAD_only_incidence_threefourthData.R`](VAR/bigtime_noMuSPAD_only_incidence_threefourthData.R) | Fits and plots the incidence-only VAR baseline for the first-three-quarters robustness split. |
| [`VAR/var_predictions_diff_splits_diff_seeds.R`](VAR/var_predictions_diff_splits_diff_seeds.R) | Original multi-seed fitting extension: produces the ten-seed VAR predictions for the five data-split origins at a selected lag order. Its final fitting calls are the ones the artificial-data refit follows. |
| [`VAR/var_train_papermetric_w2.R`](VAR/var_train_papermetric_w2.R) | Original training-metric calculation from the saved model workspaces (training-side inputs for the paper tables). |
| [`VAR/VAR_generate_tables.R`](VAR/VAR_generate_tables.R) | Base-R table generator: reads the 31 saved VAR prediction/metric-input RData files documented in `VAR/README.md` from `Compare_models/Graphics/Predicted_values/` and writes `T1_var.csv`, `S3_var.csv`, `S5_var.csv`, `S6_var.csv` and `S7_var_candidate.csv` (VAR rows only) directly into `VAR/`. Tables 1, S3, S5 and S6 match the reference; S7 remains a candidate and must not be presented as an exact reproduction; Table S4 is not generated here. With `--from-refit` it instead reads the artificial-data refit outputs. |

See `VAR/README.md` for the exact commands. Of the eight scripts documented there, `VAR_refit_artificial.R` (the portable artificial-data refit) is referenced but not committed; the other seven are committed. Its named input `Artificial_data_MuSPAD.RData` (the 408-row time-series variant) and the `VAR_train_terms.RData` table input are likewise not committed.

### LSTM

| Script | Purpose |
|---|---|
| [`LSTM/Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R`](LSTM/Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R) | Full-data seven-day preparation: per-origin train/test inputs, imputed/scaled CSVs and scaling metadata for the Python scripts; `make_Graph(model)` (~line 649) builds the full-variant prediction RData (`LSTM_7day_MuSPAD_seeds.RData`, `LSTM_7day_baseline_seeds.RData`). |
| [`LSTM/Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R`](LSTM/Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R) | Seven-day preparation for the four non-full variants via `make_split_data(p, suffix)` (`p` in {0.65, 0.675, 0.75, 0.80}); its Postprocessing section (~lines 202–434) builds the eight non-full prediction RData (old layout; see the caveats in `LSTM/README.md`). |
| [`LSTM/Python_Scripts_plus_Data_prep/Data_prep_threefourthData_diff_seeds.R`](LSTM/Python_Scripts_plus_Data_prep/Data_prep_threefourthData_diff_seeds.R) | Legacy three-quarters (75%) variant preparation; superseded by the diff-splits script; kept for provenance. |
| [`LSTM/Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py`](LSTM/Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py) | MuSPAD fit: input `(1, 7, 122)`, 7-day lag, 30-trial BayesianOptimization over the Table S1 space (1–4 stateful LSTM layers, 32–128 units in steps of 32, per-layer dropout 0.0–0.5, RMSprop, MSE, 20 epochs, batch 1, no shuffling); run per variant, origin and seed (10). |
| [`LSTM/Python_Scripts_plus_Data_prep/python_lstm_onlyIncidence_diff_seeds.py`](LSTM/Python_Scripts_plus_Data_prep/python_lstm_onlyIncidence_diff_seeds.py) | Baseline (incidence-only) fit: fixed three stateful 50-unit LSTM layers with dropout 0.2 after each layer and a time-distributed one-unit output; input `(1, 7, 1)`; run per variant, origin and seed (10). |
| [`LSTM/Python_Scripts_plus_Data_prep/python_lstm_lime_shap_different_clusters.py`](LSTM/Python_Scripts_plus_Data_prep/python_lstm_lime_shap_different_clusters.py) | LIME/SHAP explainability on clusters [3, 4, 5] (supplementary Tables S9–S12; outside the locked table protocol). |
| [`LSTM/Metrics/lstm_seedavg_batch_2026-10-03.R`](LSTM/Metrics/lstm_seedavg_batch_2026-10-03.R) | Locked seed-averaging metrics batch (six guards G1–G6): loads the ten committed prediction RData files, averages the predictions across the ten seeds (35 pooled test days) and produces all committed 10-03 deliverables (Table 1, S3–S7); the committed 10-03 run passed 6/6 guards. Still points at the old `Diff_seeds` layout, so it is not runnable as-is. |

The seven single-run 7-/14-day scripts previously listed in this section were removed from the repository in commit `538a5a5`, which introduced the diff-seeds workflow (see `LSTM/LSTM_reproduction_README.md`, §2.4, for their full history). The complete module documentation — layout, naming conventions, the paper-table map, the locked metrics protocol, and the caveats — is in `LSTM/README.md`.

### Cross-model evaluation

| Script | Purpose |
|---|---|
| [`Compare_models/Metrics/Metrics_all_models_diff_seeds.R`](Compare_models/Metrics/Metrics_all_models_diff_seeds.R) | Cross-model metrics (RMSE) for the non-seed prediction objects (VAR, LASSO, MLP) loaded from the diff-seeds run's `Predicted_values` folder and harmonized to a common time/label scheme. Still points to the original `S:` share path (non-recursive file scan) and is kept for reference. |
| [`Compare_models/Metrics/Metrics_timeseries_models_diff_splits_diff_seeds.R`](Compare_models/Metrics/Metrics_timeseries_models_diff_splits_diff_seeds.R) | Cross-model metrics (RMSE) for the time-series model predictions across the different splits and seeds; its second section still reads the old `Predicted_values\Three_fourth_data\Diff_seeds` layout. All paths remain original `S:` share paths; kept for reference. |
| [`Compare_models/Graphics/Graphic_grid_diff_seeds.R`](Compare_models/Graphics/Graphic_grid_diff_seeds.R) | Overview-graphic helper: `Create_overview_graphic(folder, model_order, show_range)` loads every prediction RData in a given folder, derives the model names from the file names (stripping `_7day`, flagging seed models) and builds the faceted real-versus-predicted comparison graphic. |

The committed prediction RData live under `Compare_models/Graphics/Predicted_values/` in five variant folders — `fullData`, `65percentData`, `675percentData`, `80percentData` and `threefourthData`: LSTM (both models, all five variants), VAR (both models, lag orders 7/14/21, seeded and non-seeded), and LASSO and MLP (full data only). The per-origin subset CSVs and the locked metric tables of the original share are not committed.

### Combined variable-importance reporting

| Script | Purpose |
|---|---|
| [`Variable_Importance/Var_imp_table_shapley_all_methods.R`](Variable_Importance/Var_imp_table_shapley_all_methods.R) | Combines the top MLP and LSTM Shapley features with LASSO and VAR coefficients, then matches machine-readable feature names to the variable codebook and adds feature values/titles for the manuscript supplement. |
| [`Variable_Importance/Var_imp_table_lime_all_methods.R`](Variable_Importance/Var_imp_table_lime_all_methods.R) | Builds the corresponding cross-model table using LIME for MLP/LSTM and coefficients for LASSO/VAR, with codebook titles and feature-value annotations. |
| [`Variable_Importance/Var_imp_lime_shapley_summary_topic_numbers.R`](Variable_Importance/Var_imp_lime_shapley_summary_topic_numbers.R) | Reads manually topic-coded top-feature spreadsheets, compares each topic’s representation among the top 50 features with its overall representation, and creates LIME and SHAP topic-composition figures for the manuscript. |

### Clustering sensitivity (MLP and LSTM)

| Script | Purpose |
|---|---|
| [`Clustering_sensitivity_MLP_LSTM/explainability_cluster_sensitivity.R`](Clustering_sensitivity_MLP_LSTM/explainability_cluster_sensitivity.R) | Base-R sensitivity check for the SHAP/LIME feature-attribution tables: reads the eight `*_k3_k4_k5_merged.csv` files (MLP/LSTM × LIME/SHAP × low/high incidence), validates their columns and quantifies the rank movement of the top features across the k = 3/4/5 pairs. |
| [`Clustering_sensitivity_MLP_LSTM/lstm_lime_category_sensitivity_from_raw.R`](Clustering_sensitivity_MLP_LSTM/lstm_lime_category_sensitivity_from_raw.R) | Base-R reproduction of the LSTM LIME category-sensitivity table: ranks features by absolute LIME attribution per k and regime (top 50 each, ties broken alphabetically), maps them to variable categories via `LSTM_LIME_feature_category_mapping.csv`, and stops if any top-50 feature is unmapped (expects 300 mapped rows). |

The eight merged attribution CSVs and the category-mapping table are committed in this folder.

## Model settings reported in the manuscript

| Model | Selected setup |
|---|---|
| LASSO | 10-fold cross-validation; `lambda.min`; L1 penalty; log-incidence target; MuSPAD predictors at `t` for incidence at `t + 7` |
| MLP | Four hidden layers, 10 ReLU units per layer, linear output, MSE loss, Adam, 30 epochs, batch size equal to 10% of training data |
| VAR | Sparse VAR; L1 penalty; time-series cross-validation; `cvcut = 0.8`; forecast horizon 7; lag orders 7, 14, and 21 |
| LSTM | Baseline (incidence only): three stateful LSTM layers with 50 units each; dropout 0.2 after each layer; time-distributed one-unit output. MuSPAD: best of 30 BayesianOptimization trials per origin over the search space documented in `LSTM/README.md`, applied identically to seeds 1–10; five rolling origins, seven-day horizon |

The manuscript’s primary full-data seven-day RMSE results are:

| Model | Training, MuSPAD | Test, MuSPAD | Training, baseline | Test, baseline |
|---|---:|---:|---:|---:|
| LASSO | 33.09 | 9.28 | 70.73 | 61.70 |
| MLP | 30.52 | 15.13 | 75.51 | 33.65 |
| LSTM | 31.34 | 4.36 | 68.84 | 3.94 |
| VAR, `p = 7` | 34.59 | 36.16 | 38.45 | 30.49 |
| VAR, `p = 14` | 36.93 | 15.54 | 38.80 | 30.41 |
| VAR, `p = 21` | 36.14 | 11.00 | 39.42 | 27.18 |

## Metric conventions

The repository contains both MSE and RMSE outputs:

- model-specific scripts often call `Metrics::mse()` and save files named `mse_*`;
- the MLP validation tables explicitly store `train_mse` and `val_mse`; and
- the two current `Compare_models/Metrics/` scripts (diff-seeds versions) call `Metrics::rmse()`; they still reference the original `S:` share paths and the old `Diff_seeds` layout and are kept for reference only.

Do not label a model-specific MSE value as RMSE without taking its square root. In particular, the DP comparison values `977.57`, `1337.32`, and related entries are on the MSE scale in the code; their corresponding RMSE values are `sqrt(MSE)`.

## Differential-privacy implementation notes

The active DP-SGD script contains the following training parameters:

- clipping threshold `C = 1`;
- noise multipliers `sigma = 1.627197265625`, `2.7294921875`, and `9.0625`;
- Poisson-style random inclusion with nominal batch size 32;
- learning rate 0.01;
- 30 epochs and eight nominal batches per epoch;
- MSE loss;
- Adam and RMSprop during validation, with Adam in the final test phase; and
- 10-fold validation with fold-level SHAP and LIME calculations.

The committed script does **not** calculate or store `epsilon`, `delta`, or a privacy-accountant trace. The mapping from the three noise multipliers to the manuscript’s privacy budgets (`epsilon = 8, 4, 1`) and the RDP-accounting assumptions therefore need to be supplied from the separate privacy-accounting analysis or supplement before the reported guarantees can be independently reproduced.


## Citation

If you use this code, please cite the accompanying manuscript:

> Krepel J, Binkyte R, Kerkouche R, Harries M, Klett-Tammen CJ, Fritz M, Kesselheim S, Kühn M, Bazarova A, Lange B, MuSPAD study group, RESPINOW study group. *Predicting COVID-19 incidence from seroprevalence and population-based cohort data using interpretable machine learning with differential privacy analysis.*

Please replace this entry with the journal citation and DOI once the article is published.
