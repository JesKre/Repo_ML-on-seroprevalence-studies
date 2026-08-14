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
├── Data_Preprocessing/   # Daily aggregation and construction of modeling datasets
├── LASSO/                # MuSPAD and time-only LASSO models
├── MLP/                  # MLP, baseline, optimizer comparison, and DP-SGD
├── VAR/                  # Sparse VAR models and incidence-only baselines
├── LSTM/                 # R/Python LSTM workflow and explanation post-processing
├── Compare_models/       # Cross-model metrics and manuscript graphics
├── Variable_Importance/  # Cross-model SHAP/LIME/coefficient tables and topic summaries
└── Variable_selection_codebook.csv
```

The script catalogue below covers every script currently present in the repository. Historical `Archive` or `Archiv` directories, if present in another branch or checkout, are intentionally excluded.

## Data access

Individual-level MuSPAD data are not distributed in this repository because they contain sensitive health information. Access can be requested through the [MuSPAD data access process at SeroHub](https://serohub.net/) and is subject to approval and applicable data-protection requirements.

The following external inputs are required:

- the cleaned individual-level MuSPAD object used as `df_merge_clean`;
- the selected-variable file corresponding to the included [`Variable_selection_codebook.csv`](Variable_selection_codebook.csv);
- RKI county-level incidence data, referred to in the scripts as `COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv`; and
- intermediate `.RData` and `.csv` files produced by earlier workflow stages.

`Variable_selection_codebook.csv` is semicolon-delimited and documents the 77 selected source variables, including source, title, data type, description, unit, and response options.

## Software requirements

The repository does not include a lockfile, so exact R package versions are not recorded. The active scripts use the following packages:

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
  -> VAR
  -> Compare_models
  -> Variable_Importance
```

## Script reference

### Data preprocessing

| Script | Purpose |
|---|---|
| [`Data_Preprocessing/01_Data_Aggregation.R`](Data_Preprocessing/01_Data_Aggregation.R) | Loads cleaned participant-level MuSPAD data, keeps the selected variables, corrects known values, converts dates and an ordinal health-status field, identifies numeric/binary/categorical fields, expands category levels, and aggregates participant records by day. It also identifies variables that are completely missing on at least 20% of sampling days and creates city-presence and daily time-series outputs. |
| [`Data_Preprocessing/02_Prepare_for_Analysis.R`](Data_Preprocessing/02_Prepare_for_Analysis.R) | Removes high-missingness and near-zero-information features, joins county-level RKI incidence, and creates the two modeling datasets. The time-agnostic dataset aligns MuSPAD values at `t` with incidence at `t + 7`; the time-aware dataset fills the complete daily calendar, carries sampling location forward, and retains current incidence for sequential models. |

### LASSO

| Script | Purpose |
|---|---|
| [`LASSO/LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R`](LASSO/LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R) | Fits the main MuSPAD LASSO model. It applies a chronological 90/10 train-test split, separate `missForest` imputation, training-derived z-scaling, a log transform of incidence, and 10-fold `cv.glmnet` selection of `lambda.min`. It back-transforms predictions, calculates errors, ranks non-zero coefficients for variable importance, and saves predictions for cross-model plots. |
| [`LASSO/LASSO_model_only_time_log_all_cities.R`](LASSO/LASSO_model_only_time_log_all_cities.R) | Fits the time-only LASSO baseline with the same chronological split, log-incidence target, 10-fold lambda selection, back-transformed predictions, metrics, and plotting outputs. |

### Multilayer perceptron and differential privacy

| Script | Purpose |
|---|---|
| [`MLP/Keras_MuSPAD_Expl_cluster_incidence.R`](MLP/Keras_MuSPAD_Expl_cluster_incidence.R) | Fits the primary MuSPAD MLP with four hidden layers of 10 ReLU units, a linear output, and MSE loss. It uses a 90/10 chronological train-test split and one randomly assigned fold for validation, then retrains on the training-validation data. The latter half clusters incidence values and computes/aggregates `iml` Shapley and LIME explanations for the lowest- and highest-incidence clusters. It also exports predictions and model-specific MSE values. |
| [`MLP/Keras_only_time.R`](MLP/Keras_only_time.R) | Fits the time-only MLP baseline using the same four-by-ten architecture, validation design, 30 epochs, prediction export, and plotting workflow. |
| [`MLP/Keras_MuSPAD_compare_DP_10foldcv_Expl.R`](MLP/Keras_MuSPAD_compare_DP_10foldcv_Expl.R) | Runs the non-private 10-fold MLP comparison between Adam and RMSprop. It saves weights at epochs 1, 15, and 30, calculates fold-level training/validation MSE, aggregates Shapley and LIME importance, retrains a final model, joins its performance with the DP results, and produces the privacy-comparison plot. |
| [`MLP/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R`](MLP/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R) | Implements custom DP-SGD for the four-by-ten MLP. It computes gradients per example, clips each gradient tensor to an L2 norm of `C = 1`, adds Gaussian noise with standard deviation `sigma * C`, and averages by a nominal batch size of 32. Validation compares Adam and RMSprop over 10 folds for noise multipliers 1.627197265625, 2.7294921875, and 9.0625; final testing uses Adam, learning rate 0.01, 30 epochs, and eight nominal batches per epoch. The script exports model weights, predictions, MAPE/SMAPE/RMSLE/RMSE, and fold-level SHAP/LIME summaries. |

### Vector autoregression

| Script | Purpose |
|---|---|
| [`VAR/bigtime_MuSPAD_allData.R`](VAR/bigtime_MuSPAD_allData.R) | Fits sparse VAR models to the full time-aware MuSPAD dataset. Validation searches `HLag` versus `L1`, `cv`/`bic`/`aic` selection, and `cvcut` 0.8/0.9 at a 7-day horizon. The test block fits the selected `L1` plus time-series-CV specification, evaluates forecasts and fitted values, and exports predictions for the manuscript comparisons. Set `p` to 7, 14, or 21 for the three reported lag variants. |
| [`VAR/bigtime_MuSPAD_threefourthData.R`](VAR/bigtime_MuSPAD_threefourthData.R) | Repeats sparse-VAR tuning and evaluation using the first-three-quarters robustness split generated by the LSTM data-preparation script. It writes validation results, saved models, MSE values, predictions, and plots. |
| [`VAR/bigtime_noMuSPAD_only_incidence_allData.R`](VAR/bigtime_noMuSPAD_only_incidence_allData.R) | Fits the full-data VAR baseline using incidence as the only series, recursively forecasts seven days, evaluates training/test MSE, and exports prediction objects and plots. Set `p` to 7, 14, or 21 to match the MuSPAD models. |
| [`VAR/bigtime_noMuSPAD_only_incidence_threefourthData.R`](VAR/bigtime_noMuSPAD_only_incidence_threefourthData.R) | Fits and plots the incidence-only VAR baseline for the first-three-quarters robustness split. |

### LSTM

| Script | Purpose |
|---|---|
| [`LSTM/Data_prep_for_Python_script_7day_prediction.R`](LSTM/Data_prep_for_Python_script_7day_prediction.R) | Prepares the full 7-day LSTM workflow. It creates imputed/scaled training, validation, and test CSVs; saves scaling parameters; creates fully imputed data for explanation; clusters incidence; writes low/high cluster indices and feature means; imports Python SHAP/LIME results; returns explanation values to the incidence scale; aggregates them over samples and time; and reconstructs training/test prediction plots and MSE outputs. |
| [`LSTM/Data_prep_for_Python_script_14day_prediction.R`](LSTM/Data_prep_for_Python_script_14day_prediction.R) | Fourteen-day counterpart of the preceding script. It prepares 14-day train/validation/test inputs, evaluates Python predictions, and rescales/aggregates SHAP and LIME results for the 14-day horizon. |
| [`LSTM/Data_prep_halfData_threefourthData.R`](LSTM/Data_prep_halfData_threefourthData.R) | Generates half-series and three-quarters-series train/validation/test CSVs and scaling metadata. Its plotting section evaluates both the MuSPAD and incidence-only LSTM outputs for the three-quarters robustness analysis and saves standardized prediction objects for model comparison. |
| [`LSTM/python_lstm_lime_shap.py`](LSTM/python_lstm_lime_shap.py) | Trains the stateful LSTM for full-data 7- and 14-day horizons and the three-quarters 7-day variant. The model has three stateful 50-unit LSTM layers, dropout 0.2 after each layer, a time-distributed linear output, RMSprop, MSE loss, batch size 1, no shuffling, and 20 epochs. It saves models and predictions. For explanation, it transfers the trained weights to an equivalent stateless network, uses `shap.GradientExplainer` with a 100-sequence background sample, and uses recurrent LIME for low/high-incidence input windows. |
| [`LSTM/Feature_means_for_SHAP_LIME_inputs.R`](LSTM/Feature_means_for_SHAP_LIME_inputs.R) | Calculates original-scale feature means for the seven-day input windows that precede low- and high-incidence output samples. It provides overall-cluster and per-time-step means used to interpret SHAP/LIME directions. |
| [`LSTM/Graphics_only_Incidence_7day_prediction.R`](LSTM/Graphics_only_Incidence_7day_prediction.R) | Reads prediction CSVs from the 7-day incidence-only LSTM baseline, reverses incidence scaling, computes training/test MSE, and exports the baseline prediction object and figures. |
| [`LSTM/Graphics_only_Incidence_14day_prediction.R`](LSTM/Graphics_only_Incidence_14day_prediction.R) | Fourteen-day version of the incidence-only LSTM post-processing and plotting workflow. |

### Cross-model evaluation

| Script | Purpose |
|---|---|
| [`Compare_models/Metrics_all_models.R`](Compare_models/Metrics_all_models.R) | Loads all saved full-data 7-day prediction objects, harmonizes their labels, and calculates MAPE, SMAPE, RMSLE, and RMSE by training/test split. It writes wide comparison tables and LaTeX table code. |
| [`Compare_models/Metrics_all_models_14day_Pred.R`](Compare_models/Metrics_all_models_14day_Pred.R) | Produces the same metric tables for the 14-day prediction outputs. |
| [`Compare_models/Metrics_all_models_three_fourth_data.R`](Compare_models/Metrics_all_models_three_fourth_data.R) | Produces the same metric tables for the first-three-quarters robustness analysis. |
| [`Compare_models/Graphic_grid.R`](Compare_models/Graphic_grid.R) | Combines saved real-versus-predicted objects into faceted model-comparison figures for the full-data and three-quarters analyses and exports PNG/EPS manuscript graphics. |

### Combined variable-importance reporting

| Script | Purpose |
|---|---|
| [`Variable_Importance/Var_imp_table_shapley_all_methods.R`](Variable_Importance/Var_imp_table_shapley_all_methods.R) | Combines the top MLP and LSTM Shapley features with LASSO and VAR coefficients, then matches machine-readable feature names to the variable codebook and adds feature values/titles for the manuscript supplement. |
| [`Variable_Importance/Var_imp_table_lime_all_methods.R`](Variable_Importance/Var_imp_table_lime_all_methods.R) | Builds the corresponding cross-model table using LIME for MLP/LSTM and coefficients for LASSO/VAR, with codebook titles and feature-value annotations. |
| [`Variable_Importance/Var_imp_lime_shapley_summary_topic_numbers.R`](Variable_Importance/Var_imp_lime_shapley_summary_topic_numbers.R) | Reads manually topic-coded top-feature spreadsheets, compares each topic’s representation among the top 50 features with its overall representation, and creates LIME and SHAP topic-composition figures for the manuscript. |

## Model settings reported in the manuscript

| Model | Selected setup |
|---|---|
| LASSO | 10-fold cross-validation; `lambda.min`; L1 penalty; log-incidence target; MuSPAD predictors at `t` for incidence at `t + 7` |
| MLP | Four hidden layers, 10 ReLU units per layer, linear output, MSE loss, Adam, 30 epochs, batch size equal to 10% of training data |
| VAR | Sparse VAR; L1 penalty; time-series cross-validation; `cvcut = 0.8`; forecast horizon 7; lag orders 7, 14, and 21 |
| LSTM | Three stateful LSTM layers with 50 units each; dropout 0.2 after each layer; time-distributed one-unit output |

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
- the three `Compare_models/Metrics_*.R` scripts call `Metrics::rmse()` and produce the RMSE tables used for the main model comparison.

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
