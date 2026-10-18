# Predicting COVID-19 incidence from MuSPAD seroprevalence data

Code accompanying the manuscript [**“Predicting COVID-19 incidence from seroprevalence and population-based cohort data using interpretable machine learning with differential privacy analysis.”**](https://www.medrxiv.org/content/10.64898/2026.04.01.26349876v1)

This repository compares four model families—LASSO regression, multilayer perceptrons (MLP), sparse vector autoregression (VAR), and long short-term memory networks (LSTM)—for predicting COVID-19 incidence using aggregated German Multilocal SeroPrevalence (MuSPAD) study data. It includes time-only or incidence-only baselines, coefficient-based and SHAP/LIME explanations, differential-privacy analyses, and scripts for regenerating evaluation tables.

**Start here:** use the documentation map below to find an analysis. For table regeneration, begin with the saved results; model retraining and reconstruction from participant-level data are separate workflows.

## Documentation map

The linked READMEs describe module-specific inputs, environments, methods, commands, and outputs.

| Area | Guide | What you will find |
| --- | --- | --- |
| Artificial data | [Artificial datasets](Artificial_datasets/README.md) | Time-agnostic and time-aware synthetic-feature datasets, CSV/RData formats, date alignment, imputation indicators, and generation procedure. |
| LASSO | [LASSO reproducibility](LASSO/README.md) | MuSPAD and time-only fitting scripts; metrics, bootstrap confidence intervals, and calibration from saved predictions. |
| MLP | [MLP scripts and table reproduction](MLP/README.md) | Original fitting and explanations, ten-seed Keras-2 refits, and performance/calibration table generation. |
| MLP with differential privacy | [DP table reproduction](MLP/Table_reproduction_DP/MLP_with_DP/README.md) | Table generation from supplied per-fold results, cached training input, seeded DP workers, and zero-noise control. |
| VAR | [VAR models and table reproduction](VAR/README.md) | MuSPAD and incidence-only models, saved-prediction table generation, and artificial-data refitting across lag orders and rolling origins. |
| LSTM | [LSTM models](LSTM/README.md) | R preparation, Python training, saved prediction layout, and evaluation overview. |
| LSTM evaluation | [Metrics and reproducibility](LSTM/Metrics/README.md) | Seed averaging, bootstrap intervals, calibration, leave-one-origin-out analysis, and prediction-interval evaluation. |
| LSTM provenance | [Historical reproduction record](LSTM/LSTM_reproduction_README.md) | Model generations, original shared-drive layout, evaluation protocol, and historical caveats. Some referenced artifacts are outside this checkout. |
| Explanation sensitivity | [Clustering sensitivity](Clustering_sensitivity_MLP_LSTM/README.md) | SHAP/LIME robustness across k = 3, 4, and 5 incidence clusters, and LSTM LIME category sensitivity. |

## Study and data overview

MuSPAD collected questionnaire and serological data in two survey rounds during 2020–2021 across eight German regions. The analysis aggregates participant records by day and combines them with regional seven-day incidence from the Robert Koch Institute (RKI).

| Analysis | Data structure | Models and baseline |
| --- | --- | --- |
| Time-agnostic | 279 survey days; predictors at t paired with incidence at t + 7; chronological training/test split | LASSO and MLP; time-only baselines |
| Time-aware | 408 consecutive days; seven-day forecasting evaluated at five origins: 65%, 67.5%, 75%, 80%, and full training period | VAR and LSTM; incidence-only baselines |

The original scripts also contain additional horizon and robustness analyses. Follow the relevant module guide for the exact scope.

Individual-level MuSPAD data are restricted and are not distributed here. The existing project documentation directs access requests to [SeroHub](https://serohub.net/); access is subject to approval. Rebuilding the original datasets additionally requires the cleaned participant object (`df_merge_clean`), the selected-variable input, and the RKI county-level incidence file referenced by the preprocessing scripts.

The [artificial datasets](Artificial_datasets/README.md) support method development without participant-level access. Their feature values are synthetic, while incidence, dates, and time indices retain observed values. They do **not** reproduce the manuscript's fitted results or preserve the original feature time dependencies. RData files contain the modeling object `df_ts_inc`; CSV exports use semicolons and decimal commas and include auxiliary date/imputation fields.

## Choose a reproduction workflow

### Regenerate results from saved predictions or summaries

This route avoids model retraining. The LASSO, MLP, DP table generator, and clustering-sensitivity scripts use base R. The following commands use the current repository layout; run them from the repository root in a POSIX shell:

```sh
# LASSO: metrics, bootstrap intervals, and point calibration
LASSO_PRED_DIR="$PWD/Compare_models/Graphics/Predicted_values/fullData" \
  Rscript LASSO/lasso_fullmetrics_2026-10-02.R
LASSO_PRED_DIR="$PWD/Compare_models/Graphics/Predicted_values/fullData" \
  Rscript LASSO/lasso_pointcalibration_2026-10-02.R

# MLP: performance, seed stability, and calibration tables
MLP_PRED_DIR="$PWD/Compare_models/Graphics/Predicted_values/fullData" \
  Rscript MLP/MLP_generate_tables.R

# DP comparison: regenerate tables from the supplied per-fold master CSVs
Rscript MLP/Table_reproduction_DP/MLP_with_DP/scripts/04_make_tables.R
```

In PowerShell, set the corresponding environment variable before running each LASSO/MLP command:

```powershell
$env:LASSO_PRED_DIR = (Resolve-Path 'Compare_models/Graphics/Predicted_values/fullData').Path
$env:MLP_PRED_DIR = $env:LASSO_PRED_DIR
```

LASSO writes to `LASSO/repro_results/`; MLP writes `MLP_table*.csv` in `MLP/`; the DP workflow writes `table7.csv` and `tableS8.csv` in its `results/` folder. The latter filename retains older supplementary numbering.

For clustering sensitivity, run from its input directory:

```sh
cd Clustering_sensitivity_MLP_LSTM
Rscript explainability_cluster_sensitivity.R
Rscript lstm_lime_category_sensitivity_from_raw.R
```

For LSTM evaluation, follow [the metrics guide](LSTM/Metrics/README.md) and configure the historical paths and CSV verification inputs it describes.

### VAR tables from saved predictions

The repository includes [VAR_generate_tables.R](VAR/VAR_generate_tables.R), [VAR_refit_artificial.R](VAR/VAR_refit_artificial.R), and the training-metric input [VAR_train_terms.RData](Compare_models/Graphics/Predicted_values/fullData/VAR_train_terms.RData).

The table generator uses base R and expects **31 RData files together in one input directory**: the 30 seeded prediction files (two models × three lag orders × five origins), plus `VAR_train_terms.RData`. The predictions are stored under each origin's `VAR_with_seeds/` subfolder; the training terms are in `fullData/`. Setting `VAR_PRED_DIR` to `fullData/` alone therefore does not supply all inputs.

From the repository root in a POSIX shell, copy these inputs into a temporary directory and point the generator there:

```sh
var_inputs=$(mktemp -d)
cp Compare_models/Graphics/Predicted_values/*/VAR_with_seeds/*.RData "$var_inputs/"
cp Compare_models/Graphics/Predicted_values/fullData/VAR_train_terms.RData "$var_inputs/"
VAR_PRED_DIR="$var_inputs" Rscript VAR/VAR_generate_tables.R
```

The script writes `T1_var.csv`, `S6_var.csv`, `S8_var.csv`, `S9_var.csv`, and `S10_var_candidate.csv` into `VAR/`. These contain VAR rows only. `S10_var_candidate.csv` remains an unverified candidate that does not fully match the paper. The temporary input directory can be deleted after the run; the repository's supplied inputs remain in their original folders.

In PowerShell, create a separate input directory, copy the same 31 files into it, set `$env:VAR_PRED_DIR` to its absolute path, and run `Rscript VAR/VAR_generate_tables.R`.

### Refit models or rebuild the original analysis

Use each model's guide for dependencies and input configuration. These workflows require more software and data than table regeneration:

- **LASSO:** R with `glmnet`, `missForest`, and the packages listed in its guide.
- **MLP refits:** legacy R `keras` and a compatible Keras-2 Python environment; the documented setup uses TensorFlow/Keras 2.15.1 and `MLP_PYTHON`.
- **LSTM training:** R preparation followed by Python TensorFlow/Keras, `keras_tuner`, NumPy, and pandas; explanation scripts also use LIME/SHAP. Check the scripts' compatibility requirements before installing.
- **DP retraining:** its own R/Python environment selected through `DP10SEED_PYTHON`; follow the dedicated guide for legacy Keras configuration.
- **VAR fitting:** R with `bigtime`; the portable artificial-data workflow is shown below.

For a VAR demonstration using the supplied 408-row artificial time series, run from the repository root in a POSIX shell:

```sh
# Explicitly select the time-aware input: the script's default points to the 279-row file.
VAR_DATA_FILE="$PWD/Artificial_datasets/Artificial_data_MuSPAD_time_aware.RData" \
  Rscript VAR/VAR_refit_artificial.R --check-input
VAR_DATA_FILE="$PWD/Artificial_datasets/Artificial_data_MuSPAD_time_aware.RData" \
  Rscript VAR/VAR_refit_artificial.R
VAR_PRED_DIR="$PWD/VAR" Rscript VAR/VAR_generate_tables.R --from-refit
```

`--check-input` validates the data without fitting. The refit runs 300 fits (two models × three lag orders × five origins × ten seeds) and writes its 31 prediction/training-input RData files into `VAR/` by default. The final command generates tables from these new files. These artificial-data results demonstrate the method; they do not recover the manuscript's numerical results. Use `VAR_REFIT_DIR` to change the refit output location and set `VAR_PRED_DIR` to that same directory for table generation.

Several original scripts retain absolute Windows `S:` paths, assume existing output folders, or depend on earlier interactive steps. Configure these before execution. Keep new fitting outputs separate from the supplied manuscript predictions.

The workflow is:

```text
Variable selection + restricted MuSPAD data + RKI incidence
    -> Data_Preprocessing
    -> time-agnostic data -> LASSO / MLP
    -> time-aware data   -> LSTM R preparation -> Python fitting -> R post-processing
                         -> VAR (including prepared inputs from the time-series workflow)
    -> saved predictions -> model-specific evaluation / comparison graphics
    -> saved explanations and coefficients -> Variable_Importance

Saved DP per-fold results -> DP tables
Merged SHAP/LIME CSVs     -> clustering-sensitivity summaries
```

## Data_Preprocessing/

[Data_Preprocessing](Data_Preprocessing/) constructs daily aggregates and modeling inputs.

| File | Purpose |
| --- | --- |
| [01_Data_Aggregation.R](Data_Preprocessing/01_Data_Aggregation.R) | Loads cleaned participant data and selected variables, prepares numeric/categorical fields, aggregates them by day, and identifies features with extensive missingness. |
| [02_Prepare_for_Analysis.R](Data_Preprocessing/02_Prepare_for_Analysis.R) | Filters features, joins RKI incidence, and constructs survey-day and complete-calendar modeling datasets. |
| [Create_aggregated_artificial_dataset.R](Data_Preprocessing/Create_aggregated_artificial_dataset.R) | Uses `missForest` and `synthpop` to create synthetic-feature versions of the modeling datasets. See the [artificial-data guide](Artificial_datasets/README.md) for the packaged files. |
| [Participants_per_day.csv](Data_Preprocessing/Participants_per_day.csv) | Supplementary daily participant-count output. |

These scripts require original inputs and path adaptation. The selected-variable input is called `Variable_selection_basic.csv` in the scripts; reconcile it with the codebook supplied in `Variable_selection/`.

### Datasets/ and Variable_selection/

[Datasets](Datasets/) contains the weekly-incidence time-series modeling input, [Time_series_data_incidence_1W_binary_classnumber_var_subset_missing_zero_2025-02-14.RData](Datasets/Time_series_data_incidence_1W_binary_classnumber_var_subset_missing_zero_2025-02-14.RData). It is separate from the explicitly labeled [artificial datasets](Artificial_datasets/). The folder does not contain every external or intermediate dataset named by the analysis scripts.

[Variable_selection](Variable_selection/) provides:

- [Variable_selection_codebook.csv](Variable_selection/Variable_selection_codebook.csv): semicolon-delimited codebook for the 74 selected source variables, including titles, types, descriptions, units, and response options.
- [Codebook_variables_excluded.xlsx](Variable_selection/Codebook_variables_excluded.xlsx): record of considered but excluded variables.

### Compare_models/

[Compare_models](Compare_models/) holds shared prediction inputs, cross-model graphics, and older comparison scripts.

| Subfolder or script | Purpose |
| --- | --- |
| [Graphics/Predicted_values/](Compare_models/Graphics/Predicted_values/) | Saved observed/predicted incidence in RData format, organized by forecast origin. |
| [Graphics/Graphic_grid_diff_seeds.R](Compare_models/Graphics/Graphic_grid_diff_seeds.R) | Creates faceted observed-versus-predicted comparison figures, including displays across seeds. |
| [Metrics/Metrics_all_models_diff_seeds.R](Compare_models/Metrics/Metrics_all_models_diff_seeds.R) | Cross-model RMSE summaries from saved predictions. |
| [Metrics/Metrics_timeseries_models_diff_splits_diff_seeds.R](Compare_models/Metrics/Metrics_timeseries_models_diff_splits_diff_seeds.R) | Time-series model comparisons across splits and seeds. |

Prediction folders are [fullData](Compare_models/Graphics/Predicted_values/fullData/), [65percentData](Compare_models/Graphics/Predicted_values/65percentData/), [675percentData](Compare_models/Graphics/Predicted_values/675percentData/) (67.5%), [threefourthData](Compare_models/Graphics/Predicted_values/threefourthData/) (75%), and [80percentData](Compare_models/Graphics/Predicted_values/80percentData/). All contain LSTM and non-seeded VAR predictions; each has a `VAR_with_seeds/` subfolder for seeded VAR predictions. `fullData/` additionally contains LASSO and MLP predictions and `VAR_train_terms.RData`, which supplies the training observations, fitted values, and preceding observations used by the VAR table generator.

The comparison scripts retain shared-drive paths and historical layouts such as `Diff_seeds/`. Use the model-specific evaluation guides for the revised table calculations.

### Variable_Importance/

[Variable_Importance](Variable_Importance/) combines model explanations into manuscript tables and topic summaries.

| Script | Purpose |
| --- | --- |
| [Var_imp_table_shapley_all_methods.R](Variable_Importance/Var_imp_table_shapley_all_methods.R) | Combines MLP/LSTM Shapley rankings with LASSO/VAR coefficients and attaches codebook labels and feature values. |
| [Var_imp_table_lime_all_methods.R](Variable_Importance/Var_imp_table_lime_all_methods.R) | Builds the corresponding combined table using LIME for the neural networks. |
| [Var_imp_lime_shapley_summary_topic_numbers.R](Variable_Importance/Var_imp_lime_shapley_summary_topic_numbers.R) | Reads manually topic-coded spreadsheets and plots topic representation among the top 50 features relative to all features. |

These scripts need upstream explanation/coefficient outputs and manually coded spreadsheets that are not all committed, plus shared-drive path changes. For a self-contained sensitivity analysis using supplied attribution CSVs, see [Clustering_sensitivity_MLP_LSTM](Clustering_sensitivity_MLP_LSTM/README.md).

### Additional folders

- [LSTM/Python_Scripts_plus_Data_prep/](LSTM/Python_Scripts_plus_Data_prep/) contains R preparation/post-processing, two Python training scripts, and a cluster-based LIME/SHAP script. Their roles are described in the [LSTM guide](LSTM/README.md).
- [MLP/DP_Analysis/](MLP/DP_Analysis/) contains the original DP-SGD implementation and the non-private Adam/RMSprop comparison, including explanation analyses. Use the [separate DP reproduction guide](MLP/Table_reproduction_DP/MLP_with_DP/README.md) for table regeneration.
- [MLP/Table_reproduction_DP/](MLP/Table_reproduction_DP/) contains the `MLP_with_DP/` package: `scripts/` holds caching, training workers, and table generation; `data/` holds the supplied cache; `results/` holds the two per-fold master CSVs.

The root also contains [LSTM.zip](LSTM.zip), an archive alongside the unpacked module. Start with the [LSTM directory and its guide](LSTM/README.md) when navigating the code.


## Citation

If you use this code, please cite the accompanying manuscript:

> Predicting COVID-19 incidence from seroprevalence and population-based cohort data using interpretable machine learning with differential privacy analysis
Jessica Krepel, Ruta Binkyte, Raouf Kerkouche, Manuela Harries, Carolina J. Klett-Tammen, Mario Fritz, Stefan Kesselheim, Martin Kühn, Alina Bazarova, Berit Lange, with the MuSPAD study group, RESPINOW study group
medRxiv 2026.04.01.26349876; doi: https://doi.org/10.64898/2026.04.01.26349876


