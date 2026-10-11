# LASSO reproducibility

This folder contains the MuSPAD LASSO model, its time-only baseline, and the
calculations for their rows in main-text Table 1 and supplementary Tables S3
and S5. It contains code and this README only; generated results are not inputs
to the bundle and are not committed.

## Scripts and their relationship

| Script | Purpose |
| --- | --- |
| `LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R` | Original MuSPAD fitting procedure; saves observed/predicted incidence |
| `LASSO_model_only_time_log_all_cities.R` | Original time-only fitting procedure; saves observed/predicted incidence |
| `lasso_fullmetrics_2026-10-02.R` | Table 1 and S3 metrics and bootstrap CIs |
| `lasso_pointcalibration_2026-10-02.R` | Table S5 calibration intercept/slope, CIs and joint test |

The last two scripts derive results from the predictions saved by the first
two. They also accept the original saved predictions without refitting. The
full-metrics script calculates the bootstrap intervals itself, so a separate
bootstrap script and saved bootstrap draws are unnecessary. The seed-sweep
comparison script and all duplicate result/documentation files are omitted.

## Inputs needed to reproduce the reported values

Only the original saved predictions are required:

- `LASSO_7day_MuSPAD.RData`, containing `graph_LASSO_7day_M`.
- `LASSO_7day_baseline.RData`, containing `graph_LASSO_7day_BL`.

These must include training and test observations, dates, and predictions
(columns `time`, `Incidence`, `set`, `real_vs_pred`). The scripts expect 251
training and 27 test observations per model. They look in
`../Compare_models/Graphics/Predicted_values/`, or the directory specified by
`LASSO_PRED_DIR`.

The two prediction files are shared repository inputs at
`Compare_models/Graphics/Predicted_values/`; do not duplicate them in `LASSO/`.
Include them when pushing the repository. The modeling dataset is needed only
for optional refitting. All result CSVs, bootstrap draws and other generated
outputs can be regenerated and are omitted from this bundle.

## Reproduce the tables from the original predictions

From `LASSO/`, using R 4.3 or later:

```sh
Rscript lasso_fullmetrics_2026-10-02.R
Rscript lasso_pointcalibration_2026-10-02.R
```

Both scripts use base R. They run independently and write their CSV results
(and the calibration plot) into `repro_results/`; this directory is created
when needed. No pre-existing output CSVs or bootstrap files are required.
For inputs elsewhere, set `LASSO_PRED_DIR`; optionally set `LASSO_OUT_DIR` for
a different output directory.

The metrics script computes RMSE, sMAPE (0-2 scale), MAE, MASE, R2 and MAPE.
MASE uses the training-series one-step persistence MAE. Test-metric 95% CIs use
10,000 moving-block bootstrap replicates, block length seven observations,
seed 42; IID intervals are included for comparison. Blocks follow observation
order, which includes gaps in calendar dates. These CIs quantify test-observation
resampling uncertainty, conditional on the saved model predictions.

Calibration fits observed incidence against predicted incidence
by OLS, with t-based coefficient CIs and a joint test of intercept=0, slope=1.
The scripts compare current predictions with published reference values and
warn about differences rather than forcing the reported numbers.

## Optional: refit the models

Install `tidyverse`, `glmnet`, `missForest` and `Metrics`. Supply the original
modeling RData file containing `df_ts_inc` via `LASSO_DATA_FILE`, or place it in
`../Datasets/` with a name matching `binary_classnumber_var_subset_missing_zero`.
The two fitting scripts preserve the original chronological 90/10 split,
log-incidence target, lambda selection and back-transformed predictions.

For example, in PowerShell:

```powershell
$env:LASSO_DATA_FILE = 'C:/path/to/original_modeling_data.RData'
$env:LASSO_PRED_DIR = 'C:/path/to/new_predictions'
$env:LASSO_SEED = '42'
Rscript LASSO_model_levelcount_log_scale_all_cities_varsubsetbasic_without_cities_var_imp.R
Rscript LASSO_model_only_time_log_all_cities.R
Rscript lasso_fullmetrics_2026-10-02.R
Rscript lasso_pointcalibration_2026-10-02.R
```

Use a separate prediction directory to preserve the original saved run.
The original fits were unseeded; `LASSO_SEED` is optional for repeatable new
fits. Refitting can change the numerical table values. Exact regeneration of
the reported values therefore uses the original predictions.

Changes from the supplied scripts are limited to path configuration, removing
redundant dependencies on generated check files and Markdown output, and
reporting published-value differences as warnings. Model fitting and metric,
bootstrap and calibration calculations are retained. Prediction-interval
coverage calculations are omitted because they are not required for these tables.
