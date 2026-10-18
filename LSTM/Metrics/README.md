# LSTM evaluation and reproducibility

## Overview

The script `lstm_seedavg_batch_2026-10-03.R` reproduces the performance evaluation of the MuSPAD-based LSTM and the incidence-only LSTM baseline.

It calculates:

- Performance metrics and confidence intervals
- Calibration intercept and slope
- Leave-one-forecast-origin-out sensitivity analyses
- Prediction intervals and their empirical coverage

The analysis uses previously saved predictions from 10 random seeds at each of five forecast origins. **No model retraining is required.**

## Input data

All required LSTM prediction files are included in the GitHub repository under:

`Compare_models/Graphics/Predicted_values/`

The files are organized into five subdirectories corresponding to the forecast origins:

| Directory | Forecast origin |
|---|---|
| `65percentData/` | 65% |
| `675percentData/` | 67.5% |
| `threefourthData/` | 75% |
| `80percentData/` | 80% |
| `fullData/` | Full training period |

Each directory contains two `.RData` files: one for the MuSPAD-based LSTM and one for the incidence-only baseline. Each file contains predictions from 10 random seeds.

The script uses these saved predictions and corresponding observed incidence values to calculate the evaluation statistics. It does not require direct access to individual-level MuSPAD data.

## Evaluation methods

**Seed averaging.** Predictions are averaged across the 10 random seeds for each forecast origin and prediction day. Primary performance metrics are calculated from these averaged predictions, rather than by averaging metrics across seeds.

**Confidence intervals.** The pooled test evaluation comprises 35 observations from five forecast origins, with seven consecutive days per origin. Confidence intervals are estimated using an origin-block bootstrap with 2,000 replicates.

**Calibration.** Calibration is assessed by regressing observed incidence on seed-averaged predictions across the 35 out-of-sample observations. The analysis reports the intercept, slope, confidence intervals, and a joint test of ideal calibration (intercept = 0, slope = 1).

**Leave-one-forecast-origin-out analysis.** Each forecast origin is excluded in turn, and performance is recalculated using the remaining four origins (28 observations). Paired differences between the MuSPAD-based and incidence-only models are evaluated using origin-block bootstrap confidence intervals.

**Prediction intervals.** Since the LSTM does not directly produce prediction intervals, 95% prediction intervals are estimated using residual bootstrap sampling around the seed-averaged predictions. Coverage is assessed overall and separately for each forecast origin.

## Reproducibility

The analysis uses fixed random seeds and 2,000 bootstrap replicates. Bootstrap seeds and model seed identifiers are defined in the script.

The script also contains internal verification checks for the consistency of the reported calculations.

To reproduce the analysis, run `lstm_seedavg_batch_2026-10-03.R` after configuring its input and output paths.

## Outputs

The script generates the following CSV files:

| File | Contents |
|---|---|
| `metrics_LSTM_w2_seedavg_test_2026-10-03.csv` | Test performance metrics and confidence intervals |
| `metrics_LSTM_w2_seedavg_train_2026-10-03.csv` | Training performance metrics and confidence intervals |
| `metrics_LSTM_w2_seedavg_calibration_2026-10-03.csv` | Calibration results |
| `metrics_LSTM_w2_seedavg_pi_2026-10-03.csv` | Prediction interval coverage |
| `loo_LSTM_seedavg_diff_ci_2026-10-03.csv` | Leave-one-origin-out comparisons and confidence intervals |
| `loo_LSTM_seedavg_perseed_2026-10-03.csv` | Per-seed leave-one-origin-out results |

Corresponding Excel files are also generated.

The output tables can be regenerated from the saved prediction files; they do not need to be stored separately in the repository.
