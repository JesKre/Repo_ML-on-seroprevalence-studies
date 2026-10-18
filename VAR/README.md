# VAR: models and table reproduction

All VAR scripts belong directly in `VAR/`. Run the commands below in PowerShell from the repository root.

## Regenerate the paper table rows

Place these **31 existing RData files** in `Compare_models/Graphics/Predicted_values/`:

- `VAR_7day_MuSPAD*_p*_seeds.RData` and `VAR_7day_baseline*_p*_seeds.RData`: 30 files covering both models, lag orders 7/14/21, and five origins (full, 65%, 67.5%, 75%, 80%). Full-origin filenames have no variant suffix.
- `VAR_train_terms.RData`: training observations, fitted values and preceding observations needed for the training metrics and MASE denominator.

Then run:

```powershell
Rscript VAR/VAR_generate_tables.R
```

This needs only base R. It creates `T1_var.csv`, `S6_var.csv`, `S8_var.csv`, `S9_var.csv` and `S10_var_candidate.csv` directly in `VAR/`. These contain the VAR rows only. The generated CSVs do not need to be committed.

The saved RData inputs are necessary for recovering the paper's numerical results without the original study data. Artificial data cannot recover those numbers.

## Refit using artificial data

Put `Artificial_data_MuSPAD.RData` in `Artificial_datasets/` at the repository root. It must contain `df_ts_inc`, with 408 ordered rows, numeric `Inzidenz`, `time_day` and predictor columns. Use the time-series dataset, rather than the 279-row time-agnostic MLP dataset.

Install `bigtime` once in R using `install.packages("bigtime")`. Then run:

```powershell
Rscript VAR/VAR_refit_artificial.R
$env:VAR_PRED_DIR = (Resolve-Path 'VAR').Path
Rscript VAR/VAR_generate_tables.R --from-refit
Remove-Item Env:VAR_PRED_DIR
```

The first command fits both models for three lag orders, five origins and ten seeds (300 fits). It creates the 31 prediction/metric-input RData files in `VAR/`, so the saved paper inputs in the shared folder are retained. The second command calculates tables from these artificial-data fits, using seed 1 as in the paper-table workflow. These are demonstration results, not the paper results.

The refitting script derives scaling directly from each training subset; it does not require private pre-scaled CSVs. It follows the final fitting calls in `var_predictions_diff_splits_diff_seeds.R`: MuSPAD uses L1, cross-validation and `cvcut=0.8`; baseline uses the original incidence-only call with package defaults. Training scoring retains the original six-day shift between fitted values and observations. The preliminary validation grid, plots and individual saved model workspaces are omitted. This is a portable demonstration, not a reconstruction of unavailable original preprocessing inputs.

For another compatible input, set `VAR_DATA_FILE` to its RData path. `VAR_REFIT_DIR` can change the refitting output folder. `VAR_PRED_DIR` changes the table generator's input folder. Software versions can affect refitting; the demonstration has been parsed and its input checked, but the 300 fits have not been run during packaging.

## What each script does

| Script | Purpose |
|---|---|
| `bigtime_MuSPAD_allData.R` | Original multivariable VAR workflow using all available training data. |
| `bigtime_MuSPAD_threefourthData.R` | Original multivariable workflow using the 75% split. |
| `bigtime_noMuSPAD_only_incidence_allData.R` | Original incidence-only baseline using all available training data. |
| `bigtime_noMuSPAD_only_incidence_threefourthData.R` | Original incidence-only baseline using the 75% split. |
| `var_predictions_diff_splits_diff_seeds.R` | Original multi-seed fitting extension; produces predictions for the five origins at a selected lag order. |
| `var_train_papermetric_w2.R` | Original training-metric calculation from saved model workspaces. |
| `VAR_refit_artificial.R` | Portable artificial-data refitting, derived from the final fitting calls in the multi-seed extension. |
| `VAR_generate_tables.R` | Generates table rows from the shared saved inputs, or from artificial-data refits. |


