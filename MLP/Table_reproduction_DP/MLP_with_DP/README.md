# Reproduce Table 7 and supplementary Table S11

This folder contains the ten-seed Adam differential-privacy analysis and the saved per-fold results used to regenerate its tables. Table generation uses base R and does not require model retraining or TensorFlow.

## Connection to the original analysis

The reference implementation is [DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R](../../DP_Analysis/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R).


| Script | Purpose |
| --- | --- |
| [01_make_data_cache.R](scripts/01_make_data_cache.R) | Optionally rebuild the input cache: convert `time_day` to numeric and remove day 168. |
| [02_run_adam_dp_grid_shard.R](scripts/02_run_adam_dp_grid_shard.R) | Run one seed over three noise settings and ten folds; save a per-seed results file. |
| [03_run_epsinf_shard.R](scripts/03_run_epsinf_shard.R) | Run one seed of the zero-noise control over ten folds. |
| [04_make_tables.R](scripts/04_make_tables.R) | Validate supplied results or combine worker files, convert fold MSEs to RMSEs, and generate tables. |

## Included inputs

| File | Contents |
| --- | --- |
| [data/dataset_local_cache.rds](data/dataset_local_cache.rds) | Post-preprocessing input: 279 rows × 123 columns, before imputation/scaling. Used for retraining. |
| [results/adam10seed_perfold_master.csv](results/adam10seed_perfold_master.csv) | 300 fold records: ten seeds × three noise settings × ten folds. |
| [results/epsinf10seed_perfold_master.csv](results/epsinf10seed_perfold_master.csv) | 100 zero-noise fold records: ten seeds × ten folds. |

The master CSVs use semicolons and decimal commas and are read with `read.csv2()`. Worker shards and generated table CSVs are not required inputs for the default table-generation command.

## Regenerate the tables

From the repository root:

```sh
Rscript MLP/Table_reproduction_DP/MLP_with_DP/scripts/04_make_tables.R
```

The script checks that every expected seed/setting/fold combination appears exactly once, noise multipliers match the expected values, and MSEs are finite and nonnegative. It writes these files under `MLP/Table_reproduction_DP/MLP_with_DP/results/`:

| Output | Meaning |
| --- | --- |
| `table7.csv` | Training and validation mean RMSE and between-seed SD for each setting. |
| `tableS8.csv` | Corresponding summaries plus the mean within-seed fold SD. This **legacy filename** corresponds to supplementary **Table S11** in the current README/manuscript numbering. |

The script retains `tableS8.csv` as its output filename; do not look for a generated `tableS11.csv`.

For each seed and setting, the script first takes the square root of **each fold MSE**, then computes the mean and sample SD across ten folds. It reports the mean and sample SD of the ten seed-specific mean RMSEs; the supplementary output additionally reports the mean of the ten within-seed fold SDs. Final statistics are rounded to one decimal, and sample SDs use `n - 1`. Taking the square root of the mean MSE would give a different result.


## Retrain from the cached input

Retraining requires R packages `reticulate`, `tensorflow`, `keras`, `tidyverse`, and `missForest`, plus a compatible legacy Keras Python environment. The previously documented training environment was R 4.6.1, Python 3.12.14, TensorFlow 2.21.0, and compatible `tf-keras`; these are provenance details, not an automatically installed or locked environment.

Set `DP10SEED_PYTHON` to the Python executable for that environment. Workers set `TF_USE_LEGACY_KERAS=True` before initializing TensorFlow.

To preserve the supplied master results, use a separate run directory. This PowerShell example copies the package into `work/dp-retraining`, then runs the original scripts **from the repository root**:

```powershell
$package = 'MLP/Table_reproduction_DP/MLP_with_DP'
$run = Join-Path (Get-Location) 'work/dp-retraining'
if (Test-Path $run) { throw 'Choose a new run directory, or resume it explicitly.' }
New-Item -ItemType Directory -Force (Split-Path $run) | Out-Null
Copy-Item $package $run -Recurse

$env:DP10SEED_ROOT = $run
$env:DP10SEED_PYTHON = 'C:/path/to/venv/Scripts/python.exe'
foreach ($seed in 1..10) {
    Rscript "$package/scripts/02_run_adam_dp_grid_shard.R" $seed
    if ($LASTEXITCODE -ne 0) { throw "DP worker failed: seed $seed" }
    Rscript "$package/scripts/03_run_epsinf_shard.R" $seed
    if ($LASTEXITCODE -ne 0) { throw "Baseline worker failed: seed $seed" }
}
Rscript "$package/scripts/04_make_tables.R" --from-shards
if ($LASTEXITCODE -ne 0) { throw 'Table generation failed' }
Remove-Item Env:DP10SEED_ROOT
```

Workers save `adam10seed_perfold_s01.csv` through `s10.csv` and `epsinf10seed_perfold_s01.csv` through `s10.csv` under the selected run directory's `results/`. They skip completed fold records when restarted. To resume, retain that directory, set `DP10SEED_ROOT` and `DP10SEED_PYTHON` again, and rerun the worker commands. Do not repeat the copy step.

After all workers finish, `04_make_tables.R --from-shards` validates the complete grid, replaces the two master CSVs in the selected run directory, and generates the tables there. Without `--from-shards`, it reads existing master CSVs instead. Unset `DP10SEED_ROOT` to return to the supplied package inputs.

## Optional cache rebuilding and configuration

The supplied cache is sufficient for retraining. Rebuild it only when intentionally changing the modeling input:

```sh
Rscript MLP/Table_reproduction_DP/MLP_with_DP/scripts/01_make_data_cache.R
```

This script selects the last filename matching `binary_classnumber_var_subset_missing_zero` in the configured data directory. Supply the appropriate original modeling dataset; it must produce the expected 279 × 123 cache after preprocessing. Rebuilding overwrites the cache in the selected package/run directory.

| Environment variable | Purpose |
| --- | --- |
| `DP10SEED_ROOT` | Package/run directory containing `data/` and `results/`; defaults to the package containing the script. |
| `DP10SEED_PYTHON` | Required Python executable for training workers. |
| `DP10SEED_DATA_DIR` | Source modeling-data directory when rebuilding a missing cache or running the cache script. |
| `MUSPAD_DATA_DIR` | Fallback modeling-data directory; defaults to the repository's `Datasets/`. |

Workers use the cache when present; changing the data-directory variables does not replace an existing cache.

