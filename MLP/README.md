# MLP scripts and reproduction of manuscript tables

This folder contains the original MLP analyses and the additional scripts
for the revised ten-seed performance results. Scripts and generated table
CSVs are directly in `MLP/`. The shared saved predictions are in
[`Compare_models/Graphics/Predicted_values/`](../Compare_models/Graphics/Predicted_values/).
The [DP reproduction workflow](Table_reproduction_DP/MLP_with_DP/README.md)
remains in its separate folder.

## Scripts in this folder

| Script | What it does | Relationship to the tables |
| --- | --- | --- |
| [Keras_MuSPAD_Expl_cluster_incidence.R](Keras_MuSPAD_Expl_cluster_incidence.R) | MuSPAD MLP: imputation, scaling, preliminary training/validation, final fitting and cluster-specific explanations | fitting/explainability analysis |
| [Keras_only_time.R](Keras_only_time.R) | time-only MLP baseline with preliminary and final fitting | Original baseline |
| [Keras_MuSPAD_compare_DP_10foldcv_Expl.R](Keras_MuSPAD_compare_DP_10foldcv_Expl.R) | Non-private RMSprop/Adam comparison over ten folds, including SHAP/LIME explanations | Comparator for the DP work; does not implement DP-SGD |
| [DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R](DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R) | Original gradient-clipping/noise implementation, optimizer comparisons and explanation analysis | Reference for the separate Tables 7/S8 reproduction |
| [repro_mlp_k2.R](repro_mlp_k2.R) | Ten-seed Keras-2 grid: three input treatments, one to four hidden layers and 30/45/60 epochs; saves test predictions and observed series | Produces test predictions for the selected MuSPAD and baseline configuration |
| [repro_mlp_k2_trainfull_rmse.R](repro_mlp_k2_trainfull_rmse.R) | Ten-seed rerun of the selected four-layer, 30-epoch models; saves full-training-block predictions | Supplies the training predictions used for the performance and stability summaries |
| [MLP_generate_tables.R](MLP_generate_tables.R) | Loads the two seed prediction objects and calculates metrics, seed variability, bootstrap CIs and OLS calibration | Generates the MLP rows of Tables 1, S3, S4 and S5; `--from-runs` first creates the prediction objects from fresh fitting outputs |

The original four scripts retain their code and local project paths. The two
fitting extensions share their data problem, architecture and two-stage design,
but follow the later ten-seed grid protocol. 

## Workflow 1: regenerate the reported table rows

Required files in `Compare_models/Graphics/Predicted_values/`:

- `MLP_7day_MuSPAD_seeds.RData`: object `graph_MLP_7day_M_seeds`.
- `MLP_7day_baseline_seeds.RData`: object `graph_MLP_7day_BL_seeds`.

Each object contains observed and predicted incidence, date, training/test
label and seed ID for ten seeds: 251 training and 27 test observations per seed.
On the original Windows checkout, the directory is
`../Compare_models/Graphics/Predicted_values/`.
The script finds it relative to the repository; another location can be supplied
with `MLP_PRED_DIR`.

With base R, from the repository root:

```sh
Rscript MLP/MLP_generate_tables.R
```

The original modeling dataset and TensorFlow are not needed for this step.
No saved result CSVs or bootstrap files are required. The script writes:

| Output directly in `MLP/` | Contents |
| --- | --- |
| `MLP_table1.csv` | Training RMSE/sMAPE mean and SD; averaged-test-prediction metrics with bootstrap CIs |
| `MLP_tableS3.csv` | Corresponding MAE, MASE and R2 results |
| `MLP_tableS4.csv` | Training mean/SD and test point estimates with between-seed metric SDs |
| `MLP_tableS5.csv` | Calibration intercept/slope and 95% CIs; joint test of intercept=0 and slope=1 |

All outputs contain only the MLP and time-only baseline rows, at full precision.
Training metrics are summarized across seeds. Test point estimates use the
average of ten prediction vectors; the S4 test SDs come from seed-specific
metrics, so the reported point need not equal their mean. Test CIs use 10,000
moving-block resamples, block length seven observations, seed 42; IID bounds
are also exported for comparison. MASE uses training-series persistence MAE;
sMAPE uses the 0-2 scale. Prediction-interval coverage is not generated.

## Workflow 2: refit from modeling data

For a new fit, supply an RData file containing `df_ts_inc`, with incidence in
column 1, `time_day` in column 2 and the expected predictor columns. The real
modeling dataset is the input used for the manuscript analysis. The file
`../Artificial_datasets/Artificial_data_MuSPAD_time_agnostic.RData` contains an artificial counterpart
(279 rows, 123 columns) and can demonstrate the same workflow, but its fitted
results are not the manuscript's results.

Refitting requires R packages `missForest`, `glmnet`, `reticulate`, `tensorflow`
and legacy `keras`, plus the supplied Keras-2 environment: TensorFlow 2.15.1,
Keras 2.15.1; R keras 2.16.1 was used. `MLP_PYTHON` specifies that environment's
Python executable. From the repository root in PowerShell:

```powershell
$env:MLP_DATA_FILE = (Resolve-Path '../Artificial_datasets/Artificial_data_MuSPAD_time_agnostic.RData').Path
$env:MLP_PYTHON = 'C:/path/to/tensorflow_environment/Scripts/python.exe'
$env:MLP_PRED_DIR = (Resolve-Path 'MLP').Path
Rscript MLP/repro_mlp_k2.R
Rscript MLP/repro_mlp_k2_trainfull_rmse.R
Rscript MLP/MLP_generate_tables.R --from-runs
```

The first script generates the grid test predictions and observed series; the second generates the selected
models' training predictions. The third combines them into the two seed RData
objects and generates all four tables. The example directs artificial-data
prediction objects into `MLP/`, preserving the shared original prediction files.
For a later original-table regeneration, unset `MLP_PRED_DIR` or point it back
to `Compare_models/Graphics/Predicted_values/`.

Generated fitting CSVs and table CSVs default to `MLP/`. `MLP_RUN_DIR` can change
the fitting-output location.


## Differential-privacy tables

Tables 7 and S8 use the dedicated workflow in
[Table_reproduction_DP/MLP_with_DP/](Table_reproduction_DP/MLP_with_DP/).
Follow its [README](Table_reproduction_DP/MLP_with_DP/README.md) for inputs,
environment and commands. It adapts the original DP script's Adam validation
phase into ten seeded runs and adds a zero-noise baseline.

