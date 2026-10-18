# LSTM module

This README documents the LSTM part of the workflow in **repository-relative** terms:
the R data-preparation scripts, the two Python fitting scripts, the committed
prediction RData (shared under `Compare_models/Graphics/Predicted_values/`), and the
locked seed-averaged metrics batch under `Metrics/`. This is a merge and
documentation pass — no code was executed, and no script was modified.

Companion documents in this folder:

- [`LSTM_reproduction_README.md`](LSTM_reproduction_README.md) — the detailed provenance
  record (end-to-end data flow, complete script inventory including superseded scripts,
  fit generations G0–G2, share-side layout, locked protocol, caveats). Prepared
  2026-10-13 against the original `S:` share checkout; it uses share-side paths
  (`S:\...`, `LOKI_ML\...`), which no longer exist in this repository.
- [`Metrics/README_lstm_seedavg_2026-10-03.txt`](Metrics/README_lstm_seedavg_2026-10-03.txt) —
  deliverables list and verification record of the 2026-10-03 metrics batch
  (guards, seeds, naming flags).
- [`READ_ME_2.docx`](READ_ME_2.docx) — the original pipeline description (Word document,
  unchanged).

## Naming conventions

- **Seeds and origins**: 10 model seeds (1–10) per origin, 5 rolling origins.
- **Test**: 7 days per origin → 35 pooled test observations; leave-one-origin-out (LOO)
  variants exclude one origin → 28 points.
- **Training**: per-origin `n_train` — full 394, 80% 319, threefourth 299, 67.5% 268,
  65% 258.
- **Inputs**: MuSPAD `(1, 7, 122)`; baseline/incidence-only `(1, 7, 1)`; forecast horizon
  7 days.
- **Prediction RData**: the four non-full variants are named
  `LSTM_7day_{MuSPAD,baseline}_seeds_{variant}.RData`; the full variant has no suffix
  (`LSTM_7day_MuSPAD_seeds.RData`, `LSTM_7day_baseline_seeds.RData`).
- **RData objects**: `graph_LSTM_7day_M` / `graph_LSTM_7day_BL` (full variant, built
  2026-09-20) or `graph_LSTM_M` / `graph_LSTM_BL` (non-full variants, rebuilt
  2026-10-01); the metrics batch resolves either naming via `objnames()`.
- **Model labels**: the baseline model is called `onlyIncidence` in the 10-03 batch
  outputs (the locked 10-01/10-02 files keep `baseline`); sMAPE is on the 0–2 scale.
- Predictions are **unclipped** (can be negative).

## Layout

```text
LSTM/
├── Metrics/                           # locked metrics batch + 22 committed files (below)
├── Python_Scripts_plus_Data_prep/     # 2 Python fitting scripts + 3 R data-prep scripts
├── Python_results/                    # fit outputs (~274 MB); kept for now, not documented
├── LSTM_reproduction_README.md        # provenance record (share-side paths)
└── READ_ME_2.docx                     # original pipeline description
```

`Metrics/` contains 22 files: the batch script `lstm_seedavg_batch_2026-10-03.R`, the
protocol record `README_lstm_seedavg_2026-10-03.txt`, and 20 metric deliverables —
`metrics_LSTM_w2_seedavg_{test,train,pi,calibration}_2026-10-03.{csv,xlsx}` (8),
`loo_LSTM_seedavg_{diff_ci,perseed}_2026-10-03.{csv,xlsx}` (4),
`LSTM_{MuSPAD,onlyIncidence}_{test,train}set_perseed_2026-10-0{7,8}.csv` (4; the
onlyIncidence test set is 10-07, the rest 10-08), and the matching
`..._metrics_mean_sd_10seeds.xlsx` (4).

## Pipeline (5 steps)

```text
1. R data prep     (3 scripts)  -> per-origin, per-variant inputs + scaling CSVs
2. Python fitting  (2 scripts)  -> per variant x origin x 10 seeds, 2 models
3. R post-processing          -> 10 committed prediction RData under Compare_models/
4. Metrics batch   (R)        -> Tables 1, S3-S7 deliverables under LSTM/Metrics/
5. Explainability  (Python)   -> S9-S12 (outside the table scope)
```

1. **Data prep (R).** `Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R`
   (full variant) and `Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R`
   (the four non-full variants via `make_split_data(p, suffix)`, `p` in
   {0.65, 0.675, 0.75, 0.80}) build the per-origin train/test inputs (test: 7 days;
   train: see `n_train` above), the imputed/scaled CSVs, and the scaling metadata that
   the Python scripts read from `Datasets\LSTM_data_subsets` (share-side; not committed).
   The legacy `Data_prep_threefourthData_diff_seeds.R` covered the 75% variant before the
   diff-splits script superseded it.
2. **Fitting (Python).** `python_lstm_MuSPAD_diff_seeds.py` (MuSPAD; 122 features, 7-day
   lag, 30-trial BayesianOptimization over the Table S1 search space) and
   `python_lstm_onlyIncidence_diff_seeds.py` (baseline; fixed architecture) are run for
   each of the 5 variants x 5 origins x 10 seeds. Each run writes a `.keras` model
   (`final_model_tplus7_seed{seed}_testset{variant}.keras`) and per-day prediction CSVs
   into `Python_results/` (share-side layout; kept here, not documented).
3. **Post-processing / RData build (R).** The per-seed, per-origin prediction CSVs are
   aggregated into the 10 committed prediction RData files under
   `Compare_models/Graphics/Predicted_values/{fullData, 65percentData, 675percentData,
   80percentData, threefourthData}/`. The full variant is built by `make_Graph(model)`
   in the full-data prep script (~line 649); the eight non-full RData are built by the
   Postprocessing section of the diff-splits script (~lines 202–434), which still writes
   to the old `Predicted_values\Diff_seeds\` layout (see caveat C4).
4. **Metrics batch (R).** `Metrics/lstm_seedavg_batch_2026-10-03.R` (1,377 lines) loads
   the 10 RData files, averages the predictions across the 10 seeds (35 pooled test days,
   per-origin train windows), and computes every Table 1 / S3–S7 deliverable under six
   guards (G1–G6). The committed 10-03 run succeeded on 2026-10-03 (04:35–04:37, EXIT 0,
   guards 6/6 PASS).
5. **Explainability (Python).** `python_lstm_lime_shap_different_clusters.py` computes
   LIME/SHAP attributions on the low/high-incidence clusters [3, 4, 5] (Tables S9–S12).
   Outside the locked table protocol; the merged attribution CSVs it feeds are committed
   in `Clustering_sensitivity_MLP_LSTM/`.

## Script inventory

| Script | Role |
|---|---|
| [`Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R`](Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R) (832 lines) | Full-data 7-day preparation: per-origin train/test inputs, imputed/scaled CSVs and scaling metadata for the Python scripts; `make_Graph(model)` (~line 649) builds the full-variant prediction RData. |
| [`Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R`](Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R) (481 lines) | 7-day preparation for the four non-full variants via `make_split_data(p, suffix)` (`p` in {0.65, 0.675, 0.75, 0.80}); its Postprocessing section (~lines 202–434) builds the eight non-full prediction RData (old layout, C4). |
| [`Python_Scripts_plus_Data_prep/Data_prep_threefourthData_diff_seeds.R`](Python_Scripts_plus_Data_prep/Data_prep_threefourthData_diff_seeds.R) (285 lines) | Legacy three-quarters (75%) variant preparation; superseded by the diff-splits script; kept for provenance. |
| [`Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py`](Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py) (343 lines) | MuSPAD fit: input `(1, 7, 122)`, 7-day lag, 30-trial BayesianOptimization over the Table S1 space (1–4 stateful LSTM layers, 32–128 units in steps of 32, per-layer dropout 0.0–0.5 in steps of 0.1, RMSprop learning rate in {1e-2, 1e-3, 1e-4}, MSE, 20 epochs, batch 1, no shuffling); writes `final_model_tplus7_seed{seed}_testset{variant}.keras`. |
| [`Python_Scripts_plus_Data_prep/python_lstm_onlyIncidence_diff_seeds.py`](Python_Scripts_plus_Data_prep/python_lstm_onlyIncidence_diff_seeds.py) (249 lines) | Baseline (incidence-only) fit: fixed three stateful 50-unit LSTM layers with dropout 0.2 after each, time-distributed one-unit output; input `(1, 7, 1)`; same file naming. |
| [`Python_Scripts_plus_Data_prep/python_lstm_lime_shap_different_clusters.py`](Python_Scripts_plus_Data_prep/python_lstm_lime_shap_different_clusters.py) (348 lines) | LIME/SHAP explainability on clusters [3, 4, 5] (Tables S9–S12; outside the table scope). |
| [`Metrics/lstm_seedavg_batch_2026-10-03.R`](Metrics/lstm_seedavg_batch_2026-10-03.R) (1,377 lines) | Locked metrics batch (six guards G1–G6); produces all committed 10-03 deliverables (Table 1, S3–S7); still points at the old `Diff_seeds` layout (C3). |

## Name equivalence (flat workflow vs. this repository)

The flat local workflow copy (`LSTM_flat_workflow\LSTM\`) corresponds to the repo as
follows; no files were re-copied in this merge beyond the two listed "copied" rows:

| Local flat-workflow script | Repo path | Status |
|---|---|---|
| `Data_prep_diff_splits_diff_seeds.R` | `Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_diff_splits_diff_seeds.R` | Equivalent (different name) |
| `python_lstm_with_Expl_diff_seeds.py` | `Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py` | Equivalent (different name) |
| `python_lstm_onlyIncidence_diff_seeds.py` | `Python_Scripts_plus_Data_prep/python_lstm_onlyIncidence_diff_seeds.py` | Byte-identical |
| Local full-data prep script | `Python_Scripts_plus_Data_prep/Data_prep_for_Python_script_7day_prediction_full_data_diff_seeds.R` | Local copy differs only by a ~13-line `read.csv` block around line 655; the repo version is authoritative |
| `Data_prep_threefourthData_diff_seeds.R` | `Python_Scripts_plus_Data_prep/Data_prep_threefourthData_diff_seeds.R` | Copied in this merge (legacy, provenance-only) |

## Paper table → committed source (T1, S1–S7)

| Paper table | Content | Committed source |
|---|---|---|
| **Table 1** (main text, LSTM row) | Test RMSE / sMAPE, 7-day, pooled 5 origins | `Metrics/metrics_LSTM_w2_seedavg_test_2026-10-03.csv` |
| **S1** | Hyperparameter search space | Defined in code: `Python_Scripts_plus_Data_prep/python_lstm_MuSPAD_diff_seeds.py` (no separate file) |
| **S2** | Selected LSTM configurations per origin (seeds 1–10 identical) | **Not committed** — share-side `Compare_models\Metric_tables\LSTM_w2\architectures_LSTM_2026-10-01.{csv,xlsx}` (vintage `canon16.09`), built from the saved `.keras` configs |
| **S3** | MAE / MASE / R² (train + test) | `Metrics/metrics_LSTM_w2_seedavg_{test,train}_2026-10-03.csv` |
| **S4** | Seed stability: point estimates ± SD across 10 seeds | Point estimates: same as S3. SD (from per-seed metrics, interactive run 2026-10-07/08): `Metrics/LSTM_{MuSPAD,onlyIncidence}_{test,train}set_perseed_2026-10-0{7,8}.csv` + `..._metrics_mean_sd_10seeds.xlsx`. SDs come from per-seed metrics, so point estimates need not equal the per-seed means |
| **S5** | Calibration (α, β of y ~ ŷ) + PI coverage/width | `Metrics/metrics_LSTM_w2_seedavg_calibration_2026-10-03.csv`, `Metrics/metrics_LSTM_w2_seedavg_pi_2026-10-03.csv` |
| **S6** | LOO sensitivity, RMSE + sMAPE (paper covers LSTM & VAR) | `Metrics/loo_LSTM_seedavg_diff_ci_2026-10-03.csv` (+ `.xlsx`), `Metrics/loo_LSTM_seedavg_perseed_2026-10-03.csv` (+ `.xlsx`) |
| **S7** | LOO sensitivity, MAE + MASE + R² (paper covers LSTM & VAR) | Same files as S6 |
| S9–S12 | LIME/SHAP explainability (outside the locked scope) | `Python_Scripts_plus_Data_prep/python_lstm_lime_shap_different_clusters.py`; merged attributions committed under `Clustering_sensitivity_MLP_LSTM/` |

Every LSTM cell in T1 and S3–S7 traces to the **10-03 seed-averaged batch** plus the
**10-07/08 per-seed files**; the locked wave-2 files in the share-side `LSTM_w2\`
folder are the contemporaneous cross-check.

## Where the committed data live

| Path | Contents |
|---|---|
| `LSTM/Python_Scripts_plus_Data_prep/` | The 6 scripts above (3 R prep, 2 Python fitting, 1 Python explainability) |
| `LSTM/Metrics/` | Batch script + protocol record + the 20 Table 1 / S3–S7 deliverables |
| `LSTM/Python_results/` | Fit outputs (~274 MB) — kept for the time being, **not documented** here |
| `Compare_models/Graphics/Predicted_values/{fullData, 65percentData, 675percentData, 80percentData, threefourthData}/` | The 10 committed LSTM prediction RData (2 per variant), co-located with the VAR/LASSO/MLP prediction files shared by the other modules (LASSO and MLP files exist for `fullData` only) |

**Not committed** (share-side, original `S:` layout): the per-origin subset CSVs and
scaling RData under `Datasets\LSTM_data_subsets` (Python inputs); and
`Compare_models\Metric_tables\`, including the locked wave-2 `LSTM_w2\` files and the
S2 `architectures_LSTM_2026-10-01.{csv,xlsx}`.

## Locked metric protocol

As implemented in the 10-03 batch (carried over verbatim from
`LSTM_reproduction_README.md`, section 6):

- **Point estimates** (T1, S3, S5, S6, S7): computed from **predictions averaged across
  the 10 seeds**, pooled over the 5 rolling origins — never the mean of per-seed metrics.
- **S4 SDs only**: from per-seed metrics evaluated separately for each seed (hence the
  S4 caption note that point estimates need not equal the per-seed means).
- **MASE denominators** (one-step persistence MAE on the corresponding training series):
  pooled `5.000946`; LOO excl65 `5.173746`, excl675 `5.094352`, excl75 `4.852672`,
  excl80 `4.836091`, exclFull `5.040132`.
- **CIs**: moving-block bootstrap resampling complete 7-day forecast-origin blocks,
  B = 2000.
- **PIs (S5)**: residual bootstrap around the seed-averaged predictor (user-sanctioned
  definition, 2026-10-02). The LSTM models produce **no native prediction intervals** —
  all PI/coverage numbers are post-hoc.
- **Bootstrap seeds**: `SEED_PTEST=20261028`; LOO excl{65,675,75,80,Full} =
  `20261030…20261103`; `SEED_PI=20261104`; `SEED_TRAIN=20261105`.
- **Data**: n_train per origin — full 394, 65% 258, 67.5% 268, 80% 319, threefourth 299;
  test weeks contain no zero-incident days. Inputs: MuSPAD `(1, 7, 122)`, baseline
  `(1, 7, 1)`. Predictions are **unclipped** (can be negative).

## Caveats (C1–C12)

- **C1** — S2 dropout: the reported dropout 0.2 is a standalone `Dropout(0.2)` layer;
  the internal (recurrent) dropout in the saved model `config.json` files is 0.0. The
  table is consistent with the models; the S2 CSV documentation is incomplete.
- **C2** — No keras-tuner oracle logs: the 30-trial tuning history was not persisted.
  Only the selected configuration is recoverable (from the S2 files / saved configs).
- **C3** — Stale batch paths: `Metrics/lstm_seedavg_batch_2026-10-03.R` still references
  the old `Predicted_values\Diff_seeds\{variant}` layout (header line 12; path
  definitions lines 102/112; load line 215; `vfolder()` lines 318/327, e.g.
  `65percentData` → `65percent_data`). The share was reorganized into per-variant
  folders on 2026-10-07 → 2026-10-13, so the batch cannot be rerun as-is. Not fixed in
  this merge (no reruns, no new scripts).
- **C4** — The RData build is not a standalone committed script: the full variant comes
  from `make_Graph(model)` in the full-data prep script (~line 649); the eight non-full
  RData come from the Postprocessing section of the diff-splits script (~lines 202–434),
  which writes to the old `Diff_seeds` layout. The 2026-10-01 interactive run survives
  only as `.Rhistory` evidence on the share.
- **C5** — Mixed vintages: the full-variant RData were built 2026-09-20 01:22
  (`graph_LSTM_7day_M` / `graph_LSTM_7day_BL`); the four non-full variants were rebuilt
  2026-10-01 14:44 (`graph_LSTM_M` / `graph_LSTM_BL`). The batch handles both via
  `objnames()`.
- **C6** — The 2026-10-07/08 per-seed metric files (S4 SDs) have no committed generator
  script; they came from an interactive run.
- **C7** — Fit generations: G0 2026-08-31; G1 MuSPAD 2026-09-16 (old file naming
  `final_model_seed{s}.keras` — the only Table-1 artifacts); G2 baseline 2026-10-01.
  `Python_results/` contains earlier-generation remnants; see the provenance record
  section 4 for details.
- **C8** — Dead code in `python_lstm_onlyIncidence_diff_seeds.py`: an uncalled
  `build_model(hp)` that references an undefined `timesteps`.
- **C9** — The first 10-03 batch launch failed with `Fehler in abs(e):
  Nicht-numerisches Argument`; the successful run was 2026-10-03 04:35–04:37, the log
  ends with `DONE` (EXIT 0, guards 6/6 PASS).
- **C10** — The share-side layout was volatile as of 2026-10-13; this repo copy is
  frozen at this commit.
- **C11** — The `Compare_models/Metrics/` scripts are stale: the all-models script lists
  the `Predicted_values` root non-recursively (pattern `^(VAR|LASSO|MLP).*\.RData$`);
  the timeseries script reads `Predicted_values\Three_fourth_data\Diff_seeds`.
- **C12** — The full-variant prep writes CSVs to the `Datasets\` root while the Python
  scripts read `Datasets\LSTM_data_subsets` (the legacy threefourth script has the same
  inconsistency); superseded, provenance-only.

## Minimum keep-in-repo set

- The 6 scripts under `Python_Scripts_plus_Data_prep/`.
- `Metrics/` (batch script + all 20 deliverables + protocol record).
- The 10 committed LSTM prediction RData under
  `Compare_models/Graphics/Predicted_values/{variant}/`.
- This README, `LSTM_reproduction_README.md`, and `READ_ME_2.docx`.
- `Python_results/` is kept for the time being but is **not required** for the table
  values — the predictions are already embedded in the committed RData.

Not in the repo (share-side): the locked `Compare_models\Metric_tables\LSTM_w2\` folder
(including the S2 architecture files), the per-origin subset CSVs + scaling RData, and
the tuner results folders.

## Rerun order (if ever needed)

1. R data prep (the three prep scripts) — requires the restricted MuSPAD data and RKI
   incidence input; writes the subset CSVs (see C12 for the `Datasets\` inconsistency).
2. Python fitting (both fitting scripts; the MuSPAD script is the long pole: 30 tuning
   trials x 5 origins x 10 seeds).
3. RData build — must first be recreated as a saved script (C4) and pointed at the
   current `Predicted_values` layout.
4. `Metrics/lstm_seedavg_batch_2026-10-03.R` — after fixing the stale `Diff_seeds`
   paths (C3).
5. Per-seed metrics for S4 — recreate as a script (C6).

No rerun of any step was performed in this merge.
