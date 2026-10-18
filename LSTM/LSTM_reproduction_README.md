# LSTM reproduction documentation — muspad-ml-loki-sp2.1

**What this document is.** Identification and documentation of which scripts produced the
LSTM numbers in the paper (main-text **Table 1**, LSTM row, and supplementary **Tables S2–S7**).
It maps every paper table to the exact scripts, model files, and output files on the share.
**No reruns were performed** for this documentation pass; everything below is verified against
file contents, file timestamps, and the saved run logs.

**Companion documents (read these first — this README deliberately does not duplicate them):**

| Document | Role |
|---|---|
| `LSTM\LSTM_Pipelines_Old_vs_Current.md` | Evolution of the LSTM pipelines (old R/keras2 vs. current Python/keras3) |
| `LSTM\LSTM_output_explanation_agent.txt` | Explanation of the Python fit outputs (model + prediction files) |
| `LSTM\Metrics\README_lstm_seedavg_2026-10-03.txt` | Metric protocol of the 10-03 batch (seed averaging, bootstrap, PI definition) |

**Naming conventions used throughout.**
- "baseline" = the *onlyIncidence* model (LSTM fed with current incidence only, no MuSPAD features).
- All current LSTM tables are **h7** (7-day forecast), **pooled over 5 rolling forecast origins**
  (35 test observations; 28 for leave-one-origin-out), **10 random seeds (1–10)**,
  and — except for the S4 standard deviations — based on **seed-averaged predictions**.
- No 14- or 21-day LSTM rows exist in the current paper tables.

---

## 1. End-to-end data flow

```
[1] R data prep (Python_Scripts_plus_Data_prep\*.R)
        |   writes to  S:\PROJACTIVE\MUSPAD-LABORERGEBNISSE\LOKI_ML\Datasets\
        |       - LSTM_data_subsets\Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_<subset>*
        |       - scaling_parameters_<subset>_*   (per-origin min/max scaling)
        v
[2] Python fitting (2 scripts, 2 fit generations)
        python_lstm_with_Expl_diff_seeds.py        (MuSPAD, keras_tuner, 30 trials/seed)   -> fitted 2026-09-16
        python_lstm_onlyIncidence_diff_seeds.py    (baseline, fixed 3xLSTM(50), no tuner)  -> fitted 2026-10-01
        |   writes to  LSTM\Python_results\Python_Original\Different_seeds\keras_tuner\results_h7{,_65percentData,_675percentData,_threefourthData,_80percentData}\
        |       - final_model_*.keras                    (zip: metadata.json + config.json + model.weights.h5)
        |       - train/test_predictions{,_only_incidence}_tplus7_seed{s}.csv   (scaled units)
        v
[3] RData build  -- INTERACTIVE R session, NO saved script (2026-10-01 14:44 for the 4 non-full variants;
    |                fullData RData is from 2026-09-20). Session text preserved in
    |                keras_tuner\results_h7_675percentData\.Rhistory.
    |   Reads the per-seed prediction CSVs, back-scales with Datasets\scaling_parameters_*,
    |   pivots to long format, saves 20 x LSTM_7day_{MuSPAD,baseline}_seeds_{subset}.RData
    |   (objects graph_LSTM_M / graph_LSTM_BL; full variant: graph_LSTM_7day_M / graph_LSTM_7day_BL)
    |   into  Compare_models\Graphics\Predicted_values\...  (see section 5 for the current layout)
    v
[4] R metrics batch  --  LSTM\Metrics\lstm_seedavg_batch_2026-10-03.R   (run 2026-10-03 04:35-04:37, log ends "DONE")
        |   Seed-averaged predictions -> T1/S3/S5 point estimates;
        |   origin-block bootstrap B=2000 -> CIs;
        |   residual bootstrap around seed-averaged predictor -> PI/coverage (S5);
        |   leave-one-origin-out (per-exclusion MASE denominators) -> S6/S7
        v
    LSTM\Metrics\metrics_LSTM_w2_seedavg_{test,train,pi,calibration}_2026-10-03.{csv,xlsx}
    LSTM\Metrics\loo_LSTM_seedavg_{diff_ci,perseed}_2026-10-03.{csv,xlsx}
        v
[5] Per-seed metrics (for S4 SDs only) -- INTERACTIVE R, no saved script (2026-10-07/08)
        v
    LSTM\Metrics\LSTM_{MuSPAD,onlyIncidence}_{test,train}set_perseed_2026-10-0{7,8}.csv
    LSTM\Metrics\LSTM_{MuSPAD,onlyIncidence}_{test,train}set_metrics_mean_sd_10seeds.xlsx

Cross-check layer (locked "wave-2" files, data only, no scripts):
    Compare_models\Metric_tables\LSTM_w2\   (metrics_LSTM_w2_*_2026-10-01/02 + loo_LSTM_2026-10-02 + architectures_LSTM_2026-10-01)
```

---

## 2. Script inventory

### 2.1 Current — these produced the paper numbers

| Script | Role | Status / notes |
|---|---|---|
| `Python_Scripts_plus_Data_prep\python_lstm_with_Expl_diff_seeds.py` | **Fitting + keras-tuning** (MuSPAD). Per rolling origin: keras_tuner, 30 trials, best config fit with all 10 seeds; h7. **Contains the hyperparameter search space reported in S1 Table.** | Current. Dead code: LIME/SHAP imports (unused in fitting). |
| `Python_Scripts_plus_Data_prep\python_lstm_onlyIncidence_diff_seeds.py` | **Fitting** (baseline). Fixed architecture 3×LSTM(50)+Dropout(0.2)+TimeDistributed(Dense(1)); no tuner. | Current. Dead code: `build_model(hp)` never called (references undefined `timesteps`). |
| `Metrics\lstm_seedavg_batch_2026-10-03.R` | **Metrics / CIs / PI / LOO** — the only metrics script; produced all of T1 (LSTM row), S3, S5, S6, S7 numbers and their files. | Current, but contains **stale `Diff_seeds` paths** — cannot be rerun as-is (see caveat C3). |
| *(interactive, no script)* RData build, 2026-10-01 14:44 | Converts per-seed prediction CSVs into the 20 `LSTM_7day_*.RData` files consumed by the batch. | Session preserved in `keras_tuner\results_h7_675percentData\.Rhistory`. Should be saved as a script (caveat C4). |
| *(interactive, no script)* per-seed metrics, 2026-10-07/08 | Per-seed RMSE/SMAPE/MASE/MAE/R² pooled over the 5 origins → SDs in S4. | No generator script anywhere on the share (caveat C6). |

### 2.2 Data prep (R, current)

| Script | Role |
|---|---|
| `Python_Scripts_plus_Data_prep\Data_prep_diff_splits_diff_seeds.R` (2026-09-28) | Builds the 5 data-subset splits (full / 65% / 67.5% / 80% / threefourth) for the seeded rolling-origin setup. |
| `Python_Scripts_plus_Data_prep\Data_prep_for_Python_script_7day_prediction_diff_seeds.R` (2026-09-21) | Writes the per-seed time-series + scaling inputs consumed by both Python fit scripts. |
| `Python_Scripts_plus_Data_prep\Data_prep_threefourthData_diff_seeds.R` (2026-08-31) | Earlier threefourth-only variant; superseded by the two above but kept for provenance. |

All outputs land **outside** the project folder, at `S:\PROJACTIVE\MUSPAD-LABORERGEBNISSE\LOKI_ML\Datasets\`
(external input for the whole chain).

### 2.3 Secondary — explainability (S9–S12, outside the table scope of this pass)

| Script | Role |
|---|---|
| `Python_Scripts_plus_Data_prep\python_lstm_lime_shap_different_clusters.py` (2026-09-18) | LIME/SHAP cluster-sensitivity (h7/h14) → feeds S9–S12 explainability tables. |
| `Python_Scripts_plus_Data_prep\Feature_means_for_SHAP_LIME_inputs.R`, `Analysis_Feature_Importance.R` | Support scripts for the explainability inputs. |

### 2.4 Superseded / historical — do not use for the paper numbers

- `LSTM_old\` — pre-kernel-rework scripts (`python_lstm.py`, `python_lstm_changed.py`, LIME/SHAP one-offs).
- `Python_Scripts_plus_Data_prep\Archiv\` — old data-prep and LIME/SHAP iterations.
- `Python_Scripts_plus_Data_prep\Other_python_script_versions\` — stateless Gradient-Explainer LIME/SHAP variants.
- `Python_Scripts_plus_Data_prep\python_lstm_lime_shap_cluster_*` (several top-level variants) — LIME/SHAP lineage.
- `R_scripts_results\` — the old **R-side keras2 pipeline** (`LSTM_varsubset_basic_*.R`, `LSTM_test_model_final.h5`, `LSTM_example.R`); fully superseded by the Python pipeline.
- `Data_prep_for_Python_script_14day_prediction.R` / `..._7day_prediction.R` (non-diff_seeds) — single-seed predecessors of the seeded prep.
- `Python_results\Archiv\*` — 2025-era outputs (LIME/SHAP CSVs, old plots).

---

## 3. Paper table → script/output mapping (all numerically verified)

| Paper table | Content | Produced by | Source file(s) |
|---|---|---|---|
| **Table 1** (main text, LSTM row) | Test RMSE / sMAPE, 7-day, pooled 5 origins | `lstm_seedavg_batch_2026-10-03.R` (seed-averaged predictions) | `LSTM\Metrics\metrics_LSTM_w2_seedavg_test_2026-10-03.{csv,xlsx}`; cross-checked vs locked `Compare_models\Metric_tables\LSTM_w2\metrics_LSTM_w2_test_2026-10-01.{csv,xlsx}` |
| **S1 Table** | Hyperparameter search space | Defined in `python_lstm_with_Expl_diff_seeds.py` (keras_tuner `HyperModel`) | (no separate file — it is code) |
| **S2 Table** | Selected LSTM configurations per origin, seeds 1–10 identical | Built from the saved `.keras` model configs | `Compare_models\Metric_tables\LSTM_w2\architectures_LSTM_2026-10-01.{csv,xlsx}` (vintage `canon16.09`; columns `model;origin;seed;vintage;n_lstm;units;dropout;return_seq;input_shape`) |
| **S3 Table** | MAE / MASE / R² (train + test) | `lstm_seedavg_batch_2026-10-03.R` | `metrics_LSTM_w2_seedavg_{test,train}_2026-10-03.{csv,xlsx}` |
| **S4 Table** | Seed stability: point estimates ± SD across 10 seeds | Point: `lstm_seedavg_batch_2026-10-03.R`; **SD: interactive per-seed run 2026-10-07/08** | Point: same as S3. SD: `LSTM\Metrics\LSTM_{MuSPAD,onlyIncidence}_{test,train}set_perseed_2026-10-0{7,8}.csv` (+ `..._metrics_mean_sd_10seeds.xlsx`). Example verified: MuSPAD test RMSE `83.78 ± 11.27`. Note (per S4 caption): SDs come from per-seed metrics, so point estimates need not equal the per-seed means. |
| **S5 Table** | Calibration (α, β of y ~ ŷ) + PI coverage/width | `lstm_seedavg_batch_2026-10-03.R` | `metrics_LSTM_w2_seedavg_calibration_2026-10-03.{csv,xlsx}`, `metrics_LSTM_w2_seedavg_pi_2026-10-03.{csv,xlsx}` |
| **S6 Table** | LOO sensitivity, RMSE + sMAPE (LSTM & VAR) | `lstm_seedavg_batch_2026-10-03.R` (LOO section) | `loo_LSTM_seedavg_diff_ci_2026-10-03.{csv,xlsx}`, `loo_LSTM_seedavg_perseed_2026-10-03.{csv,xlsx}`; cross-checked vs `LSTM_w2\loo_LSTM_2026-10-02.{csv,xlsx}` + `metrics_LSTM_w2_{test,train}_excl{65,675,75,80,Full}_2026-10-{01,02}` |
| **S7 Table** | LOO sensitivity, MAE + MASE + R² (LSTM & VAR) | `lstm_seedavg_batch_2026-10-03.R` (LOO section) | same files as S6 |
| S9–S12 (out of scope) | LIME/SHAP explainability | `python_lstm_lime_shap_different_clusters.py` | — |

Every LSTM cell in T1 and S3–S7 traces to the **10-03 seed-averaged batch** plus the
**10-07/08 per-seed files**; the locked wave-2 files in `LSTM_w2\` are the contemporaneous cross-check.

---

## 4. Fit generations (which model files actually produced the numbers)

Three generations of fitted models exist on the share. Only the last two matter for the paper:

| Gen. | Fitted | Files | Architecture (verified from `config.json`) | Size | Verdict |
|---|---|---|---|---|---|
| G0 (old baseline) | 2026-08-31 | `final_model_only_incidence_tplus7_seed{s}.keras` (old naming) | 1×LSTM(128), vintage `canon31.08` | 557,956 B | **Superseded** — do not use |
| G1 (MuSPAD) | **2026-09-16 22:46–23:32** | `final_model_seed{s}.keras` (old naming) in each `results_h7*` folder | keras_tuner-best per origin. Verified (full, seed 1): `Input(1,7,122) → LSTM(128, return_sequences, stateful, tanh, forget-bias) → Dropout(0.2) → Dense(1, linear)`; e.g. 65% origin = 2 LSTMs (64, 96). Optimizer RMSprop lr 0.001, loss MSE | ~1.05 MB | **Used for MuSPAD paper numbers** (prediction CSVs dated 16.09) |
| G2 (baseline refit) | **2026-10-01 13:15–13:41** | `final_model_only_incidence_tplus7_seed{s}_testset{variant}.keras` (current naming) | Functional: `Input(1,7,1) → [LSTM(50, return_sequences, stateful) → Dropout(0.2)] ×3 → TimeDistributed(Dense(1, linear))`; RMSprop lr 0.001, MSE | ~455 KB | **Used for baseline paper numbers** (prediction CSVs dated 10-01) |

Notes:
- Keras file format is the new-style zip (`metadata.json`, `config.json`, `model.weights.h5`) — there is **no `model.json`**; architecture checks must read `config.json`.
- `stateful: true` appears in the saved configs (artifact of how the models were built/saved; training behavior per the fit scripts is the authoritative one).
- Intermediate artifacts to not confuse with G1/G2: top-level `results_h7\` also contains old-named buggy `final_model_only_incidence_tplus7_seed1.keras` / `seed2.keras` remnants from an earlier run.
- **No keras_tuner oracle logs** are stored in the repository — the per-trial search history cannot be recovered; only the final (best) model configs are available (see caveat C2).

---

## 5. Where everything lives

| Artifact | Location |
|---|---|
| Fit outputs (models + prediction CSVs), both generations, all 5 variants | `LSTM\Python_results\Python_Original\Different_seeds\keras_tuner\results_h7{,_65percentData,_675percentData,_threefourthData,_80percentData}\` |
| The 20 `LSTM_7day_*.RData` (5 variants × MuSPAD/baseline) | `Compare_models\Graphics\Predicted_values\{fullData,65percentData,675percentData,80percentData,threefourthData}\` — **mixed with VAR RData** in the same folders; `VAR_with_seeds` subfolders were added 2026-10-13. There is **no longer a `Diff_seeds` folder** (reorganized by the collaborator after the 10-03 batch). |
| RData consumers (plots) | `Compare_models\Graphics\Graphic_grid_diff_seeds.R` (loads RData at lines 313/384/452/520/589; saves `ggplot_forecast_*_{Sys.Date()}.{png,eps}`); `Compare_models\Graphics\Graphic_grid.R` (line 128 has a stale `Predicted_values/Three_fourth_data` path); `Compare_models\Metric_tables\Metrics_all_models.R` (line 143) |
| 10-03 batch outputs (T1/S3/S5/S6/S7) + protocol README | `LSTM\Metrics\` (`lstm_seedavg_batch_2026-10-03.R`, 8× csv/xlsx `*_2026-10-03`, `README_lstm_seedavg_2026-10-03.txt`) |
| Per-seed S4 files | `LSTM\Metrics\LSTM_{MuSPAD,onlyIncidence}_{test,train}set_perseed_2026-10-0{7,8}.csv` + 4 `..._metrics_mean_sd_10seeds.xlsx` |
| Locked wave-2 metrics (data only, no scripts) | `Compare_models\Metric_tables\LSTM_w2\` — `metrics_LSTM_w2_{test,train,seedpooled,perhorizon,verification}_2026-10-01.{csv,xlsx}`, `metrics_LSTM_w2_{test,train}_excl{65,675,75,80,Full}_2026-10-{01,02}.{csv,xlsx}`, `loo_LSTM_2026-10-02.{csv,xlsx}`, `architectures_LSTM_2026-10-01.{csv,xlsx}` |
| External inputs (time series + scaling params) | `S:\PROJACTIVE\MUSPAD-LABORERGEBNISSE\LOKI_ML\Datasets\` (one level above the project folder) |
| Interactive RData session (no saved script) | `LSTM\Python_results\Python_Original\Different_seeds\keras_tuner\results_h7_675percentData\.Rhistory` |
| Working copy used for the 10-03 verification pass (not the only home) | `C:\Users\abz26\AppData\Local\Temp\opencode\lstm_w2\` (includes `batch_run3.log`/`batch_run4.log`; successful run log ends `DONE`) |

**RData details.** The 10-01 14:44 session rebuilt only the four non-full variants
(`data_subsets = c("threefourthData","65percentData","80percentData","675percentData")`);
`fullData\LSTM_7day_*.RData` are older (2026-09-20 01:22). Object names differ accordingly:
full variant = `graph_LSTM_7day_M` / `graph_LSTM_7day_BL`; other variants = `graph_LSTM_M` / `graph_LSTM_BL`
(the batch's `objnames()` check tries the `7day` names first and falls back, so both work).

---

## 6. Locked metric protocol (as implemented in the 10-03 batch)

- **Point estimates** (T1, S3, S5, S6, S7): computed from **predictions averaged across the 10 seeds**,
  pooled over the 5 rolling origins — never the mean of per-seed metrics.
- **S4 SDs only**: from per-seed metrics evaluated separately for each seed (hence the S4
  caption note that point estimates need not equal the per-seed means).
- **MASE denominators** (one-step persistence MAE on the corresponding training series):
  pooled `5.000946`; LOO excl65 `5.173746`, excl675 `5.094352`, excl75 `4.852672`, excl80 `4.836091`, exclFull `5.040132`.
- **CIs**: moving-block bootstrap resampling complete 7-day forecast-origin blocks, B = 2000.
- **PIs (S5)**: residual bootstrap around the seed-averaged predictor (user-sanctioned definition, 2026-10-02).
  The LSTM models produce **no native prediction intervals** — all PI/coverage numbers are post-hoc.
- **Bootstrap seeds**: `SEED_PTEST=20261028`; LOO excl{65,675,75,80,Full} = `20261030…20261103`;
  `SEED_PI=20261104`; `SEED_TRAIN=20261105`.
- **Data**: n_train per origin — full 394, 65% 258, 67.5% 268, 80% 319, threefourth 299;
  test weeks contain no zero-incident days. Inputs: MuSPAD `(1, 7, 122)`, baseline `(1, 7, 1)`.
  Predictions are **unclipped** (can be negative).

---

## 7. Caveats and open items

- **C1 — S2 dropout column (resolved, for the record).** The `dropout` column of
  `architectures_LSTM_2026-10-01.csv` records the **LSTM-internal** dropout, which is `0.0` for every
  layer — and `config.json` of the actual saved models confirms internal dropout = 0.0. However, the
  models also contain **standalone `Dropout(rate=0.2)` layers** that the CSV does not record
  (MuSPAD: one between the LSTM and the Dense output; baseline: one after each of the three LSTMs).
  The non-zero dropout values reported in S2 Table refer to those standalone layers — table and model
  agree; only the CSV's documentation of it is incomplete.
- **C2 — No keras_tuner oracle logs** in the repository; the 30-trial search history per origin is not
  persisted. Only the selected (best) configurations are recoverable from the saved models.
- **C3 — Stale paths in the 10-03 batch.** `lstm_seedavg_batch_2026-10-03.R` points at
  `Compare_models\Graphics\Predicted_values\Diff_seeds\{variant_data}` (path defs at lines 102 and 318;
  load at line 215; `vfolder("65percentData") → "65percent_data"`). The collaborator reorganized
  `Predicted_values` after the batch (2026-10-07 → 10-13): the 20 LSTM RData now sit directly in
  `Predicted_values\{fullData,65percentData,675percentData,80percentData,threefourthData}\` and no
  `Diff_seeds` folder exists anymore. **The batch cannot be rerun as-is** — fix the three path
  references (or restore the old folder layout) first.
- **C4 — The RData build has no saved script.** It was an interactive session (evidence:
  `results_h7_675percentData\.Rhistory`). Recommend saving it as a proper script before any future
  data update.
- **C5 — Mixed RData vintages.** fullData RData (09-20) predate the sub-variant RData (10-01 14:44) and
  use different object names (`graph_LSTM_7day_*`). The batch handles both, but any future rebuild must
  keep the naming consistent.
- **C6 — Per-seed 10-07/08 files have no generator script.** Produced interactively in the R console
  (no `.Rhistory` or script found under `LSTM\Metrics\` or `Compare_models`). Format:
  `seed,RMSE,SMAPE,MASE,MAE,R2`, 10 rows = per-seed metrics pooled over the 5 origins.
- **C7 — Intermediate-generation artifacts.** G0 (08-31) models and the buggy old-named
  `final_model_only_incidence_tplus7_seed{1,2}.keras` remnants in top-level `results_h7\` must not be
  mistaken for G1/G2 files (see section 4).
- **C8 — Dead code.** LIME/SHAP imports in `python_lstm_with_Expl_diff_seeds.py` (unused for fitting);
  `build_model(hp)` in `python_lstm_onlyIncidence_diff_seeds.py` (never called; references undefined
  `timesteps`).
- **C9 — Batch run history.** The first batch launch failed (`Fehler in abs(e): Nicht-numerisches
  Argument`, captured in `batch_run_log.txt`, UTF-16); it was fixed and the successful run completed
  2026-10-03 04:35–04:37 (log ends `DONE`; per-origin LOO and PI summaries in the final lines).
- **C10 — Live share.** The collaborator is active on S: (Predicted_values reorganization 10-07→10-13,
  `VAR_with_seeds` folders added 10-13). Timestamps and layout documented here are as of the
  2026-10-13 documentation pass and may have moved since.

---

## 8. Minimum keep-in-repo set for LSTM reproducibility

1. **Scripts (2):** `python_lstm_with_Expl_diff_seeds.py`, `python_lstm_onlyIncidence_diff_seeds.py`
2. **Data prep (3):** `Data_prep_diff_splits_diff_seeds.R`, `Data_prep_for_Python_script_7day_prediction_diff_seeds.R`, `Data_prep_threefourthData_diff_seeds.R`
3. **Fit outputs:** `keras_tuner\results_h7*\` (G1 MuSPAD + G2 baseline models and prediction CSVs)
4. **RData:** the 20 `LSTM_7day_*.RData` in `Compare_models\Graphics\Predicted_values\{variantData}\`
5. **Metrics:** `LSTM\Metrics\` (batch script, all 10-03 outputs, per-seed 10-07/08 files, protocol README)
6. **Locked wave-2:** `Compare_models\Metric_tables\LSTM_w2\` (incl. `architectures_LSTM_2026-10-01.*`)
7. **Docs:** this README + the three companion documents from the header.

## 9. Rerun order (if ever needed)

1. R data prep (section 2.2) → `LOKI_ML\Datasets\`
2. Python fitting (both scripts; the MuSPAD one is the long pole: 30 tuning trials × 5 origins × 10 seeds)
3. RData build — **must first be recreated as a saved script** (C4) and pointed at the current
   `Predicted_values` layout
4. `lstm_seedavg_batch_2026-10-03.R` — **after fixing the stale `Diff_seeds` paths** (C3)
5. Per-seed metrics for S4 — recreate as a script (C6)

No rerun of any step was performed during this documentation pass.

---

*Prepared 2026-10-13. All statements verified against file contents (`config.json` of the saved models,
`.Rhistory` sessions, batch logs, output CSVs) and file timestamps on the share; no code was executed
on S: for this documentation.*
