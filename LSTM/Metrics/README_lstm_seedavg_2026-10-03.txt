README_lstm_seedavg_2026-10-03.txt
LSTM W2 seed-averaged (seedavg) batch - deliverables and verification record
================================================================

Run date: 2026-10-03 (date stamp = run date, not the 2026-10-02 protocol date)
Batch script: lstm_seedavg_batch_2026-10-03.R (67769 B, SHA-256 45835BF9C9337089F052285F8F45156DC72CF02862645D0B4A353C8CB9B78704)
Run log: batch_run4.log - EXIT 0, GUARD SUMMARY 6/6 PASS, ALL GUARDS PASS, DONE (line 496)
No locked scripts or deliverables were modified; all work done in a fresh clone.
No model refits; predictions reused from locked 10-01 RData (model seeds 1-10 per origin).

Protocol (user-specified 2026-10-02)
----------------------------------------------------------------
1. Primary estimator: seed-average the PREDICTIONS across the 10 model seeds
   (per origin, per horizon h), then compute all metrics on the averaged
   predictions. Never average per-seed metrics, never stack seeds for primary
   analysis (no n=350 pseudoreplication).
2. Pooled test: 35 obs/pred pairs (5 origins x 7 days each); CIs by origin-block
   bootstrap (5 blocks of 7 days, B=2000, percentile, seed 20261028).
3. Per-seed metrics (pooled 35-day, 10 seeds) are secondary, reported as
   mean +/- SD; the per-seed level reproduces the locked 10-01 seedpooled CSV
   exactly (enforced by guard G2).
4. Calibration: OLS actual ~ 1 + seedavg_forecast on 35 OOS pairs (n=35, df=33);
   CIs reuse the pooled-test bootstrap draws (coarse 5-origin uncertainty);
   F_joint = joint identity-line test (intercept 0, slope 1), F(2,33);
   p assumes IID and is reference only.
5. Leave-one-origin-out (LOO): 5 exclusions, seed-average 4 origins x 7 days = 28
   points per exclusion; CIs by 4-origin block bootstrap, one shared stream per
   exclusion -> paired delta (MuSPAD - onlyIncidence; negative = MuSPAD better).
6. PI: residual bootstrap around the seed-averaged predictor (no native LSTM PIs
   exist; user-sanctioned 2026-10-02), B=2000 iid residual draws per day, type-7
   quantiles 0.025/0.975; normal-alternative PI as supplement. Coverage CIs:
   binom.test (Clopper-Pearson) + origin-block bootstrap (same stream).
7. Training: 1538 pooled days (per-origin n_train: 65%=258, 67.5%=268, 75%=299,
   80%=319, full=394); 8 metrics (no DA/skill); CIs by origin-block bootstrap on
   train windows (seed 20261105).

Files (semicolon-delimited, %.15g, NA as "NA"; each CSV has a same-name .xlsx twin)
----------------------------------------------------------------
1. metrics_LSTM_w2_seedavg_test_2026-10-03        : 35-day pooled (10 metrics + CIs) + 5 per-origin rows (7 metrics, CIs NA)
2. metrics_LSTM_w2_seedavg_train_2026-10-03       : 1538-day train (8 metrics + CIs) + per-origin rows + per-seed mean/sd rows
3. metrics_LSTM_w2_seedavg_calibration_2026-10-03 : pooled seedavg OLS (n=35) + 20 per-seed rows (CIs NA)
4. loo_LSTM_seedavg_diff_ci_2026-10-03            : 5 exclusions; per-model metrics + CIs and paired delta + CIs (no p column)
5. metrics_LSTM_w2_seedavg_pi_2026-10-03          : PI coverage, residual bootstrap + normal, pooled + per origin
6. loo_LSTM_seedavg_perseed_2026-10-03            : per-seed LOO metrics (seeds 1-10 + mean/sd row per exclusion)

Naming / convention flags (deviations from locked 10-01 files)
----------------------------------------------------------------
- Baseline model named "onlyIncidence" in all files of this batch (locked 10-01/10-02
  files keep "baseline"; the batch maps onlyIncidence -> "baseline" only for RData
  object names and 10-01 anchor labels).
- sMAPE on the 0-2 scale (chat report also shows x100).
- MASE denominator: pooled 5.00094585779517 = 10-01 pooled MAE_naive_train, identical
  for both models; LOO denominators = 4-origin day-pooled MAE_naive_train
  (excl65 5.173746 / excl675 5.094352 / excl75 4.852672 / excl80 4.836091 / exclFull 5.040132).
- Row labels "1:10" (mean) and "1:10_sd" (sd) throughout; the 10-01 seedpooled anchor
  uses set labels "Test"/"Training".
- Date stamp 2026-10-03 = run date.

Seed registry (B=2000, percentile; re-runs reuse seeds -> byte-identical outputs,
proven 3x: run3 == run4 pre-tolerance == run4 post-tolerance)
----------------------------------------------------------------
- 20261028 : pooled test 5-origin block bootstrap (draws reused for calibration CIs)
- 20261030 : LOO excl 65percentData            (shared stream for paired delta)
- 20261031 : LOO excl 675percentData           (shared stream)
- 20261101 : LOO excl threefourthData (75%)    (shared stream)
- 20261102 : LOO excl 80percentData            (shared stream)
- 20261103 : LOO excl full                     (shared stream)
- 20261104 : PI residual bootstrap + coverage block bootstrap (same stream)
- 20261105 : train pooled CI
- 20261106 : fallback reserved (NOT consumed)
- 20261029 : unused
- Model seeds 1-10 (per origin): locked 10-01 RData, not refit.

Guards (all executed before the digest; any failure stops before output)
----------------------------------------------------------------
- G1 PASS: locked 10-01 test + train CSVs reproduced from raw RData
  (max|diff| = 4.911e-11 over 2016 point values + 16 CI endpoints)
- G2 PASS: seed-average identity exact (max|pbar - colMeans| = 0.000e+00);
  per-seed 35-day level + mean/sd reproduce locked seedpooled CSV
  (max|diff| = 3.865e-12 over 240 values)
- G3 PASS: saved per-draw bootstrap matrices reproduce reported CIs;
  draw b=1 recomputed from raw blocks (max|diff| = 0.000e+00, tol 1e-14);
  delta points = diff of points, delta CIs = quantiles of saved paired differences
- G4 PASS: PI endpoints == seedavg + type-7 quantiles of saved draw matrix
  (max|diff| = 0.000e+00, tol 1e-14); covered recomputed exactly;
  binom.test reproducible; per-origin covered sums == pooled; normal PI reproducible
- G5 PASS: manual OLS == lm; R2cal dual-route; F_joint dual-route;
  p = pf(F,2,33); rmse_identity == pooled test RMSE
  (max|diff| = 1.252e-13 over 2 fits, tol 1e-10)
- G6 PASS: pooled MASE denom == 10-01 pooled MAE_naive_train (|diff| = 1.776e-15)
  and == day-pooled weighted mean of per-origin values (|diff| = 8.882e-16);
  identical MuSPAD vs onlyIncidence; LOO denoms = 4 remaining origins (5/5)

Tolerance note (flagged in report): G1/G2 tolerance changed from absolute 1e-12 to
relative 1e-12 vs max|stored| of the locked CSV read back. Rationale: the locked CSVs
store %.15g, so storage rounding is ~5e-15 relative scale; observed max|diff| sits at
relative ~1.2e-15 (G1) and ~3.2e-16 (G2) = pure storage rounding, no numerical
mismatch. Guard-design fix only - zero change to outputs (all 6 CSVs byte-identical
to pre-change hashes, see table below).

SHA-256 (computed after final content; S: mirror verified 2/2 MATCH per file)
----------------------------------------------------------------
loo_LSTM_seedavg_diff_ci_2026-10-03.csv           |  6210 | 36347A8DA53798043D434DF744824406485196A9E9B333C1FF3C875557EBCF24
loo_LSTM_seedavg_diff_ci_2026-10-03.xlsx          |  8835 | 296159080444533CEA1955B349A5C906390681E52D762C6AA738D710D0EB5C54
loo_LSTM_seedavg_perseed_2026-10-03.csv           | 18143 | 8DC045229DEF14EAA32261C01B47C097A1DE3880D0696CBA0D75ADBE8CF9F071
loo_LSTM_seedavg_perseed_2026-10-03.xlsx          | 17399 | 3029835BDD5096E6FDE3762AA6E5E9C1E063DA5AA6377F9CD9F6E8B4B6794E55
metrics_LSTM_w2_seedavg_calibration_2026-10-03.csv | 6232 | 7540D214B4B8102DACAA46CBF27A763F8175415D09A6F1039F62D72DAC52A238
metrics_LSTM_w2_seedavg_calibration_2026-10-03.xlsx | 9742 | EA9CBBB97364E152C7EB38C8DB72F4ABC4ED35AE402FE2B8B0BB73F5F0901013
metrics_LSTM_w2_seedavg_pi_2026-10-03.csv         |  9942 | 76C32CBD28BC0CFD5BA4C67099844D181AA9F9F8BB36C9864FEE87D30CED740D
metrics_LSTM_w2_seedavg_pi_2026-10-03.xlsx        |  8012 | B71C0D2EE26A1BC4A3FC8D2E1974D3FE7D5F7912DCF983C6C56DDA1655BC8176
metrics_LSTM_w2_seedavg_test_2026-10-03.csv       | 11949 | 8F9035D2C3E44D365ECBD91A72E195C44A2D67A8AC36CEB05D21013D6310AA45
metrics_LSTM_w2_seedavg_test_2026-10-03.xlsx      | 14581 | DA3C0C710B9D52FDDE61FB73602DD63BBB1A86D8D61D98A9FCD5F19280C624AB
metrics_LSTM_w2_seedavg_train_2026-10-03.csv      | 10008 | CC1A54D65AB7B7BFB473F9B7F7EBDD0335E4AADB2A476A9D7ED2A4B60BBA204F
metrics_LSTM_w2_seedavg_train_2026-10-03.xlsx     | 13168 | 6386C82F7CE07574DC25621C58DDEB156D8ED105FB8ABED8C3BADFA5571FF371
lstm_seedavg_batch_2026-10-03.R (script record)   | 67769 | 45835BF9C9337089F052285F8F45156DC72CF02862645D0B4A353C8CB9B78704
