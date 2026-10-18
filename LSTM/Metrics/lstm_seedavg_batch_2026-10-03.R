# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

#!/usr/bin/env Rscript
# ============================================================================
# LSTM seed-averaged wave-2 metrics (SEEDAVG) — run 2026-10-03, R 4.6.1 x64.
# Spec: 2026-10-02 user instruction — mirror the VAR pipeline with one extra
# layer: seeds -> AVERAGE PREDICTIONS -> origins -> origin-block CIs.
#   * PRIMARY analysis is on the seed-averaged predictions
#     pbar = (1/10) * sum_s pred_s (never averaged metrics, never stacked
#     seeds for the primary result).
#   * Per-seed metrics are SECONDARY, reported as mean +/- SD across seeds.
#
# Sources (read-only; identical to the locked 2026-10-01 clean run):
#   <external location>
#     <variant folder>\LSTM_7day_{MuSPAD,baseline}_seeds<tok>.RData
#     object: graph_LSTM_7day_M/_BL (full_data) or graph_LSTM_M/_BL (others)
#     long format: time, real_vs_pred, Incidence, set (Training/Test), seed
#     (1..10); raw weekly incidence units
#   NOTE: the baseline model IS the onlyIncidence LSTM
#   (python_lstm_onlyIncidence_diff_seeds.py); it is reported here as
#   "onlyIncidence" (= "baseline" in the 2026-10-01/02 files,
#    RData objects graph_LSTM*_BL).
#
# Prediction intervals: NO native LSTM PIs exist anywhere (verified
# 2026-10-02: the Python seed generators emit point predictions only).
# 95% PIs = residual bootstrap around the SEED-AVERAGED predictor
# (user-sanctioned 2026-10-02), mirroring the MLP G4 mechanics:
#   per origin o: pool = in-sample residuals of the seed-averaged fit
#   (n_train_o = 258/268/299/319/394); E = matrix(sample(pool, B*7, TRUE),
#   nrow = B, ncol = 7); PI day t = pbar_te[o,t] + type-7 quantile
#   0.025/0.975 of column t. Normal-alternative PI as supplement
#   (pbar + qnorm(0.025/0.975) x sd(pool)). Coverage CIs: binom.test
#   (Clopper-Pearson) + origin-block bootstrap of the 5x7 covered blocks.
#
# Seeds (B = 2000 throughout, percentile CIs):
#   20261028  pooled-test origin-block draws (5 blocks x 7 days) -> pooled
#             test CIs (F1); the SAME draws are reused for the calibration
#             CIs (F2) (VAR convention: "reused draws")
#   20261030 / 20261031 / 20261101 / 20261102 / 20261103
#             LOO exclusions excl65 / excl675 / excl75 / excl80 / exclFull:
#             ONE shared draw stream per exclusion (4 blocks x 7 days), the
#             same origin draws applied to both models -> paired deltas
#             (MuSPAD - onlyIncidence) WITH CIs
#   20261104  PI residual bootstrap (model loop MuSPAD -> onlyIncidence,
#             origin loop full, 65, 67.5, 80, 75), then — same stream —
#             coverage origin-block bootstrap
#   20261105  pooled-train origin-block draws (5 blocks = full training
#             windows) -> train CIs
#   20261106  reserved fallback (unused)
#
# Conventions (identical to the VAR / 2026-10-01 LSTM analysis unless noted):
#   sMAPE on the 0-2 scale = mean(2|e|/(|a|+|f|)), 0/0 -> 0
#     (x100 = VAR-style 0-200 scale; both are reported in the chat report)
#   MASE denominator = day-pooled 1-day persistence MAE over the in-sample
#     training windows of the considered origins (seed-independent,
#     identical for both models): pooled = 5 origins (1533 lags),
#     LOO = the 4 remaining origins; per-seed levels use the pooled
#     denominator (2026-10-01 seed-pooled convention)
#   R2 around the mean of the (resampled) set; bias = mean(pred - real),
#   positive = over-prediction; delta = MuSPAD - onlyIncidence
#   (negative = MuSPAD better)
#   DA = share of matching 1-step direction changes (y0 = last training
#   actual); skill = 1 - sum(e^2)/sum(naive e^2), naive = 1-day persistence
#
# Outputs (all in lstm_w2/, semicolon / %.15g house style):
#   F1 metrics_LSTM_w2_seedavg_test_2026-10-03.csv
#   F2 metrics_LSTM_w2_seedavg_calibration_2026-10-03.csv
#   F3 loo_LSTM_seedavg_diff_ci_2026-10-03.csv
#   F4 metrics_LSTM_w2_seedavg_pi_2026-10-03.csv
#   F5 loo_LSTM_seedavg_perseed_2026-10-03.csv
#   F6 metrics_LSTM_w2_seedavg_train_2026-10-03.csv
#
# Guards (logged; stop at the end if any FAIL):
#   G1  reproduce the locked 2026-10-01 test + train CSVs from raw RData
#       (100/10/2 + 100/10/2 rows, all point metrics + 8 pooled CI
#       endpoints) <= 1e-12
#   G2  seed-average identity (pbar == colMeans, exact) and per-seed 35-day
#       level reproduces the locked metrics_LSTM_w2_seedpooled_2026-10-01.csv
#       (20 rows x 10 metrics + mean/sd rows) <= 1e-12
#   G3  bootstrap integrity: reported CIs == quantiles of the saved
#       per-draw matrices (<= 1e-14); draw b=1 recomputed from raw blocks
#       == saved row 1 (<= 1e-12); low <= high everywhere; delta points ==
#       diff of points and delta CIs == quantiles of the saved paired
#       differences (<= 1e-14)
#   G4  PI mechanics: lo/hi == pbar + type-7 quantiles of the saved draw
#       matrix (<= 1e-14); covered recomputed from saved lo/hi (exact);
#       binom.test CP reproducible (exact); per-origin covered sums ==
#       pooled (exact); normal PI reproducible; coverage block-bootstrap
#       draw-1 reproducible
#   G5  calibration internals: manual OLS == lm (<= 1e-12); F_joint
#       dual-route (SS difference vs (th-th0)'X'X(th-th0), X'X direct)
#       < 1e-10; n = 35, df = 33; R2cal == lm r.squared; rmse_identity ==
#       pooled RMSE (<= 1e-12)
#   G6  MASE denominators: pooled denom == MAE_naive_train of the 2026-10-01
#       pooled rows and == day-pooled mean of the per-origin values
#       (<= 1e-12); identical across models (exact); LOO denom = the 4
#       remaining origins, identical across models (exact)
# ============================================================================

options(stringsAsFactors = FALSE, warn = 1)

base   <- REPO_ROOT
proj   <- file.path(REPO_ROOT)
pv_dir <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values")
local  <- Sys.getenv("MUSPAD_VERIFICATION_DIR", file.path(REPO_ROOT, "Compare_models/Metric_tables/LSTM_w2"))
dir.create(local, showWarnings = FALSE, recursive = TRUE)
verification_files <- file.path(local, c("metrics_LSTM_w2_test_2026-10-01.csv",
  "metrics_LSTM_w2_train_2026-10-01.csv", "metrics_LSTM_w2_seedpooled_2026-10-01.csv"))
if (!all(file.exists(verification_files))) stop(
  "Missing historical verification CSVs: ", paste(verification_files[!file.exists(verification_files)], collapse = ", "),
  ". Set MUSPAD_VERIFICATION_DIR to the folder containing these inputs.")

DATE   <- "2026-10-03"
H      <- 7L
nseeds <- 10
B      <- 2000

variants <- c("full", "65percentData", "675percentData", "80percentData", "threefourthData")
vfolder  <- function(v) if (v == "full") "fullData" else v
vtoken   <- function(v) if (v == "full") "" else paste0("_", v)

models   <- c("MuSPAD", "onlyIncidence")
rdatname <- function(m, v) sprintf("LSTM_7day_%s_seeds%s.RData",
                                   if (m == "MuSPAD") "MuSPAD" else "baseline",
                                   vtoken(v))
objnames <- function(m) {
  if (m == "MuSPAD") c("graph_LSTM_7day_M", "graph_LSTM_M")
  else               c("graph_LSTM_7day_BL", "graph_LSTM_BL")
}
lab1001 <- function(m) if (m == "MuSPAD") "MuSPAD" else "baseline"

SEED_PTEST <- 20261028
SEED_LOO   <- c(excl65 = 20261030, excl675 = 20261031, excl75 = 20261101,
                excl80 = 20261102, exclFull = 20261103)
SEED_PI    <- 20261104
SEED_TRAIN <- 20261105

excl_key   <- c(excl65 = "65percentData", excl675 = "675percentData",
                excl75 = "threefourthData", excl80 = "80percentData",
                exclFull = "full")

MET10 <- c("MAE", "RMSE", "MSE", "R2", "MAPE", "sMAPE", "MASE", "bias",
           "DA", "skill")
MET6  <- c("RMSE", "MAE", "sMAPE", "MASE", "R2", "bias")

# ---------------- helpers (verbatim from metrics_lstm_w2.R) ----------------
paired <- function(allw, setv, seed) {
  d  <- allw[allw$seed == seed & allw$set == setv, ]
  re <- d[d$real_vs_pred == "real", ]; pr <- d[d$real_vs_pred == "predicted", ]
  stopifnot(nrow(re) == nrow(pr))
  re <- re[order(re$time), ]; pr <- pr[order(pr$time), ]
  stopifnot(identical(as.character(re$time), as.character(pr$time)))
  data.frame(time = as.character(re$time),
             real = as.numeric(re$Incidence),
             predicted = as.numeric(pr$Incidence))
}

smape_term <- function(a, f) {
  den <- abs(a) + abs(f)
  ifelse(den == 0, 0, 2 * abs(f - a) / den)      # 0-2 scale, 0/0 -> 0
}

da_vec <- function(real, pred, y0) {
  dy  <- real - c(y0, real[-length(real)])
  dyp <- pred - c(y0, real[-length(real)])
  as.numeric((dyp * dy > 0) | (dyp == 0 & dy == 0))
}

write_hs <- function(df, fn) {
  cols <- lapply(df, function(col) {
    if (is.numeric(col)) sprintf("%.15g", col) else as.character(col)
  })
  M <- do.call(cbind, cols)
  lines <- c(paste(paste0('"', names(df), '"'), collapse = ";"),
             apply(M, 1, function(r) paste(r, collapse = ";")))
  writeLines(lines, fn)
  cat("wrote:", fn, "\n")
}

metrics_from_blocks_test <- function(bs, naivemae_den) {
  real <- unlist(lapply(bs, function(b) b$real))
  e    <- unlist(lapply(bs, function(b) b$e))
  e2   <- unlist(lapply(bs, function(b) b$e2))
  list(MAE   = mean(abs(e)),
       RMSE  = sqrt(mean(e2)),
       MSE   = mean(e2),
       R2    = 1 - sum(e2) / sum((real - mean(real))^2),
       MAPE  = mean(unlist(lapply(bs, function(b) b$mapec))),
       sMAPE = mean(unlist(lapply(bs, function(b) b$smapc))),
       bias  = mean(e),
       MASE  = mean(abs(e)) / naivemae_den,
       DA    = mean(unlist(lapply(bs, function(b) b$da))),
       skill = 1 - sum(e2) / sum(unlist(lapply(bs, function(b) b$naive_e2))))
}

metrics_from_blocks_train <- function(bs, naivemae_den) {
  real <- unlist(lapply(bs, function(b) b$real))
  e    <- unlist(lapply(bs, function(b) b$e))
  e2   <- unlist(lapply(bs, function(b) b$e2))
  list(MAE   = mean(abs(e)),
       RMSE  = sqrt(mean(e2)),
       MSE   = mean(e2),
       R2    = 1 - sum(e2) / sum((real - mean(real))^2),
       MAPE  = mean(unlist(lapply(bs, function(b) b$mapec))),
       sMAPE = mean(unlist(lapply(bs, function(b) b$smapc))),
       bias  = mean(e),
       MASE  = mean(abs(e)) / naivemae_den)
}

metrics6 <- function(bs, den) {
  bm <- metrics_from_blocks_test(bs, den)
  unlist(bm[MET6])
}

# ---------------- load + per-cell loop (mirrors metrics_lstm_w2.R) --------
cells <- list()    # one entry per (model, origin, seed)
origs <- list()    # per model: list over variants

for (m in models) {
  origs[[m]] <- list()
  for (v in variants) {
    fn <- file.path(pv_dir, vfolder(v), rdatname(m, v))
    stopifnot(file.exists(fn))
    e0 <- new.env(); load(fn, envir = e0)
    found <- objnames(m)[objnames(m) %in% ls(e0)]
    if (length(found) != 1)
      stop(sprintf("expected exactly one of [%s] in %s; file contains: %s",
                   paste(objnames(m), collapse = ", "), fn, paste(ls(e0), collapse = ", ")))
    allw <- as.data.frame(e0[[found]])
    allw$time <- as.Date(as.character(allw$time))

    stopifnot(all(sort(unique(allw$seed)) == 1:nseeds))
    stopifnot(all(sort(unique(allw$set)) == c("Test", "Training")))
    cnt <- table(allw$seed, allw$set, allw$real_vs_pred)
    for (sv in c("Training", "Test"))
      stopifnot(all(cnt[, sv, , drop = FALSE] == cnt[1, sv, 1]))

    reals <- lapply(1:nseeds, function(s) {
      d <- allw[allw$seed == s & allw$real_vs_pred == "real", c("time", "Incidence")]
      d[order(d$time), , drop = FALSE]
    })
    for (s in 2:nseeds) {
      j <- merge(reals[[1]], reals[[s]], by = "time", suffixes = c("_1", "_2"))
      stopifnot(nrow(j) == nrow(reals[[1]]),
                max(abs(j$Incidence_1 - j$Incidence_2)) == 0)
    }

    tr1 <- paired(allw, "Training", 1)
    te1 <- paired(allw, "Test", 1)
    stopifnot(nrow(te1) == H)
    tr_days <- as.Date(tr1$time); te_days <- as.Date(te1$time)
    exp_days <- as.integer(seq(tr_days[length(tr_days)] + 1, by = "day", length.out = H))
    stopifnot(identical(as.integer(te_days), exp_days))

    o <- list(model = m, variant = v,
              n_train = nrow(tr1), n_test = H,
              train_from = tr_days[1], train_to = tr_days[length(tr_days)],
              origin_date = tr_days[length(tr_days)],
              test_from = te_days[1], test_to = te_days[H],
              train_days = tr_days, test_days = te_days,
              real_train = tr1$real,
              y0 = tr1$real[nrow(tr1)],
              n_zero_train = sum(tr1$real == 0),
              n_zero_test  = sum(te1$real == 0))
    o$train_naive     <- abs(o$real_train[-1] - o$real_train[-length(o$real_train)])
    o$mae_naive_train <- mean(o$train_naive)
    o$real_test   <- te1$real
    o$naive_test  <- c(o$y0, o$real_test[-H])
    o$naive_e2    <- (o$real_test - o$naive_test)^2
    o$mae_naive_test <- mean(abs(o$real_test - o$naive_test))
    o$mse_naive_test <- mean(o$naive_e2)

    tr_p <- matrix(NA_real_, nseeds, o$n_train)
    te_p <- matrix(NA_real_, nseeds, H)
    for (s in 1:nseeds) {
      tr <- paired(allw, "Training", s)
      te <- paired(allw, "Test", s)
      stopifnot(identical(tr$real, o$real_train), identical(te$real, o$real_test))
      e    <- te$predicted - te$real
      tr_p[s, ] <- tr$predicted
      te_p[s, ] <- te$predicted
      cells[[length(cells) + 1]] <- list(
        model = m, variant = v, org = match(v, variants), seed = s,
        n_train = nrow(tr), n_test = nrow(te),
        origin_date = o$origin_date, test_time = o$test_days,
        real = te$real, pred = te$predicted,
        e = e, abs_e = abs(e), e2 = e^2,
        mapec = 100 * abs(e) / abs(te$real),
        smapc = smape_term(te$real, te$predicted),
        naivemae_train = o$mae_naive_train,
        naivemae_test  = o$mae_naive_test,
        da = da_vec(te$real, te$predicted, o$y0),
        naive_e2 = o$naive_e2,
        tr_real = tr$real, tr_pred = tr$predicted,
        tr_mape = 100 * abs(tr$predicted - tr$real) / abs(tr$real),
        tr_smap = smape_term(tr$real, tr$predicted),
        n_zero_test = o$n_zero_test)
    }
    o$pred_tr_mat <- tr_p
    o$pred_te_mat <- te_p
    o$pbar_tr <- colMeans(tr_p)
    o$pbar_te <- colMeans(te_p)
    origs[[m]][[v]] <- o
    cat(sprintf("loaded %s | %s | n_train %d | n_zero_test %d\n", m, v, o$n_train, o$n_zero_test))
  }
}

# ---- guards: nested training windows (prefixes of full) ----
for (m in models) {
  for (v in variants) {
    of <- origs[[m]][["full"]]; ov <- origs[[m]][[v]]
    stopifnot(identical(ov$train_days, of$train_days[seq_len(ov$n_train)]))
  }
}
# ---- guards: cross-model identity (days + actuals) ----
for (v in variants) {
  a <- origs[["MuSPAD"]][[v]]; b2 <- origs[["onlyIncidence"]][[v]]
  stopifnot(identical(a$train_days, b2$train_days),
            identical(a$test_days,  b2$test_days),
            max(abs(a$real_train - b2$real_train)) == 0,
            max(abs(a$real_test  - b2$real_test))  == 0)
}

# ---- cross-check: test days vs VAR p7 files of the same origin ----
vardir <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values")
var_rdatname <- function(m, v) sprintf("VAR_7day_%s%s_p7_seeds.RData",
                                       if (m == "MuSPAD") "MuSPAD" else "baseline",
                                       vtoken(v))
var_objnames <- function(m) {
  if (m == "MuSPAD") c("graph_VAR_7day_M", "graph_VAR_M")
  else               c("graph_VAR_7day_bl", "graph_VAR_bl")
}
for (v in variants) {
  fn <- file.path(vardir, vfolder(v), "VAR_with_seeds", var_rdatname("MuSPAD", v))
  if (file.exists(fn)) {
    e0 <- new.env(); load(fn, envir = e0)
    found <- var_objnames("MuSPAD")[var_objnames("MuSPAD") %in% ls(e0)]
    stopifnot(length(found) == 1)
    wd <- as.data.frame(e0[[found]])
    wd$time <- as.Date(as.character(wd$time))
    vte <- sort(unique(wd$time[wd$set == "Test"]))
    lte <- origs[["MuSPAD"]][[v]]$test_days
    stopifnot(identical(as.integer(lte), as.integer(vte)))
  }
}

# ---- MASE denominators (seed-independent; identical for both models) ----
den_pool <- mean(unlist(lapply(variants, function(v) origs[["MuSPAD"]][[v]]$train_naive)))
stopifnot(den_pool == mean(unlist(lapply(variants, function(v) origs[["onlyIncidence"]][[v]]$train_naive))))
den_loo <- setNames(lapply(excl_key, function(vx)
  mean(unlist(lapply(setdiff(variants, vx), function(v) origs[["MuSPAD"]][[v]]$train_naive)))),
  names(excl_key))

# ---- seed-averaged blocks (THE primary objects) ----
# test: 5 blocks of 7 days; train: 5 blocks = full training windows
sablk <- list(); satrk <- list()
for (m in models) {
  sablk[[m]] <- setNames(lapply(variants, function(v) {
    o <- origs[[m]][[v]]
    e <- o$pbar_te - o$real_test
    list(real = o$real_test, e = e, e2 = e^2,
         mapec = 100 * abs(e) / abs(o$real_test),
         smapc = smape_term(o$real_test, o$pbar_te),
         da = da_vec(o$real_test, o$pbar_te, o$y0),
         naive_e2 = o$naive_e2)
  }), variants)
  satrk[[m]] <- setNames(lapply(variants, function(v) {
    o <- origs[[m]][[v]]
    e <- o$pbar_tr - o$real_train
    list(real = o$real_train, e = e, e2 = e^2,
         mapec = 100 * abs(e) / abs(o$real_train),
         smapc = smape_term(o$real_train, o$pbar_tr))
  }), variants)
}

# point metrics on the seed-averaged pairs
pt35  <- setNames(lapply(models, function(m) metrics_from_blocks_test(sablk[[m]], den_pool)), models)
pt7   <- list()
for (m in models) for (v in variants)
  pt7[[paste0(m, ";", v)]] <- metrics_from_blocks_test(sablk[[m]][v], origs[[m]][[v]]$mae_naive_train)
ptr35 <- setNames(lapply(models, function(m) metrics_from_blocks_train(satrk[[m]], den_pool)), models)
ptr7  <- list()
for (m in models) for (v in variants)
  ptr7[[paste0(m, ";", v)]] <- metrics_from_blocks_train(satrk[[m]][v], origs[[m]][[v]]$mae_naive_train)

err_free <- function(re, e, den) {
  real <- re; pred <- re + e
  c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MSE = mean(e^2),
    R2 = 1 - sum(e^2) / sum((real - mean(real))^2),
    MAPE = mean(100 * abs(e) / abs(real)),
    sMAPE = mean(smape_term(real, pred)),
    MASE = mean(abs(e)) / den, bias = mean(e))
}

# per-seed 35-day (secondary): mirrors the 2026-10-01 seedpooled view
perseed <- list()
for (m in models) {
  PS <- matrix(NA_real_, nseeds, 10)
  colnames(PS) <- MET10
  for (s in 1:nseeds) {
    ix <- which(sapply(cells, function(c) c$model == m & c$seed == s))
    stopifnot(length(ix) == 5)
    re <- unlist(cells[ix] |> lapply(function(c) c$real))
    e  <- unlist(cells[ix] |> lapply(function(c) c$e))
    m2 <- err_free(re, e, den_pool)
    da_all <- unlist(cells[ix] |> lapply(function(c) c$da))
    skill  <- 1 - sum(unlist(cells[ix] |> lapply(function(c) c$e2))) /
               sum(unlist(cells[ix] |> lapply(function(c) c$naive_e2)))
    PS[s, ] <- c(m2, DA = mean(da_all), skill = skill)
  }
  perseed[[m]] <- PS
}

# ============================================================================
# F1: pooled seed-averaged TEST table + origin-block bootstrap (seed 20261028)
# ============================================================================
set.seed(SEED_PTEST)
ptest_boot <- list(); ptest_D <- list()
for (m in models) {
  boot <- matrix(NA_real_, B, 10); D <- matrix(0L, B, 5)
  for (b in 1:B) {
    s <- sample(1:5, 5, replace = TRUE); D[b, ] <- s
    bm <- metrics_from_blocks_test(sablk[[m]][s], den_pool)
    boot[b, ] <- unlist(bm[MET10])
  }
  ptest_boot[[m]] <- boot; ptest_D[[m]] <- D
}
ptest_ci <- setNames(lapply(models, function(m) {
  ci <- t(apply(ptest_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
  attr(ci, "dimnames") <- list(MET10, c("low", "high")); ci
}), models)

f1_cols <- c("level", "model", "seed", "origin", "n_test",
             "MAE", "MAE_ci95_low", "MAE_ci95_high",
             "RMSE", "RMSE_ci95_low", "RMSE_ci95_high",
             "MSE", "MSE_ci95_low", "MSE_ci95_high",
             "R2", "R2_ci95_low", "R2_ci95_high",
             "MAPE", "MAPE_ci95_low", "MAPE_ci95_high",
             "sMAPE", "sMAPE_ci95_low", "sMAPE_ci95_high",
             "MASE", "MASE_ci95_low", "MASE_ci95_high",
             "bias", "bias_ci95_low", "bias_ci95_high",
             "DA", "DA_ci95_low", "DA_ci95_high",
             "skill_vs_naive_test", "skill_vs_naive_test_ci95_low",
             "skill_vs_naive_test_ci95_high",
             "MASE_denom", "n_zero_real_test", "test_from", "test_to", "bootstrap")

f1_row <- function(level, m, seedlab, v, n, met, ci = NULL, den, nzero,
                   tfrom = "", tto = "", boot = "") {
  r <- data.frame(
    level = level, model = m, seed = seedlab, origin = if (is.null(v)) "all" else v,
    n_test = n,
    MAE = met$MAE, MAE_ci95_low = if (is.null(ci)) NA else ci["MAE", 1],
    MAE_ci95_high = if (is.null(ci)) NA else ci["MAE", 2],
    RMSE = met$RMSE, RMSE_ci95_low = if (is.null(ci)) NA else ci["RMSE", 1],
    RMSE_ci95_high = if (is.null(ci)) NA else ci["RMSE", 2],
    MSE = met$MSE, MSE_ci95_low = if (is.null(ci)) NA else ci["MSE", 1],
    MSE_ci95_high = if (is.null(ci)) NA else ci["MSE", 2],
    R2 = met$R2, R2_ci95_low = if (is.null(ci)) NA else ci["R2", 1],
    R2_ci95_high = if (is.null(ci)) NA else ci["R2", 2],
    MAPE = met$MAPE, MAPE_ci95_low = if (is.null(ci)) NA else ci["MAPE", 1],
    MAPE_ci95_high = if (is.null(ci)) NA else ci["MAPE", 2],
    sMAPE = met$sMAPE, sMAPE_ci95_low = if (is.null(ci)) NA else ci["sMAPE", 1],
    sMAPE_ci95_high = if (is.null(ci)) NA else ci["sMAPE", 2],
    MASE = met$MASE, MASE_ci95_low = if (is.null(ci)) NA else ci["MASE", 1],
    MASE_ci95_high = if (is.null(ci)) NA else ci["MASE", 2],
    bias = met$bias, bias_ci95_low = if (is.null(ci)) NA else ci["bias", 1],
    bias_ci95_high = if (is.null(ci)) NA else ci["bias", 2],
    DA = met$DA, DA_ci95_low = if (is.null(ci)) NA else ci["DA", 1],
    DA_ci95_high = if (is.null(ci)) NA else ci["DA", 2],
    skill_vs_naive_test = met$skill,
    skill_vs_naive_test_ci95_low = if (is.null(ci)) NA else ci["skill", 1],
    skill_vs_naive_test_ci95_high = if (is.null(ci)) NA else ci["skill", 2],
    MASE_denom = den, n_zero_real_test = nzero,
    test_from = tfrom, test_to = tto, bootstrap = boot,
    stringsAsFactors = FALSE)
  r[f1_cols]
}

f1_rows <- list()
for (m in models) {
  ci <- ptest_ci[[m]]
  f1_rows[[length(f1_rows) + 1]] <- f1_row(
    "seedavg", m, "1:10", NULL, 35, pt35[[m]], ci, den_pool,
    sum(sapply(variants, function(v) origs[[m]][[v]]$n_zero_test)),
    boot = sprintf("origin-block 5x7, %d reps, percentile, seed %d, seed-averaged predictions (10 seeds), same draws reused for calibration CIs", B, SEED_PTEST))
}
for (m in models) for (v in variants) {
  o <- origs[[m]][[v]]
  f1_rows[[length(f1_rows) + 1]] <- f1_row(
    "origin_seedavg", m, "1:10", v, 7, pt7[[paste0(m, ";", v)]], NULL,
    o$mae_naive_train, o$n_zero_test,
    tfrom = as.character(o$test_from), tto = as.character(o$test_to))
}
for (m in models) {
  PS <- perseed[[m]]
  for (s in 1:nseeds)
    f1_rows[[length(f1_rows) + 1]] <- f1_row(
      "per_seed", m, s, NULL, 35, as.list(PS[s, ]), NULL, den_pool,
      sum(sapply(variants, function(v) origs[[m]][[v]]$n_zero_test)))
  f1_rows[[length(f1_rows) + 1]] <- f1_row(
    "per_seed_mean", m, "1:10", NULL, 35,
    as.list(colMeans(PS)), NULL, den_pool,
    sum(sapply(variants, function(v) origs[[m]][[v]]$n_zero_test)))
  f1_rows[[length(f1_rows) + 1]] <- f1_row(
    "per_seed_sd", m, "1:10_sd", NULL, 35,
    as.list(apply(PS, 2, sd)), NULL, den_pool,
    sum(sapply(variants, function(v) origs[[m]][[v]]$n_zero_test)))
}
f1_tab <- do.call(rbind, f1_rows)[, f1_cols, drop = FALSE]
write_hs(f1_tab, file.path(local, paste0("metrics_LSTM_w2_seedavg_test_", DATE, ".csv")))

# ============================================================================
# F2: calibration (OLS actual ~ 1 + seedavg_forecast on the 35 pairs)
#     CIs: refit OLS on each saved pooled-test draw (reused, seed 20261028)
# ============================================================================
ols_fit <- function(y, f) {
  X <- cbind(1, f)
  XtX <- crossprod(X); Xty <- crossprod(X, y)
  th  <- as.numeric(solve(XtX, Xty))
  r   <- y - X %*% th
  RSS <- sum(r^2); TSS <- sum((y - mean(y))^2)
  c(alpha = th[1], beta = th[2], RSS = RSS, R2cal = 1 - RSS / TSS,
    XtX = XtX)
}
calib_fit <- function(y, f) {
  XtX <- crossprod(cbind(1, f)); Xty <- crossprod(cbind(1, f), y)
  th  <- as.numeric(solve(XtX, Xty))
  r   <- y - cbind(1, f) %*% th
  RSS <- sum(r^2); TSS <- sum((y - mean(y))^2)
  list(alpha = th[1], beta = th[2], RSS = RSS, R2cal = 1 - RSS / TSS,
       XtX = XtX)
}
fjoint <- function(fit, y, f) {
  n <- length(y)
  RSS_id <- sum((y - f)^2)
  Fss <- ((RSS_id - fit$RSS) / 2) / (fit$RSS / (n - 2))
  th0 <- c(0, 1); th <- c(fit$alpha, fit$beta)
  Fq  <- (as.numeric(t(th - th0) %*% fit$XtX %*% (th - th0)) / 2) / (fit$RSS / (n - 2))
  list(F = Fss, Fq = Fq, p = pf(Fss, 2, n - 2, lower.tail = FALSE))
}

real35 <- setNames(lapply(models, function(m)
  unlist(lapply(variants, function(v) origs[[m]][[v]]$real_test))), models)
pbar35 <- setNames(lapply(models, function(m)
  unlist(lapply(variants, function(v) origs[[m]][[v]]$pbar_te))), models)

calib_pt <- list()
for (m in models) {
  fit <- calib_fit(real35[[m]], pbar35[[m]])
  fj  <- fjoint(fit, real35[[m]], pbar35[[m]])
  calib_pt[[m]] <- c(intercept = fit$alpha, slope = fit$beta, R2cal = fit$R2cal,
                     F_joint = fj$F, p_ref = fj$p, Fq = fj$Fq,
                     rmse_identity = sqrt(mean((real35[[m]] - pbar35[[m]])^2)))
}

# bootstrap refit on the saved pooled-test draws (reused)
draw_idx <- function(s) {
  Mpos <- matrix(s - 1L, nrow = 5, ncol = 7) * 7L +
          matrix(1:7, nrow = 5, ncol = 7, byrow = TRUE)
  as.vector(Mpos)
}
calib_boot <- list()
for (m in models) {
  cb <- matrix(NA_real_, B, 3)
  for (b in 1:B) {
    fitb <- calib_fit(real35[[m]][draw_idx(ptest_D[[m]][b, ])],
                      pbar35[[m]][draw_idx(ptest_D[[m]][b, ])])
    cb[b, ] <- c(fitb$alpha, fitb$beta, fitb$R2cal)
  }
  calib_boot[[m]] <- cb
}
calib_ci <- setNames(lapply(models, function(m) {
  ci <- t(apply(calib_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
  attr(ci, "dimnames") <- list(c("intercept", "slope", "R2cal"), c("low", "high")); ci
}), models)
calib_q <- setNames(lapply(models, function(m) {
  q <- t(apply(calib_boot[[m]], 2, function(x) quantile(x, c(0.25, 0.50, 0.75))))
  attr(q, "dimnames") <- list(c("intercept", "slope", "R2cal"), c("q25", "q50", "q75")); q
}), models)

f2_note <- "pooled OLS actual ~ 1 + seedavg_forecast on 35 OOS pairs (5 origins x 7 h, seed-averaged predictions), CIs from origin-block bootstrap (2000 reps, seed 20261028, draws reused from the pooled test CI, coarse 5-origin uncertainty, not precise inference), F_joint = joint test that the calibration line equals the identity line (intercept 0, slope 1), F(2,33), p assumes IID and is reference only"

f2_cols <- c("section", "level", "model", "seed", "n", "n_origins",
             "intercept", "intercept_ci95_low", "intercept_ci95_high",
             "intercept_q25", "intercept_q50", "intercept_q75",
             "slope", "slope_ci95_low", "slope_ci95_high",
             "slope_q25", "slope_q50", "slope_q75",
             "R2cal", "R2cal_ci95_low", "R2cal_ci95_high",
             "F_joint", "p_classical_ref_IID", "rmse_identity",
             "n_boot_excluded", "note")

f2_row <- function(level, m, seedlab, vals, ci = NULL, q = NULL, note = "",
                   nboot = NA_real_) {
  r <- data.frame(
    section = "seedavg", level = level, model = m, seed = seedlab,
    n = 35, n_origins = 5,
    intercept = vals[["intercept"]],
    intercept_ci95_low = if (is.null(ci)) NA else ci["intercept", 1],
    intercept_ci95_high = if (is.null(ci)) NA else ci["intercept", 2],
    intercept_q25 = if (is.null(q)) NA else q["intercept", 1],
    intercept_q50 = if (is.null(q)) NA else q["intercept", 2],
    intercept_q75 = if (is.null(q)) NA else q["intercept", 3],
    slope = vals[["slope"]],
    slope_ci95_low = if (is.null(ci)) NA else ci["slope", 1],
    slope_ci95_high = if (is.null(ci)) NA else ci["slope", 2],
    slope_q25 = if (is.null(q)) NA else q["slope", 1],
    slope_q50 = if (is.null(q)) NA else q["slope", 2],
    slope_q75 = if (is.null(q)) NA else q["slope", 3],
    R2cal = vals[["R2cal"]],
    R2cal_ci95_low = if (is.null(ci)) NA else ci["R2cal", 1],
    R2cal_ci95_high = if (is.null(ci)) NA else ci["R2cal", 2],
    F_joint = vals[["F_joint"]], p_classical_ref_IID = vals[["p_ref"]],
    rmse_identity = if (is.null(vals[["rmse_identity"]])) NA else vals[["rmse_identity"]],
    n_boot_excluded = nboot, note = note, stringsAsFactors = FALSE)
  r[f2_cols]
}

f2_rows <- list()
for (m in models) {
  f2_rows[[length(f2_rows) + 1]] <- f2_row(
    "seedavg", m, "1:10", calib_pt[[m]], calib_ci[[m]], calib_q[[m]],
    note = f2_note, nboot = 0)
}
for (m in models) {
  CS <- matrix(NA_real_, nseeds, 6)
  colnames(CS) <- c("intercept", "slope", "R2cal", "F_joint", "p_ref", "rmse_identity")
  for (s in 1:nseeds) {
    ix <- which(sapply(cells, function(c) c$model == m & c$seed == s))
    re <- unlist(cells[ix] |> lapply(function(c) c$real))
    pe <- unlist(cells[ix] |> lapply(function(c) c$pred))
    fit <- calib_fit(re, pe)
    fj  <- fjoint(fit, re, pe)
    CS[s, ] <- c(fit$alpha, fit$beta, fit$R2cal, fj$F, fj$p,
                 sqrt(mean((re - pe)^2)))
  }
  for (s in 1:nseeds)
    f2_rows[[length(f2_rows) + 1]] <- f2_row("per_seed", m, s, as.list(CS[s, ]))
  f2_rows[[length(f2_rows) + 1]] <- f2_row(
    "per_seed_mean", m, "1:10",
    as.list(c(intercept = mean(CS[, 1]), slope = mean(CS[, 2]),
              R2cal = mean(CS[, 3]), F_joint = mean(CS[, 4]),
              p_ref = mean(CS[, 5]), rmse_identity = mean(CS[, 6]))))
  f2_rows[[length(f2_rows) + 1]] <- f2_row(
    "per_seed_sd", m, "1:10_sd",
    as.list(c(intercept = sd(CS[, 1]), slope = sd(CS[, 2]),
              R2cal = sd(CS[, 3]), F_joint = sd(CS[, 4]),
              p_ref = sd(CS[, 5]), rmse_identity = sd(CS[, 6]))))
}
f2_tab <- do.call(rbind, f2_rows)[, f2_cols, drop = FALSE]
write_hs(f2_tab, file.path(local, paste0("metrics_LSTM_w2_seedavg_calibration_", DATE, ".csv")))

# ============================================================================
# F3: LOO (5 exclusions) — paired origin-block bootstrap, shared draws
# ============================================================================
loo_res <- list()
for (ek in names(excl_key)) {
  v_ex <- excl_key[[ek]]
  keep <- setdiff(variants, v_ex)
  sM <- sablk[["MuSPAD"]][keep]; sB <- sablk[["onlyIncidence"]][keep]
  den_e <- den_loo[[ek]]
  stopifnot(den_e == mean(unlist(lapply(keep, function(v) origs[["onlyIncidence"]][[v]]$train_naive))))

  pM <- metrics6(sM, den_e); pB <- metrics6(sB, den_e)

  set.seed(SEED_LOO[[ek]])
  D  <- matrix(0L, B, 4)
  bM <- matrix(NA_real_, B, 6); bB <- matrix(NA_real_, B, 6)
  colnames(bM) <- MET6; colnames(bB) <- MET6
  for (b in 1:B) {
    s <- sample(1:4, 4, replace = TRUE); D[b, ] <- s
    bM[b, ] <- metrics6(sM[s], den_e)
    bB[b, ] <- metrics6(sB[s], den_e)
  }
  ciM <- t(apply(bM, 2, function(x) quantile(x, c(0.025, 0.975))))
  ciB <- t(apply(bB, 2, function(x) quantile(x, c(0.025, 0.975))))
  dM  <- bM - bB
  ciD <- t(apply(dM, 2, function(x) quantile(x, c(0.025, 0.975))))
  loo_res[[ek]] <- list(v_ex = v_ex, keep = keep, den = den_e,
                        pM = pM, pB = pB, D = D, bM = bM, bB = bB,
                        ciM = ciM, ciB = ciB, ciD = ciD)
}

f3_cols <- c("excluded_origin", "n_points", "n_origins")
for (mt in MET6) for (mo in models) f3_cols <- c(f3_cols,
  paste0(mt, "_", mo), paste0(mt, "_", mo, "_ci95_low"), paste0(mt, "_", mo, "_ci95_high"))
for (mt in MET6) f3_cols <- c(f3_cols, paste0("delta_", mt),
  paste0("delta_", mt, "_ci95_low"), paste0("delta_", mt, "_ci95_high"))
f3_cols <- c(f3_cols, "MASE_denom", "bootstrap_seed")

f3_rows <- list()
for (ek in names(excl_key)) {
  r <- loo_res[[ek]]
  row <- list(excluded_origin = r$v_ex, n_points = 28, n_origins = 4)
  for (i in seq_along(MET6)) {
    row[[paste0(MET6[i], "_MuSPAD")]]                 <- r$pM[MET6[i]]
    row[[paste0(MET6[i], "_MuSPAD_ci95_low")]]        <- r$ciM[MET6[i], 1]
    row[[paste0(MET6[i], "_MuSPAD_ci95_high")]]       <- r$ciM[MET6[i], 2]
    row[[paste0(MET6[i], "_onlyIncidence")]]          <- r$pB[MET6[i]]
    row[[paste0(MET6[i], "_onlyIncidence_ci95_low")]] <- r$ciB[MET6[i], 1]
    row[[paste0(MET6[i], "_onlyIncidence_ci95_high")]]<- r$ciB[MET6[i], 2]
    row[[paste0("delta_", MET6[i])]]                   <- r$pM[MET6[i]] - r$pB[MET6[i]]
    row[[paste0("delta_", MET6[i], "_ci95_low")]]      <- r$ciD[MET6[i], 1]
    row[[paste0("delta_", MET6[i], "_ci95_high")]]     <- r$ciD[MET6[i], 2]
  }
  row[["MASE_denom"]] <- r$den
  row[["bootstrap_seed"]] <- SEED_LOO[[ek]]
  f3_rows[[length(f3_rows) + 1]] <- as.data.frame(row, stringsAsFactors = FALSE)
}
f3_tab <- do.call(rbind, f3_rows)[, f3_cols, drop = FALSE]
write_hs(f3_tab, file.path(local, paste0("loo_LSTM_seedavg_diff_ci_", DATE, ".csv")))

# ============================================================================
# F5: LOO per-seed supplement (no CIs)
# ============================================================================
f5_cols <- c("excluded_origin", "model", "seed", "n_points",
             "MAE", "RMSE", "sMAPE", "MASE", "R2", "bias", "MASE_denom")
f5_rows <- list()
for (ek in names(excl_key)) {
  keep <- loo_res[[ek]]$keep
  den_e <- loo_res[[ek]]$den
  for (m in models) {
    MS <- matrix(NA_real_, nseeds, 6)
    colnames(MS) <- c("MAE", "RMSE", "sMAPE", "MASE", "R2", "bias")
    for (s in 1:nseeds) {
      ix <- which(sapply(cells, function(c) c$model == m & c$seed == s & c$variant %in% keep))
      stopifnot(length(ix) == 4)
      re <- unlist(cells[ix] |> lapply(function(c) c$real))
      e  <- unlist(cells[ix] |> lapply(function(c) c$e))
      MS[s, ] <- c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)),
                   sMAPE = mean(unlist(cells[ix] |> lapply(function(c) c$smapc))),
                   MASE = mean(abs(e)) / den_e,
                   R2 = 1 - sum(unlist(cells[ix] |> lapply(function(c) c$e2))) /
                          sum((re - mean(re))^2),
                   bias = mean(e))
    }
    for (s in 1:nseeds)
      f5_rows[[length(f5_rows) + 1]] <- data.frame(
        excluded_origin = loo_res[[ek]]$v_ex, model = m, seed = s, n_points = 28,
        MAE = MS[s, 1], RMSE = MS[s, 2], sMAPE = MS[s, 3], MASE = MS[s, 4],
        R2 = MS[s, 5], bias = MS[s, 6], MASE_denom = den_e,
        stringsAsFactors = FALSE)
    f5_rows[[length(f5_rows) + 1]] <- data.frame(
      excluded_origin = loo_res[[ek]]$v_ex, model = m, seed = "1:10", n_points = 28,
      MAE = mean(MS[, 1]), RMSE = mean(MS[, 2]), sMAPE = mean(MS[, 3]),
      MASE = mean(MS[, 4]), R2 = mean(MS[, 5]), bias = mean(MS[, 6]),
      MASE_denom = den_e, stringsAsFactors = FALSE)
    f5_rows[[length(f5_rows) + 1]] <- data.frame(
      excluded_origin = loo_res[[ek]]$v_ex, model = m, seed = "1:10_sd", n_points = 28,
      MAE = sd(MS[, 1]), RMSE = sd(MS[, 2]), sMAPE = sd(MS[, 3]),
      MASE = sd(MS[, 4]), R2 = sd(MS[, 5]), bias = sd(MS[, 6]),
      MASE_denom = den_e, stringsAsFactors = FALSE)
  }
}
f5_tab <- do.call(rbind, f5_rows)[, f5_cols, drop = FALSE]
write_hs(f5_tab, file.path(local, paste0("loo_LSTM_seedavg_perseed_", DATE, ".csv")))

# ============================================================================
# F4: prediction intervals (residual bootstrap around seedavg predictor)
#     seed 20261104: PI draws first (model loop x origin loop), then the
#     coverage origin-block bootstrap continues on the same stream
# ============================================================================
set.seed(SEED_PI)
piE <- list(); pires <- list()
for (m in models) {
  for (v in variants) {
    o <- origs[[m]][[v]]
    pool <- o$real_train - o$pbar_tr
    E <- matrix(sample(pool, size = B * 7, replace = TRUE), nrow = B, ncol = 7)
    lo <- vapply(1:7, function(t) quantile(E[, t], 0.025, type = 7) + o$pbar_te[t], numeric(1))
    hi <- vapply(1:7, function(t) quantile(E[, t], 0.975, type = 7) + o$pbar_te[t], numeric(1))
    covered <- (o$real_test >= lo) & (o$real_test <= hi)
    sd_p <- sd(pool)
    lo_n <- o$pbar_te + qnorm(0.025) * sd_p
    hi_n <- o$pbar_te + qnorm(0.975) * sd_p
    covered_n <- (o$real_test >= lo_n) & (o$real_test <= hi_n)
    piE[[paste0(m, ";", v)]] <- E
    pires[[paste0(m, ";", v)]] <- list(
      pool_n = length(pool), pool_sd = sd_p,
      lo = lo, hi = hi, covered = as.integer(covered),
      lo_n = lo_n, hi_n = hi_n, covered_n = as.integer(covered_n))
  }
}
# coverage origin-block bootstrap (same stream)
covBB <- list(); covD <- list(); covmat_l <- list()
for (m in models) {
  covered5 <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$covered))
  cm <- matrix(NA_real_, B, 1); Dcov <- matrix(0L, B, 5)
  for (b in 1:B) {
    s <- sample(1:5, 5, replace = TRUE); Dcov[b, ] <- s
    cm[b, 1] <- mean(covered5[draw_idx(s)])
  }
  covBB[[m]] <- quantile(cm, c(0.025, 0.975))
  covD[[m]] <- Dcov; covmat_l[[m]] <- cm
}

f4_cols <- c("model", "variant", "origin", "n", "covered", "coverage",
             "coverage_binom_low", "coverage_binom_high",
             "coverage_blockboot_low", "coverage_blockboot_high",
             "mean_width", "median_width", "min_width", "max_width",
             "nominal", "pool_sd", "pi_seed", "pi_note")
f4_note_res  <- "residual bootstrap around the seed-averaged predictor (no native LSTM PIs exist, user-sanctioned 2026-10-02), pool = in-sample residuals of the seed-averaged fit, B=2000 iid draws per day, type-7 quantiles 0.025/0.975, coverage CIs: binom.test (Clopper-Pearson) + origin-block bootstrap (same stream, seed 20261104)"
f4_note_norm <- "normal alternative: seed-averaged forecast + qnorm(0.025/0.975) x sd(in-sample residuals), coverage CI binom.test (Clopper-Pearson) only"

f4_rows <- list()
for (m in models) {
  for (vr in c("residual_bootstrap", "normal")) {
    agg <- if (vr == "residual_bootstrap") {
      lo7  <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$lo))
      hi7  <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$hi))
      cov7 <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$covered))
    } else {
      lo7  <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$lo_n))
      hi7  <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$hi_n))
      cov7 <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$covered_n))
    }
    w_all <- hi7 - lo7
    bt_all <- binom.test(sum(cov7), 35)
    f4_rows[[length(f4_rows) + 1]] <- data.frame(
      model = m, variant = vr, origin = "all", n = 35, covered = sum(cov7),
      coverage = mean(cov7),
      coverage_binom_low = bt_all$conf.int[1], coverage_binom_high = bt_all$conf.int[2],
      coverage_blockboot_low = if (vr == "residual_bootstrap") covBB[[m]][1] else NA,
      coverage_blockboot_high = if (vr == "residual_bootstrap") covBB[[m]][2] else NA,
      mean_width = mean(w_all), median_width = median(w_all),
      min_width = min(w_all), max_width = max(w_all),
      nominal = 0.95, pool_sd = NA, pi_seed = SEED_PI,
      pi_note = if (vr == "residual_bootstrap") f4_note_res else f4_note_norm,
      stringsAsFactors = FALSE)
    for (j in seq_along(variants)) {
      v <- variants[j]
      P <- pires[[paste0(m, ";", v)]]
      lo7v  <- if (vr == "residual_bootstrap") P$lo else P$lo_n
      hi7v  <- if (vr == "residual_bootstrap") P$hi else P$hi_n
      cov7v <- if (vr == "residual_bootstrap") P$covered else P$covered_n
      wv    <- hi7v - lo7v
      btv   <- binom.test(sum(cov7v), 7)
      f4_rows[[length(f4_rows) + 1]] <- data.frame(
        model = m, variant = vr, origin = v, n = 7, covered = sum(cov7v),
        coverage = mean(cov7v),
        coverage_binom_low = btv$conf.int[1], coverage_binom_high = btv$conf.int[2],
        coverage_blockboot_low = NA, coverage_blockboot_high = NA,
        mean_width = mean(wv), median_width = median(wv),
        min_width = min(wv), max_width = max(wv),
        nominal = 0.95, pool_sd = P$pool_sd, pi_seed = SEED_PI,
        pi_note = if (vr == "residual_bootstrap") f4_note_res else f4_note_norm,
        stringsAsFactors = FALSE)
    }
  }
}
f4_tab <- do.call(rbind, f4_rows)[, f4_cols, drop = FALSE]
write_hs(f4_tab, file.path(local, paste0("metrics_LSTM_w2_seedavg_pi_", DATE, ".csv")))

# ============================================================================
# F6: seed-averaged TRAIN table (origin-block CI, seed 20261105)
# ============================================================================
set.seed(SEED_TRAIN)
train_boot <- list(); train_D <- list()
for (m in models) {
  boot <- matrix(NA_real_, B, 8); D <- matrix(0L, B, 5)
  for (b in 1:B) {
    s <- sample(1:5, 5, replace = TRUE); D[b, ] <- s
    bm <- metrics_from_blocks_train(satrk[[m]][s], den_pool)
    boot[b, ] <- unlist(bm[c("MAE", "RMSE", "MSE", "R2", "MAPE", "sMAPE", "MASE", "bias")])
  }
  train_boot[[m]] <- boot; train_D[[m]] <- D
}
train_ci <- setNames(lapply(models, function(m) {
  ci <- t(apply(train_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
  attr(ci, "dimnames") <- list(c("MAE", "RMSE", "MSE", "R2", "MAPE", "sMAPE", "MASE", "bias"),
                               c("low", "high")); ci
}), models)

f6_cols <- c("level", "model", "seed", "origin", "n_rows", "window_from", "window_to",
             "MAE", "MAE_ci95_low", "MAE_ci95_high",
             "RMSE", "RMSE_ci95_low", "RMSE_ci95_high",
             "MSE", "MSE_ci95_low", "MSE_ci95_high",
             "R2", "R2_ci95_low", "R2_ci95_high",
             "MAPE", "MAPE_ci95_low", "MAPE_ci95_high",
             "sMAPE", "sMAPE_ci95_low", "sMAPE_ci95_high",
             "MASE", "MASE_ci95_low", "MASE_ci95_high",
             "bias", "bias_ci95_low", "bias_ci95_high",
             "MASE_denom", "bootstrap")
MT8 <- c("MAE", "RMSE", "MSE", "R2", "MAPE", "sMAPE", "MASE", "bias")

f6_row <- function(level, m, seedlab, v, n, met, ci = NULL, den, wfrom = "", wto = "",
                   boot = "") {
  r <- data.frame(level = level, model = m, seed = seedlab,
                  origin = if (is.null(v)) "all" else v, n_rows = n,
                  window_from = wfrom, window_to = wto,
                  stringsAsFactors = FALSE)
  for (mt in MT8) {
    r[[mt]] <- met[[mt]]
    r[[paste0(mt, "_ci95_low")]]  <- if (is.null(ci)) NA else ci[mt, 1]
    r[[paste0(mt, "_ci95_high")]] <- if (is.null(ci)) NA else ci[mt, 2]
  }
  r[["MASE_denom"]] <- den
  r[["bootstrap"]] <- boot
  r[f6_cols]
}

f6_rows <- list()
for (m in models) {
  ntot <- sum(sapply(variants, function(v) origs[[m]][[v]]$n_train))
  f6_rows[[length(f6_rows) + 1]] <- f6_row(
    "seedavg", m, "1:10", NULL, ntot, ptr35[[m]], train_ci[[m]], den_pool,
    boot = sprintf("origin-block 5x(train window), %d reps, percentile, seed %d, seed-averaged predictions (10 seeds)", B, SEED_TRAIN))
}
for (m in models) for (v in variants) {
  o <- origs[[m]][[v]]
  f6_rows[[length(f6_rows) + 1]] <- f6_row(
    "origin_seedavg", m, "1:10", v, o$n_train, ptr7[[paste0(m, ";", v)]], NULL,
    o$mae_naive_train,
    wfrom = as.character(o$train_from), wto = as.character(o$train_to))
}
for (m in models) {
  TS <- matrix(NA_real_, nseeds, 8)
  colnames(TS) <- MT8
  for (s in 1:nseeds) {
    ix <- which(sapply(cells, function(c) c$model == m & c$seed == s))
    stopifnot(length(ix) == 5)
    re <- unlist(cells[ix] |> lapply(function(c) c$tr_real))
    e  <- unlist(cells[ix] |> lapply(function(c) c$tr_pred - c$tr_real))
    TS[s, ] <- c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MSE = mean(e^2),
                 R2 = 1 - sum(e^2) / sum((re - mean(re))^2),
                 MAPE = mean(unlist(cells[ix] |> lapply(function(c) c$tr_mape))),
                 sMAPE = mean(unlist(cells[ix] |> lapply(function(c) c$tr_smap))),
                 MASE = mean(abs(e)) / den_pool, bias = mean(e))
  }
  ntot <- sum(sapply(variants, function(v) origs[[m]][[v]]$n_train))
  for (s in 1:nseeds)
    f6_rows[[length(f6_rows) + 1]] <- f6_row("per_seed", m, s, NULL, ntot,
                                             as.list(TS[s, ]), NULL, den_pool)
  f6_rows[[length(f6_rows) + 1]] <- f6_row("per_seed_mean", m, "1:10", NULL, ntot,
                                           as.list(colMeans(TS)), NULL, den_pool)
  f6_rows[[length(f6_rows) + 1]] <- f6_row("per_seed_sd", m, "1:10_sd", NULL, ntot,
                                           as.list(apply(TS, 2, sd)), NULL, den_pool)
}
f6_tab <- do.call(rbind, f6_rows)[, f6_cols, drop = FALSE]
write_hs(f6_tab, file.path(local, paste0("metrics_LSTM_w2_seedavg_train_", DATE, ".csv")))

# ============================================================================
# GUARDS
# ============================================================================
guard_log <- list()
log_guard <- function(name, ok, text) {
  guard_log[[length(guard_log) + 1]] <<- list(name = name, ok = ok, text = text)
  cat(sprintf("[%s] %s: %s\n", name, if (isTRUE(ok)) "PASS" else "FAIL", text))
  invisible(NULL)
}
tol12 <- 1e-12; tol14 <- 1e-14

test1001     <- read.csv(file.path(local, "metrics_LSTM_w2_test_2026-10-01.csv"),
                         sep = ";", quote = "\"", stringsAsFactors = FALSE)
train1001    <- read.csv(file.path(local, "metrics_LSTM_w2_train_2026-10-01.csv"),
                         sep = ";", quote = "\"", stringsAsFactors = FALSE)
seedpooled01 <- read.csv(file.path(local, "metrics_LSTM_w2_seedpooled_2026-10-01.csv"),
                         sep = ";", quote = "\"", stringsAsFactors = FALSE)

# ---- G1: reproduce locked 2026-10-01 test + train files ----
{
  dmax <- 0
  # origin_seed level (100 rows)
  for (c in cells) {
    r1 <- test1001[test1001$level == "origin_seed" &
                   test1001$model == lab1001(c$model) &
                   test1001$origin == c$variant &
                   as.integer(test1001$seed) == c$seed, , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    e  <- c$pred - c$real
    mt <- list(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MSE = mean(e^2),
            R2 = 1 - sum(e^2) / sum((c$real - mean(c$real))^2),
            MAPE = mean(c$mapec), sMAPE = mean(c$smapc),
            MASE = mean(abs(e)) / c$naivemae_train, bias = mean(e))
    dmax <- max(dmax,
      abs(mt$MAE - r1$MAE), abs(mt$RMSE - r1$RMSE), abs(mt$MSE - r1$MSE),
      abs(mt$R2 - r1$R2), abs(mt$MAPE - r1$MAPE), abs(mt$sMAPE - r1$sMAPE),
      abs(mt$MASE - r1$MASE), abs(mt$bias - r1$bias),
      abs(mean(c$da) - r1$DA),
      abs(1 - sum(c$e2) / sum(c$naive_e2) - r1$skill_vs_naive_test))
  }
  # origin level (10 rows)
  for (m in models) for (v in variants) {
    o <- origs[[m]][[v]]
    ix <- which(sapply(cells, function(c) c$model == m & c$variant == v))
    re <- rep(o$real_test, nseeds)
    e  <- unlist(cells[ix] |> lapply(function(c) c$e))
    m2 <- c(MAE = mean(abs(e)), RMSE = sqrt(mean(unlist(cells[ix] |> lapply(function(c) c$e2)))),
            MSE = mean(unlist(cells[ix] |> lapply(function(c) c$e2))),
            R2 = 1 - sum(unlist(cells[ix] |> lapply(function(c) c$e2))) /
                  sum((re - mean(re))^2),
            MAPE = mean(unlist(cells[ix] |> lapply(function(c) c$mapec))),
            sMAPE = mean(unlist(cells[ix] |> lapply(function(c) c$smapc))),
            bias = mean(e), MASE = mean(abs(e)) / o$mae_naive_train)
    r1 <- test1001[test1001$level == "origin" & test1001$model == lab1001(m) &
                   test1001$origin == v, , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    dmax <- max(dmax, abs(m2["MAE"] - r1$MAE), abs(m2["RMSE"] - r1$RMSE),
                abs(m2["MSE"] - r1$MSE), abs(m2["R2"] - r1$R2),
                abs(m2["MAPE"] - r1$MAPE), abs(m2["sMAPE"] - r1$sMAPE),
                abs(m2["MASE"] - r1$MASE), abs(m2["bias"] - r1$bias),
                abs(mean(unlist(cells[ix] |> lapply(function(c) c$da))) - r1$DA),
                abs(1 - sum(unlist(cells[ix] |> lapply(function(c) c$e2))) /
                      sum(unlist(cells[ix] |> lapply(function(c) c$naive_e2))) - r1$skill_vs_naive_test))
  }
  # pooled level (2 rows, incl. CI endpoints): regenerate the 10-01 blocks
  # (10-seed-pooled 70-day blocks) and compare all 10 point metrics; then
  # regenerate the 10-01 bootstrap draws (seed 20261002, shared stream
  # MuSPAD -> onlyIncidence) and compare the RMSE + sMAPE CI endpoints
  set.seed(20261002)
  for (m in models) {
    blocks1001 <- lapply(variants, function(v) {
      o  <- origs[[m]][[v]]
      ixv <- which(sapply(cells, function(c) c$model == m & c$variant == v))
      list(real = rep(o$real_test, nseeds),
           e    = unlist(cells[ixv] |> lapply(function(c) c$e)),
           e2   = unlist(cells[ixv] |> lapply(function(c) c$e2)),
           mapec = unlist(cells[ixv] |> lapply(function(c) c$mapec)),
           smapc = unlist(cells[ixv] |> lapply(function(c) c$smapc)),
           da     = unlist(cells[ixv] |> lapply(function(c) c$da)),
           naive_e2 = rep(o$naive_e2, nseeds))
    })
    pm <- metrics_from_blocks_test(blocks1001, den_pool)
    r1 <- test1001[test1001$level == "pooled" & test1001$model == lab1001(m), , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    dmax <- max(dmax,
      abs(pm$MAE - r1$MAE), abs(pm$RMSE - r1$RMSE), abs(pm$MSE - r1$MSE),
      abs(pm$R2 - r1$R2), abs(pm$MAPE - r1$MAPE), abs(pm$sMAPE - r1$sMAPE),
      abs(pm$MASE - r1$MASE), abs(pm$bias - r1$bias),
      abs(pm$DA - r1$DA), abs(pm$skill - r1$skill_vs_naive_test))
    boot <- matrix(NA_real_, B, 10)
    colnames(boot) <- MET10
    for (b in 1:B) {
      s  <- sample(1:5, 5, replace = TRUE)
      bm <- metrics_from_blocks_test(blocks1001[s], den_pool)
      boot[b, ] <- unlist(bm[MET10])
    }
    ci <- t(apply(boot, 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, abs(ci["RMSE", 1] - r1$RMSE_ci95_low),
                abs(ci["RMSE", 2] - r1$RMSE_ci95_high),
                abs(ci["sMAPE", 1] - r1$sMAPE_ci95_low),
                abs(ci["sMAPE", 2] - r1$sMAPE_ci95_high))
  }
  # train file (100 + 10 + 2 rows)
  for (c in cells) {
    e   <- c$tr_pred - c$tr_real
    mt  <- list(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MSE = mean(e^2),
             R2 = 1 - sum(e^2) / sum((c$tr_real - mean(c$tr_real))^2),
             MAPE = mean(c$tr_mape), sMAPE = mean(c$tr_smap),
             MASE = mean(abs(e)) / c$naivemae_train, bias = mean(e))
    r1 <- train1001[train1001$level == "origin_seed" &
                    train1001$model == lab1001(c$model) &
                    train1001$origin == c$variant &
                    as.integer(train1001$seed) == c$seed, , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    dmax <- max(dmax, abs(mt$MAE - r1$MAE), abs(mt$RMSE - r1$RMSE),
                abs(mt$MSE - r1$MSE), abs(mt$R2 - r1$R2), abs(mt$MAPE - r1$MAPE),
                abs(mt$sMAPE - r1$sMAPE), abs(mt$MASE - r1$MASE),
                abs(mt$bias - r1$bias))
  }
  for (m in models) for (v in variants) {
    o  <- origs[[m]][[v]]
    ix <- which(sapply(cells, function(c) c$model == m & c$variant == v))
    re <- rep(o$real_train, nseeds)
    e  <- unlist(cells[ix] |> lapply(function(c) c$tr_pred - c$tr_real))
    m2 <- c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MSE = mean(e^2),
            R2 = 1 - sum(e^2) / sum((re - mean(re))^2),
            MAPE = mean(unlist(cells[ix] |> lapply(function(c) c$tr_mape))),
            sMAPE = mean(unlist(cells[ix] |> lapply(function(c) c$tr_smap))),
            bias = mean(e), MASE = mean(abs(e)) / o$mae_naive_train)
    r1 <- train1001[train1001$level == "origin" & train1001$model == lab1001(m) &
                    train1001$origin == v, , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    dmax <- max(dmax, abs(m2["MAE"] - r1$MAE), abs(m2["RMSE"] - r1$RMSE),
                abs(m2["MSE"] - r1$MSE), abs(m2["R2"] - r1$R2),
                abs(m2["MAPE"] - r1$MAPE), abs(m2["sMAPE"] - r1$sMAPE),
                abs(m2["MASE"] - r1$MASE), abs(m2["bias"] - r1$bias))
  }
  set.seed(20261003)
  for (m in models) {
    blocks1001t <- lapply(variants, function(v) {
      o  <- origs[[m]][[v]]
      ixv <- which(sapply(cells, function(c) c$model == m & c$variant == v))
      list(real = rep(o$real_train, nseeds),
           e    = unlist(cells[ixv] |> lapply(function(c) c$tr_pred - c$tr_real)),
           e2   = unlist(cells[ixv] |> lapply(function(c) (c$tr_pred - c$tr_real)^2)),
           mapec = unlist(cells[ixv] |> lapply(function(c) c$tr_mape)),
           smapc = unlist(cells[ixv] |> lapply(function(c) c$tr_smap)))
    })
    allm <- metrics_from_blocks_train(blocks1001t, den_pool)
    boot <- matrix(NA_real_, B, 8)
    colnames(boot) <- MT8
    for (b in 1:B) {
      s  <- sample(1:5, 5, replace = TRUE)
      bm <- metrics_from_blocks_train(blocks1001t[s], den_pool)
      boot[b, ] <- unlist(bm[MT8])
    }
    ci <- t(apply(boot, 2, function(x) quantile(x, c(0.025, 0.975))))
    r1 <- train1001[train1001$level == "pooled" & train1001$model == lab1001(m), , drop = FALSE]
    stopifnot(nrow(r1) == 1)
    dmax <- max(dmax, abs(allm$MAE - r1$MAE), abs(allm$RMSE - r1$RMSE),
                abs(allm$MSE - r1$MSE), abs(allm$R2 - r1$R2),
                abs(allm$MAPE - r1$MAPE), abs(allm$sMAPE - r1$sMAPE),
                abs(allm$MASE - r1$MASE), abs(allm$bias - r1$bias),
                abs(ci["RMSE", 1] - r1$RMSE_ci95_low),
                abs(ci["RMSE", 2] - r1$RMSE_ci95_high),
                abs(ci["sMAPE", 1] - r1$sMAPE_ci95_low),
                abs(ci["sMAPE", 2] - r1$sMAPE_ci95_high))
  }
  vmx1 <- max(abs(unlist(test1001[sapply(test1001, is.numeric)])),
              abs(unlist(train1001[sapply(train1001, is.numeric)])), na.rm = TRUE)
  log_guard("G1", dmax <= 1e-12 * vmx1,
    sprintf("10-01 locked test + train CSVs reproduced from raw RData: max|diff| = %.3e over 2016 point values + 16 CI endpoints (rel tol 1e-12 vs max|stored| = %.4g)", dmax, vmx1))
}

# ---- G2: seed-average identity + per-seed level vs locked seedpooled file ----
{
  dmax_id <- max(sapply(models, function(m)
    max(sapply(variants, function(v) {
      o <- origs[[m]][[v]]
      max(max(abs(o$pbar_te - colMeans(o$pred_te_mat))),
          max(abs(o$pbar_tr - colMeans(o$pred_tr_mat))))
    }))))
  dmax_sp <- 0
  for (m in models) {
    PS <- perseed[[m]]
    for (s in 1:nseeds) {
      r1 <- seedpooled01[seedpooled01$level == "seed_pooled" &
                         seedpooled01$set == "Test" &
                         seedpooled01$model == lab1001(m) &
                         as.integer(seedpooled01$seed) == s, , drop = FALSE]
      stopifnot(nrow(r1) == 1)
      dmax_sp <- max(dmax_sp,
        abs(PS[s, "MAE"] - r1$MAE), abs(PS[s, "RMSE"] - r1$RMSE),
        abs(PS[s, "MSE"] - r1$MSE), abs(PS[s, "R2"] - r1$R2),
        abs(PS[s, "MAPE"] - r1$MAPE), abs(PS[s, "sMAPE"] - r1$sMAPE),
        abs(PS[s, "MASE"] - r1$MASE), abs(PS[s, "bias"] - r1$bias),
        abs(PS[s, "DA"] - r1$DA), abs(PS[s, "skill"] - r1$skill_vs_naive))
    }
    rmean <- seedpooled01[seedpooled01$level == "seed_mean_sd" &
                          seedpooled01$set == "Test" &
                          seedpooled01$model == lab1001(m) &
                          seedpooled01$seed == "1:10", , drop = FALSE]
    rsd   <- seedpooled01[seedpooled01$level == "seed_mean_sd" &
                          seedpooled01$set == "Test" &
                          seedpooled01$model == lab1001(m) &
                          seedpooled01$seed == "1:10_sd", , drop = FALSE]
    stopifnot(nrow(rmean) == 1, nrow(rsd) == 1)
    dmax_sp <- max(dmax_sp,
      max(abs(colMeans(PS) - c(rmean$MAE, rmean$RMSE, rmean$MSE, rmean$R2,
                               rmean$MAPE, rmean$sMAPE, rmean$MASE, rmean$bias,
                               rmean$DA, rmean$skill_vs_naive))),
      max(abs(apply(PS, 2, sd) - c(rsd$MAE, rsd$RMSE, rsd$MSE, rsd$R2,
                                   rsd$MAPE, rsd$sMAPE, rsd$MASE, rsd$bias,
                                   rsd$DA, rsd$skill_vs_naive))))
  }
  vmx2 <- max(abs(unlist(seedpooled01[sapply(seedpooled01, is.numeric)])), na.rm = TRUE)
  log_guard("G2", dmax_id == 0 && dmax_sp <= 1e-12 * vmx2,
    sprintf("seed-average identity exact (max|pbar - colMeans| = %.3e); per-seed 35-day level + mean/sd reproduce locked seedpooled CSV: max|diff| = %.3e over 240 values (rel tol 1e-12 vs max|stored| = %.4g)",
            dmax_id, dmax_sp, vmx2))
}

# ---- G3: bootstrap integrity (F1, F2, F3, F6) ----
{
  dmax <- 0
  # F1 pooled test: CI == quantiles of saved draws; draw 1 reproducible
  for (m in models) {
    ci_re <- t(apply(ptest_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, max(abs(ci_re - ptest_ci[[m]])))
    bm1 <- metrics_from_blocks_test(sablk[[m]][ptest_D[[m]][1, ]], den_pool)
    dmax <- max(dmax, max(abs(unlist(bm1[MET10]) - ptest_boot[[m]][1, ])))
  }
  # F2 calibration: CI == quantiles; draw 1 reproducible
  for (m in models) {
    ci_re <- t(apply(calib_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, max(abs(ci_re - calib_ci[[m]])))
    fit1 <- calib_fit(real35[[m]][draw_idx(ptest_D[[m]][1, ])],
                      pbar35[[m]][draw_idx(ptest_D[[m]][1, ])])
    dmax <- max(dmax, max(abs(c(fit1$alpha, fit1$beta, fit1$R2cal) - calib_boot[[m]][1, ])))
  }
  # F3 LOO: delta points, delta CIs, draw 1
  for (ek in names(excl_key)) {
    r <- loo_res[[ek]]
    r1f <- f3_tab[f3_tab$excluded_origin == r$v_ex, , drop = FALSE]
    stopifnot(nrow(r1f) == 1)
    for (mk in MET6)
      dmax <- max(dmax, abs(r1f[[paste0("delta_", mk)]] -
                          (r1f[[paste0(mk, "_MuSPAD")]] - r1f[[paste0(mk, "_onlyIncidence")]])))
    ciD_re <- t(apply(r$bM - r$bB, 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, max(abs(ciD_re - r$ciD)))
    ciM_re <- t(apply(r$bM, 2, function(x) quantile(x, c(0.025, 0.975))))
    ciB_re <- t(apply(r$bB, 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, max(abs(ciM_re - r$ciM)), max(abs(ciB_re - r$ciB)))
    s1 <- r$D[1, ]
    den_e <- r$den
    bm1 <- metrics6(sablk[["MuSPAD"]][r$keep][s1], den_e)
    bb1 <- metrics6(sablk[["onlyIncidence"]][r$keep][s1], den_e)
    dmax <- max(dmax, max(abs(bm1 - r$bM[1, ])), max(abs(bb1 - r$bB[1, ])))
  }
  # F6 train: CI == quantiles; draw 1 reproducible
  for (m in models) {
    ci_re <- t(apply(train_boot[[m]], 2, function(x) quantile(x, c(0.025, 0.975))))
    dmax <- max(dmax, max(abs(ci_re - train_ci[[m]])))
    bm1 <- metrics_from_blocks_train(satrk[[m]][train_D[[m]][1, ]], den_pool)
    dmax <- max(dmax, max(abs(unlist(bm1[MT8]) - train_boot[[m]][1, ])))
  }
  # CI ordering low <= high on every reported CI pair
  ord_ok <- TRUE
  for (tab in list(f1_tab, f2_tab, f3_tab, f4_tab, f6_tab)) {
    for (cn in names(tab)) {
      if (grepl("_ci95_low$", cn)) {
        hi <- sub("_ci95_low$", "_ci95_high$", cn)
        if (hi %in% names(tab) && is.numeric(tab[[cn]]) && is.numeric(tab[[hi]])) {
          lo_v <- tab[[cn]]; hi_v <- tab[[hi]]
          both <- !is.na(lo_v) & !is.na(hi_v)
          if (any(both) && any(lo_v[both] > hi_v[both])) ord_ok <- FALSE
        }
      }
    }
  }
  log_guard("G3", dmax <= tol14 && ord_ok,
    sprintf("all saved per-draw matrices reproduce reported CIs (max|diff| = %.3e), draw b=1 recomputed from raw blocks for every bootstrap (F1/F2/F3/F6) (max|diff| = %.3e, tol 1e-14); delta points = diff of points and delta CIs = quantiles of saved paired differences; low <= high on all %d CI pairs",
            dmax, dmax,
            sum(vapply(c(f1_tab, f2_tab, f3_tab, f4_tab, f6_tab),
                       function(t) sum(grepl("_ci95_low$", names(t))), integer(1)), integer(1))))
}

# ---- G4: PI mechanics ----
{
  dmax <- 0
  mchk <- "MuSPAD"; vchk <- "675percentData"
  P <- pires[[paste0(mchk, ";", vchk)]]; E <- piE[[paste0(mchk, ";", vchk)]]
  o <- origs[[mchk]][[vchk]]
  lo_r <- vapply(1:7, function(t) quantile(E[, t], 0.025, type = 7) + o$pbar_te[t], numeric(1))
  hi_r <- vapply(1:7, function(t) quantile(E[, t], 0.975, type = 7) + o$pbar_te[t], numeric(1))
  dmax <- max(dmax, max(abs(lo_r - P$lo)), max(abs(hi_r - P$hi)))
  cov_r <- (o$real_test >= P$lo) & (o$real_test <= P$hi)
  stopifnot(all(as.integer(cov_r) == P$covered))
  # binom.test determinism (pooled, both models, both variants)
  bt_ok <- TRUE
  for (m in models) {
    for (vr in c("residual_bootstrap", "normal")) {
      cov7 <- unlist(lapply(variants, function(v)
        if (vr == "residual_bootstrap") pires[[paste0(m, ";", v)]]$covered
        else pires[[paste0(m, ";", v)]]$covered_n))
      bt <- binom.test(sum(cov7), 35)
      r1 <- f4_tab[f4_tab$model == m & f4_tab$variant == vr & f4_tab$origin == "all", , drop = FALSE]
      stopifnot(nrow(r1) == 1)
      bt_ok <- bt_ok && abs(bt$conf.int[1] - r1$coverage_binom_low) <= tol14 &&
                       abs(bt$conf.int[2] - r1$coverage_binom_high) <= tol14 &&
                       r1$covered == sum(cov7)
    }
  }
  # per-origin covered sums == pooled
  sum_ok <- all(mapply(function(m) {
    for (vr in c("residual_bootstrap", "normal")) {
      cov7 <- unlist(lapply(variants, function(v)
        if (vr == "residual_bootstrap") pires[[paste0(m, ";", v)]]$covered
        else pires[[paste0(m, ";", v)]]$covered_n))
      r1 <- f4_tab[f4_tab$model == m & f4_tab$variant == vr & f4_tab$origin == "all", "covered", drop = TRUE]
      r5 <- f4_tab[f4_tab$model == m & f4_tab$variant == vr & f4_tab$origin %in% variants, "covered", drop = TRUE]
      if (r1 != sum(r5)) return(FALSE)
    }
    TRUE
  }, models))
  # normal PI reproducible
  nrm_ok <- TRUE
  for (m in models) for (v in variants) {
    o <- origs[[m]][[v]]
    pool <- o$real_train - o$pbar_tr
    lo_n <- o$pbar_te + qnorm(0.025) * sd(pool)
    hi_n <- o$pbar_te + qnorm(0.975) * sd(pool)
    P <- pires[[paste0(m, ";", v)]]
    nrm_ok <- nrm_ok && max(abs(lo_n - P$lo_n)) <= tol14 && max(abs(hi_n - P$hi_n)) <= tol14
  }
  # coverage block-bootstrap draw 1 reproducible
  for (m in models) {
    covered5 <- unlist(lapply(variants, function(v) pires[[paste0(m, ";", v)]]$covered))
    s1 <- covD[[m]][1, ]
    cm1 <- mean(covered5[draw_idx(s1)])
    dmax <- max(dmax, abs(cm1 - covmat_l[[m]][1, ]))
    q_re <- quantile(covmat_l[[m]], c(0.025, 0.975))
    dmax <- max(dmax, max(abs(q_re - covBB[[m]])))
  }
  log_guard("G4", dmax <= tol14 && bt_ok && sum_ok && nrm_ok,
    sprintf("PI endpoints == seedavg + type-7 quantiles of saved draw matrix (max|diff| = %.3e, tol 1e-14, check origin 67.5%%); covered recomputed from saved lo/hi (exact); binom.test CP reproducible on 4 pooled rows; per-origin covered sums == pooled; normal PI reproducible (10 origins); coverage block-bootstrap draw-1 reproducible", dmax))
}

# ---- G5: calibration internals ----
{
  dmax <- 0
  for (m in models) {
    y <- real35[[m]]; f <- pbar35[[m]]
    fit <- calib_fit(y, f)
    lm0 <- lm(y ~ f)
    co0 <- coef(lm0)
    dmax <- max(dmax, abs(fit$alpha - co0[1]), abs(fit$beta - co0[2]))
    stopifnot(length(y) == 35, length(f) == 35)
    dmax <- max(dmax, abs(fit$R2cal - summary(lm0)$r.squared))
    dmax <- max(dmax, abs(fit$R2cal - (1 - fit$RSS / sum((y - mean(y))^2))))
    fj <- fjoint(fit, y, f)
    dmax <- max(dmax, abs(fj$F - fj$Fq))
    stopifnot(abs(fj$p - pf(fj$F, 2, 33, lower.tail = FALSE)) <= tol14)
    dmax <- max(dmax, abs(fj$F - calib_pt[[m]]["F_joint"]),
                abs(fj$p - calib_pt[[m]]["p_ref"]))
    dmax <- max(dmax, abs(calib_pt[[m]]["rmse_identity"] - pt35[[m]]$RMSE))
  }
  log_guard("G5", dmax <= 1e-10,
    sprintf("manual OLS == lm (intercept/slope), R2cal == lm r.squared and 1 - RSS/TSS, F_joint dual-route SS-diff vs (th-th0)'X'X(th-th0) (X'X direct), p = pf(F,2,33), rmse_identity == pooled RMSE: max|diff| = %.3e over 2 fits (tol 1e-10)", dmax))
}

# ---- G6: MASE denominators ----
{
  r1 <- test1001[test1001$level == "pooled", , drop = FALSE]
  stopifnot(nrow(r1) == 2)
  d_1001 <- max(abs(den_pool - r1$MAE_naive_train))
  # day-pooled mean of the per-origin 10-01 values (weights n_train - 1)
  wv <- sapply(variants, function(v) origs[["MuSPAD"]][[v]]$n_train - 1)
  na_tr <- sapply(variants, function(v) test1001[
    test1001$level == "origin" & test1001$model == "MuSPAD" &
    test1001$origin == v, "MAE_naive_train"])
  d_w <- abs(den_pool - sum(wv * na_tr) / sum(wv))
  loo_ok <- TRUE
  for (ek in names(excl_key)) {
    keep <- setdiff(variants, excl_key[[ek]])
    w4 <- sapply(keep, function(v) origs[["MuSPAD"]][[v]]$n_train - 1)
    n4 <- sapply(keep, function(v) test1001[
      test1001$level == "origin" & test1001$model == "MuSPAD" &
      test1001$origin == v, "MAE_naive_train"])
    if (abs(den_loo[[ek]] - sum(w4 * n4) / sum(w4)) > tol12) loo_ok <- FALSE
  }
  log_guard("G6", d_1001 <= tol12 && d_w <= tol12 && loo_ok,
    sprintf("pooled MASE denom %.10g == 10-01 pooled MAE_naive_train (|diff| = %.3e) and == day-pooled mean of per-origin values (|diff| = %.3e, weights 257/267/298/318/393); denom identical MuSPAD vs onlyIncidence (exact); LOO denoms = 4 remaining origins, identical across models (5/5)",
            den_pool, d_1001, d_w))
}

nfail <- sum(!vapply(guard_log, function(g) isTRUE(g$ok), logical(1)))
cat(sprintf("\n==== GUARD SUMMARY: %d/%d PASS ====\n", length(guard_log) - nfail, length(guard_log)))
if (nfail > 0) stop(sprintf("%d guard(s) FAILED", nfail))
cat("ALL GUARDS PASS\n")

# ============================================================================
# digest for the chat report
# ============================================================================
cat("\n================ DIGEST ================\n")
for (m in models) {
  pm <- pt35[[m]]; ci <- ptest_ci[[m]]
  cat(sprintf("%s | seedavg 35d: RMSE %.4f [%.4f, %.4f] | MAE %.4f [%.4f, %.4f] | sMAPE %.4f [%.4f, %.4f] | MASE %.4f | R2 %.4f | bias %.4f | MAPE %.4f\n",
      m, pm$RMSE, ci["RMSE", 1], ci["RMSE", 2],
      pm$MAE, ci["MAE", 1], ci["MAE", 2],
      pm$sMAPE, ci["sMAPE", 1], ci["sMAPE", 2], pm$MASE, pm$R2, pm$bias, pm$MAPE))
  cat(sprintf("   per-seed mean(SD): RMSE %.4f (%.4f) MAE %.4f (%.4f) sMAPE %.4f (%.4f)\n",
      mean(perseed[[m]][, "RMSE"]), sd(perseed[[m]][, "RMSE"]),
      mean(perseed[[m]][, "MAE"]), sd(perseed[[m]][, "MAE"]),
      mean(perseed[[m]][, "sMAPE"]), sd(perseed[[m]][, "sMAPE"])))
  cat(sprintf("   calibration: int %.4f [%.4f, %.4f] slope %.4f [%.4f, %.4f] R2cal %.4f F(2,33)=%.4f p=%.4g\n",
      calib_pt[[m]]["intercept"], calib_ci[[m]]["intercept", 1], calib_ci[[m]]["intercept", 2],
      calib_pt[[m]]["slope"], calib_ci[[m]]["slope", 1], calib_ci[[m]]["slope", 2],
      calib_pt[[m]]["R2cal"], calib_pt[[m]]["F_joint"], calib_pt[[m]]["p_ref"]))
  cat(sprintf("   train 1538d: RMSE %.4f [%.4f, %.4f] MAE %.4f R2 %.4f sMAPE %.4f\n",
      ptr35[[m]]$RMSE, train_ci[[m]]["RMSE", 1], train_ci[[m]]["RMSE", 2],
      ptr35[[m]]$MAE, ptr35[[m]]$R2, ptr35[[m]]$sMAPE))
}
for (ek in names(excl_key)) {
  r <- loo_res[[ek]]
  cat(sprintf("LOO %-15s: dRMSE %+.4f [%.4f, %.4f] | dMAE %+.4f [%.4f, %.4f] | dsMAPE %+.4f [%.4f, %.4f] | dMASE %+.4f | dR2 %+.4f | dbias %+.4f [%.4f, %.4f]\n",
      r$v_ex,
      r$pM["RMSE"] - r$pB["RMSE"], r$ciD["RMSE", 1], r$ciD["RMSE", 2],
      r$pM["MAE"] - r$pB["MAE"], r$ciD["MAE", 1], r$ciD["MAE", 2],
      r$pM["sMAPE"] - r$pB["sMAPE"], r$ciD["sMAPE", 1], r$ciD["sMAPE", 2],
      r$pM["MASE"] - r$pB["MASE"],
      r$pM["R2"] - r$pB["R2"],
      r$pM["bias"] - r$pB["bias"], r$ciD["bias", 1], r$ciD["bias", 2]))
}
for (m in models) {
  r1 <- f4_tab[f4_tab$model == m & f4_tab$variant == "residual_bootstrap" & f4_tab$origin == "all", , drop = FALSE]
  r2 <- f4_tab[f4_tab$model == m & f4_tab$variant == "normal" & f4_tab$origin == "all", , drop = FALSE]
  cat(sprintf("PI %-13s: resboot coverage %.3f [binom %.3f, %.3f | block %.3f, %.3f] width mean %.2f | normal coverage %.3f [%.3f, %.3f]\n",
      m, r1$coverage, r1$coverage_binom_low, r1$coverage_binom_high,
      r1$coverage_blockboot_low, r1$coverage_blockboot_high, r1$mean_width,
      r2$coverage, r2$coverage_binom_low, r2$coverage_binom_high))
}
cat("MASE denom pooled: ", den_pool, "\n")
for (ek in names(excl_key)) cat("MASE denom LOO", ek, ": ", den_loo[[ek]], "\n")
cat("\nDONE\n")
