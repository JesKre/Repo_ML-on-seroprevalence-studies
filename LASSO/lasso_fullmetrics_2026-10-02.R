# Tables 1 and S3: metrics, test-observation bootstrap CIs.
# Adapted from the supplied fullmetrics script; numerical calculations retained.
# Bootstrap checks against saved intermediate files are omitted: all intervals
# are calculated here from the original prediction pairs. No input CSVs required.
# Outputs: lasso_fullmetrics_2026-10-02.csv.
# See README.md for inputs, run order, methods and limitations.

# Repository-relative paths; optional overrides for external inputs/outputs.
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(script_file) != 1) stop("Run this script with Rscript.")
LASSO_DIR <- normalizePath(dirname(script_file), mustWork = TRUE)
REPO_ROOT <- dirname(LASSO_DIR)
PV_DIR <- Sys.getenv("LASSO_PRED_DIR", unset = file.path(REPO_ROOT, "Compare_models", "Graphics", "Predicted_values"))
OUT_DIR <- Sys.getenv("LASSO_OUT_DIR", unset = file.path(LASSO_DIR, "repro_results"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

options(warn = 1)


B_CI  <- 10000; BLOCK <- 7L; SEED_CI <- 42
DATE  <- "2026-10-02"
MODELS <- c("MuSPAD", "onlyTime")

models <- list(
  list(name = "MuSPAD",   file = "LASSO_7day_MuSPAD.RData",   obj = "graph_LASSO_7day_M"),
  list(name = "onlyTime", file = "LASSO_7day_baseline.RData", obj = "graph_LASSO_7day_BL"))

# ---------------------------------------------------------------------------
# Load training + test pairs from the saved RData
# ---------------------------------------------------------------------------
load_pairs <- function(m) {
  env <- new.env()
  input <- file.path(PV_DIR, m$file)
  if (!file.exists(input)) stop("Required saved predictions not found: ", input,
                               ". Set LASSO_PRED_DIR to their directory; see README.md.")
  load(input, envir = env)
  d <- env[[m$obj]]
  out <- list()
  for (s in c("Training", "Test")) {
    x <- d[d[["set"]] == s, ]
    r <- x[x[["real_vs_pred"]] == "real", c("time", "Incidence")]
    p <- x[x[["real_vs_pred"]] == "predicted", c("time", "Incidence")]
    names(r)[2] <- "real"; names(p)[2] <- "predicted"
    pm <- merge(r, p, by = "time")
    pm <- pm[order(pm$time), , drop = FALSE]
    pm$real      <- as.numeric(pm$real)
    pm$predicted <- as.numeric(pm$predicted)
    out[[s]] <- pm
  }
  out
}

data <- lapply(models, load_pairs)
names(data) <- vapply(models, `[[`, character(1), "name")

# G4: structure guards -------------------------------------------------------
for (nm in MODELS) {
  tr <- data[[nm]]$Training; te <- data[[nm]]$Test
  stopifnot(nrow(tr) == 251, nrow(te) == 27)
  stopifnot(!anyNA(c(tr$real, tr$predicted, te$real, te$predicted)))
  stopifnot(all(tr$real > 0), all(tr$predicted > 0), all(te$real > 0), all(te$predicted > 0))
  stopifnot(length(unique(tr$time)) == 251, length(unique(te$time)) == 27)
}
stopifnot(identical(data[["MuSPAD"]]$Training$time, data[["onlyTime"]]$Training$time))
stopifnot(identical(data[["MuSPAD"]]$Test$time,     data[["onlyTime"]]$Test$time))
cat("G4 PASS: 251 train + 27 test pairs per model; no NA/zero; identical dates across models\n")
flush(stdout())

# ---------------------------------------------------------------------------
# MASE denominators (identical for both models: same training series)
# ---------------------------------------------------------------------------
tr_real <- data[["MuSPAD"]]$Training$real
tr_date <- as.Date(data[["MuSPAD"]]$Training$time)
d_gap   <- as.numeric(diff(tr_date))
cal_idx <- which(d_gap == 1)                       # pair (t, t-1) with calendar-adjacent dates
denom_consec  <- mean(abs(diff(tr_real)))          # 250 consecutive pairs (PRIMARY)
denom_caldays <- mean(abs(tr_real[cal_idx + 1] - tr_real[cal_idx]))  # 204 calendar-adjacent pairs
stopifnot(length(denom_consec) == 1, length(denom_caldays) == 1,
          denom_consec > 0, denom_caldays > 0)
cat(sprintf("MASE denominators: consecutive pairs (n=%d) = %.6f | calendar-adjacent (n=%d) = %.6f\n",
            length(diff(tr_real)), denom_consec, length(cal_idx), denom_caldays))
flush(stdout())

# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
metrics6 <- function(real, pred, mase_denom) {
  e     <- real - pred
  rmse  <- sqrt(mean(e^2))
  smape <- mean(2 * abs(e) / (abs(real) + abs(pred)))
  r2    <- 1 - sum(e^2) / sum((real - mean(real))^2)
  mae   <- mean(abs(e))
  mape  <- mean(abs(e) / abs(real))
  mase  <- mae / mase_denom
  c(RMSE = rmse, sMAPE = smape, R2 = r2, MAE = mae, MAPE = mape, MASE = mase)
}

# Paper values (repro_results/repro_summary_vs_paper.csv, 2 decimals)
paper <- list(
  MuSPAD = list(RMSE  = c(training = 33.09, test = 9.28),
                sMAPE = c(training =  0.39, test = 1.07),
                MAPE  = c(training =  0.45, test = 0.68)),
  onlyTime = list(RMSE  = c(training = 70.73, test = 61.70),
                  sMAPE = c(training =  0.91, test = 1.46),
                  MAPE  = c(training =  2.42, test = 6.56)))

pt <- list()
for (nm in MODELS) {
  pt[[nm]]$Training <- metrics6(data[[nm]]$Training$real, data[[nm]]$Training$predicted, denom_consec)
  pt[[nm]]$Test     <- metrics6(data[[nm]]$Test$real,     data[[nm]]$Test$predicted,     denom_consec)
}

# G1: paper cross-check (2-decimal tables) -----------------------------------
for (nm in MODELS) {
  for (s in c("training", "test")) {
    for (met in c("RMSE", "sMAPE", "MAPE")) {
      pv <- paper[[nm]][[met]][s]
      if (abs(pt[[nm]][[if (s == "training") "Training" else "Test"]][met] - pv) > 0.005 + 1e-9)
        warning("Current predictions differ from the published ", nm, " ", s, " ", met,
                "; outputs describe the supplied predictions.")
      cat(sprintf("G1  %s %-8s %-6s RData = %.4f  paper = %.2f  (|diff| = %.4f)\n",
                  nm, met, s, pt[[nm]][[if (s == "training") "Training" else "Test"]][met], pv, abs(pt[[nm]][[if (s == "training") "Training" else "Test"]][met] - pv)))
    }
  }
}
cat("Reference-value comparisons completed\n")
flush(stdout())

# ---------------------------------------------------------------------------
# Bootstrap CIs on the test set -- exact replication of the locked protocol,
# extended to all six metrics
# ---------------------------------------------------------------------------
moving_block_indices <- function(n, b, B, seed) {
  K         <- ceiling(n / b)
  max_start <- n - b + 1
  set.seed(seed)
  starts <- matrix(sample.int(max_start, size = B * K, replace = TRUE), nrow = B)
  offs <- matrix(0:(b - 1), ncol = b, nrow = B, byrow = TRUE)
  idx  <- matrix(0L, nrow = B, ncol = K * b)
  for (k in seq_len(K))
    idx[, (k - 1) * b + seq_len(b)] <- starts[, k] + offs
  idx[, seq_len(n), drop = FALSE]
}
iid_indices <- function(n, B, seed) {
  set.seed(seed)
  matrix(sample.int(n, size = B * n, replace = TRUE), nrow = B)
}
boot_all <- function(idx, real, pred, denom) {
  Rm <- matrix(real[idx], nrow = NROW(idx))
  Pm <- matrix(pred[idx], nrow = NROW(idx))
  E  <- Rm - Pm
  list(
    RMSE  = sqrt(rowMeans(E^2)),
    sMAPE = rowMeans(2 * abs(E) / (abs(Rm) + abs(Pm))),
    R2    = 1 - rowSums(E^2) / rowSums((Rm - rowMeans(Rm))^2),
    MAE   = rowMeans(abs(E)),
    MAPE  = rowMeans(abs(E) / abs(Rm)),
    MASE  = rowMeans(abs(E)) / denom)
}
ci95 <- function(x) as.numeric(quantile(x, c(0.025, 0.975)))

bootres <- list()
for (nm in MODELS) {
  real <- data[[nm]]$Test$real
  pred <- data[[nm]]$Test$predicted
  n    <- length(real)
  stopifnot(n == 27)

  mb  <- boot_all(moving_block_indices(n, BLOCK, B_CI, SEED_CI), real, pred, denom_consec)
  iid <- boot_all(iid_indices(n, B_CI, SEED_CI), real, pred, denom_consec)

  bootres[[nm]] <- list(mb = mb, iid = iid)
  flush(stdout())
}
cat("Bootstrap intervals computed directly from saved prediction pairs\n")
flush(stdout())

# ---------------------------------------------------------------------------
# Full metrics table (long format)
# ---------------------------------------------------------------------------
csv_field15 <- function(v) {
  s <- if (is.numeric(v)) sprintf("%.15g", v) else as.character(v)
  if (grepl('[";\n]', s)) s <- paste0('"', gsub('"', '""', s), '"')
  s
}
write_csv15 <- function(df, path) {
  n <- nrow(df)
  body <- vapply(seq_len(n), function(i) {
    row <- df[i, , drop = FALSE]
    paste(vapply(row, csv_field15, character(1)), collapse = ";")
  }, character(1))
  writeLines(c(paste(names(df), collapse = ";"), body), path)
}

mk_rows <- function(nm, s, pe, mb = NULL, iid = NULL) {
  mets <- c("RMSE", "sMAPE", "R2", "MAE", "MAPE", "MASE")
  do.call(rbind, lapply(mets, function(met) {
    pv  <- if (met %in% c("RMSE", "sMAPE", "MAPE")) paper[[nm]][[met]][[s]] else NA_real_
    note <- switch(met,
      RMSE  = "comparison to published reference; see abs_diff_vs_paper",
      sMAPE = "0-2 scale: mean(2|a-f|/(|a|+|f|))",
      R2    = "around the set's own mean; test window is a quiet post-outbreak period (actuals 4.9-23.3), SST small",
      MAE   = "",
      MAPE  = "finite: all predictions > 0 (exp of log-scale fit) and all actuals > 0; baseline inflated by small test actuals",
      MASE  = "MASE = MAE / in-sample 1-step persistence MAE (250 consecutive training pairs); alternative calendar-adjacent denom: see MASE_denom_* rows",
      "")
    data.frame(model = nm, set = s, metric = met, value = pe[met],
               ci_mb_lo = if (is.null(mb)) NA_real_ else ci95(mb[[met]])[1],
               ci_mb_hi = if (is.null(mb)) NA_real_ else ci95(mb[[met]])[2],
               ci_iid_lo = if (is.null(iid)) NA_real_ else ci95(iid[[met]])[1],
               ci_iid_hi = if (is.null(iid)) NA_real_ else ci95(iid[[met]])[2],
               paper_value = pv,
               abs_diff_vs_paper = if (is.na(pv)) NA_real_ else abs(pe[met] - pv),
               note = note,
               stringsAsFactors = FALSE)
  }))
}

fm_rows <- list()
for (nm in MODELS) {
  fm_rows[[length(fm_rows) + 1L]] <- mk_rows(nm, "training", pt[[nm]]$Training)
  fm_rows[[length(fm_rows) + 1L]] <- mk_rows(nm, "test",     pt[[nm]]$Test,
                                             mb = bootres[[nm]]$mb, iid = bootres[[nm]]$iid)
}
fm_rows[[length(fm_rows) + 1L]] <- data.frame(
  model = "BOTH", set = "training_insample", metric = "MASE_denom_consec",
  value = denom_consec, ci_mb_lo = NA_real_, ci_mb_hi = NA_real_,
  ci_iid_lo = NA_real_, ci_iid_hi = NA_real_, paper_value = NA_real_,
  abs_diff_vs_paper = NA_real_,
  note = "PRIMARY MASE denominator: in-sample 1-step persistence MAE, 250 consecutive pairs in series order; identical for both models (same training series)",
  stringsAsFactors = FALSE)
fm_rows[[length(fm_rows) + 1L]] <- data.frame(
  model = "BOTH", set = "training_insample", metric = "MASE_denom_caldays",
  value = denom_caldays, ci_mb_lo = NA_real_, ci_mb_hi = NA_real_,
  ci_iid_lo = NA_real_, ci_iid_hi = NA_real_, paper_value = NA_real_,
  abs_diff_vs_paper = NA_real_,
  note = "ALTERNATIVE MASE denominator: persistence MAE over the 204 calendar-adjacent (1-day) pairs only; identical for both models",
  stringsAsFactors = FALSE)
fm <- do.call(rbind, fm_rows)
fm_path <- file.path(OUT_DIR, paste0("lasso_fullmetrics_", DATE, ".csv"))
write_csv15(fm, fm_path)
cat("Wrote", fm_path, "(", nrow(fm), "rows )\n")
flush(stdout())

cat("\nLASSO full metrics summary\n")
print(fm, row.names = FALSE)
cat("\nDone. Files in:", OUT_DIR, "\n")
