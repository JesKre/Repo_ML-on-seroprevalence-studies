# Table S5: OLS point calibration of the saved test predictions.
# Numerical calibration calculations retained from the supplied script.
# The redundant check against a generated PI-coverage CSV has been removed.
# This script can run independently of the metrics script. See README.md.

# Repository-relative paths; optional overrides for external inputs/outputs.
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(script_file) != 1) stop("Run this script with Rscript.")
LASSO_DIR <- normalizePath(dirname(script_file), mustWork = TRUE)
REPO_ROOT <- dirname(LASSO_DIR)
PV_DIR <- Sys.getenv("LASSO_PRED_DIR", unset = file.path(REPO_ROOT, "Compare_models", "Graphics", "Predicted_values"))
OUT_DIR <- Sys.getenv("LASSO_OUT_DIR", unset = file.path(LASSO_DIR, "repro_results"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

options(warn = 1)

DATE    <- "2026-10-02"
MODELS  <- c("MuSPAD", "onlyTime")

models <- list(
  list(name = "MuSPAD",   file = "LASSO_7day_MuSPAD.RData",   obj = "graph_LASSO_7day_M"),
  list(name = "onlyTime", file = "LASSO_7day_baseline.RData", obj = "graph_LASSO_7day_BL"))

# ---------------------------------------------------------------------------
# Load the 27 test pairs from the saved RData (same loader as the main run)
# ---------------------------------------------------------------------------
load_test <- function(m) {
  env <- new.env()
  input <- file.path(PV_DIR, m$file)
  if (!file.exists(input)) stop("Required saved predictions not found: ", input,
                               ". Set LASSO_PRED_DIR to their directory; see README.md.")
  load(input, envir = env)
  d <- env[[m$obj]]
  x <- d[d[["set"]] == "Test", ]
  r <- x[x[["real_vs_pred"]] == "real", c("time", "Incidence")]
  p <- x[x[["real_vs_pred"]] == "predicted", c("time", "Incidence")]
  names(r)[2] <- "real"; names(p)[2] <- "predicted"
  pm <- merge(r, p, by = "time")
  pm <- pm[order(pm$time), , drop = FALSE]
  pm$real      <- as.numeric(pm$real)
  pm$predicted <- as.numeric(pm$predicted)
  pm
}

data <- lapply(models, load_test)
names(data) <- vapply(models, `[[`, character(1), "name")

# G0: structure --------------------------------------------------------------
for (nm in MODELS) {
  te <- data[[nm]]
  stopifnot(nrow(te) == 27, !anyNA(c(te$real, te$predicted)))
  stopifnot(all(te$real > 0), all(te$predicted > 0), length(unique(te$time)) == 27)
}
stopifnot(identical(data[["MuSPAD"]]$time, data[["onlyTime"]]$time))
cat("G0 PASS: 27 test pairs per model; no NA/zero; identical dates\n")
flush(stdout())

# G1: RMSE cross-checks -------------------------------------------------------
rmse_ref   <- c(MuSPAD = 9.27839400034959, onlyTime = 61.7042354950353)  # fullmetrics CSV
paper_rmse <- c(MuSPAD = 9.28,              onlyTime = 61.70)            # paper tables
for (nm in MODELS) {
  te   <- data[[nm]]
  rmse <- sqrt(mean((te$real - te$predicted)^2))
  if (abs(rmse - rmse_ref[nm]) > 1e-9)
    warning("Current ", nm, " predictions differ from the published run; calibration uses current pairs.")
  cat(sprintf("G1  %s: pair RMSE = %.12f (fullmetrics = %.12f, paper = %.2f)\n",
              nm, rmse, rmse_ref[nm], paper_rmse[nm]))
}
flush(stdout())

# ---------------------------------------------------------------------------
# OLS calibration: y ~ 1 + x
# ---------------------------------------------------------------------------
calib <- function(te) {
  y <- te$real; x <- te$predicted; n <- length(y)
  fit <- lm(y ~ x)
  cf  <- coef(fit)
  se  <- summary(fit)$coefficients[, 2]
  ci  <- confint(fit)                       # 95%, t-based, df = n - 2
  r2  <- summary(fit)$r.squared
  s2  <- summary(fit)$sigma^2               # = SS_ols / (n - 2)
  ss_id  <- sum((y - x)^2)                  # RSS of the identity line
  ss_ols <- s2 * (n - 2)
  # joint test H0: (alpha, beta) = (0, 1), two equivalent computations
  F_ss <- ((ss_id - ss_ols) / 2) / (ss_ols / (n - 2))
  Xmat <- cbind(1, x)
  th   <- c(cf[1], cf[2]); th0 <- c(0, 1)
  Q    <- as.numeric(t(th - th0) %*% (t(Xmat) %*% Xmat) %*% (th - th0)) / s2
  F_q  <- Q / 2
  # manual coefficients (independent recomputation)
  xb <- mean(x); yb <- mean(y)
  beta_m <- sum((x - xb) * (y - yb)) / sum((x - xb)^2)
  alpha_m <- yb - beta_m * xb
  stopifnot(abs(cf[1] - alpha_m) <= 1e-10, abs(cf[2] - beta_m) <= 1e-10,
            ss_ols <= ss_id + 1e-9,
            abs(F_ss - F_q) <= 1e-10 * max(F_ss, F_q) + 1e-12)
  list(intercept = cf[1], intercept_se = se[1],
       intercept_ci_lo = ci[1, 1], intercept_ci_hi = ci[1, 2],
       slope = cf[2], slope_se = se[2],
       slope_ci_lo = ci[2, 1], slope_ci_hi = ci[2, 2],
       r2 = r2, ss_id = ss_id, ss_ols = ss_ols,
       F = F_ss, p = pf(F_ss, 2, n - 2, lower.tail = FALSE),
       fit = fit)
}

res <- list()
for (nm in MODELS) res[[nm]] <- calib(data[[nm]])
cat("G3 PASS: OLS internals consistent (manual = lm; SS_ols <= SS_id; F_ss = F_q)\n")
flush(stdout())

# ---------------------------------------------------------------------------
# CSV
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

mk_row <- function(nm) {
  te <- data[[nm]]; c0 <- res[[nm]]
  data.frame(
    model = nm, n = nrow(te),
    y_mean = mean(te$real), pred_mean = mean(te$predicted), pred_sd = sd(te$predicted),
    intercept = c0$intercept, intercept_se = c0$intercept_se,
    intercept_ci95_lo = c0$intercept_ci_lo, intercept_ci95_hi = c0$intercept_ci_hi,
    slope = c0$slope, slope_se = c0$slope_se,
    slope_ci95_lo = c0$slope_ci_lo, slope_ci95_hi = c0$slope_ci_hi,
    r2_calibration = c0$r2,
    rmse_identity = sqrt(c0$ss_id / nrow(te)),
    f_joint = c0$F, f_pvalue = c0$p,
    note = "OLS y ~ 1 + pred on the 27 held-out test days (2021-07-16..2021-08-23), natural incidence scale; ideal values: intercept = 0, slope = 1; CIs t-based, df = 25; joint F tests (intercept, slope) = (0, 1) against F(df1 = 2, df2 = 25); rmse_identity = RMSE along the y = x line")
}
tab <- do.call(rbind, lapply(MODELS, mk_row))
csv_path <- file.path(OUT_DIR, paste0("lasso_pointcalibration_", DATE, ".csv"))
write_csv15(tab, csv_path)
cat("Wrote", csv_path, "\n")
flush(stdout())

# ---------------------------------------------------------------------------
# Figure: observed vs predicted, identity + OLS line, one panel per model
# ---------------------------------------------------------------------------
fig_path <- file.path(OUT_DIR, paste0("lasso_pointcalibration_", DATE, ".png"))
png(fig_path, width = 1600, height = 640, res = 100)
par(mfrow = c(1, 2), mar = c(4.4, 4.6, 2.4, 1), oma = c(0, 0, 2.2, 0))
for (nm in MODELS) {
  te <- data[[nm]]; c0 <- res[[nm]]
  lim <- ceiling(max(c(te$real, te$predicted)) / 5) * 5
  plot(te$predicted, te$real,
       xlim = c(0, lim), ylim = c(0, lim),
       xlab = "Predicted incidence", ylab = "Observed incidence",
       pch = 19, cex = 0.9, col = "#2c7fb8",
       main = paste0(nm, "  (n = 27)"))
  abline(0, 1, col = "grey45", lty = 2, lwd = 1.5)
  abline(c0$fit, col = "firebrick", lwd = 2)
  txt <- paste0(
    "intercept = ", sprintf("%.2f", c0$intercept),
    "  [", sprintf("%.2f", c0$intercept_ci_lo), ", ", sprintf("%.2f", c0$intercept_ci_hi), "]\n",
    "slope     = ", sprintf("%.2f", c0$slope),
    "  [", sprintf("%.2f", c0$slope_ci_lo), ", ", sprintf("%.2f", c0$slope_ci_hi), "]\n",
    "R2 (cal)  = ", sprintf("%.3f", c0$r2))
  if (nm == "MuSPAD") {
    text(0.8 * lim, 0.92 * lim, txt, pos = 2, adj = c(0.5, 0), cex = 0.72)
    legend(0.55 * lim, 0.10 * lim,
           legend = c("identity (ideal)", "OLS calibration"),
           lty = c(2, 1), col = c("grey45", "firebrick"), lwd = c(1.5, 2),
           bty = "n", cex = 0.66, pt.lwd = 1.5)
  } else {
    text(0.06 * lim, 0.92 * lim, txt, pos = 4, adj = c(0, 0), cex = 0.72)
    legend(0.06 * lim, 0.78 * lim,
           legend = c("identity (ideal)", "OLS calibration"),
           lty = c(2, 1), col = c("grey45", "firebrick"), lwd = c(1.5, 2),
           bty = "n", cex = 0.66, pt.lwd = 1.5)
  }
}
mtext("Observed vs predicted incidence - 27 held-out test days (2021-07-16..2021-08-23), LASSO",
      outer = TRUE, cex = 0.95)
dev.off()
cat("Wrote", fig_path, "\n")
flush(stdout())

# ---------------------------------------------------------------------------
# Summary for the log
# ---------------------------------------------------------------------------
for (nm in MODELS) {
  c0 <- res[[nm]]
  cat(sprintf(
    "%s: intercept = %.4f (SE %.4f; 95%% CI %.4f..%.4f) | slope = %.4f (SE %.4f; 95%% CI %.4f..%.4f) | R2_cal = %.4f | joint F(2,25) = %.4f, p = %.4g | rmse_identity = %.4f\n",
    nm, c0$intercept, c0$intercept_se, c0$intercept_ci_lo, c0$intercept_ci_hi,
    c0$slope, c0$slope_se, c0$slope_ci_lo, c0$slope_ci_hi,
    c0$r2, c0$F, c0$p, sqrt(c0$ss_id / 27)))
}
cat("DONE\n")
