# ==============================================================================
# VAR (bigtime) 7-day predictions on the percent-split LSTM datasets
# ------------------------------------------------------------------------------
# Purpose:  New 7-day VAR predictions for the 5 data variants used by the
#           multi-seed LSTM runs (full, 65 %, 67,5 %, 75 %/threefourth, 80 %),
#           keeping the original VAR procedure (sparseVAR grid p=7/h=7 +
#           final fits p in {7, 14, 21}, h=7 - as in the paper's Tables 1-6)
#           and adding the LSTM multi-seed improvement
#           (seeds 1-10, set.seed before every final fit).
#
# Data:     Pre-scaled CSVs from Datasets\LSTM_data_subsets - the SAME files
#           the LSTM seed runs use, so VAR and LSTM are directly comparable:
#             - full:        MuSPAD_*_LSTM.csv                    (23.07.2025)
#                            scaling_parameters_2025-07-23.RData  (Datasets\)
#             - percent:     MuSPAD_*_LSTM_{suffix}.csv           (16.09.2026)
#                            scaling_parameters_{suffix}_2026-09-16.RData
#           (CSVs: col 1 = row index [dropped], then 122 vars:
#            Inzidenz + 121 features, scaled with the variant's train params;
#            _traindat_v_ is scaled with its own subset parameters, as in the
#            original VAR template.)
#
# Models:   (a) MuSPAD (all 122 columns)
#               grid:    VARpen c("HLag","L1") x selection c("cv","bic","aic")
#                        x cvcut c(0.8,0.9), p=7, h=7, fit on train_v,
#                        validated on the 7-day val window (original grid)
#               final:   sparseVAR(p, selection="cv", VARpen="L1",
#                        cvcut=0.8, h=7) on full train, seeds 1-10,
#                        p in {7, 14, 21} (2nd command-line argument, default 21)
#           (b) onlyIncidence (Inzidenz only)
#               raw imputed Inzidenz recovered from the MuSPAD CSVs (unscale
#               col 1 with the MuSPAD train params), then scaled with its own
#               train params (original procedure of
#               bigtime_noMuSPAD_only_incidence_1_model_recursive.R)
#               final:   sparseVAR(p, selection="cv", h=7), seeds 1-10,
#                        p in {7, 14, 21}
#                        (no VARpen/cvcut args, as in the original script)
#
# Notes:    - sequential loop instead of furrr (deterministic, no worker env)
#           - set.seed(1) before each variant's grid (reproducibility)
#           - set.seed(seed) before each final sparseVAR fit
#           - only raw RData is loaded for the time_day column (dates)
#
# Outputs:  VAR\Validation\   grid CSVs (12 rows each, write.csv2)
#           VAR\MSE\          per-seed MSE tables (10 rows each) + summary
#           VAR\Models\       one RData per final fit (VARfit + scaling +
#                             matrices + seed/variant metadata)
#           Compare_models\Graphics\Predicted_values\Diff_seeds\
#                             long-format RData (time, Incidence, real_vs_pred,
#                             set, seed) - same convention as the LSTM
#                             Diff_seeds RData
#           Compare_models\Graphics\Single_graphics\ + VAR\
#                             PNGs (28x24 cm, res 800): per-seed lines +
#                             mean line + real; adaptive ylim (was 0-375)
# ==============================================================================

library(bigtime)
library(glmnet)
library(Metrics)
library(tidyverse)

# ------------------------------------------------------------------------------
# Setup

base     <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML"
sub      <- file.path(base, "Datasets", "LSTM_data_subsets")
var_dir  <- file.path(base, "muspad-ml-loki-sp2.1", "VAR")
cmp_dir  <- file.path(base, "muspad-ml-loki-sp2.1", "Compare_models", "Graphics")
pv_dir   <- file.path(cmp_dir, "Predicted_values", "Diff_seeds")
sg_dir   <- file.path(cmp_dir, "Single_graphics")

seeds    <- 1:10
p        <- 21
h        <- 7
run_date <- format(Sys.Date(), "%Y-%m-%d")

dir.create(pv_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(var_dir, "Validation"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(var_dir, "MSE"),        showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(var_dir, "Models"),     showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")),
                             ..., sep = " ", "\n")

# retry wrapper for output writes: a transient SMB hiccup during a long run
# must not kill the process (observed 2026-09-22: the share dropped mid-run
# and every run died at the first output write). Inputs are guaranteed
# readable by the gate below; writes are retried 10 x 10 s before giving up.
try_write <- function(expr, what) {
  for (i in 1:10) {
    err <- tryCatch({ expr; NULL },
                    error = function(e) {
                      while (length(dev.list()) > 0) suppressWarnings(dev.off())
                      conditionMessage(e)
                    })
    if (is.null(err)) return(invisible())
    log_msg(sprintf("  WRITE RETRY %d/10 (%s): %s", i, what, err))
    Sys.sleep(10)
  }
  stop("write failed after 10 retries: ", what)
}

# raw series - only needed for the time_day column
raw_file <- file.path(
  base, "Datasets",
  "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_2025-03-05.RData")
# NOTE: load(raw_file) is deferred to the input gate further below

variants <- list(
  list(tag = "",                csv_sfx = "",
       scale_file = file.path(base, "Datasets", "scaling_parameters_2025-07-23.RData")),
  list(tag = "65percentData",   csv_sfx = "_65percentData",
       scale_file = file.path(sub, "scaling_parameters_65percentData_2026-09-16.RData")),
  list(tag = "675percentData",  csv_sfx = "_675percentData",
       scale_file = file.path(sub, "scaling_parameters_675percentData_2026-09-16.RData")),
  list(tag = "threefourthData", csv_sfx = "_threefourthData",
       scale_file = file.path(sub, "scaling_parameters_threefourthData_2026-09-16.RData")),
  list(tag = "80percentData",   csv_sfx = "_80percentData",
       scale_file = file.path(sub, "scaling_parameters_80percentData_2026-09-16.RData"))
)

# optional single-variant mode (for parallel runs, one Rscript process per variant):
#   Rscript var_predictions_diff_splits_diff_seeds.R <full|65percentData|675percentData|threefourthData|80percentData> [p]
# 2nd argument: final-fit lag order p (default 21; the paper uses p in 7/14/21)
trailing_args <- commandArgs(trailingOnly = TRUE)
variant_arg <- if (length(trailing_args) >= 1) trailing_args[1] else "all"
variants <- Filter(function(v) {
  if (variant_arg == "all") TRUE
  else if (variant_arg == "full") v$tag == ""
  else v$tag == variant_arg
}, variants)
if (length(variants) == 0) stop("unknown variant: ", variant_arg)
if (length(trailing_args) >= 2) p <- as.integer(trailing_args[2])

# ------------------------------------------------------------------------------
# Input gate: wait (up to 30 min) until ALL inputs are readable.
# A run started while the SMB share is still settling would otherwise load
# some inputs fine and die at the first output write (see 2026-09-22).
input_files <- c(raw_file, unlist(lapply(variants, function(v) c(
  file.path(sub, paste0("MuSPAD_traindat_LSTM",   v$csv_sfx, ".csv")),
  file.path(sub, paste0("MuSPAD_traindat_v_LSTM", v$csv_sfx, ".csv")),
  file.path(sub, paste0("MuSPAD_valdat_LSTM",     v$csv_sfx, ".csv")),
  file.path(sub, paste0("MuSPAD_testdat_LSTM",    v$csv_sfx, ".csv")),
  v$scale_file
))))
gate_t0 <- Sys.time()
repeat {
  sz      <- tryCatch(file.info(input_files)$size,
                      error = function(e) rep(NA_real_, length(input_files)))
  missing <- input_files[is.na(sz)]
  if (length(missing) == 0) break
  if (as.numeric(difftime(Sys.time(), gate_t0, units = "mins")) > 30)
    stop("inputs not readable after 30 min: ", paste(missing, collapse = "\n  "))
  log_msg("inputs not readable yet, retrying in 30 s ...")
  Sys.sleep(30)
}
log_msg("all inputs readable; starting")
load(raw_file)   # df_ts_all_inc (408 x 123)

# ------------------------------------------------------------------------------
# Helpers

read_scaled_csv <- function(fn) {
  as.matrix(read.csv(fn, check.names = FALSE)[, -1])   # drop row-index column
}

# original grid evaluator (bigtime_MuSPAD_varsubset_basic_no_cities_1_model_
# recursive_new_with_val.R), with an NA guard for failed forecasts
evaluate_model <- function(VARpen, selection, cvcut,
                           scaled_train_mat_v, scaled_val_mat, scale_inc, center_inc) {
  VARfit <- sparseVAR(Y = scaled_train_mat_v, cvcut = cvcut, p = 7,
                      selection = selection, VARpen = VARpen, h = 7)

  pred_val <- tryCatch({
    recursiveforecast(mod = VARfit, h = 7)$fcst[, 1] * scale_inc + center_inc
  }, error = function(e) rep(NA, 7))
  real_val <- scaled_val_mat[, 1] * scale_inc + center_inc
  mse_val  <- if (any(is.na(pred_val))) NA_real_ else mse(real_val, pred_val)

  if (is.null(VARfit$Phihat)) {
    mse_train <- NA_real_
  } else {
    n_lags    <- dim(VARfit$Phihat)[2] / ncol(scaled_train_mat_v)
    start_idx <- n_lags + 7
    real_train      <- scaled_train_mat_v[start_idx:nrow(scaled_train_mat_v), 1] * scale_inc + center_inc
    predicted_train <- fitted(VARfit)[, "Inzidenz"] * scale_inc + center_inc
    mse_train       <- mse(real_train, predicted_train)
  }

  tibble(VARpen = VARpen, selection = selection, cv_cut = cvcut,
         mse_val = mse_val, mse_train = mse_train)
}

# long format + multi-seed plot + RData/PNG output (one call per model)
write_long_and_plot <- function(fit_list, real_train, real_test,
                                time_train, time_test,
                                obj_name, rdata_name, png_name) {
  parts <- list()
  for (s in seeds) {
    parts[[s]] <- rbind(
      data.frame(real = real_train,  predicted = fit_list[[s]]$pred_train,
                 time = time_train,  set = "Training", seed = s),
      data.frame(real = real_test,   predicted = fit_list[[s]]$pred_test,
                 time = time_test,   set = "Test",     seed = s)
    )
  }
  df_all <- do.call(rbind, parts)

  long <- df_all %>%
    pivot_longer(cols = c(real, predicted),
                 names_to = "real_vs_pred", values_to = "Incidence") %>%
    select(time, real_vs_pred, Incidence, set, seed) %>%
    mutate(real_vs_pred = factor(real_vs_pred, levels = c("predicted", "real")),
           set = factor(set, levels = c("Test", "Training")))
  # same structure and object-name convention as the LSTM Diff_seeds RData
  e_out <- new.env()
  assign(obj_name, long, envir = e_out)
  try_write(save(list = obj_name, file = file.path(pv_dir, paste0(rdata_name, ".RData")),
                 envir = e_out),
            "Diff_seeds RData")

  mean_pred <- aggregate(predicted ~ time + set, data = df_all, FUN = mean)
  plot_df <- rbind(
    long,
    data.frame(time = mean_pred$time, Incidence = mean_pred$predicted,
               real_vs_pred = "predicted_mean", set = mean_pred$set, seed = NA)
  )
  y_max <- ceiling(max(plot_df$Incidence[!is.na(plot_df$Incidence)]) * 1.15 / 50) * 50

  plot <- ggplot(plot_df, aes(x = time, y = Incidence)) +
    geom_line(data = subset(plot_df, real_vs_pred == "predicted"),
              aes(color = "predicted (seeds 1-10)"), alpha = 0.35, linewidth = 0.4) +
    geom_line(data = subset(plot_df, real_vs_pred == "predicted_mean"),
              aes(color = "predicted (mean)"), linewidth = 1) +
    geom_line(data = subset(plot_df, real_vs_pred == "real"),
              aes(color = "real"), linewidth = 0.8) +
    geom_point(data = subset(plot_df, real_vs_pred == "real"),
               aes(color = "real", shape = set), size = 2.5) +
    scale_color_manual(name = "Autoregressive",
                       values = c("real" = "black",
                                  "predicted (mean)" = "#E69F00",
                                  "predicted (seeds 1-10)" = "grey60")) +
    scale_shape_manual(values = c(4, 16)) +
    theme_bw() +
    ylim(0, y_max) +
    theme(legend.position = "bottom")

  try_write({
    for (dir in c(sg_dir, var_dir)) {
      png(file = file.path(dir, paste0(png_name, ".png")),
          width = 28, height = 24, unit = "cm", res = 800)
      print(plot)
      dev.off()
    }
  }, "PNG output")
  invisible(long)
}

# ------------------------------------------------------------------------------
# Main loop over variants

summary_list <- list()

for (vi in seq_along(variants)) {
  v     <- variants[[vi]]
  tag   <- v$tag
  label <- if (tag == "") "full" else tag
  tag_s <- if (tag == "") "" else paste0(tag, "_")
  log_msg("========================================================")
  log_msg("=== variant:", label, "(p =", p, ") ===")

  # ---- data (same pre-scaled CSVs as the LSTM seed runs)
  scaled_train_mat   <- read_scaled_csv(file.path(sub, paste0("MuSPAD_traindat_LSTM",   v$csv_sfx, ".csv")))
  scaled_train_mat_v <- read_scaled_csv(file.path(sub, paste0("MuSPAD_traindat_v_LSTM", v$csv_sfx, ".csv")))
  scaled_val_mat     <- read_scaled_csv(file.path(sub, paste0("MuSPAD_valdat_LSTM",     v$csv_sfx, ".csv")))
  scaled_test_mat    <- read_scaled_csv(file.path(sub, paste0("MuSPAD_testdat_LSTM",    v$csv_sfx, ".csv")))

  e <- new.env()
  load(v$scale_file, envir = e)
  scale_center_train   <- e$scale_center_train
  scale_scale_train    <- e$scale_scale_train
  scale_center_train_v <- e$scale_center_train_v
  scale_scale_train_v  <- e$scale_scale_train_v
  test_indices         <- e$test_indices
  val_indices          <- e$val_indices

  stopifnot(
    ncol(scaled_train_mat) == 122,
    nrow(scaled_train_mat) == test_indices[1] - 1,
    nrow(scaled_val_mat)   == 7,
    nrow(scaled_test_mat)  == 7,
    identical(names(scale_center_train), colnames(scaled_train_mat))
  )
  t_test <- as.Date("2020-01-01") + df_ts_all_inc[test_indices, 2]
  log_msg(sprintf("train %d rows (train_v %d), val rows %d:%d, test %s .. %s",
                  nrow(scaled_train_mat), nrow(scaled_train_mat_v),
                  val_indices[1], val_indices[2],
                  as.character(min(t_test)), as.character(max(t_test))))

  # ---------------------------------------------------------------- grid (MuSPAD)
  set.seed(1)
  param_grid <- expand.grid(VARpen = c("HLag", "L1"),
                            selection = c("cv", "bic", "aic"),
                            cvcut = c(0.8, 0.9),
                            stringsAsFactors = FALSE)
  scale_inc  <- scale_scale_train_v[which(names(scale_scale_train_v) == "Inzidenz")]
  center_inc <- scale_center_train_v[which(names(scale_center_train_v) == "Inzidenz")]

  results <- do.call(rbind, lapply(seq_len(nrow(param_grid)), function(i) {
    log_msg(sprintf("  grid %2d/12: VARpen=%-4s selection=%-3s cvcut=%.1f",
                    i, param_grid$VARpen[i], param_grid$selection[i], param_grid$cvcut[i]))
    evaluate_model(param_grid$VARpen[i], param_grid$selection[i], param_grid$cvcut[i],
                   scaled_train_mat_v, scaled_val_mat, scale_inc, center_inc)
  }))
  results$variant <- label
  res_ord <- results %>% arrange(mse_val)
  try_write(write.csv2(res_ord, file.path(var_dir, "Validation",
    sprintf("Hyperparameter_validation_VAR_%sp7_h7_diffseeds_%s.csv", tag_s, run_date))),
    "grid validation CSV")
  log_msg(sprintf("  grid best: %s / %s / cvcut %.1f -> mse_val %.2f",
                  res_ord$VARpen[1], res_ord$selection[1], res_ord$cv_cut[1], res_ord$mse_val[1]))

  # ---------------------------------------------------------------- final MuSPAD, seeds 1-10
  scale_inc_t  <- scale_scale_train[which(names(scale_scale_train) == "Inzidenz")]
  center_inc_t <- scale_center_train[which(names(scale_center_train) == "Inzidenz")]

  start_idx_train <- p + h
  real_t     <- scaled_test_mat[, 1] * scale_inc_t + center_inc_t
  real_train <- scaled_train_mat[start_idx_train:nrow(scaled_train_mat), 1] * scale_inc_t + center_inc_t
  time_test  <- t_test
  time_train <- as.Date("2020-01-01") + df_ts_all_inc[start_idx_train:(test_indices[1] - 1), 2]

  muspad_rows    <- list()
  muspad_fit_list <- list()
  for (s in seeds) {
    log_msg(sprintf("  MuSPAD seed %2d/10", s))
    set.seed(s)
    VARfit <- sparseVAR(Y = scaled_train_mat, p = p, selection = "cv",
                        VARpen = "L1", cvcut = 0.8, h = h)

    pred_test <- recursiveforecast(mod = VARfit, h = h)$fcst[, 1] * scale_inc_t + center_inc_t
    mse_test  <- mse(real_t, pred_test)
    if (is.null(VARfit$Phihat)) {
      mse_train  <- NA_real_
      pred_train <- rep(NA_real_, length(real_train))
    } else {
      pred_train <- fitted(VARfit)[, "Inzidenz"] * scale_inc_t + center_inc_t
      mse_train  <- mse(real_train, pred_train)
    }

    try_write(save(VARfit, scale_center_train, scale_scale_train,
         scaled_train_mat, scaled_test_mat,
         seed = s, p = p, h = h, variant = label,
         file = file.path(var_dir, "Models",
                          sprintf("model_VAR_%sp%d_h%d_seed%d_%s.RData",
                                  tag_s, p, h, s, run_date))),
      "MuSPAD model RData")
    muspad_rows[[s]] <- data.frame(variant = label, model = "MuSPAD", p = p, h = h,
                                   seed = s, mse_train = mse_train, mse_test = mse_test)
    muspad_fit_list[[s]] <- list(pred_train = pred_train, pred_test = pred_test)
  }
  mse_muspad <- do.call(rbind, muspad_rows)
  try_write(write.csv2(mse_muspad, file.path(var_dir, "MSE",
    sprintf("mse_VAR_MuSPAD_%sp%d_h%d_diffseeds_%s.csv", tag_s, p, h, run_date)),
    row.names = FALSE),
    "MuSPAD MSE CSV")
  log_msg("  MuSPAD mse_test by seed: ", paste(round(mse_muspad$mse_test, 1), collapse = " "))

  # ---------------------------------------------------------------- final onlyIncidence, seeds 1-10
  # recover raw imputed Inzidenz from the MuSPAD-scaled CSVs, then re-scale
  # with the series' own train parameters (original onlyIncidence procedure)
  inc_train_raw    <- scaled_train_mat[, 1] * scale_inc_t + center_inc_t
  traindat_inc     <- scale(inc_train_raw)
  scaled_train_inc <- as.matrix(traindat_inc)
  colnames(scaled_train_inc) <- "Incidence"
  inc_test_raw <- scaled_test_mat[, 1] * scale_inc_t + center_inc_t
  scaled_test_inc <- as.matrix(scale(
    inc_test_raw,
    center = attr(traindat_inc, "scaled:center"),
    scale  = attr(traindat_inc, "scaled:scale")))
  colnames(scaled_test_inc) <- "Incidence"

  scale_c_inc <- attr(traindat_inc, "scaled:center")[[1]]
  scale_s_inc <- attr(traindat_inc, "scaled:scale")[[1]]
  real_inc_t     <- scaled_test_inc[, 1] * scale_s_inc + scale_c_inc
  real_inc_train <- scaled_train_inc[start_idx_train:nrow(scaled_train_inc), 1] * scale_s_inc + scale_c_inc

  onlyinc_rows     <- list()
  onlyinc_fit_list <- list()
  for (s in seeds) {
    log_msg(sprintf("  onlyIncidence seed %2d/10", s))
    set.seed(s)
    VARfit_inc <- sparseVAR(Y = scaled_train_inc, selection = "cv", p = p, h = h)

    pred_inc_test <- recursiveforecast(mod = VARfit_inc, h = h)$fcst[, 1] * scale_s_inc + scale_c_inc
    mse_inc_test  <- mse(real_inc_t, pred_inc_test)
    if (is.null(VARfit_inc$Phihat)) {
      mse_inc_train  <- NA_real_
      pred_inc_train <- rep(NA_real_, length(real_inc_train))
    } else {
      pred_inc_train <- fitted(VARfit_inc)[, "Incidence"] * scale_s_inc + scale_c_inc
      mse_inc_train  <- mse(real_inc_train, pred_inc_train)
    }

    try_write(save(VARfit_inc, scale_c_inc, scale_s_inc,
         scaled_train_inc, scaled_test_inc,
         seed = s, p = p, h = h, variant = label,
         file = file.path(var_dir, "Models",
                          sprintf("model_VAR_onlyIncidence_%sp%d_h%d_seed%d_%s.RData",
                                  tag_s, p, h, s, run_date))),
      "onlyIncidence model RData")
    onlyinc_rows[[s]] <- data.frame(variant = label, model = "onlyIncidence", p = p, h = h,
                                    seed = s, mse_train = mse_inc_train, mse_test = mse_inc_test)
    onlyinc_fit_list[[s]] <- list(pred_train = pred_inc_train, pred_test = pred_inc_test)
  }
  mse_onlyinc <- do.call(rbind, onlyinc_rows)
  try_write(write.csv2(mse_onlyinc, file.path(var_dir, "MSE",
    sprintf("mse_VAR_onlyIncidence_%sp%d_h%d_diffseeds_%s.csv", tag_s, p, h, run_date)),
    row.names = FALSE),
    "onlyIncidence MSE CSV")
  log_msg("  onlyIncidence mse_test by seed: ", paste(round(mse_onlyinc$mse_test, 1), collapse = " "))

  # ---------------------------------------------------------------- outputs
  write_long_and_plot(
    fit_list = muspad_fit_list,
    real_train = real_train, real_test = real_t,
    time_train = time_train, time_test = time_test,
    obj_name = "graph_VAR_7day_M",
    rdata_name = sprintf("VAR_7day_MuSPAD_%sp%d_seeds", if (tag == "") "" else paste0(tag, "_"), p),
    png_name   = sprintf("ggplot_forecast_bigtime_muspad_varsubsetbasic_%sp%d_h%d_diffseeds_%s",
                         tag_s, p, h, run_date)
  )
  write_long_and_plot(
    fit_list = onlyinc_fit_list,
    real_train = real_inc_train, real_test = real_inc_t,
    time_train = time_train, time_test = time_test,
    obj_name = "graph_VAR_7day_bl",
    rdata_name = sprintf("VAR_7day_baseline_%sp%d_seeds", if (tag == "") "" else paste0(tag, "_"), p),
    png_name   = sprintf("ggplot_forecast_bigtime_only_incidence_%sp%d_h%d_diffseeds_%s",
                         tag_s, p, h, run_date)
  )

  summary_list[[label]] <- rbind(mse_muspad, mse_onlyinc)
  log_msg("variant", label, "done")
}

# ------------------------------------------------------------------------------
# Summary

summary_all <- do.call(rbind, summary_list)
# per-variant summary file when running in single-variant mode
sum_tag <- if (variant_arg == "all") "" else
  paste0(if (variants[[1]]$tag == "") "full" else variants[[1]]$tag, "_")
summary_file <- file.path(var_dir, "MSE",
  sprintf("mse_VAR_diffseeds_summary_%sp%d_h%d_%s.csv", sum_tag, p, h, run_date))
try_write(write.csv2(summary_all, summary_file, row.names = FALSE), "summary CSV")

log_msg("========================================================")
log_msg("ALL DONE. Summary:", summary_file)
summary_tbl <- summary_all %>%
  group_by(variant, model) %>%
  summarise(mse_test_mean = mean(mse_test),
            mse_test_min  = min(mse_test),
            mse_test_max  = max(mse_test), .groups = "drop")
log_msg(paste(capture.output(print(summary_tbl)), collapse = "\n"))
