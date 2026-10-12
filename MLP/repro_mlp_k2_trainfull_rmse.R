script_file <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE))
stopifnot(length(script_file)==1)
ROOT <- normalizePath(dirname(script_file),mustWork=TRUE)
DATA_FILE <- Sys.getenv("MLP_DATA_FILE",unset="")
PYTHON <- Sys.getenv("MLP_PYTHON",unset="")
if(!nzchar(DATA_FILE)||!file.exists(DATA_FILE)) stop("Set MLP_DATA_FILE to the original modeling RData file.")
if(!nzchar(PYTHON)||!file.exists(PYTHON)) stop("Set MLP_PYTHON to the TensorFlow 2.15.1 Python executable.")
outdir <- Sys.getenv("MLP_RUN_DIR",unset=ROOT)
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
# Per-seed Stage-2 in-sample ("trainfull", 251-row) RMSE for the MLP k2 table row.
#
# Short re-run (20 models) of the EXACT protocol of repro_mlp_k2_smape.R
# (identical data, preprocessing, 251/25/27 split, scaling, architecture,
# optimizer, batch, epochs, seeds, two-stage fit with fresh Stage-2 init):
#   - imputations: missForest (MuSPAD) and onlyTime (baseline) ONLY
#   - layers = 4, epochs = 30 (the reported configuration)
#   - seeds 1:10
# with ONE addition: per-seed Stage-2 in-sample predictions on the full
# 251-row train+val block are stored, and per-seed trainfull RMSE/MAE in
# count units are computed. Per-seed test predictions (27 obs) are stored
# too, and per-seed test MSE is cross-checked against the canonical
# 09-25 run (seed_predictions_k2_36cells.csv) to verify protocol fidelity.
#
# NOTE: per-seed values are NOT bit-reproducible across re-runs on this
# machine (TF 2.15.1 CPU / oneDNN); per-cell means are stable. Only
# mean-level (over 10 seeds) statements are made.
#
# Table convention:
#   "Training" column = Stage-2 final model, in-sample on the full 251-day
#   train+val block (matches the paper's 251-day Training definition).
#   "Test" column    = fixed 27-day block, values taken from the canonical
#   09-25 run (letter-consistent), cross-checked here.
options(warn = 1)
suppressPackageStartupMessages({
  library(missForest)
  library(reticulate)
})
reticulate::use_python(PYTHON, required=TRUE)
suppressPackageStartupMessages({
  library(tensorflow)
  library(keras)
})
Sys.unsetenv("TF_USE_LEGACY_KERAS")
cat("python:", py_config()$python, "\n")
py_run_string("import tensorflow as tf, keras; print('tf version:', tf.__version__, '| keras version:', keras.__version__)")

load(DATA_FILE)
df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
# The synthetic input is already preprocessed and may not contain day 168.
drop_day168 <- which(df_ts_inc$time_day == 168)
if (length(drop_day168)) df_ts_inc <- df_ts_inc[-drop_day168, , drop = FALSE]
cat("dim after day-168 removal:", dim(df_ts_inc), "\n")

n_tot <- nrow(df_ts_inc)
idx_tv <- 1:as.integer(n_tot / 10 * 9)
idx_te <- as.integer((n_tot / 10 * 9 + 1):n_tot)
stopifnot(length(idx_tv) == 251, length(idx_te) == 27)

LAYER <- 4
EPOCHS <- 30
f <- 10
repeats <- 10

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

smape02 <- function(a, f) {
  a <- as.numeric(a); f <- as.numeric(f)
  d <- abs(a) + abs(f)
  num <- 2 * abs(a - f)
  c(mean = mean(num / ifelse(d > 0, d, 1) * as.numeric(d > 0)),
    n_zero_denom = sum(d == 0))
}

seed_rows <- list()
pred_long <- list()
t0 <- Sys.time()

for (model_type in c("missForest", "onlyTime")) {

  if (model_type == "missForest") {
    train_val_dat_1 <- df_ts_inc[idx_tv, -1]
    train_val_dat <- missForest(train_val_dat_1)$ximp
    testdat_1 <- df_ts_inc[idx_te, -1]
    testdat <- missForest(testdat_1)$ximp
  } else {
    train_val_dat <- as.data.frame(df_ts_inc[idx_tv, 2])
    colnames(train_val_dat) = colnames(df_ts_inc)[2]
    testdat <- as.data.frame(df_ts_inc[idx_te, 2])
    colnames(testdat) = colnames(df_ts_inc)[2]
  }

  train_val_dat_y <- df_ts_inc[idx_tv, 1]
  testdat_y <- df_ts_inc[idx_te, 1]

  set.seed(10)
  folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = f, labels = FALSE))
  traindat_1 <- train_val_dat[-which(folds == 10, arr.ind = TRUE), ]
  traindat_scale <- scale(traindat_1)
  traindat <- as.matrix(traindat_scale)
  traindat_y <- train_val_dat_y[-which(folds == 10, arr.ind = TRUE)]
  valdat_1 <- train_val_dat[which(folds == 10, arr.ind = TRUE), ]
  valdat <- as.matrix(scale(valdat_1, center = attr(traindat_scale, "scaled:center"),
                            scale = attr(traindat_scale, "scaled:scale")))
  valdat_y <- train_val_dat_y[which(folds == 10, arr.ind = TRUE)]
  train_val_dat_scale <- scale(train_val_dat)
  train_val_dat_s <- as.matrix(train_val_dat_scale)
  testdat_s <- as.matrix(scale(testdat, center = attr(train_val_dat_scale, "scaled:center"),
                               scale = attr(train_val_dat_scale, "scaled:scale")))

  cat(model_type, ": n_train:", nrow(traindat), "n_val:", nrow(valdat), "n_test:", nrow(testdat),
      "n_feat:", ncol(traindat), "batch:", round(nrow(traindat) / 10), "\n")

  for (r in 1:repeats) {
    # ---- model 1: train/val (Stage 1, as in original) ----
    set.seed(r)
    tensorflow::tf$random$set_seed(r)
    model <- keras_model_sequential()
    model %>% layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu")
    if (LAYER > 1) {
      for (i in 2:LAYER) model %>% layer_dense(units = 10, activation = "relu")
    }
    model %>% layer_dense(units = 1)
    model %>% compile(optimizer = 'rmsprop', loss = 'mse', metrics = list('mean_squared_error'))
    history <- model %>% fit(traindat, traindat_y, epochs = EPOCHS,
                             validation_data = list(valdat, valdat_y),
                             batch_size = round((dim(traindat)[1]) / 10, 0), verbose = 0)
    eval_train <- model %>% evaluate(traindat, traindat_y, verbose = 0)
    eval_val <- model %>% evaluate(valdat, valdat_y, verbose = 0)

    # ---- model 2: full train+val -> test (Stage 2, as in original) ----
    set.seed(r)
    tensorflow::tf$random$set_seed(r)
    model <- keras_model_sequential()
    model %>% layer_dense(units = 10, input_shape = dim(train_val_dat_s)[2], activation = "relu")
    if (LAYER > 1) {
      for (i in 2:LAYER) model %>% layer_dense(units = 10, activation = "relu")
    }
    model %>% layer_dense(units = 1)
    model %>% compile(optimizer = 'rmsprop', loss = 'mse', metrics = list('mean_squared_error'))
    history <- model %>% fit(train_val_dat_s, train_val_dat_y, epochs = EPOCHS,
                             batch_size = round((dim(traindat)[1]) / 10, 0), verbose = 0)
    eval_test <- model %>% evaluate(testdat_s, testdat_y, verbose = 0)

    # per-seed Stage-2 predictions (NEW): test (27) + in-sample full train+val (251)
    pred_te <- as.numeric(model %>% predict(testdat_s, verbose = 0))
    pred_tv <- as.numeric(model %>% predict(train_val_dat_s, verbose = 0))

    tf_rmse <- sqrt(mean((pred_tv - as.numeric(train_val_dat_y))^2))
    tf_mae  <- mean(abs(pred_tv - as.numeric(train_val_dat_y)))
    tf_sm   <- smape02(train_val_dat_y, pred_tv)
    te_rmse <- sqrt(mean((pred_te - as.numeric(testdat_y))^2))
    te_mse  <- mean((pred_te - as.numeric(testdat_y))^2)

    d <- data.frame(
      Model_Type = model_type, Layers = LAYER, Epochs = EPOCHS, seed = r,
      train_mse_s1 = eval_train[2], val_mse_s1 = eval_val[2],
      test_mse = eval_test[2], check_pred_mse = te_mse,
      trainfull_rmse = tf_rmse, trainfull_mae = tf_mae, trainfull_smape = tf_sm["mean"],
      test_rmse = te_rmse
    )
    seed_rows[[length(seed_rows) + 1]] <- d

    # raw predictions (long form) for audit
    pred_long[[length(pred_long) + 1]] <- data.frame(
      Model_Type = model_type, seed = r,
      split = c(rep("trainfull", length(pred_tv)), rep("test", length(pred_te))),
      i = c(seq_along(pred_tv), seq_along(pred_te)),
      y = c(as.numeric(train_val_dat_y), as.numeric(testdat_y)),
      pred = c(pred_tv, pred_te),
      stringsAsFactors = FALSE)

    cat(sprintf("  seed %2d: trainfull RMSE %8.4f (MAE %8.4f, sMAPE %.4f) | test RMSE %8.4f (MSE %10.4f)  (%s)\n",
                r, tf_rmse, tf_mae, tf_sm["mean"], te_rmse, te_mse,
                format(Sys.time() - t0, digits = 4)))
  }
}

newdf <- do.call(rbind, seed_rows)
stopifnot(nrow(newdf) == 20)
write.table(newdf, file.path(outdir, "mlp_trainfull_rmse_k2_L4E30_perseed_2026-09-30.csv"),
            sep = ";", row.names = FALSE)
preddf <- do.call(rbind, pred_long)
write.table(preddf, file.path(outdir, "mlp_trainfull_predictions_k2_L4E30_2026-09-30.csv"),
            sep = ";", row.names = FALSE)

# ================= verification vs canonical 09-25 run (per-seed test MSE) =
canon <- read.delim(file.path(outdir, "seed_predictions_k2_36cells.csv"), sep = ";", dec = ".",
                    stringsAsFactors = FALSE)
m <- merge(canon[, c("Model_Type", "Layers", "Epochs", "seed", "test_mse", "check_pred_mse")],
           newdf[, c("Model_Type", "Layers", "Epochs", "seed", "test_mse", "check_pred_mse")],
           by = c("Model_Type", "Layers", "Epochs", "seed"), suffixes = c("_canon", "_new"))
stopifnot(nrow(m) == 20)
cat("\n=== verification: fresh re-run vs canonical 09-25 run (per-seed test MSE) ===\n")
for (mtc in c("test_mse", "check_pred_mse")) {
  a <- m[[paste0(mtc, "_new")]]; b <- m[[paste0(mtc, "_canon")]]
  ad <- max(abs(a - b))
  rd <- max(abs(a - b) / pmax(abs(b), 1e-12))
  cat(sprintf("%-15s max_abs_diff %.6e | max_rel_diff %.6e\n", mtc, ad, rd))
}
for (mtc in c("missForest", "onlyTime")) {
  sub <- m[m$Model_Type == mtc, ]
  cat(sprintf("%-10s | mean test_mse: canon %.4f vs new %.4f (rel %.4f) | sd: %.4f vs %.4f\n",
              mtc, mean(sub$test_mse_canon), mean(sub$test_mse_new),
              abs(mean(sub$test_mse_new) - mean(sub$test_mse_canon)) / mean(sub$test_mse_canon),
              sd(sub$test_mse_canon), sd(sub$test_mse_new)))
}

# ================= aggregation: mean (SD) over 10 seeds =================
cat("\n=== trainfull (Stage-2 in-sample, 251 days) + test, mean (SD) over 10 seeds, k2 L4/E30 ===\n")
agg <- do.call(rbind, lapply(c("missForest", "onlyTime"), function(mtc) {
  sub <- newdf[newdf$Model_Type == mtc, ]
  data.frame(
    Model_Type = mtc,
    trainfull_rmse_mean = mean(sub$trainfull_rmse), trainfull_rmse_sd = sd(sub$trainfull_rmse),
    trainfull_mae_mean = mean(sub$trainfull_mae), trainfull_mae_sd = sd(sub$trainfull_mae),
    trainfull_smape_mean = mean(sub$trainfull_smape), trainfull_smape_sd = sd(sub$trainfull_smape),
    test_rmse_mean_fresh = mean(sub$test_rmse), test_rmse_sd_fresh = sd(sub$test_rmse)
  )
}))
write.table(agg, file.path(outdir, "mlp_trainfull_rmse_k2_L4E30_summary_2026-09-30.csv"),
            sep = ";", row.names = FALSE)
print(agg, row.names = FALSE)
cat("\nDONE. elapsed:", format(Sys.time() - t0, digits = 4), "\n")
