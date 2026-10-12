script_file <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE))
stopifnot(length(script_file)==1)
ROOT <- normalizePath(dirname(script_file),mustWork=TRUE)
DATA_FILE <- Sys.getenv("MLP_DATA_FILE",unset="")
PYTHON <- Sys.getenv("MLP_PYTHON",unset="")
if(!nzchar(DATA_FILE)||!file.exists(DATA_FILE)) stop("Set MLP_DATA_FILE to the original modeling RData file.")
if(!nzchar(PYTHON)||!file.exists(PYTHON)) stop("Set MLP_PYTHON to the TensorFlow 2.15.1 Python executable.")
outdir <- Sys.getenv("MLP_RUN_DIR",unset=ROOT)
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
# Keras-2 reproduction attempt of MLP_muspad_seeds.xlsx
# Goal: test whether the team's original run used Keras 2 (TF <= 2.15 era) by
# re-running the FULL 36-cell grid on Python TF 2.15.1 (bundled Keras 2.15.1).
#
# Mirrors 01_RSkript_Keras_ANN_...R verbatim in PROTOCOL (data, split, scaling,
# architecture, optimizer, batch size, two-stage fit, seeds):
#  - R-level calls EXACTLY as in the original: model %>% layer_dense / compile /
#    fit / evaluate / predict (R keras pipe path; verified to accept plain
#    vector y, double batch_size, and to run on the Keras-2 backend of TF 2.15.1)
#  - y passed as plain R numeric vectors (as in the original)
#  - batch_size = round(n_train/10, 0) = 23.0 double, epochs from c(30,45,60)
#    (as in the original)
#  - Sys.unsetenv("TF_USE_LEGACY_KERAS") after library(keras): R keras 2.16.1's
#    .onLoad sets it unconditionally, which breaks lazy tf.keras access on TF 2.15.1
# Additions (do not affect training): per-seed test predictions saved.
options(warn = 1)
suppressPackageStartupMessages({
  library(missForest)
  library(glmnet)
  library(reticulate)
})
reticulate::use_python(PYTHON, required=TRUE)
suppressPackageStartupMessages({
  library(tensorflow)
  library(keras)
})
# R keras 2.16.1 .onLoad sets TF_USE_LEGACY_KERAS=1 unconditionally; under TF
# 2.15.1 that breaks the lazy tf.keras attribute path -> unset (propagates to python)
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
idx_tv <- 1:as.integer(n_tot / 10 * 9)             # rows 1:251
idx_te <- as.integer((n_tot / 10 * 9 + 1):n_tot)   # rows 252:278 (truncated fractional seq, as in original)
cat("n rows:", n_tot, "| 90%% cut (raw):", n_tot / 10 * 9,
    "| train_val rows 1:", max(idx_tv), "(", length(idx_tv), "rows) |",
    "test rows", min(idx_te), ":", max(idx_te), "(", length(idx_te), "rows)", "\n")
stopifnot(length(idx_tv) == 251, length(idx_te) == 27)

layers_list <- c(1, 2, 3, 4)
epochs_list <- c(30, 45, 60)
f <- 10
repeats <- 10

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

results <- data.frame(Model_Type = character(), Layers = integer(), Epochs = integer(),
                      Mean_Train_MSE = numeric(), SD_Train_MSE = numeric(),
                      Mean_Val_MSE = numeric(), SD_Val_MSE = numeric(),
                      Mean_Test_MSE = numeric(), SD_Test_MSE = numeric())
seed_rows <- list()
t0 <- Sys.time()

for (model_type in c("meanImp", "missForest", "onlyTime")) {

  if (model_type == "meanImp") {
    train_val_dat <- as.data.frame(makeX(train = df_ts_inc[idx_tv, -1],
                                         test = df_ts_inc[idx_te, -1], na.impute = T)$x)
    testdat <- as.data.frame(makeX(train = df_ts_inc[idx_tv, -1],
                                   test = df_ts_inc[idx_te, -1], na.impute = T)$xtest)
  } else if (model_type == "missForest") {
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

  train_val_dat_y <- df_ts_inc[idx_tv, 1]   # plain vector, as in original
  testdat_y <- df_ts_inc[idx_te, 1]         # plain vector, as in original

  # fixed train/val split (identical for every config and seed)
  set.seed(10)
  folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = f, labels = FALSE))
  traindat_1 <- train_val_dat[-which(folds == 10, arr.ind = TRUE), ]
  traindat_scale <- scale(traindat_1)
  traindat <- as.matrix(traindat_scale)
  traindat_y <- train_val_dat_y[-which(folds == 10, arr.ind = TRUE)]   # vector
  valdat_1 <- train_val_dat[which(folds == 10, arr.ind = TRUE), ]
  valdat <- as.matrix(scale(valdat_1, center = attr(traindat_scale, "scaled:center"),
                            scale = attr(traindat_scale, "scaled:scale")))
  valdat_y <- train_val_dat_y[which(folds == 10, arr.ind = TRUE)]      # vector
  train_val_dat_scale <- scale(train_val_dat)
  train_val_dat_s <- as.matrix(train_val_dat_scale)
  testdat_s <- as.matrix(scale(testdat, center = attr(train_val_dat_scale, "scaled:center"),
                               scale = attr(train_val_dat_scale, "scaled:scale")))

  cat(model_type, ": n_train:", nrow(traindat), "n_val:", nrow(valdat), "n_test:", nrow(testdat),
      "n_feat:", ncol(traindat), "batch:", round(nrow(traindat) / 10), "\n")

  for (layers in layers_list) {
    for (epochs in epochs_list) {
      train_mse_list <- c(); val_mse_list <- c(); test_mse_list <- c()
      cat("\nTraining", model_type, "Model: Layers =", layers, ", Epochs =", epochs, "\n")
      for (r in 1:repeats) {
        # ---- model 1: train/val (as in original) ----
        set.seed(r)
        tensorflow::tf$random$set_seed(r)
        model <- keras_model_sequential()
        model %>% layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu")
        if (layers > 1) {
          for (i in 2:layers) model %>% layer_dense(units = 10, activation = "relu")
        }
        model %>% layer_dense(units = 1)
        model %>% compile(optimizer = 'rmsprop', loss = 'mse', metrics = list('mean_squared_error'))
        history <- model %>% fit(traindat, traindat_y, epochs = epochs,
                                 validation_data = list(valdat, valdat_y),
                                 batch_size = round((dim(traindat)[1]) / 10, 0), verbose = 0)
        eval_train <- model %>% evaluate(traindat, traindat_y, verbose = 0)
        eval_val <- model %>% evaluate(valdat, valdat_y, verbose = 0)
        train_mse_list <- c(train_mse_list, eval_train[2])
        val_mse_list <- c(val_mse_list, eval_val[2])

        # ---- model 2: full train+val -> test (as in original) ----
        set.seed(r)
        tensorflow::tf$random$set_seed(r)
        model <- keras_model_sequential()
        model %>% layer_dense(units = 10, input_shape = dim(train_val_dat_s)[2], activation = "relu")
        if (layers > 1) {
          for (i in 2:layers) model %>% layer_dense(units = 10, activation = "relu")
        }
        model %>% layer_dense(units = 1)
        model %>% compile(optimizer = 'rmsprop', loss = 'mse', metrics = list('mean_squared_error'))
        history <- model %>% fit(train_val_dat_s, train_val_dat_y, epochs = epochs,
                                 batch_size = round((dim(traindat)[1]) / 10, 0), verbose = 0)
        eval_test <- model %>% evaluate(testdat_s, testdat_y, verbose = 0)
        test_mse_list <- c(test_mse_list, eval_test[2])

        # per-seed test predictions (extra output, not in original)
        pred_r <- as.numeric(model %>% predict(testdat_s, verbose = 0))

        seed_rows[[length(seed_rows) + 1]] <- data.frame(
          Model_Type = model_type, Layers = layers, Epochs = epochs, seed = r,
          train_mse = eval_train[2], val_mse = eval_val[2], test_mse = eval_test[2],
          check_pred_mse = mean((pred_r - as.numeric(testdat_y))^2)
        )
        seed_rows[[length(seed_rows)]]$preds <- list(pred_r)

        cat(sprintf("  seed %2d: train %.3f val %.3f test %.3f  (%s)\n",
                    r, eval_train[2], eval_val[2], eval_test[2],
                    format(Sys.time() - t0, digits = 4)))
      }
      results <- rbind(results, data.frame(
        Model_Type = model_type, Layers = layers, Epochs = epochs,
        Mean_Train_MSE = mean(train_mse_list), SD_Train_MSE = sd(train_mse_list),
        Mean_Val_MSE = mean(val_mse_list), SD_Val_MSE = sd(val_mse_list),
        Mean_Test_MSE = mean(test_mse_list), SD_Test_MSE = sd(test_mse_list)))
      # checkpoint after every cell (a crash late in the run must not lose results)
      write.table(results, file.path(outdir, "results_repro_k2_36cells.csv"), sep = ";", row.names = FALSE)
    }
  }
}

# ---------------- save ----------------
seed_df <- do.call(rbind, lapply(seed_rows, function(d) {
  p <- d$preds[[1]]
  d$preds <- NULL
  for (i in seq_along(p)) d[[paste0("pred", i)]] <- p[i]
  d
}))
write.table(results, file.path(outdir, "results_repro_k2_36cells.csv"), sep = ";", row.names = FALSE)
write.table(seed_df, file.path(outdir, "seed_predictions_k2_36cells.csv"), sep = ";", row.names = FALSE)

# context block (identical to w2 run; re-written for self-contained downstream metrics)
y_tv_all <- as.numeric(df_ts_inc[idx_tv, 1])
y_te_all <- as.numeric(df_ts_inc[idx_te, 1])
td_tv_all <- df_ts_inc[idx_tv, 2]
td_te_all <- df_ts_inc[idx_te, 2]

write.table(data.frame(time_day = td_tv_all, y = y_tv_all),
            file.path(outdir, "series_trainval.csv"), sep = ";", row.names = FALSE)
write.table(data.frame(time_day = td_te_all, y = y_te_all),
            file.path(outdir, "series_test.csv"), sep = ";", row.names = FALSE)

persistence_train <- abs(y_tv_all[-1] - y_tv_all[-length(y_tv_all)])
rw_train_anchored <- c(y_tv_all[length(y_tv_all)], y_te_all[-length(y_te_all)])
write.table(data.frame(
  n_trainval = length(y_tv_all), n_test = length(y_te_all),
  mean_trainval_y = mean(y_tv_all), mean_test_y = mean(y_te_all),
  mae_naive_train_nextobs = mean(persistence_train),
  rmse_naive_train_nextobs = sqrt(mean(persistence_train^2)),
  mse_naive_train_nextobs = mean(persistence_train^2),
  mae_naive_test_nextobs = mean(abs(y_te_all - rw_train_anchored)),
  rmse_naive_test_nextobs = sqrt(mean((y_te_all - rw_train_anchored)^2)),
  sse_randomwalk_test = sum((y_te_all - rw_train_anchored)^2),
  sst_test_mean = sum((y_te_all - mean(y_te_all))^2),
  sst_train_mean = sum((y_te_all - mean(y_tv_all))^2)
), file.path(outdir, "context_naive_benchmarks.csv"), sep = ";", row.names = FALSE)

cat("\nDONE. elapsed:", format(Sys.time() - t0, digits = 4), "\n")
