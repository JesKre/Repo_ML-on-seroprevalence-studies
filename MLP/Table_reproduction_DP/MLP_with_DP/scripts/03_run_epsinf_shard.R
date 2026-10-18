# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

# New sigma=0 control for the existing reference DP loop; not a run of the old standalone non-private script.
# ============================================================================
# Epsilon = Inf baseline, 10-seed version (Adam only) -- one process per seed
# ----------------------------------------------------------------------------
# Same protocol as the 10-seed Adam DP study (02_run_adam_dp_grid_shard.R), but
# with sigma = 0 (the exact sigma -> 0 limit of the DP-SGD loop):
#   - same data, preprocessing, missForest (seed 20260101), folds (seed 10)
#   - same architecture, lr 0.01, 30 epochs, 9-batch off-by-one loop
#   - same Poisson subsampling (p = 32/n), per-sample MSE grads, clipping C=1
#   - noise term still drawn, but stddev = sigma*C = 0 => exactly 0
#
# Seed scheme (generalization of the already-delivered s=1 baseline, which
# used base_offset = 10+k = 11..20):
#     base_offset = (s-1)*100 + 10 + k,  k = fold 1..10
#     glorot layer l : 100000 + 5*(base_offset-1) + l
#     subsampling    : 300000 + base_offset   (R RNG)
#     noise (stddev 0): 600000 + base_offset  (TF RNG)
# i.e. the i=2 slot of the 10-seed DP study scheme. Consequence:
#   - s=1 reproduces the delivered s=1 baseline bit-identically
#   - each seed's epsinf run shares init + subsampling with that seed's
#     DP cell (seed s, sigma=2.7295, eps=4); only the noise differs
#
# Usage: Rscript 03_run_epsinf_shard.R <s>   (s = 1..10)
# Writes ONLY epsinf10seed_perfold_s<NN>.csv (own file => no append races).
# Checkpointed: completed folds are skipped on restart.
# ============================================================================

# --- repo paths (this copy): package root is one level above scripts/ -------
PKG_ROOT <- Sys.getenv("DP10SEED_ROOT", unset = "")
if (PKG_ROOT == "") {
  script_file <- sub("^--file=", "",
                     grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  stopifnot(length(script_file) == 1)
  PKG_ROOT <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
}
PY_EXE <- Sys.getenv("DP10SEED_PYTHON",
                     unset = "")

if (!nzchar(PY_EXE)) stop("Set DP10SEED_PYTHON to a compatible Python executable; see the DP README.")

Sys.setenv(TF_USE_LEGACY_KERAS = "True")
reticulate::use_python(PY_EXE, required = TRUE)

suppressMessages({
  library(keras)
  library(tensorflow)
  library(tidyverse)
  library(missForest)
})

my_seed <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1]))
stopifnot(!is.na(my_seed), my_seed >= 1, my_seed <= 10)
s_tag <- sprintf("%02d", my_seed)

OUT_DIR    <- file.path(PKG_ROOT, "results")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
SHARD_CSV  <- file.path(OUT_DIR, sprintf("epsinf10seed_perfold_s%s.csv", s_tag))
DATA_CACHE <- file.path(PKG_ROOT, "data", "dataset_local_cache.rds")
PROG_LOG   <- file.path(OUT_DIR, sprintf("epsinf10seed_progress_s%s.log", s_tag))
unlink(PROG_LOG)
log_prog <- function(...) {
  line <- paste0(format(Sys.time(), "%H:%M:%S"), " | ", paste0(...))
  cat(line, "\n")
  cat(line, "\n", file = PROG_LOG, append = TRUE)
}
done_keys <- character(0)
if (file.exists(SHARD_CSV)) {
  d <- read.csv2(SHARD_CSV)
  done_keys <- paste(d$seed, d$epsilon, d$fold, sep = "|")
}

t0 <- Sys.time()
log_prog(sprintf("START epsinf10seed seed=%d, already-done cells: %d / 10",
                 my_seed, sum(paste(rep(my_seed, 10), rep("Inf", 10), 1:10, sep = "|") %in% done_keys)))

# ----------------------------------------------------------------------------
# Data (identical to the other shards; local RDS cache so restarts survive the
# external data directory being disconnected)
# ----------------------------------------------------------------------------
if (file.exists(DATA_CACHE)) {
  df_ts_inc <- readRDS(DATA_CACHE)
  log_prog("data: LOCAL CACHE, dim df_ts_inc:", dim(df_ts_inc))
} else {
  setwd(Sys.getenv("DP10SEED_DATA_DIR", unset = DATA_DIR))
  file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n = 1)
  cat("data file:", file, "\n")
  load(file)
  df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
  df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day == "168"), ]
  saveRDS(df_ts_inc, DATA_CACHE)
  log_prog("data:", file, " dim df_ts_inc:", dim(df_ts_inc), " -> cached locally")
}

train_val_dat <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9), -1]
set.seed(20260101)
train_val_dat <- missForest(train_val_dat)$ximp
train_val_dat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9), 1]
log_prog("missForest done, elapsed:", round(as.numeric(Sys.time() - t0), 1), "s")

# Folds: IDENTICAL to original
n_fold <- 10
set.seed(10)
folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = n_fold, labels = FALSE))

traindat_list <- list(); traindat_y_list <- list()
valdat_list <- list();   valdat_y_list <- list()

for (k in 1:n_fold) {
  traindat_1 <- train_val_dat[-which(folds == k, arr.ind = TRUE), ]
  traindat_scale <- scale(traindat_1)
  not_all_na <- function(x) any(!is.na(x))
  not_all_nan <- function(x) any(!is.nan(x))
  traindat <- as.data.frame(traindat_scale) %>%
    select(where(not_all_na)) %>%
    select(where(not_all_nan))
  traindat_list[[k]] <- as.matrix(traindat)
  traindat_y_list[[k]] <- train_val_dat_y[-which(folds == k, arr.ind = TRUE)]
  valdat_1 <- train_val_dat[which(folds == k, arr.ind = TRUE), ]
  valdat_scale <- scale(valdat_1, center = attr(traindat_scale, "scaled:center"),
                        scale = attr(traindat_scale, "scaled:scale"))
  valdat <- as.data.frame(valdat_scale) %>% select(colnames(traindat))
  valdat_list[[k]] <- as.matrix(valdat)
  valdat_y_list[[k]] <- train_val_dat_y[which(folds == k, arr.ind = TRUE)]
}
log_prog("folds built:", table(folds))

# ----------------------------------------------------------------------------
# DP-SGD training (same logic as the other shards, seeded)
# ----------------------------------------------------------------------------
mse_loss <- function(y_true, y_pred) k_mean(k_square(y_pred - y_true))

clip_gradients <- function(gradients, C) {
  l2_norm <- function(x) tf$sqrt(tf$reduce_sum(tf$square(x)))
  lapply(gradients, function(grad) {
    grad_norm <- l2_norm(grad)
    factor <- tf$minimum(1.0, C / grad_norm)
    grad * factor
  })
}

train_with_dp_seeded <- function(model, x_train, y_train, x_val, y_val,
                                 epochs, batch_size, num_batches, optimizer,
                                 C, sigma, noise_gen) {
  num_samples <- nrow(x_train)
  sampling_probability <- batch_size / num_samples
  history <- data.frame(epoch = integer(), loss = numeric(), val_loss = numeric())
  for (epoch in 1:epochs) {
    epoch_loss <- 0
    i <- 0
    while (i <= num_batches) {           # as in original: 8 -> runs 9 batches
      i <- i + 1
      flags <- sample(c(FALSE, TRUE), size = num_samples, replace = TRUE,
                      prob = c(1 - sampling_probability, sampling_probability))
      x_batch <- x_train[flags, , drop = FALSE]
      y_batch <- y_train[flags]
      batch_gradients <- lapply(model$trainable_variables, function(var) tf$zeros_like(var))
      for (sample_idx in 1:nrow(x_batch)) {
        x_sample <- x_batch[sample_idx, , drop = FALSE]
        y_sample <- y_batch[sample_idx]
        with(tf$GradientTape() %as% tape, {
          predictions <- model(x_sample)
          loss <- mse_loss(y_sample, predictions)
        })
        sample_gradients <- tape$gradient(loss, model$trainable_variables)
        clipped <- clip_gradients(sample_gradients, C)
        batch_gradients <- mapply("+", batch_gradients, clipped, SIMPLIFY = FALSE)
        epoch_loss <- epoch_loss + as.numeric(loss$numpy())
      }
      noisy_gradients <- lapply(batch_gradients, function(grad) {
        grad + noise_gen$normal(shape = tf$shape(grad), mean = 0.0, stddev = sigma * C)
      })
      averaged_gradients <- lapply(noisy_gradients, function(grad) grad / batch_size)
      optimizer$apply_gradients(zip_lists(averaged_gradients, model$trainable_variables))
    }
    epoch_loss <- epoch_loss / num_samples
    val_loss <- mse_loss(y_val, model(x_val))
    history <- rbind(history, data.frame(epoch = epoch, loss = epoch_loss,
                                         val_loss = as.numeric(val_loss$numpy())))
  }
  history
}

learning_rate <- 0.01
sigma <- 0                                  # <- the ONLY change: no noise (epsilon = Inf)

for (k in 1:n_fold) {
  s <- my_seed
  base_offset <- (s - 1) * 100 + 10 + k     # i=2 slot of the 10-seed scheme
  key <- paste(s, "Inf", k, sep = "|")
  if (key %in% done_keys) { log_prog(sprintf("cell %2d (seed %2d, fold %2d) already done, skip", k, s, k)); next }

  t_cell <- Sys.time()
  log_prog(sprintf("cell %2d START seed=%2d sigma=%.0f eps=Inf fold %2d", k, s, sigma, k))

  traindat   <- traindat_list[[k]]
  traindat_y <- traindat_y_list[[k]]
  valdat     <- valdat_list[[k]]
  valdat_y   <- valdat_y_list[[k]]

  # init seeds: distinct per (cell, layer); layers 1..5
  init_seeds <- 100000 + 5 * (base_offset - 1) + (1:5)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu",
                kernel_initializer = initializer_glorot_uniform(seed = init_seeds[1])) %>%
    layer_dense(units = 10, activation = "relu",
                kernel_initializer = initializer_glorot_uniform(seed = init_seeds[2])) %>%
    layer_dense(units = 10, activation = "relu",
                kernel_initializer = initializer_glorot_uniform(seed = init_seeds[3])) %>%
    layer_dense(units = 10, activation = "relu",
                kernel_initializer = initializer_glorot_uniform(seed = init_seeds[4])) %>%
    layer_dense(units = 1,
                kernel_initializer = initializer_glorot_uniform(seed = init_seeds[5]))

  C <- 1; batch_size <- 32; epochs <- 30; num_batches <- 8
  optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = learning_rate)
  model %>% compile(optimizer = optimizer, loss = mse_loss)

  set.seed(300000 + base_offset)                                    # Poisson subsampling
  noise_gen <- tf$random$Generator$from_seed(600000 + base_offset)  # drawn, but * sigma = 0

  history_df <- train_with_dp_seeded(model, traindat, traindat_y, valdat, valdat_y,
                                     epochs, batch_size, num_batches, optimizer,
                                     C, sigma, noise_gen)
  train_mse <- (model %>% evaluate(traindat, traindat_y))[1]
  val_mse   <- (model %>% evaluate(valdat, valdat_y))[1]

  row <- data.frame(
    seed    = s,
    optimizer = "Adam",
    sigma   = 0,
    epsilon = "Inf",
    fold    = k,
    train_loss_Epochs_1_15_30 = paste(round(history_df$loss[c(1, 15, 30)], 2), collapse = " ; "),
    val_loss_Epochs_1_15_30   = paste(round(history_df$val_loss[c(1, 15, 30)], 2), collapse = " ; "),
    num_epochs = 30,
    train_mse = round(as.numeric(train_mse), 2),
    val_mse   = round(as.numeric(val_mse), 2),
    stringsAsFactors = FALSE
  )
  append_it <- file.exists(SHARD_CSV) && file.info(SHARD_CSV)$size > 0
  write.table(row, SHARD_CSV, sep = ";", dec = ",", quote = TRUE,
              append = append_it, col.names = !append_it, row.names = FALSE)
  log_prog(sprintf("[cell %2d | seed=%2d eps=Inf fold %2d] train_mse=%8.2f val_mse=%8.2f  (%.0fs)",
                   k, s, k, row$train_mse, row$val_mse, as.numeric(Sys.time() - t_cell)))
}

log_prog("total elapsed:", round(as.numeric(Sys.time() - t0), 1), "s")
log_prog(sprintf("EPSINF10_DONE s%s", s_tag))
