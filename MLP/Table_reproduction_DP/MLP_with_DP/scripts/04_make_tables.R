# Generate Tables 7 and S8 directly from the supplied per-fold masters.
# Base R only. Run from any working directory: Rscript /path/to/this/script.R
PKG_ROOT <- Sys.getenv("DP10SEED_ROOT", unset = "")
if (!nzchar(PKG_ROOT)) {
  script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  stopifnot(length(script_file) == 1)
  PKG_ROOT <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
}
RES <- file.path(PKG_ROOT, "results")
validate_master <- function(x, filename, epsilons, sigmas) {
  required <- c("seed", "epsilon", "sigma", "fold", "train_mse", "val_mse")
  if (!all(required %in% names(x))) stop("Missing columns: ", filename)
  x$epsilon <- as.numeric(x$epsilon)
  keys <- paste(x$seed, x$epsilon, x$fold, sep = "|")
  expected <- expand.grid(seed = 1:10, epsilon = epsilons, fold = 1:10)
  expected_keys <- paste(expected$seed, expected$epsilon, expected$fold, sep = "|")
  if (anyDuplicated(keys) || !setequal(keys, expected_keys))
    stop("Duplicate, missing, or unexpected seed/epsilon/fold cells: ", filename)
  for (j in seq_along(epsilons)) {
    actual <- x$sigma[x$epsilon == epsilons[j]]
    if (any(!is.finite(actual)) || any(abs(actual - sigmas[j]) > 1e-10))
      stop("Unexpected sigma for epsilon in ", filename)
  }
  mse <- as.matrix(x[c("train_mse", "val_mse")])
  if (any(!is.finite(mse)) || any(mse < 0)) stop("Invalid MSE: ", filename)
  x[required]
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) && !identical(args, "--from-shards"))
  stop("Usage: Rscript 04_make_tables.R [--from-shards]")
from_shards <- identical(args, "--from-shards")
read_results <- function(prefix, epsilons, sigmas) {
  filename <- paste0(prefix, "10seed_perfold_master.csv")
  if (from_shards) {
    paths <- file.path(RES, sprintf(paste0(prefix, "10seed_perfold_s%02d.csv"), 1:10))
    if (!all(file.exists(paths))) stop("Missing shard files: ", paste(paths[!file.exists(paths)], collapse = ", "))
    raw <- do.call(rbind, lapply(paths, read.csv2, stringsAsFactors = FALSE))
  } else {
    raw <- read.csv2(file.path(RES, filename), stringsAsFactors = FALSE)
  }
  validated <- validate_master(raw, filename, epsilons, sigmas)
  list(raw = raw, validated = validated, filename = filename)
}
dp_result <- read_results("adam", c(8, 4, 1), c(1.627197265625, 2.7294921875, 9.0625))
inf_result <- read_results("epsinf", Inf, 0)
# Write regenerated masters only after BOTH grids pass validation.
if (from_shards) {
  for (r in list(dp_result, inf_result))
    write.table(r$raw, file.path(RES, r$filename), sep = ";", dec = ",",
                row.names = FALSE, quote = TRUE)
}
dp <- dp_result$validated
baseline <- inf_result$validated
x <- rbind(dp, baseline)
rows <- list()
for (metric in c("Training", "Validation")) {
  mse_col <- if (metric == "Training") "train_mse" else "val_mse"
  for (eps in c(Inf, 8, 4, 1)) {
    g <- x[x$epsilon == eps, ]
    values <- split(sqrt(g[[mse_col]]), g$seed)
    means <- vapply(values, mean, numeric(1))
    fold_sds <- vapply(values, sd, numeric(1))
    rows[[length(rows) + 1L]] <- data.frame(
      metric = metric, epsilon = eps, mean_rmse = mean(means),
      sd_across_seeds = sd(means), mean_sd_across_folds = mean(fold_sds))
  }
}
summary <- do.call(rbind, rows)
tableS8 <- summary
tableS8[3:5] <- lapply(tableS8[3:5], round, digits = 1)
tr <- tableS8[tableS8$metric == "Training", ]
va <- tableS8[tableS8$metric == "Validation", ]
table7 <- data.frame(epsilon = tr$epsilon,
  training_mean_rmse = tr$mean_rmse, training_sd_across_seeds = tr$sd_across_seeds,
  validation_mean_rmse = va$mean_rmse, validation_sd_across_seeds = va$sd_across_seeds)
write.csv(table7, file.path(RES, "table7.csv"), row.names = FALSE)
write.csv(tableS8, file.path(RES, "tableS8.csv"), row.names = FALSE)
print(table7, row.names = FALSE)
print(tableS8, row.names = FALSE)
cat("\nWrote results/table7.csv and results/tableS8.csv.\n")
cat("Known reported-table difference: baseline training mean fold SD rounds to 4.1, not the supplement's 3.8.\n")
