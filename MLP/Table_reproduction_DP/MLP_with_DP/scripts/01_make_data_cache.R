# ============================================================================
# One-shot: build data/dataset_local_cache.rds from the original dataset.
# ----------------------------------------------------------------------------
# The cache is the exact post-preprocessing state (df_ts_inc, 279 x 123)
# that the shard workers consume:
#   - original file: last match of "binary_classnumber_var_subset_missing_zero"
#     in S:\PROJACTIVE\MUSPAD-LABORERGEBNISSE\LOKI_ML\Datasets
#   - time_day coerced to numeric, row with time_day == 168 dropped
#
# The cache is shipped with this package (data/dataset_local_cache.rds), so
# you only need this script if you want to rebuild it or refresh it.
# ============================================================================
PKG_ROOT <- Sys.getenv("DP10SEED_ROOT", unset = "")
if (PKG_ROOT == "") {
  script_file <- sub("^--file=", "",
                     grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  stopifnot(length(script_file) == 1)
  PKG_ROOT <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
}

setwd(Sys.getenv("DP10SEED_DATA_DIR", unset = "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets"))
file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n = 1)
stopifnot(file.exists(file))
cat("data file:", file, "\n")
load(file)
df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day == "168"), ]

out_dir <- file.path(PKG_ROOT, "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(out_dir, "dataset_local_cache.rds")
saveRDS(df_ts_inc, out)
cat("cached:", file, " dim df_ts_inc:", dim(df_ts_inc), " -> ", out, "\n")
stopifnot(nrow(df_ts_inc) == 279, ncol(df_ts_inc) == 123)
cat("CACHE OK\n")
