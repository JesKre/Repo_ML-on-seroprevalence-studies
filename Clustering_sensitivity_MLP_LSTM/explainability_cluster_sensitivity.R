#!/usr/bin/env Rscript
# Sensitivity of SHAP/LIME feature attributions to k=3,4,5.
# Input: eight *_k3_k4_k5_merged.csv files in the input directory.
# Usage: Rscript explainability_cluster_sensitivity.R [input_dir] [output_dir]
# Base R only; no additional packages required.

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[1] else "."
output_dir <- if (length(args) >= 2) args[2] else input_dir
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

models <- c("MLP", "LSTM")
methods <- c("lime", "shap")
regimes <- c("low", "high")
ks <- c("3", "4", "5")
pairs <- list(c("3", "4"), c("3", "5"), c("4", "5"))

read_merged <- function(path, model) {
  d <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                fileEncoding = "UTF-8")
  if (model == "LSTM") {
    required <- c("Feature", paste0("k=", ks), paste0("Rank k=", ks))
    stopifnot(all(required %in% names(d)))
    feature <- d[["Feature"]]
    imp <- sapply(ks, function(k) as.numeric(d[[paste0("k=", k)]]))
    ranks <- sapply(ks, function(k) as.numeric(d[[paste0("Rank k=", k)]]))
  } else {
    required <- c("feature", paste0("importance_k", ks), paste0("rank_k", ks))
    stopifnot(all(required %in% names(d)))
    feature <- d[["feature"]]
    imp <- sapply(ks, function(k) as.numeric(d[[paste0("importance_k", k)]]))
    ranks <- sapply(ks, function(k) as.numeric(d[[paste0("rank_k", k)]]))
  }
  colnames(imp) <- colnames(ranks) <- ks
  if (anyNA(feature) || anyDuplicated(feature)) stop("Missing/duplicate feature in ", path)
  if (anyNA(imp) || anyNA(ranks)) stop("Missing importance/rank in ", path)
  if (any(!is.finite(imp)) || any(!is.finite(ranks))) stop("Non-finite values in ", path)
  # Check that ranks are complete permutations and correspond to abs(importance).
  for (k in ks) {
    if (!identical(sort(ranks[, k]), as.numeric(seq_len(nrow(d))))) {
      warning("Rank column is not a permutation: ", basename(path), " k=", k)
    }
    derived <- rank(-abs(imp[, k]), ties.method = "first")
    if (any(derived != ranks[, k])) {
      warning("Supplied ranks differ from ranks of absolute attributions in ",
              basename(path), " k=", k, "; using supplied ranks")
    }
  }
  list(feature = feature, imp = imp, ranks = ranks)
}

pairwise <- list()
summary_rows <- list()
cv_features <- list()
ix <- 0L
for (model in models) for (method in methods) for (regime in regimes) {
  file <- sprintf("%s_%s_%s_k3_k4_k5_merged.csv", model, method, regime)
  path <- file.path(input_dir, file)
  if (!file.exists(path)) stop("Missing input: ", path)
  d <- read_merged(path, model)
  n <- length(d$feature)
  # CV across the three absolute attribution magnitudes, using sample SD (n-1).
  magnitudes <- abs(d$imp)
  mean_mag <- rowMeans(magnitudes)
  sd_mag <- apply(magnitudes, 1, sd)
  cv <- ifelse(mean_mag > 0, sd_mag / mean_mag, NA_real_)
  for (top_n in c(20L, 50L)) {
    if (n < top_n) stop("Fewer than ", top_n, " features in ", file)
    union_top <- apply(d$ranks <= top_n, 1, any)
    cv_selected <- cv[union_top]
    if (anyNA(cv_selected)) warning("Zero mean magnitude in ", file,
                                    " (excluded from CV median)")
    median_cv <- median(cv_selected, na.rm = TRUE) * 100
    cv_features[[length(cv_features) + 1L]] <- data.frame(
      model = model, method = toupper(method), regime = regime,
      top_n = top_n, feature = d$feature[union_top],
      importance_k3 = d$imp[union_top, "3"],
      importance_k4 = d$imp[union_top, "4"],
      importance_k5 = d$imp[union_top, "5"],
      cv_percent = cv[union_top] * 100
    )
    for (pair in pairs) {
      k1 <- pair[1]; k2 <- pair[2]
      s1 <- d$feature[d$ranks[, k1] <= top_n]
      s2 <- d$feature[d$ranks[, k2] <= top_n]
      intersection <- length(intersect(s1, s2))
      union_count <- length(union(s1, s2))
      ix <- ix + 1L
      pairwise[[ix]] <- data.frame(
        model = model, method = toupper(method), regime = regime,
        k1 = as.integer(k1), k2 = as.integer(k2), top_n = top_n,
        spearman_rho = cor(d$ranks[, k1], d$ranks[, k2], method = "spearman"),
        overlap_count = intersection,
        overlap_percent = 100 * intersection / top_n,
        jaccard = intersection / union_count,
        median_cv_percent = median_cv,
        cv_feature_count = sum(!is.na(cv_selected))
      )
    }
    rows <- do.call(rbind, tail(pairwise, length(pairs)))
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      model = model, method = toupper(method), regime = regime, top_n = top_n,
      spearman_min = min(rows$spearman_rho),
      spearman_max = max(rows$spearman_rho),
      overlap_min = min(rows$overlap_count),
      overlap_max = max(rows$overlap_count),
      overlap_percent_min = min(rows$overlap_percent),
      overlap_percent_max = max(rows$overlap_percent),
      jaccard_min = min(rows$jaccard), jaccard_max = max(rows$jaccard),
      median_cv_percent = median_cv
    )
  }
}

write.csv(do.call(rbind, pairwise),
          file.path(output_dir, "explainability_pairwise_metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, summary_rows),
          file.path(output_dir, "explainability_summary_metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, cv_features),
          file.path(output_dir, "explainability_feature_cv.csv"), row.names = FALSE)
cat("Saved pairwise metrics, summary metrics, and per-feature CV to:",
    normalizePath(output_dir), "\n")
