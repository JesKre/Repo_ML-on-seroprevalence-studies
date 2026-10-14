#!/usr/bin/env Rscript
# ============================================================================
# var_train_papermetric_w2.R — 2026-09-29
#
# The PAPER's TRAINING-set metric (fitted() scored six days ahead: the
# one-step in-sample fit at time t is scored against the observation at t+6)
# applied to all 30 W2 rolling-origin fits (seed 1, 2026-09-23).
#
#   fitted(fit) = first n-p-6 one-step in-sample fits (times p+1 .. n-6)
#   RMSE_train  = rmse( fitted(fit)  vs  observations at times p+7 .. n )
#
# Metric set (computed on the shifted scoring window obs p+7..n):
#   MAE, RMSE, MSE, sMAPE (0-200), R2 (SST around window / pooled mean),
#   MASE (= MAE / 1-day persistence MAE, same window / pooled terms)
#
# Rows:
#   level="origin" : 30 per-origin rows (paper metric on each window)
#   level="pooled" : 6 pooled rows - pooled per-day error terms across the
#                    5 origins (each term from its own origin's refit model),
#                    pooled R2 around the pooled mean (S1 convention),
#                    pooled MASE on pooled persistence terms
#
# CIs (percentile, B=2000):
#   origin rows : day-pair bootstrap - resample (obs_t, obs_{t-1}, fit_t)
#                 triples within the window; single RNG stream, seed 20260929
#   pooled rows : origin-cluster bootstrap - sample(1:5, 5, replace=TRUE),
#                 R2 around the replicate's own pooled mean; seed 20260930
#
# Expected reproduction:
#   full split, paper Table 1 RMSE : onlyInc 38.45/38.80/39.42 (exact),
#                                    MuSPAD  34.59/36.93/36.14 (approx)
#   full split, paper Table 2 sMAPE (0-2): onlyInc 0.51/0.49/0.48,
#                                          MuSPAD  0.62/0.66/0.64
#   threefourth split, paper Table 3 RMSE: onlyInc 53.04/53.31/67.21 (exact),
#                                          MuSPAD 33.80/31.19/34.59 (approx)
#
# Output: metrics_VAR_w2_train_paper_shifted_2026-09-29.csv
#         (on S: + local mirror; xlsx afterwards)
#         log: C:\Users\abz26\AppData\Local\Temp\opencode\var_train_papermetric_log.txt
# ============================================================================
suppressMessages(library(bigtime))
options(scipen = 999)

mdir <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR/Models"
mout <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR/Metrics"
lout <- "C:/Users/abz26/AppData/Local/Temp/opencode"
if (!dir.exists(mdir)) stop("Models dir not reachable: ", mdir, " (check S: drive / NEON)")
if (!dir.exists(mout)) stop("Metrics dir not reachable: ", mout)

sink(file.path(lout, "var_train_papermetric_log.txt"), split = 2)
on.exit(sink(type = "output"))

models <- c("MuSPAD", "onlyIncidence")
ps     <- c(7L, 14L, 21L)
vorder <- c("65percentData", "675percentData", "threefourthData", "80percentData", "full")
vdate  <- c("65percentData" = "2021-03-24", "675percentData" = "2021-04-03",
            "threefourthData" = "2021-05-04", "80percentData" = "2021-05-24",
            "full" = "2021-08-07")
ntrain_expect <- c("65percentData" = 265L, "675percentData" = 275L,
                   "threefourthData" = 306L, "80percentData" = 326L,
                   "full" = 401L)

# printed paper TRAINING values
paper_rmse <- list(
  "MuSPAD|full"                 = c("7" = 34.59, "14" = 36.93, "21" = 36.14),
  "onlyIncidence|full"          = c("7" = 38.45, "14" = 38.80, "21" = 39.42),
  "MuSPAD|threefourthData"      = c("7" = 33.80, "14" = 31.19, "21" = 34.59),
  "onlyIncidence|threefourthData" = c("7" = 53.04, "14" = 53.31, "21" = 67.21))
paper_sma02 <- list(
  "MuSPAD|full"        = c("7" = 0.62, "14" = 0.66, "21" = 0.64),
  "onlyIncidence|full" = c("7" = 0.51, "14" = 0.49, "21" = 0.48))

B <- 2000L

getscaling <- function(e, ism) {
  if (ism) return(c(as.numeric(e$scale_s_inc), as.numeric(e$scale_c_inc)))
  sv <- e$scale_scale_train; cv <- e$scale_center_train
  s  <- if (!is.null(names(sv)) && "Inzidenz" %in% names(sv)) sv[["Inzidenz"]] else sv[1]
  cc <- if (!is.null(names(cv)) && "Inzidenz" %in% names(cv)) cv[["Inzidenz"]] else cv[1]
  c(as.numeric(s), as.numeric(cc))
}

metrics6 <- function(fO, aO, aOprev) {
  ee   <- fO - aO
  den  <- abs(fO) + abs(aO)
  term <- 2 * abs(ee) / den
  iz   <- which(den == 0); if (length(iz)) term[iz] <- 0
  pers <- mean(abs(aO - aOprev))
  list(MAE = mean(abs(ee)), RMSE = sqrt(mean(ee^2)), MSE = mean(ee^2),
       sMAPE = 100 * mean(term),
       R2 = 1 - sum(ee^2) / sum((aO - mean(aO))^2),
       MASE = mean(abs(ee)) / pers, PERS = pers)
}

# day-pair bootstrap CI (uses the global RNG stream; B from outer scope)
daypair_ci <- function(fO, aO, aOprev) {
  nt <- length(fO)
  ii <- sample.int(nt, B * nt, replace = TRUE)
  aM  <- matrix(aO[ii], nrow = B)
  apM <- matrix(aOprev[ii], nrow = B)
  fM  <- matrix(fO[ii], nrow = B)
  eM  <- fM - aM
  denb <- abs(fM) + abs(aM)
  sm  <- matrix(2 * abs(eM) / denb, nrow = B)
  sm[denb == 0] <- 0
  mse_b <- rowMeans(eM^2)
  amean <- rowMeans(aM)
  lst <- list(MAE = rowMeans(abs(eM)), RMSE = sqrt(mse_b), MSE = mse_b,
              sMAPE = 100 * rowMeans(sm),
              R2 = 1 - mse_b * nt / rowSums((aM - amean)^2),
              MASE = rowMeans(abs(eM)) / rowMeans(abs(aM - apM)))
  t(vapply(lst, function(x) as.numeric(quantile(x, c(0.025, 0.975))), numeric(2)))
}

set.seed(20260929)  # single day-pair bootstrap stream for the 30 origin rows

rows  <- vector("list", 30)
terms <- new.env()
ri <- 0
for (mname in models) {
  ism <- (mname == "onlyIncidence")
  for (p in ps) {
    for (v in vorder) {
      vseg <- if (v == "full") "" else paste0(v, "_")
      fn   <- file.path(mdir, paste0("model_VAR_", if (ism) "onlyIncidence_" else "",
                                     vseg, "p", p, "_h7_seed1_2026-09-23.RData"))
      if (!file.exists(fn)) stop("missing RData: ", fn)
      e <- new.env(); load(fn, envir = e)
      fit <- if (ism) e$VARfit_inc else e$VARfit
      sc  <- getscaling(e, ism); s <- sc[1]; c0 <- sc[2]

      Y <- fit$Y; n <- nrow(Y)
      stopifnot(n == ntrain_expect[[v]])

      fv  <- fitted(fit)
      fvv <- as.numeric(if (is.matrix(fv)) fv[, 1] else fv)
      iW  <- (p + 7):n
      stopifnot(length(fvv) == length(iW), length(fvv) == n - p - 6)

      fO     <- fvv * s + c0
      aO     <- as.numeric(Y[iW, 1]) * s + c0
      aOprev <- as.numeric(Y[(p + 6):(n - 1), 1]) * s + c0

      m  <- metrics6(fO, aO, aOprev)
      ci <- daypair_ci(fO, aO, aOprev)

      assign(paste0("T_", mname, "_", p, "_", v),
             list(f = fO, a = aO, ap = aOprev), envir = terms)

      ri <- ri + 1
      rows[[ri]] <- data.frame(level = "origin", model = mname, p = p, origin = v,
                               origin_date = vdate[[v]], n_train = n, n_terms = length(fO),
                               MAE_train_paper = m$MAE, RMSE_train_paper = m$RMSE,
                               MSE_train_paper = m$MSE, sMAPE_train_paper = m$sMAPE,
                               R2_train_paper = m$R2, MASE_train_paper = m$MASE,
                               persist_MAE_train = m$PERS,
                               MAE_train_paper_ci_lo = ci["MAE", 1],   MAE_train_paper_ci_hi = ci["MAE", 2],
                               RMSE_train_paper_ci_lo = ci["RMSE", 1], RMSE_train_paper_ci_hi = ci["RMSE", 2],
                               MSE_train_paper_ci_lo = ci["MSE", 1],   MSE_train_paper_ci_hi = ci["MSE", 2],
                               sMAPE_train_paper_ci_lo = ci["sMAPE", 1], sMAPE_train_paper_ci_hi = ci["sMAPE", 2],
                               R2_train_paper_ci_lo = ci["R2", 1],     R2_train_paper_ci_hi = ci["R2", 2],
                               MASE_train_paper_ci_lo = ci["MASE", 1], MASE_train_paper_ci_hi = ci["MASE", 2],
                               stringsAsFactors = FALSE)
      cat(sprintf("[%2d/30] %-14s p=%2d %-16s n=%3d  MAE=%.4f  RMSE=%.4f  MSE=%.2f  sMAPE(0-2)=%.4f  R2=%.4f  MASE=%.4f (pers %.4f)\n",
                  ri, mname, p, v, n, m$MAE, m$RMSE, m$MSE, m$sMAPE / 100, m$R2, m$MASE, m$PERS))
    }
  }
}
dfo <- do.call(rbind, rows)

# ---------------- pooled across the 5 rolling origins ----------------------
set.seed(20260930)  # origin-cluster bootstrap stream for the 6 pooled rows

pool_rows <- vector("list", 6)
pi <- 0
for (mname in models) for (p in ps) {
  blk <- lapply(vorder, function(v) get(paste0("T_", mname, "_", p, "_", v), envir = terms))
  fP  <- do.call(c, lapply(blk, `[[`, "f"))
  aP  <- do.call(c, lapply(blk, `[[`, "a"))
  apP <- do.call(c, lapply(blk, `[[`, "ap"))
  mP  <- metrics6(fP, aP, apP)

  off <- c(0L, cumsum(vapply(blk, function(b) length(b$f), 1L)))
  ciP <- matrix(NA_real_, nrow = 6, ncol = B)
  for (b in seq_len(B)) {
    sel <- sample(1:5, 5, replace = TRUE)
    idx <- do.call(c, lapply(sel, function(o) (off[o] + 1):(off[o + 1])))
    mS  <- metrics6(fP[idx], aP[idx], apP[idx])
    ciP[, b] <- c(mS$MAE, mS$RMSE, mS$MSE, mS$sMAPE, mS$R2, mS$MASE)
  }
  ciPt <- t(apply(ciP, 1, quantile, probs = c(0.025, 0.975)))
  rownames(ciPt) <- c("MAE", "RMSE", "MSE", "sMAPE", "R2", "MASE")

  pi <- pi + 1
  pool_rows[[pi]] <- data.frame(level = "pooled", model = mname, p = p, origin = "pooled_5origins",
                                origin_date = "", n_train = NA, n_terms = length(fP),
                                MAE_train_paper = mP$MAE, RMSE_train_paper = mP$RMSE,
                                MSE_train_paper = mP$MSE, sMAPE_train_paper = mP$sMAPE,
                                R2_train_paper = mP$R2, MASE_train_paper = mP$MASE,
                                persist_MAE_train = mP$PERS,
                                MAE_train_paper_ci_lo = ciPt["MAE", 1],   MAE_train_paper_ci_hi = ciPt["MAE", 2],
                                RMSE_train_paper_ci_lo = ciPt["RMSE", 1], RMSE_train_paper_ci_hi = ciPt["RMSE", 2],
                                MSE_train_paper_ci_lo = ciPt["MSE", 1],   MSE_train_paper_ci_hi = ciPt["MSE", 2],
                                sMAPE_train_paper_ci_lo = ciPt["sMAPE", 1], sMAPE_train_paper_ci_hi = ciPt["sMAPE", 2],
                                R2_train_paper_ci_lo = ciPt["R2", 1],     R2_train_paper_ci_hi = ciPt["R2", 2],
                                MASE_train_paper_ci_lo = ciPt["MASE", 1], MASE_train_paper_ci_hi = ciPt["MASE", 2],
                                stringsAsFactors = FALSE)
  cat(sprintf("[P%02d/6] %-14s p=%2d POOLED n_terms=%4d  MAE=%.4f  RMSE=%.4f  MSE=%.2f  sMAPE(0-2)=%.4f  R2=%.4f  MASE=%.4f (pers %.4f)\n",
              pi, mname, p, length(fP), mP$MAE, mP$RMSE, mP$MSE, mP$sMAPE / 100, mP$R2, mP$MASE, mP$PERS))
}
df <- rbind(dfo, do.call(rbind, pool_rows))

# ---------------- reproduction checks ----------------
cat("\n=== REPRODUCTION: FULL SPLIT vs PAPER Table 1 (RMSE) / Table 2 (sMAPE, 0-2) ===\n")
for (mname in models) for (p in ps) {
  r <- df[df$level == "origin" & df$model == mname & df$p == p & df$origin == "full", ]
  cat(sprintf("%-14s p=%2d | RMSE  repro %7.4f vs paper %6.2f | sMAPE repro %6.4f vs paper %5.2f\n",
              mname, p, r$RMSE_train_paper,
              paper_rmse[[paste0(mname, "|full")]][as.character(p)],
              r$sMAPE_train_paper / 100,
              paper_sma02[[paste0(mname, "|full")]][as.character(p)]))
}
cat("\n=== REPRODUCTION: THREEFOURTH SPLIT vs PAPER Table 3 (RMSE) ===\n")
for (mname in models) for (p in ps) {
  r <- df[df$level == "origin" & df$model == mname & df$p == p & df$origin == "threefourthData", ]
  cat(sprintf("%-14s p=%2d | RMSE repro %7.4f vs paper %6.2f\n",
              mname, p, r$RMSE_train_paper,
              paper_rmse[[paste0(mname, "|threefourthData")]][as.character(p)]))
}
cat("\n=== HEADLINE: pooled (5 origins) vs full split ===\n")
for (mname in models) for (p in ps) {
  pr <- df[df$level == "pooled" & df$model == mname & df$p == p, ]
  fr <- df[df$level == "origin" & df$model == mname & df$p == p & df$origin == "full", ]
  cat(sprintf("%-14s p=%2d | POOLED n=%4d  MAE=%8.4f  RMSE=%8.4f [%.4f .. %.4f]  MSE=%10.2f  sMAPE02=%.4f  R2=%8.4f  MASE=%8.4f [%.4f .. %.4f]\n",
              mname, p, pr$n_terms, pr$MAE_train_paper, pr$RMSE_train_paper,
              pr$RMSE_train_paper_ci_lo, pr$RMSE_train_paper_ci_hi, pr$MSE_train_paper,
              pr$sMAPE_train_paper / 100, pr$R2_train_paper, pr$MASE_train_paper,
              pr$MASE_train_paper_ci_lo, pr$MASE_train_paper_ci_hi))
  cat(sprintf("%-14s p=%2d | FULL   n=%4d  MAE=%8.4f  RMSE=%8.4f [%.4f .. %.4f]  MSE=%10.2f  sMAPE02=%.4f  R2=%8.4f  MASE=%8.4f [%.4f .. %.4f]\n",
              mname, p, fr$n_terms, fr$MAE_train_paper, fr$RMSE_train_paper,
              fr$RMSE_train_paper_ci_lo, fr$RMSE_train_paper_ci_hi, fr$MSE_train_paper,
              fr$sMAPE_train_paper / 100, fr$R2_train_paper, fr$MASE_train_paper,
              fr$MASE_train_paper_ci_lo, fr$MASE_train_paper_ci_hi))
}

fname <- "metrics_VAR_w2_train_paper_shifted_2026-09-29.csv"
write.table(df, file.path(mout, fname), sep = ";", dec = ".", row.names = FALSE, quote = FALSE)
write.table(df, file.path(lout, fname), sep = ";", dec = ".", row.names = FALSE, quote = FALSE)
cat("\nwrote:", file.path(mout, fname), "\n")
cat("DONE\n")
