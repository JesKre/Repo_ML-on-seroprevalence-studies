# Generate VAR table rows from saved predictions; base R only.
# S10 is a candidate calculation, not a verified paper reproduction.
t_start <- Sys.time()

## ---- self-locate -------------------------------------------------------------
.args   <- commandArgs(trailingOnly = FALSE)
.farg   <- grep("^--file=", .args, value = TRUE)
if (length(.farg)) {
  .script <- normalizePath(sub("^--file=", "", .farg[1]))
} else {
  .script <- normalizePath("VAR_generate_tables.R")
}
.gendir  <- dirname(.script)
.root    <- normalizePath(file.path(.gendir, ".."))
.datadir <- Sys.getenv("VAR_PRED_DIR", file.path(.root, "Compare_models", "Graphics", "Predicted_values"))
.outdir  <- .gendir
.from_refit <- "--from-refit" %in% commandArgs(trailingOnly = TRUE)
dir.create(.outdir, showWarnings = FALSE, recursive = TRUE)

## ---- constants -----------------------------------------------------------------
ord5   <- c("full", "65percentData", "675percentData",
            "80percentData", "threefourthData")   # TEST pool order (T1/S6 test, S9)
vorder <- c("65percentData", "675percentData", "threefourthData",
            "80percentData", "full")              # TRAIN pool order (T1/S6 train, S10)
models <- c("MuSPAD", "onlyIncidence")
ps     <- c(7L, 14L, 21L)
B      <- 2000L
ntrain_expect <- c("65percentData" = 265L, "675percentData" = 275L,
                   "threefourthData" = 306L, "80percentData" = 326L,
                   "full"            = 401L)
terms_p <- c("7" = 1508L, "14" = 1473L, "21" = 1438L)   # pooled staged train terms
dm5_p   <- c("7" = 1503L, "14" = 1468L, "21" = 1433L)   # pooled train diffs (K-1/origin)
mlab    <- function(m) if (m == "MuSPAD") "MuSPAD" else "Baseline"
grp_of  <- c(`65percentData` = "65%",  `675percentData` = "67.5%",
             `threefourthData` = "75%", `80percentData` = "80%",
             `full` = "Full")
cfgs9 <- list(list(E = "65percentData",   seed = 20261017L),
              list(E = "675percentData",  seed = 20261008L),
              list(E = "threefourthData", seed = 20261018L),
              list(E = "80percentData",   seed = 20261019L),
              list(E = "full",            seed = 20261020L))
cfgs10 <- list(list(E = "65percentData",   seed = 20261021L),
               list(E = "675percentData",  seed = 20261010L),
               list(E = "threefourthData", seed = 20261022L),
               list(E = "80percentData",   seed = 20261023L),
               list(E = "full",            seed = 20261024L))

## ---- helpers ---------------------------------------------------------------------
get_all <- function(fn, obj) {
  e <- new.env(); load(fn, envir = e)
  as.data.frame(e[[obj]])
}

## time-aligned (real, predicted) rows of one set/seed of a seeds file
paired <- function(allw, setv, seed = 1) {
  d  <- allw[allw$seed == seed & allw$set == setv, ]
  re <- d[d$real_vs_pred == "real", ]
  pr <- d[d$real_vs_pred == "predicted", ]
  stopifnot(nrow(re) == nrow(pr))
  re <- re[order(re$time), ]; pr <- pr[order(pr$time), ]
  stopifnot(identical(as.character(re$time), as.character(pr$time)))
  data.frame(time = as.character(re$time),
             real = as.numeric(re$Incidence),
             predicted = as.numeric(pr$Incidence))
}

## verbatim metric block of the originals (train-side window metrics)
metrics6 <- function(fO, aO, aOprev) {
  ee   <- fO - aO
  den  <- abs(fO) + abs(aO)
  term <- 2 * abs(ee) / den
  term[which(den == 0)] <- 0
  pers <- mean(abs(aO - aOprev))
  list(MAE   = mean(abs(ee)),
       RMSE  = sqrt(mean(ee^2)),
       MSE   = mean(ee^2),
       sMAPE = 100 * mean(term),
       R2    = 1 - sum(ee^2) / sum((aO - mean(aO))^2),
       MASE  = mean(abs(ee)) / pers,
       PERS  = pers)
}

## test-side pooled metrics over a set of origin blocks (den = MASE denominator)
metrics_from_blocks <- function(bl, den) {
  real <- do.call(c, lapply(bl, `[[`, "real"))
  e    <- do.call(c, lapply(bl, `[[`, "e"))
  e2   <- do.call(c, lapply(bl, `[[`, "e2"))
  sm   <- do.call(c, lapply(bl, `[[`, "smapc"))
  list(MAE   = mean(abs(e)),
       RMSE  = sqrt(mean(e2)),
       sMAPE = mean(sm),
       MASE  = mean(abs(e)) / den,
       R2    = 1 - sum(e2) / sum((real - mean(real))^2))
}

fixz <- function(s) sub("^-0(\\.0+)?$", "0", s)
fmt  <- function(x, dec = 2L) fixz(sprintf(paste0("%.", dec, "f"), x))
cell <- function(pt, lo, hi, dec = 2L, sep = ", ")
  sprintf("%s (%s%s%s)", fmt(pt, dec), fmt(lo, dec), sep, fmt(hi, dec))

## ;-separated CSV: quoted header, numerics at %.15g, character as-is
write_hs <- function(path, df) {
  cols <- lapply(as.data.frame(df), function(col)
    if (is.character(col)) col else sprintf("%.15g", as.numeric(col)))
  hdr  <- paste0('"', names(df), '"')
  body <- apply(do.call(cbind, cols), 1, paste, collapse = ";")
  writeLines(c(paste(hdr, collapse = ";"), body), path)
}

gkey <- character(0); gval <- character(0)
addg <- function(k, v) { gkey <<- c(gkey, k); gval <<- c(gval, v) }
getv <- function(k) gval[which(gkey == k, arr.ind = TRUE)]

## ---- 2. test-side cells from the 30 seed files ---------------------------------------
vtoken    <- function(v) if (v == "full") "" else paste0("_", v)
seed_file <- function(m, p, v)
  file.path(.datadir, sprintf("VAR_7day_%s%s_p%d_seeds.RData",
                              if (m == "MuSPAD") "MuSPAD" else "baseline",
                              vtoken(v), p))
objname   <- function(m) if (m == "MuSPAD") "graph_VAR_7day_M" else "graph_VAR_7day_bl"

cells <- list()
for (m in models) for (p in ps) {
  cells[[m]][[p]] <- list()
  for (v in vorder) {
    fn   <- seed_file(m, p, v)
    stopifnot(file.exists(fn))
    allw <- get_all(fn, objname(m))
    ## 10-seed identity on the Test set (seeds 2..10 vs seed 1, bit-exact)
    t1 <- allw[allw$seed == 1 & allw$set == "Test", ]
    t1 <- t1[order(t1$time), ]
    if (!.from_refit) for (s in 2:10) {
      ts <- allw[allw$seed == s & allw$set == "Test", ]
      ts <- ts[order(ts$time), ]
      stopifnot(max(abs(ts$Incidence - t1$Incidence)) == 0)
    }
    te <- paired(allw, "Test"); tr <- paired(allw, "Training")
    stopifnot(nrow(te) == 7L,
              nrow(tr) == ntrain_expect[[v]] - p - 6L)
    e  <- te$predicted - te$real
    sm <- 100 * 2 * abs(e) / (abs(te$real) + abs(te$predicted))
    cells[[m]][[p]][[v]] <- list(
      real = te$real, pred = te$predicted, e = e, e2 = e^2, smapc = sm,
      ## in-sample 1-day persistence differences (K-1 per origin)
      train_naive = abs(tr$real[-1] - tr$real[-nrow(tr)]),
      ## seed-1 Training rows, kept for the staged-terms bit-equality assert
      train_real = tr$real, train_pred = tr$predicted)
  }
}

## ---- 3. MASE denominators (test side, dm5) -------------------------------------------
dm5 <- list()
for (m in models) for (p in ps) {
  tn <- do.call(c, lapply(ord5, function(v) cells[[m]][[p]][[v]]$train_naive))
  stopifnot(length(tn) == dm5_p[[as.character(p)]])
  dm5[[m]][[p]] <- mean(tn)
}
## both models share the training series -> denominators must be identical
for (p in ps)
  stopifnot(abs(dm5[["MuSPAD"]][[p]] - dm5[["onlyIncidence"]][[p]]) <
              1e-12 * dm5[["MuSPAD"]][[p]])

## ---- 4. test bootstrap + Table S8 (seed 20260923) -------------------------------------
test_res <- list()
set.seed(20260923)
for (m in models) for (p in ps) {
  b5 <- lapply(ord5, function(v) cells[[m]][[p]][[v]])
  dm <- dm5[[m]][[p]]
  pt <- metrics_from_blocks(b5, dm)
  ## Table S8: one pooled OLS on the 35 OOS pairs (ord5 pool order)
  y  <- do.call(c, lapply(b5, `[[`, "real"))
  x  <- do.call(c, lapply(b5, `[[`, "pred"))
  cf0 <- coef(lm(y ~ x))                    # cf0[1] = intercept, cf0[2] = slope
  boot <- matrix(NA_real_, B, 5)
  colnames(boot) <- c("MAE", "RMSE", "sMAPE", "MASE", "R2")
  s5i <- numeric(B); s5s <- numeric(B)
  for (b in seq_len(B)) {
    s  <- sample(1:5, 5, replace = TRUE)
    bm <- metrics_from_blocks(b5[s], dm)
    boot[b, ] <- c(bm$MAE, bm$RMSE, bm$sMAPE, bm$MASE, bm$R2)
    ## S8 reuses the SAME draw: select 5 origin BLOCKS (never day indices)
    yb <- do.call(c, lapply(s, function(i) b5[[i]]$real))
    xb <- do.call(c, lapply(s, function(i) b5[[i]]$pred))
    fbb <- suppressWarnings(tryCatch(lm(yb ~ xb), error = function(e) NULL))
    cf  <- if (is.null(fbb)) c(NA_real_, NA_real_) else coef(fbb)
    s5i[b] <- cf[1]; s5s[b] <- cf[2]
  }
  ci  <- t(apply(boot, 2, function(col) quantile(col, c(0.025, 0.975))))
  rownames(ci) <- colnames(boot); colnames(ci) <- c("lo", "hi")
  ci5 <- rbind(intercept = quantile(s5i, c(0.025, 0.975), na.rm = TRUE),
               slope     = quantile(s5s, c(0.025, 0.975), na.rm = TRUE))
  test_res[[paste0(m, ";", p)]] <- list(pt = pt, ci = ci, cf0 = cf0, ci5 = ci5)
}

## ---- 5. train side: staged terms, point + bootstrap (seed 20260930) --------------------
te2 <- new.env()
load(file.path(.datadir, "VAR_train_terms.RData"), envir = te2)
terms <- te2$terms
stopifnot(length(terms) == 30L)

for (m in models) for (p in ps) for (v in vorder) {
  t <- terms[[paste0(m, ";", p, ";", v)]]
  stopifnot(t$n == ntrain_expect[[v]],
            t$K == ntrain_expect[[v]] - p - 6L)
  ## staged terms are BIT-EQUAL to the seed-file Training rows (seed 1)
  cll <- cells[[m]][[p]][[v]]
  stopifnot(max(abs(cll$train_real  - t$a)) == 0,
            max(abs(cll$train_pred  - t$f)) == 0)
}
for (m in models) for (p in ps) {
  tot <- sum(vapply(vorder,
                    function(v) terms[[paste0(m, ";", p, ";", v)]]$K, numeric(1)))
  stopifnot(tot == terms_p[[as.character(p)]])
}

train_res <- list()
for (m in models) for (p in ps) {
  tb  <- lapply(vorder, function(v) terms[[paste0(m, ";", p, ";", v)]])
  fP  <- do.call(c, lapply(tb, `[[`, "f"))
  aP  <- do.call(c, lapply(tb, `[[`, "a"))
  apP <- do.call(c, lapply(tb, `[[`, "ap"))
  train_res[[paste0(m, ";", p)]] <-
    list(pt = metrics6(fP, aP, apP), fP = fP, aP = aP, apP = apP,
         Ks = vapply(tb, function(t) t$K, numeric(1)))
}
set.seed(20260930)
for (m in models) for (p in ps) {
  r   <- train_res[[paste0(m, ";", p)]]
  off <- c(0L, cumsum(r$Ks))
  boot <- matrix(NA_real_, B, 5)
  colnames(boot) <- c("MAE", "RMSE", "sMAPE", "MASE", "R2")
  for (b in seq_len(B)) {
    sel <- sample(1:5, 5, replace = TRUE)
    ## each resampled block contributes its own K terms incl. lead-in
    idx <- do.call(c, lapply(sel, function(o) (off[o] + 1):(off[o + 1])))
    mS  <- metrics6(r$fP[idx], r$aP[idx], r$apP[idx])
    boot[b, ] <- c(mS$MAE, mS$RMSE, mS$sMAPE, mS$MASE, mS$R2)
  }
  r$ci <- t(apply(boot, 2, function(col) quantile(col, c(0.025, 0.975))))
  rownames(r$ci) <- colnames(boot); colnames(r$ci) <- c("lo", "hi")
  train_res[[paste0(m, ";", p)]] <- r
}

## ---- 6. Table S9: leave-one-origin-out, TEST side ---------------------------------------
s9 <- list()
for (cfg in cfgs9) {
  g  <- grp_of[[cfg$E]]
  vs <- setdiff(ord5, cfg$E)            # remaining 4 origins, ord5 relative order
  ## point estimates for all 6 rows FIRST (no RNG)
  for (m in models) for (p in ps) {
    b4  <- lapply(vs, function(v) cells[[m]][[p]][[v]])
    dm4 <- mean(do.call(c, lapply(b4, `[[`, "train_naive")))
    s9[[g]][[m]][[p]] <- list(pt = metrics_from_blocks(b4, dm4),
                              b4 = b4, dm4 = dm4)
  }
  set.seed(cfg$seed)
  for (m in models) for (p in ps) {
    r    <- s9[[g]][[m]][[p]]
    boot <- matrix(NA_real_, B, 2)
    colnames(boot) <- c("RMSE", "sMAPE")
    for (b in seq_len(B)) {
      s  <- sample(1:4, 4, replace = TRUE)
      bm <- metrics_from_blocks(r$b4[s], r$dm4)
      boot[b, ] <- c(bm$RMSE, bm$sMAPE)
    }
    r$ci <- t(apply(boot, 2, function(col) quantile(col, c(0.025, 0.975))))
    rownames(r$ci) <- colnames(boot); colnames(r$ci) <- c("lo", "hi")
    s9[[g]][[m]][[p]] <- r
  }
}

## ---- 7. Diagnostic S10: leave-one-origin-out TEST metrics ----
s10 <- list()
for (cfg in cfgs10) {
  g <- grp_of[[cfg$E]]
  vs <- setdiff(ord5,cfg$E)
  for(m in models) for(p in ps) {
    b4 <- lapply(vs,function(v)cells[[m]][[p]][[v]])
    dm4 <- mean(do.call(c,lapply(b4,`[[`,"train_naive")))
    s10[[g]][[m]][[p]] <- list(pt=metrics_from_blocks(b4,dm4),b4=b4,dm4=dm4)
  }
  set.seed(cfg$seed)
  for(m in models) for(p in ps) {
    r <- s10[[g]][[m]][[p]]
    boot <- matrix(NA_real_,B,3,dimnames=list(NULL,c("MAE","MASE","R2")))
    for(b in seq_len(B)) {
      sel<-sample(1:4,4,replace=TRUE)
      met<-metrics_from_blocks(r$b4[sel],r$dm4)
      boot[b,]<-c(met$MAE,met$MASE,met$R2)
    }
    r$ci<-t(apply(boot,2,function(col)quantile(col,c(.025,.975))))
    rownames(r$ci)<-colnames(boot);colnames(r$ci)<-c("lo","hi")
    s10[[g]][[m]][[p]]<-r
  }
}

## ---- 8. assemble the 222 formatted cells -------------------------------------------------
## T1 / S6: 2+3 metrics x 2 models x 3 p x 2 sides = 24 + 36
tab13 <- function(tag, metrics, sep) {
  for (mt in metrics) for (m in models) for (p in ps) {
    tr <- train_res[[paste0(m, ";", p)]]
    te <- test_res[[paste0(m, ";", p)]]
    for (side in c("train", "test")) {
      rr <- if (side == "train") tr else te
      pt <- rr$pt[[mt]]; ci <- rr$ci[mt, ]
      if (mt == "sMAPE") { pt <- pt / 100; ci <- ci / 100 }
      addg(paste0(tag, ";", mt, ";", mlab(m), ";", p, ";", side),
           cell(pt, ci[1], ci[2], sep = sep))
    }
  }
}
tab13("T1", c("RMSE", "sMAPE"),  sep = "-")
tab13("S6", c("MAE", "MASE", "R2"), sep = ", ")

## S8: 2 params x 2 models x 3 p = 12 (paper row order: M/B per p)
for (m in models) for (p in ps) {
  te <- test_res[[paste0(m, ";", p)]]
  addg(paste0("S8;intercept;", mlab(m), ";", p),
       cell(te$cf0[1], te$ci5["intercept", 1], te$ci5["intercept", 2]))
  addg(paste0("S8;slope;", mlab(m), ";", p),
       cell(te$cf0[2], te$ci5["slope", 1], te$ci5["slope", 2]))
}

## S9: 5 groups x 3 p x 4 cells = 60
for (cfg in cfgs9) {
  g <- grp_of[[cfg$E]]
  for (p in ps) for (m in models) {
    r <- s9[[g]][[m]][[p]]
    addg(paste0("S9;RMSE;", g, ";", p, ";", mlab(m)),
         cell(r$pt$RMSE, r$ci["RMSE", 1], r$ci["RMSE", 2]))
    addg(paste0("S9;sMAPE;", g, ";", p, ";", mlab(m)),
         cell(r$pt$sMAPE / 100, r$ci["sMAPE", 1] / 100, r$ci["sMAPE", 2] / 100))
  }
}

## S10: 5 groups x 3 p x 6 cells = 90
for (cfg in cfgs10) {
  g <- grp_of[[cfg$E]]
  for (p in ps) for (m in models) {
    r <- s10[[g]][[m]][[p]]
    addg(paste0("S10;MAE;",  g, ";", p, ";", mlab(m)),
         cell(r$pt$MAE,  r$ci["MAE", 1], r$ci["MAE",  2], dec = 1L))
    addg(paste0("S10;MASE;", g, ";", p, ";", mlab(m)),
         cell(r$pt$MASE, r$ci["MASE", 1], r$ci["MASE", 2], dec = 1L))
    addg(paste0("S10;R2;",   g, ";", p, ";", mlab(m)),
         cell(r$pt$R2,   r$ci["R2",   1], r$ci["R2",   2], dec = 2L))
  }
}
stopifnot(length(gkey) == 222L, !any(duplicated(gkey)))

## ---- 9. CSV outputs (machine-readable, numerics at full precision) ------------------------
mk13 <- function(tag, metrics) {
  M <- character(0); ML <- character(0); P <- integer(0); S <- character(0)
  PT <- numeric(0); LO <- numeric(0); HI <- numeric(0); CE <- character(0)
  for (mt in metrics) for (m in models) for (p in ps) for (side in c("train", "test")) {
    r  <- if (side == "train") train_res[[paste0(m, ";", p)]] else
      test_res[[paste0(m, ";", p)]]
    pt <- r$pt[[mt]]; ci <- r$ci[mt, ]
    if (mt == "sMAPE") { pt <- pt / 100; ci <- ci / 100 }
    M  <- c(M, mt);  ML <- c(ML, mlab(m)); P <- c(P, p); S <- c(S, side)
    PT <- c(PT, pt); LO <- c(LO, ci[1]);   HI <- c(HI, ci[2])
    CE <- c(CE, cell(pt, ci[1], ci[2], sep = if (tag == "T1") "-" else ", "))
  }
  df <- data.frame(metric = M, model = ML, p = P, side = S,
                   point = PT, ci_lo = LO, ci_hi = HI, cell = CE,
                   stringsAsFactors = FALSE)
  write_hs(file.path(.outdir, paste0(tag, "_var.csv")), df)
  invisible(df)
}
mk13("T1", c("RMSE", "sMAPE"))
mk13("S6", c("MAE", "MASE", "R2"))

mk5 <- function() {
  rows <- list()
  for (m in models) for (p in ps) {
    r <- test_res[[paste0(m, ";", p)]]
    rows[[length(rows) + 1L]] <- c(pp = "intercept", m = mlab(m), p = p,
                                   pt = r$cf0[1], lo = r$ci5["intercept", 1],
                                   hi = r$ci5["intercept", 2],
                                   ce = cell(r$cf0[1], r$ci5["intercept", 1],
                                             r$ci5["intercept", 2]))
    rows[[length(rows) + 1L]] <- c(pp = "slope", m = mlab(m), p = p,
                                   pt = r$cf0[2], lo = r$ci5["slope", 1],
                                   hi = r$ci5["slope", 2],
                                   ce = cell(r$cf0[2], r$ci5["slope", 1],
                                             r$ci5["slope", 2]))
  }
  df <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
  df$p  <- as.integer(df$p)
  df$pt <- as.numeric(df$pt); df$lo <- as.numeric(df$lo); df$hi <- as.numeric(df$hi)
  names(df) <- c("param", "model", "p", "point", "ci_lo", "ci_hi", "cell")
  write_hs(file.path(.outdir, "S8_var.csv"), df)
  invisible(df)
}
mk5()

mk9 <- function() {
  G <- character(0); P <- integer(0); ML <- character(0)
  RM <- character(0); SM <- character(0)
  for (cfg in cfgs9) {
    g <- grp_of[[cfg$E]]
    for (p in ps) for (m in models) {
      r <- s9[[g]][[m]][[p]]
      G <- c(G, g); P <- c(P, p); ML <- c(ML, mlab(m))
      RM <- c(RM, cell(r$pt$RMSE, r$ci["RMSE", 1], r$ci["RMSE", 2]))
      SM <- c(SM, cell(r$pt$sMAPE / 100, r$ci["sMAPE", 1] / 100,
                       r$ci["sMAPE", 2] / 100))
    }
  }
  df <- data.frame(group = G, p = P, model = ML, RMSE = RM, sMAPE = SM,
                   stringsAsFactors = FALSE)
  write_hs(file.path(.outdir, "S9_var.csv"), df)
  invisible(df)
}
mk9()

mk10 <- function() {
  MT <- character(0); G <- character(0); P <- integer(0); ML <- character(0)
  PT <- numeric(0); LO <- numeric(0); HI <- numeric(0); CE <- character(0)
  for (cfg in cfgs10) {
    g <- grp_of[[cfg$E]]
    for (p in ps) for (m in models) {
      r <- s10[[g]][[m]][[p]]
      for (mt in c("MAE", "MASE", "R2")) {
        dec <- if (mt == "R2") 2L else 1L
        MT <- c(MT, mt); G <- c(G, g); P <- c(P, p); ML <- c(ML, mlab(m))
        PT <- c(PT, r$pt[[mt]]); LO <- c(LO, r$ci[mt, 1]); HI <- c(HI, r$ci[mt, 2])
        CE <- c(CE, cell(r$pt[[mt]], r$ci[mt, 1], r$ci[mt, 2], dec = dec))
      }
    }
  }
  df <- data.frame(metric = MT, group = G, p = P, model = ML,
                   point = PT, ci_lo = LO, ci_hi = HI, cell = CE,
                   stringsAsFactors = FALSE)
  write_hs(file.path(.outdir, "S10_var_candidate.csv"), df)
  invisible(df)
}
mk10()

## ---- 10. markdown summary (console) -----------------------------------------------------------
line <- function(...) cat(paste0(..., "\n"), sep = "")
line("## Table 1 (VAR rows)")
line("")
line("| metric | p | train MuSPAD | test MuSPAD | train Baseline | test Baseline |")
line("|---|---|---|---|---|---|")
for (p in ps) for (mt in c("RMSE", "sMAPE")) {
  line(sprintf("| %s | %d | %s | %s | %s | %s |", mt, p,
               getv(paste0("T1;", mt, ";MuSPAD;",  p, ";train")),
               getv(paste0("T1;", mt, ";MuSPAD;",  p, ";test")),
               getv(paste0("T1;", mt, ";Baseline;", p, ";train")),
               getv(paste0("T1;", mt, ";Baseline;", p, ";test"))))
}
line("")
line("## Table S6 (VAR rows)")
line("")
line("| metric | p | train MuSPAD | test MuSPAD | train Baseline | test Baseline |")
line("|---|---|---|---|---|---|")
for (p in ps) for (mt in c("MAE", "MASE", "R2")) {
  line(sprintf("| %s | %d | %s | %s | %s | %s |", mt, p,
               getv(paste0("S6;", mt, ";MuSPAD;",  p, ";train")),
               getv(paste0("S6;", mt, ";MuSPAD;",  p, ";test")),
               getv(paste0("S6;", mt, ";Baseline;", p, ";train")),
               getv(paste0("S6;", mt, ";Baseline;", p, ";test"))))
}
line("")
line("## Table S8 (VAR rows)")
line("")
line("| model | p | intercept | slope |")
line("|---|---|---|---|")
for (m in models) for (p in ps) {
  line(sprintf("| %s | %d | %s | %s |", mlab(m), p,
               getv(paste0("S8;intercept;", mlab(m), ";", p)),
               getv(paste0("S8;slope;",     mlab(m), ";", p))))
}
line("")
line("## Table S9 (VAR rows, leave-one-origin-out)")
line("")
line("| group | p | RMSE MuSPAD | RMSE Baseline | sMAPE MuSPAD | sMAPE Baseline |")
line("|---|---|---|---|---|---|")
for (cfg in cfgs9) {
  g <- grp_of[[cfg$E]]
  for (p in ps) {
    line(sprintf("| %s | %d | %s | %s | %s | %s |", g, p,
                 getv(paste0("S9;RMSE;",  g, ";", p, ";MuSPAD")),
                 getv(paste0("S9;RMSE;",  g, ";", p, ";Baseline")),
                 getv(paste0("S9;sMAPE;", g, ";", p, ";MuSPAD")),
                 getv(paste0("S9;sMAPE;", g, ";", p, ";Baseline"))))
  }
}
line("")
line("## Table S10 (VAR rows, leave-one-origin-out TEST; candidate, unverified)")
line("")
line("| group | p | MAE M | MAE B | MASE M | MASE B | R2 M | R2 B |")
line("|---|---|---|---|---|---|---|---|")
for (cfg in cfgs10) {
  g <- grp_of[[cfg$E]]
  for (p in ps) {
    line(sprintf("| %s | %d | %s | %s | %s | %s | %s | %s |", g, p,
                 getv(paste0("S10;MAE;",  g, ";", p, ";MuSPAD")),
                 getv(paste0("S10;MAE;",  g, ";", p, ";Baseline")),
                 getv(paste0("S10;MASE;", g, ";", p, ";MuSPAD")),
                 getv(paste0("S10;MASE;", g, ";", p, ";Baseline")),
                 getv(paste0("S10;R2;",   g, ";", p, ";MuSPAD")),
                 getv(paste0("S10;R2;",   g, ";", p, ";Baseline"))))
  }
}

cat("\nTables written. S10_var_candidate.csv does not fully match the paper.\n")
