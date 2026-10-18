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