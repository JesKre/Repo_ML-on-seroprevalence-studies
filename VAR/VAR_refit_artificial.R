#!/usr/bin/env Rscript
# Portable artificial-data demonstration of the final fits in
# var_predictions_diff_splits_diff_seeds.R. No private paths or scaled CSVs.
args <- commandArgs(FALSE)
root <- dirname(normalizePath(sub('^--file=', '', grep('^--file=', args, value=TRUE)[1])))
infile <- Sys.getenv('VAR_DATA_FILE', file.path(root, '..', 'Artificial_datasets', 'Artificial_data_MuSPAD_time_aware.RData'))
outdir <- Sys.getenv('VAR_REFIT_DIR', root)
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
e <- new.env(); load(infile, envir=e)
if (!exists('df_ts_inc', envir=e, inherits=FALSE)) stop('Input must contain df_ts_inc.')
d <- as.data.frame(e$df_ts_inc)
if (nrow(d) != 408L || !all(c('Inzidenz','time_day') %in% names(d)))
  stop('Use the 408-row time-series artificial dataset, with Inzidenz and time_day.')
if (!all(vapply(d,is.numeric,logical(1))) || any(!is.finite(as.matrix(d))))
  stop('This demonstration requires complete numeric artificial data.')
dates <- as.Date('2020-01-01') + d$time_day
x <- as.matrix(d[,c('Inzidenz',setdiff(names(d),c('Inzidenz','time_day')))])
if ('--check-input' %in% commandArgs(TRUE)) {
  cat('Input accepted:',nrow(x),'rows;',ncol(x),'model columns. No models fitted.\n')
  quit(save='no')
}
if (!requireNamespace('bigtime',quietly=TRUE)) stop('Install bigtime before refitting.')
variants <- c('65percentData'=265L,'675percentData'=275L,
              'threefourthData'=306L,'80percentData'=326L,'full'=401L)
terms <- list()
long <- function(actual,predicted,time,set,seed) {
  rbind(data.frame(time,real_vs_pred='real',Incidence=actual,set,seed),
        data.frame(time,real_vs_pred='predicted',Incidence=predicted,set,seed))
}
for (v in names(variants)) {
  n <- variants[[v]]
  tr <- x[seq_len(n),,drop=FALSE]; te <- x[n+seq_len(7),,drop=FALSE]
  # Estimate all scaling parameters on training observations only.
  keep <- apply(tr,2,sd)>0
  if (!keep[1]) stop('Constant incidence target.')
  tr <- tr[,keep,drop=FALSE]; te <- te[,keep,drop=FALSE]
  z <- scale(tr); centers <- attr(z,'scaled:center'); scales <- attr(z,'scaled:scale')
  for (m in c('MuSPAD','onlyIncidence')) for (p in c(7L,14L,21L)) {
    Y <- if(m=='MuSPAD') z else z[,1,drop=FALSE]
    if(m=='onlyIncidence') colnames(Y) <- 'Incidence'
    rows <- list()
    for (seed in 1:10) {
      cat(v,m,'p=',p,'seed=',seed,'\n')
      set.seed(seed)
      fit <- if(m=='MuSPAD') bigtime::sparseVAR(Y=Y,p=p,h=7,
          selection='cv',VARpen='L1',cvcut=.8) else
          bigtime::sparseVAR(Y=Y,p=p,h=7,selection='cv')
      if (is.null(fit$Phihat)) stop('Fit has no coefficients.')
      f <- as.numeric(stats::fitted(fit)[,1])*scales[1]+centers[1]
      pred <- as.numeric(bigtime::recursiveforecast(mod=fit,h=7)$fcst[,1])*scales[1]+centers[1]
      # Preserve the original paper scoring window (six-day shifted train fits).
      idx <- (p+7):n
      stopifnot(length(f)==length(idx),length(pred)==7L)
      rows[[seed]] <- rbind(long(tr[idx,1],f,dates[idx],'Training',seed),
          long(te[,1],pred,dates[n+seq_len(7)],'Test',seed))
      if(seed==1) terms[[paste(m,p,v,sep=';')]] <- list(f=f,a=tr[idx,1],
          ap=tr[(p+6):(n-1),1],n=n,K=length(idx),s=unname(scales[1]),c0=unname(centers[1]))
    }
    obj <- if(m=='MuSPAD') 'graph_VAR_7day_M' else 'graph_VAR_7day_bl'
    env <- new.env(); env[[obj]] <- do.call(rbind,rows)
    label <- if(m=='MuSPAD') 'MuSPAD' else 'baseline'
    seg <- if(v=='full') '' else paste0('_',v)
    save(list=obj,envir=env,file=file.path(outdir,paste0('VAR_7day_',label,seg,'_p',p,'_seeds.RData')))
  }
}
save(terms,file=file.path(outdir,'VAR_train_terms.RData'))
cat('Artificial-data predictions written to',outdir,'\n')
