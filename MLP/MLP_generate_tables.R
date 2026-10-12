# Generate the MLP rows of Tables 1, S3, S4 and S5 from seed prediction objects.
# Base R only. Original fitting/explainability scripts are retained separately.
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE))
stopifnot(length(script_file)==1)
ROOT <- normalizePath(dirname(script_file),mustWork=TRUE)
PRED_DIR <- Sys.getenv("MLP_PRED_DIR",unset=file.path(dirname(ROOT),"Compare_models","Graphics","Predicted_values"))

args <- commandArgs(trailingOnly=TRUE)
if(length(args) && !identical(args,"--from-runs")) stop("Usage: Rscript MLP_generate_tables.R [--from-runs]")
if(identical(args,"--from-runs")) {
  RUN_DIR <- Sys.getenv("MLP_RUN_DIR",unset=ROOT)
  sp <- read.csv(file.path(RUN_DIR,"seed_predictions_k2_36cells.csv"),sep=";")
  tf <- read.csv(file.path(RUN_DIR,"mlp_trainfull_predictions_k2_L4E30_2026-09-30.csv"),sep=";")
  st <- read.csv(file.path(RUN_DIR,"series_test.csv"),sep=";")
  tv <- read.csv(file.path(RUN_DIR,"series_trainval.csv"),sep=";")
  predcols <- paste0("pred",1:27)
  stopifnot(nrow(st)==27,nrow(tv)==251,all(predcols %in% names(sp)))
  graphs <- list()
  for(mt in c("missForest","onlyTime")) {
    test <- sp[sp$Model_Type==mt & sp$Layers==4 & sp$Epochs==30,]
    stopifnot(nrow(test)==10,setequal(test$seed,1:10),!anyDuplicated(test$seed))
    rows <- list()
    for(seed in 1:10) {
      tr <- tf[tf$Model_Type==mt & tf$seed==seed & tf$split=="trainfull",]
      tr <- tr[order(tr$i),]
      stopifnot(nrow(tr)==251,identical(as.integer(tr$i),1:251),all(tr$y==tv$y))
      for(split in c("Training","Test")) {
        y <- if(split=="Training")tv$y else st$y
        pred <- if(split=="Training")tr$pred else as.numeric(test[test$seed==seed,predcols])
        time <- as.Date("2020-01-01")+7+if(split=="Training")tv$time_day else st$time_day
        rows[[length(rows)+1L]] <- data.frame(time=time,real_vs_pred="real",Incidence=y,set=split,seed=seed)
        rows[[length(rows)+1L]] <- data.frame(time=time,real_vs_pred="predicted",Incidence=pred,set=split,seed=seed)
      }
    }
    graphs[[mt]] <- do.call(rbind,rows)
  }
  dir.create(PRED_DIR,recursive=TRUE,showWarnings=FALSE)
  graph_MLP_7day_M_seeds <- graphs$missForest
  graph_MLP_7day_BL_seeds <- graphs$onlyTime
  save(graph_MLP_7day_M_seeds,file=file.path(PRED_DIR,"MLP_7day_MuSPAD_seeds.RData"))
  save(graph_MLP_7day_BL_seeds,file=file.path(PRED_DIR,"MLP_7day_baseline_seeds.RData"))
}

read_predictions <- function(filename,object) {
  path <- file.path(PRED_DIR,filename)
  if(!file.exists(path)) stop("Missing seed predictions: ",path,". See README.md.")
  env <- new.env();load(path,envir=env)
  if(!exists(object,envir=env,inherits=FALSE)) stop("Missing object: ",object)
  d <- as.data.frame(env[[object]])
  required <- c("time","real_vs_pred","Incidence","set","seed")
  stopifnot(all(required %in% names(d)),setequal(d$seed,1:10),
    setequal(d$set,c("Training","Test")),setequal(d$real_vs_pred,c("real","predicted")),
    all(is.finite(d$Incidence)),!anyNA(d$time),nrow(d)==5560)
  out <- list()
  for(split in c("Training","Test")) {
    matrices <- list();observations <- NULL;dates <- NULL
    for(seed in 1:10) {
      z <- d[d$seed==seed & d$set==split,]
      r <- z[z$real_vs_pred=="real",c("time","Incidence")]
      p <- z[z$real_vs_pred=="predicted",c("time","Incidence")]
      stopifnot(!anyDuplicated(r$time),!anyDuplicated(p$time),setequal(r$time,p$time))
      pairs <- merge(r,p,by="time",suffixes=c("_real","_pred"))
      pairs <- pairs[order(pairs$time),]
      stopifnot(nrow(pairs)==if(split=="Training")251 else 27)
      if(seed==1) {observations<-pairs$Incidence_real;dates<-pairs$time}
      stopifnot(identical(dates,pairs$time),identical(observations,pairs$Incidence_real))
      matrices[[seed]] <- pairs$Incidence_pred
    }
    out[[split]] <- list(real=observations,time=dates,pred=do.call(rbind,matrices))
  }
  out
}
data <- list(MuSPAD=read_predictions("MLP_7day_MuSPAD_seeds.RData","graph_MLP_7day_M_seeds"),
             baseline=read_predictions("MLP_7day_baseline_seeds.RData","graph_MLP_7day_BL_seeds"))
for(split in c("Training","Test")) {
  stopifnot(identical(data$MuSPAD[[split]]$time,data$baseline[[split]]$time),
            identical(data$MuSPAD[[split]]$real,data$baseline[[split]]$real))
}
mase_den <- mean(abs(diff(data$MuSPAD$Training$real)))
stopifnot(is.finite(mase_den),mase_den>0)
metrics <- function(pred,y) {
  e <- pred-y
  c(RMSE=sqrt(mean(e^2)),sMAPE=mean(2*abs(e)/(abs(pred)+abs(y))),
    MAE=mean(abs(e)),MASE=mean(abs(e))/mase_den,
    R2=1-sum(e^2)/sum((y-mean(y))^2))
}
# Same paired bootstrap draws as the original supplied bootstrap script.
gen_indices <- function(n,b,B,seed) {
  set.seed(seed);K<-ceiling(n/b)
  starts<-matrix(sample.int(n-b+1L,B*K,replace=TRUE),nrow=B,ncol=K)
  offs<-matrix(seq_len(b)-1L,nrow=B,ncol=b,byrow=TRUE)
  idx<-matrix(0L,nrow=B,ncol=K*b)
  for(k in seq_len(K)) idx[,(k-1L)*b+seq_len(b)]<-starts[,k]+offs
  idx[,seq_len(n),drop=FALSE]
}
B<-10000L;idx_mb<-gen_indices(27L,7L,B,42);idx_iid<-gen_indices(27L,1L,B,42)
bootstrap <- function(idx,y,pred) {
  yd<-matrix(y[idx],nrow=nrow(idx));pd<-matrix(pred[idx],nrow=nrow(idx));e<-yd-pd
  values<-list(RMSE=sqrt(rowMeans(e^2)),sMAPE=rowMeans(2*abs(e)/(abs(yd)+abs(pd))),
    MAE=rowMeans(abs(e)),MASE=rowMeans(abs(e))/mase_den,
    R2=1-rowSums(e^2)/rowSums((yd-rowMeans(yd))^2))
  lapply(values,quantile,probs=c(.025,.975),type=7)
}
calibration <- function(x,y,model) {
  n<-length(y);fit<-lm(y~x);cf<-summary(fit)$coefficients;ci<-confint(fit)
  ss_id<-sum((y-x)^2);ss_ols<-sum(residuals(fit)^2)
  Fj<-((ss_id-ss_ols)/2)/(ss_ols/(n-2))
  data.frame(model=model,intercept=cf[1,1],intercept_ci_lo=ci[1,1],intercept_ci_hi=ci[1,2],
    slope=cf[2,1],slope_ci_lo=ci[2,1],slope_ci_hi=ci[2,2],
    f_joint=Fj,p_joint=pf(Fj,2,n-2,lower.tail=FALSE),n=n)
}
rows<-list();stability<-list();cal<-list()
for(model in names(data)) {
  tr<-data[[model]]$Training;te<-data[[model]]$Test
  training<-sapply(1:10,function(seed)metrics(tr$pred[seed,],tr$real))
  perseed_test<-sapply(1:10,function(seed)metrics(te$pred[seed,],te$real))
  pbar<-colMeans(te$pred);point<-metrics(pbar,te$real)
  mb<-bootstrap(idx_mb,te$real,pbar);iid<-bootstrap(idx_iid,te$real,pbar)
  for(metric in names(point)) {
    rows[[length(rows)+1L]]<-data.frame(model=model,metric=metric,
      training_mean=mean(training[metric,]),training_sd=sd(training[metric,]),
      test_value=point[metric],test_ci_lo=mb[[metric]][1],test_ci_hi=mb[[metric]][2],
      test_iid_lo=iid[[metric]][1],test_iid_hi=iid[[metric]][2])
    stability[[length(stability)+1L]]<-data.frame(model=model,metric=metric,
      training_mean=mean(training[metric,]),training_sd=sd(training[metric,]),
      test_value=point[metric],test_sd=sd(perseed_test[metric,]))
  }
  cal[[model]]<-calibration(pbar,te$real,model)
}
results<-do.call(rbind,rows)
write.csv(results[results$metric %in% c("RMSE","sMAPE"),],file.path(ROOT,"MLP_table1.csv"),row.names=FALSE)
write.csv(results[results$metric %in% c("MAE","MASE","R2"),],file.path(ROOT,"MLP_tableS3.csv"),row.names=FALSE)
write.csv(do.call(rbind,stability),file.path(ROOT,"MLP_tableS4.csv"),row.names=FALSE)
write.csv(do.call(rbind,cal),file.path(ROOT,"MLP_tableS5.csv"),row.names=FALSE)
cat("Wrote the MLP rows of Tables 1, S3, S4 and S5 directly to",ROOT,"\n")
