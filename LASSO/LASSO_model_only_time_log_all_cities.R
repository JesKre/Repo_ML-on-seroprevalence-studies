


library(tidyverse)
library(glmnet)
library(missForest)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n=1)
load(file)

df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day=="168"),]

head(colnames(df_ts_inc))
tail(colnames(df_ts_inc))

dim(df_ts_inc)

# use logarithm on incidences
df_ts_inc[,1] <- log(df_ts_inc[,1])




predictor <- as.data.frame(cbind(0,df_ts_inc[,2]))


traindat <- predictor[1:(dim(df_ts_inc)[1]/10*9),]

traindat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),1]

testdat <- predictor[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),]

testdat_y <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),1]


lambda_seq <- 10^seq(3, -3, by = -.1)

cv_output <- cv.glmnet(x = as.matrix(traindat), y = traindat_y,
                       alpha = 1, lambda = lambda_seq, 
                       nfolds = 10)

# identifying best lambda
best_lam <- cv_output$lambda.min

#produce plot of test MSE by lambda value
plot(cv_output)

# Rebuilding the model with best lambda value identified
lasso_best <- glmnet(traindat, traindat_y, alpha = 1, lambda = best_lam)

coef(lasso_best, s = "best_lam")


########## 1.)

# Predict on test data
pred_LASSO <- exp(glmnet::predict.glmnet(lasso_best, s = best_lam, newx = as.matrix(testdat)))
real <- exp(testdat_y)

time <- 
  as.Date("2020-01-01") +
  testdat$V2 +
  7

(real_vs_pred <- data.frame(real = real, predicted = as.numeric(pred_LASSO), time = time))

# Mean squared error:

library(Metrics)

(mse_test <- mse(real_vs_pred$real, real_vs_pred$predicted))

# Mean squared relative error:

real_vs_pred_error <- real_vs_pred %>%
  mutate(relative_error = (real-predicted)/real,
         squared_relative_error = relative_error^2)

mean(real_vs_pred_error$squared_relative_error)


r_p_graph_test <- real_vs_pred %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence")%>%
  mutate(set = "Test")



########## 2.)

# "Predict on" training data
pred_LASSO <- exp(glmnet::predict.glmnet(lasso_best, s = best_lam, newx = as.matrix(traindat)))
real <- exp(traindat_y)

time <- 
  as.Date("2020-01-01") +
  traindat$V2 +
  7

(real_vs_pred <- data.frame(real = real, predicted = as.numeric(pred_LASSO), time = time))

# Mean squared error:

(mse_train <- mse(real_vs_pred$real, real_vs_pred$predicted))

# Mean squared relative error:

real_vs_pred_error <- real_vs_pred %>%
  mutate(relative_error = (real-predicted)/real,
         squared_relative_error = relative_error^2)

mean(real_vs_pred_error$squared_relative_error)



r_p_graph_train <- real_vs_pred %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence")%>%
  mutate(set = "Training")


graph_LASSO_7day_BL <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values")
save(graph_LASSO_7day_BL, file = "LASSO_7day_baseline.RData")



setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LASSO\\Graphics")

png(file=paste0("LASSO_only_time_log_",Sys.Date(),".png"), width=38, height=34, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw(base_size = 18) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LASSO")) +
  theme(legend.title=element_text(size=22), 
        legend.text=element_text(size=18))

dev.off()


mse_LASSO_onlyTime <- data.frame(mse_train, mse_test)

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LASSO")
write.csv2(mse_LASSO_onlyTime, "mse_LASSO_onlyTime.csv", row.names = FALSE)


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Single_graphics")

png(file=paste0("LASSO_only_time_log_",Sys.Date(),".png"), width=38, height=34, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw(base_size = 18) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LASSO")) +
  theme(legend.title=element_text(size=22), 
        legend.text=element_text(size=18))

dev.off()


