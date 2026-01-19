
#------------------------------------------------------------------------------
#---- Time series modelling with bigtime R package ----------------------------
#------------------------------------------------------------------------------

# Obtaining sparse estimates of large time series models

library(bigtime)
library(glmnet)
library(Metrics)
library(tidyverse)

# Dataset

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missForest"), n=1)
load(file)
# df_ts_all_inc_imp

# df_ts_all_inc_imp <- df_ts_all_inc_imp[1:100, 1:10]

# Predict 7 days ahead 

test_indices <- (dim(df_ts_all_inc_imp)[1]-6):dim(df_ts_all_inc_imp)[1]

traindat_1 <- df_ts_all_inc_imp[-test_indices,1]

# Scale the training data
traindat <- scale(traindat_1)

scaled_train_mat <- as.matrix(traindat)
colnames(scaled_train_mat) <- "Incidence"


testdat_1 <- df_ts_all_inc_imp[test_indices,1]

# Use the scaling parameters from the training data to scale the test data
testdat <- scale(testdat_1, 
                 center = attr(traindat, "scaled:center"), 
                 scale = attr(traindat, "scaled:scale"))

scaled_test_mat <- as.matrix(testdat)
colnames(scaled_test_mat) <- "Incidence"

# identical(scaled_test_mat, scale(scaled_test_mat))

# Fit the model

# Sparse Estimation of the Vector AutoRegressive (VAR) Model
# using time series cross-validation

p <- 21
h <- 7

VARfit <- sparseVAR(Y=scaled_train_mat, selection = "cv", p=p, h=h)


(Lhat <- lagmatrix(fit=VARfit)) # get estimated lagmatrix

# Prediction
# get seven-step ahead forecasts

scale_center <- attr(traindat, "scaled:center")[[1]] 
scale_scale <- attr(traindat, "scaled:scale")[[1]]

# save
result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models"
file <- paste0(result_path, "/model_VAR_onlyIncidence_p",p,"_h",h,"_",Sys.Date(),".RData")
save(VARfit, predicted, scale_center, scale_scale, 
     scaled_train_mat, scaled_test_mat, 
     file = file)

# load
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models")
(file <- tail(list.files(pattern = paste0("model_VAR_onlyIncidence_p",p,"_h",h)), n=1))
load(file = file)

predicted <- recursiveforecast(mod = VARfit, h=h)$fcst[,1] * scale_scale + scale_center

real <- scaled_test_mat[,1] * scale_scale + scale_center

time <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[test_indices,2]

(real_vs_pred_test <- data.frame(real = real, predicted = predicted, time = time))

(mse_test <- mse(real, predicted))
# 929

diagnostics_plot(VARfit, variable = "Incidence")

fitted(VARfit)[,"Incidence"]


real_train <- scaled_train_mat[(p+h):dim(scaled_train_mat)[1],1] * scale_scale + scale_center
predicted_train <- fitted(VARfit)[,"Incidence"] * scale_scale + scale_center

(mse_train <- mse(real_train, predicted_train))
# 1478

mse_VAR_onlyIncidence <- data.frame(mse_train, mse_test)

# Save results
# setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")
# write.csv2(mse_VAR_onlyIncidence, paste0("mse_VAR_onlyIncidence_p",p,"_h",h,".csv"), row.names = FALSE)


time_train <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[(p+h):(test_indices[1]-1),2]

(real_vs_pred_train <- data.frame(real = real_train, predicted = predicted_train, time = time_train))


#------------------------------------------------------------------------------
# Graphics

r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")


graph_VAR_7day_bl <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values")
save(graph_VAR_7day_bl, file = paste0("VAR_onlyIncidence_p",p,"_h",h,".RData"))


ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  ylim(0, 375) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="Autoregressive"))


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Single_graphics")

png(file=paste0("ggplot_forecast_bigtime_only_incidence_p",p,"_h",h,"_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  ylim(0, 375) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="Autoregressive"))

dev.off()


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")

png(file=paste0("ggplot_forecast_bigtime_only_incidence_p",p,"_h",h,"_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  ylim(0, 375) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="Autoregressive"))

dev.off()



