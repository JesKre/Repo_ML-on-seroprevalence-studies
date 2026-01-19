
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

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

(file <- tail(list.files(pattern = "scaling_parameters_threefourthData_"), n=1))
load(file = file)

scaled_train_mat <- as.matrix(read.csv("MuSPAD_traindat_LSTM_threefourthData.csv")[,2])

scaled_train_mat_v <- as.matrix(read.csv("MuSPAD_traindat_v_LSTM_threefourthData.csv")[,2])

scaled_val_mat <- as.matrix(read.csv("MuSPAD_valdat_LSTM_threefourthData.csv")[,2])

scaled_test_mat <- as.matrix(read.csv("MuSPAD_testdat_LSTM_threefourthData.csv")[,2])



# Fit the model

# Sparse Estimation of the Vector AutoRegressive (VAR) Model
# using time series cross-validation

p <- 21
h <- 7

VARfit <- sparseVAR(Y=scaled_train_mat, selection = "cv", p=p, h=h)


(Lhat <- lagmatrix(fit=VARfit)) # get estimated lagmatrix

# Prediction
# get seven-step ahead forecasts

scale_inc <- scale_scale_train[which(names(scale_scale_train)=="Inzidenz")]
center_inc <- scale_center_train[which(names(scale_center_train)=="Inzidenz")]

predicted <- recursiveforecast(mod = VARfit, h=h)$fcst[,1] * scale_inc + center_inc

real <- scaled_test_mat[,1] * scale_inc + center_inc


time <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[test_indices,2]

(real_vs_pred_test <- data.frame(real = real, predicted = predicted, time = time))

(mse_test <- mse(real, predicted))
# 139

diagnostics_plot(VARfit)

fitted(VARfit)


real_train <- scaled_train_mat[(p+h):dim(scaled_train_mat)[1],] * scale_inc + center_inc
predicted_train <- fitted(VARfit) * scale_inc + center_inc

time_train <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[(p+h):(test_indices[1]-1),2]

(real_vs_pred_train <- data.frame(real = real_train, predicted = predicted_train, time = time_train))


(mse_train <- mse(real_train, predicted_train))
# 2557.674

mse_VAR_onlyIncidence <- data.frame(mse_train, mse_test)

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")
write.csv2(mse_VAR_onlyIncidence, paste0("mse_VAR_onlyIncidence_threefourthData_p",p,"_h",h,".csv"), row.names = FALSE)

# save
result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models\\"
file <- paste0(result_path, "/model_VAR_onlyIncidence_threefourthData_p",p,"_h",h,"_",Sys.Date(),".RData")
save(VARfit, predicted,
     mse_VAR_onlyIncidence, center_inc, scale_inc, 
     scaled_train_mat, scaled_test_mat, 
     file = file)


#------------------------------------------------------------------------------
# Graphics

p <- 21
h <- 7

# load
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models")
(file <- tail(list.files(pattern = paste0("model_VAR_onlyIncidence_threefourthData_p",p,"_h",h,"_")), n=1))
load(file = file)

real <- scaled_test_mat[,1] * scale_inc + center_inc

time <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[test_indices,2]

(real_vs_pred_test <- data.frame(real = real, predicted = predicted, time = time))

real_train <- scaled_train_mat[(p+h):dim(scaled_train_mat)[1],] * scale_inc + center_inc
predicted_train <- fitted(VARfit) * scale_inc + center_inc

time_train <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[(p+h):(test_indices[1]-1),2]

(real_vs_pred_train <- data.frame(real = real_train, predicted = predicted_train, time = time_train))


r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")


graph_VAR_7day_bl <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data")
save(graph_VAR_7day_bl, file = paste0("VAR_onlyIncidence_p",sprintf("%02d", p),"_h",h,".RData"))


ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  ylim(0, 375) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="Autoregressive"))


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")

png(file=paste0("ggplot_forecast_bigtime_only_incidence_threefourthData_p",p,"_h",h,"_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

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



