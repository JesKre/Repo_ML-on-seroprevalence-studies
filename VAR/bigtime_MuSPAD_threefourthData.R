
#------------------------------------------------------------------------------
#---- Time series modelling with bigtime R package ----------------------------
#------------------------------------------------------------------------------

# Obtaining sparse estimates of large time series models

library(missForest)
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

scaled_train_mat <- as.matrix(read.csv("MuSPAD_traindat_LSTM_threefourthData.csv")[,-1])

scaled_train_mat_v <- as.matrix(read.csv("MuSPAD_traindat_v_LSTM_threefourthData.csv")[,-1])

scaled_val_mat <- as.matrix(read.csv("MuSPAD_valdat_LSTM_threefourthData.csv")[,-1])

scaled_test_mat <- as.matrix(read.csv("MuSPAD_testdat_LSTM_threefourthData.csv")[,-1])



#------------------------------------------------------------------------------
# Validation

# Fit the model

# Sparse Estimation of the Vector AutoRegressive (VAR) Model
# using time series cross-validation

library(purrr)
library(dplyr)

param_grid <- expand.grid(
  VARpen = c("HLag", "L1"),
  selection = c("cv", "bic", "aic"),
  cvcut = c(0.8, 0.9),
  stringsAsFactors = FALSE
)

VARpen = "HLag"
selection = "cv"
cvcut = 0.8


evaluate_model <- function(VARpen, selection, cvcut, scaled_train_mat_v, scaled_val_mat, scale_inc, center_inc, p, h) {
  
  # Fit the model
  VARfit <- sparseVAR(Y = scaled_train_mat_v, cvcut = cvcut, p = 7, selection = selection, VARpen = VARpen, h = 7)
  
  # Forecast validation period
  pred_val <- tryCatch({
    recursiveforecast(mod = VARfit, h = 7)$fcst[, 1] * scale_inc + center_inc
  }, error = function(e) return(rep(NA, 7)))
  
  real_val <- scaled_val_mat[, 1] * scale_inc + center_inc
  mse_val <- mse(real_val, pred_val)
  
  # Training predictions (aligning with forecastable period)
  if (is.null(VARfit$Phihat)) {
    mse_train <- NA
  } else {
    n_lags <- dim(VARfit$Phihat)[2] / ncol(scaled_train_mat_v)
    start_idx <- n_lags + h
    real_train <- scaled_train_mat_v[start_idx:nrow(scaled_train_mat_v), 1] * scale_inc + center_inc
    predicted_train <- fitted(VARfit)[, "Inzidenz"] * scale_inc + center_inc
    mse_train <- mse(real_train, predicted_train)
  }
  
  # Return results
  tibble(
    VARpen = VARpen,
    selection = selection,
    cv_cut = cvcut,
    mse_val = mse_val,
    mse_train = mse_train
  )
}


scale_inc <- scale_scale_train_v[which(names(scale_scale_train_v)=="Inzidenz")]
center_inc <- scale_center_train_v[which(names(scale_center_train_v)=="Inzidenz")]


p <- 7
h <- 7

library(furrr)
plan(multisession)

# results <- future_pmap_dfr(param_grid, evaluate_model)

results <- future_pmap_dfr(
  .l = param_grid,
  .f = ~evaluate_model(..1, ..2, ..3, scaled_train_mat_v, scaled_val_mat, scale_inc, center_inc, p, h),
  .options = furrr_options(seed = TRUE)
)

results

res_ord <- results %>% arrange(mse_val)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")

write.csv2(res_ord, paste0("Hyperparameter_validation_VAR_threefourthData_p",p,"_h",h,".csv"))


#------------------------------------------------------------------------------
# Testing

# Fit the model

# Sparse Estimation of the Vector AutoRegressive (VAR) Model
# using time series cross-validation

h <- 7
p <- 21

VARfit <- sparseVAR(Y=scaled_train_mat, p = p, selection = "cv", VARpen = "L1", cvcut = 0.8, h = h)


Lhat <- lagmatrix(fit=VARfit) # get estimated lagmatrix

# save
result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models\\"
file <- paste0(result_path, "/model_VAR_threefourthData_p",p,"_h",h,"_",Sys.Date(),".RData")
save(VARfit, scale_center_train, scale_scale_train, 
     scaled_train_mat, scaled_test_mat, 
     file = file)


# load
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR\\Models")
(file <- tail(list.files(pattern = paste0("model_VAR_threefourthData_p",p,"_h",h)), n=1))
load(file = file)


# Prediction
# get seven-step ahead forecasts

scale_inc <- scale_scale_train[which(names(scale_scale_train)=="Inzidenz")]
center_inc <- scale_center_train[which(names(scale_center_train)=="Inzidenz")]

predicted <- recursiveforecast(mod = VARfit, h=7)$fcst[,1] * scale_inc + center_inc

real <- scaled_test_mat[,1] * scale_inc + center_inc

time_x <- as.Date("2020-01-01") + df_ts_all_inc_imp[test_indices,2]

(real_vs_pred_test <- data.frame(real = real, predicted = predicted, time = time_x))

(mse_test <- mse(real, predicted))
# 1338

diagnostics_plot(VARfit, variable = "Inzidenz")

fitted(VARfit)[,"Inzidenz"]

# dev.new()
# lagmatrix(fit=VARfit, returnplot=TRUE)

real_train <- scaled_train_mat[(p+h):dim(scaled_train_mat)[1],1] * scale_inc + center_inc
predicted_train <- fitted(VARfit)[,"Inzidenz"] * scale_inc + center_inc

(mse_train <- mse(real_train, predicted_train))
# 1142

time_train <- 
  as.Date("2020-01-01") +
  df_ts_all_inc_imp[(p+h):(test_indices[1]-1),2]

(real_vs_pred_train <- data.frame(real = real_train, predicted = predicted_train, time = time_train))


mse_VAR_MuSPAD <- data.frame(mse_train, mse_test)

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/VAR")
write.csv2(mse_VAR_MuSPAD, paste0("mse_VAR_MuSPAD_threefourthData_p",p,"_h",h,".csv"), row.names = FALSE)



#------------------------------------------------------------------------------
# Graphics

r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")

graph_VAR_7day_M <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data")
save(graph_VAR_7day_M, file = paste0("VAR_7day_MuSPAD_p",sprintf("%02d", p),"_h",h,".RData"))

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

png(file=paste0("ggplot_forecast_bigtime_muspad_varsubsetbasic_threefourthData_p",p,"_h",h,"_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

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



