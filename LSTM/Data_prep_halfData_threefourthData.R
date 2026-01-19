
library(missForest)
library(tidyverse)

# Dataset

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

# Predict 7 days ahead 

# Predict 7 days ahead 

test_indices <- (dim(df_ts_all_inc)[1]/2+1):(dim(df_ts_all_inc)[1]/2+7)

val_indices <- (dim(df_ts_all_inc)[1]/2-6):(dim(df_ts_all_inc)[1]/2)

train_indices <- 1:(dim(df_ts_all_inc)[1]/2)

train_indices_v <- 1:(dim(df_ts_all_inc)[1]/2-5)

# impute training data
traindat_1 <- missForest(df_ts_all_inc[train_indices,-2])$ximp


# Scale the training data
traindat_scale <- scale(traindat_1)

traindat <- as.data.frame(traindat_scale)

which_na <- apply(traindat, 2, function(x) length(which(is.na(x))))

traindat <- traindat[,(which_na == 0)]

scale_center_train <- attr(traindat_scale, "scaled:center")
scale_scale_train <- attr(traindat_scale, "scaled:scale")

# -> training into train and val

traindat_scale_v <- scale(traindat_1[train_indices_v,])

traindat_v <- as.data.frame(traindat_scale_v)

which_na <- apply(traindat_v, 2, function(x) length(which(is.na(x))))

traindat_v <- traindat_v[,(which_na == 0)]

scale_center_train_v <- attr(traindat_scale_v, "scaled:center")
scale_scale_train_v <- attr(traindat_scale_v, "scaled:scale")

valdat_scale <- scale(traindat_1[val_indices,], 
                      center = scale_center_train_v,
                      scale = scale_scale_train_v)

valdat <- as.data.frame(valdat_scale)

which_na <- apply(valdat, 2, function(x) length(which(is.na(x))))

valdat <- valdat[,(which_na == 0)]


# impute test data
testdat_1 <- missForest(df_ts_all_inc[test_indices,-2])$ximp


# Use the scaling parameters from the training data to scale the test data
testdat_scale <- scale(testdat_1, 
                       center = scale_center_train, 
                       scale = scale_scale_train)

testdat <- as.data.frame(testdat_scale)

which_na <- apply(testdat, 2, function(x) length(which(is.na(x))))

testdat <- testdat[,(which_na == 0)]


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
write.csv(traindat, "MuSPAD_traindat_LSTM_halfData.csv")
write.csv(testdat, "MuSPAD_testdat_LSTM_halfData.csv")
write.csv(traindat_v, "MuSPAD_traindat_v_LSTM_halfData.csv")
write.csv(valdat, "MuSPAD_valdat_LSTM_halfData.csv")

result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\"
file <- paste0(result_path, "scaling_parameters_halfData_",Sys.Date(),".RData")

save(scale_center_train, scale_scale_train, scale_center_train_v, scale_scale_train_v,
     test_indices, val_indices,
     file = file)

#------------------------------------------------------------------------------

# 3/4 Data


library(missForest)
library(tidyverse)

# Dataset

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

# Predict 7 days ahead 

# Predict 7 days ahead 

test_indices <- (dim(df_ts_all_inc)[1]*3/4+1):(dim(df_ts_all_inc)[1]*3/4+7)

val_indices <- (dim(df_ts_all_inc)[1]*3/4-6):(dim(df_ts_all_inc)[1]*3/4)

train_indices <- 1:(dim(df_ts_all_inc)[1]*3/4)

train_indices_v <- 1:(dim(df_ts_all_inc)[1]*3/4-5)

# impute training data
traindat_1 <- missForest(df_ts_all_inc[train_indices,-2])$ximp


# Scale the training data
traindat_scale <- scale(traindat_1)

traindat <- as.data.frame(traindat_scale)

which_na <- apply(traindat, 2, function(x) length(which(is.na(x))))

traindat <- traindat[,(which_na == 0)]

scale_center_train <- attr(traindat_scale, "scaled:center")
scale_scale_train <- attr(traindat_scale, "scaled:scale")

# -> training into train and val

traindat_scale_v <- scale(traindat_1[train_indices_v,])

traindat_v <- as.data.frame(traindat_scale_v)

which_na <- apply(traindat_v, 2, function(x) length(which(is.na(x))))

traindat_v <- traindat_v[,(which_na == 0)]

scale_center_train_v <- attr(traindat_scale_v, "scaled:center")
scale_scale_train_v <- attr(traindat_scale_v, "scaled:scale")

valdat_scale <- scale(traindat_1[val_indices,], 
                      center = scale_center_train_v,
                      scale = scale_scale_train_v)

valdat <- as.data.frame(valdat_scale)

which_na <- apply(valdat, 2, function(x) length(which(is.na(x))))

valdat <- valdat[,(which_na == 0)]


# impute test data
testdat_1 <- missForest(df_ts_all_inc[test_indices,-2])$ximp


# Use the scaling parameters from the training data to scale the test data
testdat_scale <- scale(testdat_1, 
                       center = scale_center_train, 
                       scale = scale_scale_train)

testdat <- as.data.frame(testdat_scale)

which_na <- apply(testdat, 2, function(x) length(which(is.na(x))))

testdat <- testdat[,(which_na == 0)]


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
write.csv(traindat, "MuSPAD_traindat_LSTM_threefourthData.csv")
write.csv(testdat, "MuSPAD_testdat_LSTM_threefourthData.csv")
write.csv(traindat_v, "MuSPAD_traindat_v_LSTM_threefourthData.csv")
write.csv(valdat, "MuSPAD_valdat_LSTM_threefourthData.csv")

result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\"
file <- paste0(result_path, "scaling_parameters_threefourthData_",Sys.Date(),".RData")

save(scale_center_train, scale_scale_train, scale_center_train_v, scale_scale_train_v,
     test_indices, val_indices,
     file = file)





#------------------------------------------------------------------------------
# Graphics


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
traindat <- read.csv("MuSPAD_traindat_LSTM_threefourthData.csv")[,-1]
testdat <- read.csv("MuSPAD_testdat_LSTM_threefourthData.csv")[,-1]
traindat_v <- read.csv("MuSPAD_traindat_v_LSTM_threefourthData.csv")[,-1]
valdat <- read.csv("MuSPAD_valdat_LSTM_threefourthData.csv")[,-1]


library(missForest)
library(tidyverse)
library(Metrics)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
(file <- tail(list.files(pattern = "scaling_parameters_threefourthData_[1-9]"), n=1))

load(file = file)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

prediction <- 7

make_Graph <- function(model){
  setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results/Python_Original/results_h7_threefourthData")

if(model == "MuSPAD"){
  f <- "_0p2_AB_noreset"
}else if(model == "onlyIncidence"){
  f <- "_only_incidence"
}

# real_vs_pred_train <- read.csv("train_predictions_0p2_AB_noreset_tplus7.csv")
real_vs_pred_train <- read.csv(paste0("train_predictions",f,"_tplus7.csv"))

scale_center_inc <- scale_center_train[which(names(scale_center_train) == "Inzidenz")]

scale_scale_inc <- scale_scale_train[which(names(scale_scale_train) == "Inzidenz")]

#str(real_vs_pred_train)

library(magrittr)

real_vs_pred_train$predicted %<>% 
  str_remove_all("\\[") %<>% 
  str_remove_all("\\]") %<>% 
  as.numeric()

#str(real_vs_pred_train)


back_scaling <- function(x) x * scale_scale_inc + scale_center_inc

real_vs_pred_train <- data.frame(sapply(real_vs_pred_train, back_scaling))


(mse_train <- mse(real_vs_pred_train$real, real_vs_pred_train$predicted))


t_train <- df_ts_all_inc[1:(test_indices[1]-1),2]

real_vs_pred_train$time <- 
  as.Date("2020-01-01") +
  t_train[(prediction+1):(length(t_train))]

r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")


# real_vs_pred_test <- read.csv("test_predictions_0p2_AB_noreset_tplus7.csv")
real_vs_pred_test <- read.csv(paste0("test_predictions",f,"_tplus7.csv"))

#dim(real_vs_pred_test)

real_vs_pred_test$predicted %<>% 
  str_remove_all("\\[") %<>% 
  str_remove_all("\\]") %<>% 
  as.numeric()

#str(real_vs_pred_test)

real_vs_pred_test <- data.frame(sapply(real_vs_pred_test, back_scaling))

(mse_test <- mse(real_vs_pred_test$real, real_vs_pred_test$predicted))

# mse_LSTM_MuSPAD <- data.frame(mse_train, mse_test)
mse_LSTM <- data.frame(mse_train, mse_test)

if(model == "MuSPAD"){
  g <- "_MuSPAD"
}else if(model == "onlyIncidence"){
  g <- "_onlyIncidence"
}

# Save mse results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")
# write.csv2(mse_LSTM_MuSPAD, "mse_LSTM_MuSPAD_threefourthData.csv", row.names = FALSE)
write.csv2(mse_LSTM, paste0("mse_LSTM",g,"_threefourthData.csv"), row.names = FALSE)


t_test <- df_ts_all_inc[test_indices,2]

real_vs_pred_test$time <- 
  as.Date("2020-01-01") +
  t_test

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")


r_p_graph <- rbind(r_p_graph_train, r_p_graph_test)

return(r_p_graph)

}

#------------------------------------------------------------------------------

graph_LSTM_7day_M <- make_Graph(model = "MuSPAD")

model = "MuSPAD"

if(model == "MuSPAD"){
  h <- "_muspad_varsubsetbasic"
}else if(model == "onlyIncidence"){
  h <- "_onlyIncidence"
}

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data")
save(graph_LSTM_7day_M, file = "LSTM_7day_MuSPAD.RData")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")

png(file=paste0("ggplot_forecast_lstm",h,"_threefourthData_", Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(graph_LSTM_7day_M, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LSTM"))

dev.off()

#------------------------------------------------------------------------------

graph_LSTM_7day_BL <- make_Graph(model = "onlyIncidence")

model = "onlyIncidence"

if(model == "MuSPAD"){
  h <- "_muspad_varsubsetbasic"
}else if(model == "onlyIncidence"){
  h <- "_onlyIncidence"
}

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data")
save(graph_LSTM_7day_BL, file = "LSTM_7day_baseline.RData")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")

png(file=paste0("ggplot_forecast_lstm",h,"_threefourthData_", Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(graph_LSTM_7day_BL, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LSTM"))

dev.off()


