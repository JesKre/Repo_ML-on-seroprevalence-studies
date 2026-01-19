
#------------------------------------------------------------------------------
# Graphics


library(missForest)
library(tidyverse)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
(file <- tail(list.files(pattern = "scaling_parameters_14_[1-9]"), n=1))

load(file = file)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

prediction <- 14

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results/Python_Original/results_h14")

real_vs_pred_train <- read.csv("train_predictions_only_incidence_tplus14.csv")

scale_center_inc <- scale_center_train[which(names(scale_center_train) == "Inzidenz")]

scale_scale_inc <- scale_scale_train[which(names(scale_scale_train) == "Inzidenz")]

str(real_vs_pred_train)

library(magrittr)

real_vs_pred_train$predicted %<>% 
  str_remove_all("\\[") %<>% 
  str_remove_all("\\]") %<>% 
  as.numeric()

str(real_vs_pred_train)


back_scaling <- function(x) x * scale_scale_inc + scale_center_inc

real_vs_pred_train <- data.frame(sapply(real_vs_pred_train, back_scaling))


(mse_train <- mse(real_vs_pred_train$real, real_vs_pred_train$predicted))
# 5378

t_train <- df_ts_all_inc[-test_indices,2]

real_vs_pred_train$time <- 
  as.Date("2020-01-01") +
  t_train[(prediction+1):(length(t_train))]

r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results/Python_Original/results_h14")

real_vs_pred_test <- read.csv("test_predictions_only_incidence_tplus14.csv")

dim(real_vs_pred_test)

real_vs_pred_test$predicted %<>% 
  str_remove_all("\\[") %<>% 
  str_remove_all("\\]") %<>% 
  as.numeric()

str(real_vs_pred_test)

real_vs_pred_test <- data.frame(sapply(real_vs_pred_test, back_scaling))

(mse_test <- mse(real_vs_pred_test$real, real_vs_pred_test$predicted))
# 10.7

mse_LSTM_onlyIncidence <- data.frame(mse_train, mse_test)

# Save mse results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")
write.csv2(mse_LSTM_onlyIncidence, "mse_LSTM_onlyIncidence_14.csv", row.names = FALSE)


t_test <- df_ts_all_inc[test_indices,2]

real_vs_pred_test$time <- 
  as.Date("2020-01-01") +
  t_test

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")

graph_LSTM_14day_BL <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/14day_Pred")
save(graph_LSTM_14day_BL, file = "LSTM_14day_baseline.RData")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")

png(file=paste0("ggplot_forecast_lstm_onlyIncidence_14dayhorizon_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LSTM"))

dev.off()



