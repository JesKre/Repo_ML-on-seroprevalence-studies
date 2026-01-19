


library(missForest)
library(tidyverse)

# Dataset

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

# Predict 14 days ahead

prediction <- 14

test_indices <- (dim(df_ts_all_inc)[1]-prediction+1):dim(df_ts_all_inc)[1]

val_indices <- (dim(df_ts_all_inc)[1]-2*prediction+1):(dim(df_ts_all_inc)[1]-prediction)


# impute training data
traindat_1 <- missForest(df_ts_all_inc[-test_indices,-2])$ximp


# Scale the training data
traindat_scale <- scale(traindat_1)

traindat <- as.data.frame(traindat_scale)

scale_center_train <- attr(traindat_scale, "scaled:center")
scale_scale_train <- attr(traindat_scale, "scaled:scale")

# -> training into train and val

traindat_scale_v <- scale(traindat_1[-val_indices,])

traindat_v <- as.data.frame(traindat_scale_v)

scale_center_train_v <- attr(traindat_scale_v, "scaled:center")
scale_scale_train_v <- attr(traindat_scale_v, "scaled:scale")

valdat_scale <- scale(traindat_1[val_indices,], 
                      center = scale_center_train_v,
                      scale = scale_scale_train_v)

valdat <- as.data.frame(valdat_scale)


# impute test data
testdat_1 <- missForest(df_ts_all_inc[test_indices,-2])$ximp


# Use the scaling parameters from the training data to scale the test data
testdat_scale <- scale(testdat_1, 
                       center = scale_center_train, 
                       scale = scale_scale_train)

testdat <- as.data.frame(testdat_scale)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
write.csv(traindat, "MuSPAD_traindat_LSTM_14.csv")
write.csv(testdat, "MuSPAD_testdat_LSTM_14.csv")
write.csv(traindat_v, "MuSPAD_traindat_v_LSTM_14.csv")
write.csv(valdat, "MuSPAD_valdat_LSTM_14.csv")

result_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\"
file <- paste0(result_path, "scaling_parameters_14_",Sys.Date(),".RData")

save(scale_center_train, scale_scale_train, scale_center_train_v, scale_scale_train_v,
     test_indices, val_indices,
     file = file)



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

real_vs_pred_train <- read.csv("train_predictions_0p2_AB_noreset_tplus14.csv")

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


t_train <- df_ts_all_inc[-test_indices,2]

real_vs_pred_train$time <- 
  as.Date("2020-01-01") +
  t_train[(prediction+1):(length(t_train))]

r_p_graph_train <- real_vs_pred_train %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results/Python_Original/results_h14")

real_vs_pred_test <- read.csv("test_predictions_0p2_AB_noreset_tplus14.csv")

dim(real_vs_pred_test)

real_vs_pred_test$predicted %<>% 
  str_remove_all("\\[") %<>% 
  str_remove_all("\\]") %<>% 
  as.numeric()

str(real_vs_pred_test)

real_vs_pred_test <- data.frame(sapply(real_vs_pred_test, back_scaling))

(mse_test <- mse(real_vs_pred_test$real, real_vs_pred_test$predicted))

mse_LSTM_MuSPAD <- data.frame(mse_train, mse_test)

# Save mse results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")
write.csv2(mse_LSTM_MuSPAD, "mse_LSTM_MuSPAD_14.csv", row.names = FALSE)


t_test <- df_ts_all_inc[test_indices,2]

real_vs_pred_test$time <- 
  as.Date("2020-01-01") +
  t_test

r_p_graph_test <- real_vs_pred_test %>%
  pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")

graph_LSTM_14day_M <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/14day_Pred")
save(graph_LSTM_14day_M, file = "LSTM_14day_MuSPAD.RData")


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/LSTM/Python_results")

png(file=paste0("ggplot_forecast_lstm_muspad_varsubsetbasic_14dayhorizon_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="LSTM"))

dev.off()




#------------------------------------------------------------------------------
# Feature importance tables


# 1.) Feature means based on the samples of incidence clusters
#     = based on the output samples
# 
# library(missForest)
# library(tidyverse)
# 
# # Get feature means (overall, low inc cluster, high inc cluster)
# 
# # feature means overall
# mean_overall <-  apply(df_imp_1, 2, mean)
# mean_overall_df <- data.frame(Feature = names(mean_overall), mean_overall = mean_overall)
# 
# setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")
# 
# write.csv2(mean_overall_df, "feature_means_overall_orig_scale.csv")
# 
# 
# # feature means low inc cluster
# feat_mean_low_vec <- apply(df_imp_1[clus_low,], 2, mean)
# feat_mean_low_df <- data.frame(Feature = names(feat_mean_low_vec), mean_cluster = feat_mean_low_vec) %>%
#   left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)
# 
# # feature means high inc cluster
# feat_mean_high_vec <- apply(df_imp_1[clus_high,], 2, mean)
# feat_mean_high_df <- data.frame(Feature = names(feat_mean_high_vec), mean_cluster = feat_mean_high_vec) %>%
#   left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)
# 
# 
# setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")
# 
# write.csv2(feat_mean_low_df, "feature_means_low_overall_orig_scale.csv")
# 
# write.csv2(feat_mean_high_df, "feature_means_high_overall_orig_scale.csv")
# 

# 2.) Feature means based on the input samples 
#     = samples of 13 to 27 days before incidence cluster samples 

# See R-script "Feature_means_for_SHAP_LIME_inputs"


# Since the inputs are going into the model and parameters of the model are then 
# based on these values, while the values of the output time is never shown to the model,
# one should look at the feature values and means of the inputs for interpretation. 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
(file <- tail(list.files(pattern = "scaling_parameters_fully_scaled_data_"), n=1))

load(file = file)

# Feature Importance

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")

feat_mean_low_df <- read.csv2("feature_means_low_overall_orig_scale_inputs.csv")[,-1]

feat_mean_high_df <- read.csv2("feature_means_high_overall_orig_scale_inputs.csv")[,-1]


feature_means_perTime_low_long <- read.csv2("feature_means_perTime_low_overall_orig_scale_inputs.csv")[,-1]

feature_means_perTime_high_long <- read.csv2("feature_means_perTime_high_overall_orig_scale_inputs.csv")[,-1]


library(missForest)
library(tidyverse)



#------------------------------------------------------------------------------
# Shap values: mean over input and output samples and rescaling

#------------------------------------------------------------------------------
# 1.) Low incidence cluster


orig_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results\\Python_Original\\results_h14"
transformed_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results\\transformed_h14"

setwd(orig_path)

shap_python_low <- read.csv("shap_tplus14_all_samples_low.csv")

head(shap_python_low)
dim(shap_python_low)


# --------------------------------------------------------------------------

# standardised scale

shapley_low <- shap_python_low %>%
  group_by(Feature) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  arrange(desc(abs(SHAP_Value_mean)))

setwd(transformed_path)

write.csv2(shapley_low, "shap_tplus14_allMean_standardised_low.csv", fileEncoding = "latin1")


# back to original scale

# scale_center_inc <- scale_center[which(names(scale_center) == "Inzidenz")]
scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]

scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


# shapley values per sample and time step

shapley_low_orig <- shap_python_low %>%
  mutate(SHAP_Value = SHAP_Value * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value))) %>%
  left_join(feat_mean_low_df)

write.csv2(shapley_low_orig, "shap_tplus14_orig_scale_low.csv", fileEncoding = "latin1")

# Mean over time steps and samples

shapley_low_orig_scale <- shap_python_low %>%
  group_by(Feature) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value_mean))) %>%
  left_join(feat_mean_low_df)

write.csv2(shapley_low_orig_scale, "shap_tplus14_allMean_orig_scale_low.csv", fileEncoding = "latin1")


# Mean over samples, per time step * feature

shapley_low_sampleMean <- shap_python_low %>%
  group_by(Feature, Time_Step) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value_mean))) %>%
  left_join(feature_means_perTime_low_long)

write.csv2(shapley_low_sampleMean, "shap_tplus14_sampleMean_orig_scale_low.csv", fileEncoding = "latin1")



#------------------------------------------------------------------------------
# 2.) High incidence cluster

setwd(orig_path)

shap_python_high <- read.csv("shap_tplus14_all_samples_high.csv")

head(shap_python_high)
dim(shap_python_high)
# ----------------------------------------------------------------------------

# standardised scale

shapley_high <- shap_python_high %>%
  group_by(Feature) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  arrange(desc(abs(SHAP_Value_mean)))

setwd(transformed_path)

write.csv2(shapley_high, "shap_tplus14_allMean_standardised_high.csv", fileEncoding = "latin1")


# back to original scale

scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]

scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


# shapley values per sample and time step

shapley_high_orig <- shap_python_high %>%
  mutate(SHAP_Value = SHAP_Value * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value))) %>%
  left_join(feat_mean_high_df)

write.csv2(shapley_high_orig, "shap_tplus14_orig_scale_high.csv", fileEncoding = "latin1")

# Mean over time steps and samples

shapley_high_orig_scale <- shap_python_high %>%
  group_by(Feature) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value_mean))) %>%
  left_join(feat_mean_high_df)

write.csv2(shapley_high_orig_scale, "shap_tplus14_allMean_orig_scale_high.csv", fileEncoding = "latin1")


# Mean over samples, per time step * feature

shapley_high_sampleMean <- shap_python_high %>%
  group_by(Feature, Time_Step) %>%
  summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
  mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(SHAP_Value_mean))) %>%
  left_join(feature_means_perTime_high_long)

write.csv2(shapley_high_sampleMean, "shap_tplus14_sampleMean_orig_scale_high.csv", fileEncoding = "latin1")


#------------------------------------------------------------------------------

# LIME values: mean over input and output samples and rescaling

#------------------------------------------------------------------------------
# 1.) Low incidence cluster

setwd(orig_path)

lime_python_low <- read.csv("lime_tplus14_all_samples_low.csv")

colnames(lime_python_low)
dim(lime_python_low)

lime_python_low <- lime_python_low %>%
  rename(Feature_Time_Value = Feature,
         LIME_Value = LIME.Value) %>%
  mutate(
    Feature = Feature_Time_Value %>%
      str_extract("[a-zA-Z].*") %>% # keeps only the text part starting from a letter 
      # (removes numbers at beginning)
      str_split("_[^_]*$") %>%      # splits at the last underscore + suffix (e.g. _t-1)
      map_chr(1), # after the split, extracts the first element
    Time_Step = Feature_Time_Value %>%
      str_extract("t-[0-13]*")
  ) %>%
  select(Feature_Time_Value, Feature, Time_Step, LIME_Value, Sample)

# standardised scale

lime_low <- lime_python_low %>%
  group_by(Feature) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  arrange(desc(abs(LIME_Value_mean)))

setwd(transformed_path)

write.csv2(lime_low, "lime_tplus14_allMean_standardised_low.csv", fileEncoding = "latin1")


# back to original scale

scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]
scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


# lime values per sample and time step

lime_low_orig<- lime_python_low %>%
  mutate(LIME_Value = LIME_Value * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value))) %>%
  left_join(feat_mean_low_df)

write.csv2(lime_low_orig, "lime_tplus14_orig_scale_low.csv", fileEncoding = "latin1")


# Mean over time steps and samples

lime_low_orig_scale <- lime_python_low %>%
  group_by(Feature) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value_mean))) %>%
  left_join(feat_mean_low_df)

setwd(transformed_path)

write.csv2(lime_low_orig_scale, "lime_tplus14_allMean_orig_scale_low.csv", fileEncoding = "latin1")


# Mean over samples, per time step * feature

lime_low_sampleMean <- lime_python_low %>%
  group_by(Feature, Time_Step) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value_mean))) %>%
  left_join(feature_means_perTime_high_long)

write.csv2(lime_low_sampleMean, "lime_tplus14_sampleMean_orig_scale_low.csv", fileEncoding = "latin1")


#------------------------------------------------------------------------------
# 2.) High incidence cluster

setwd(orig_path)

lime_python_high <- read.csv("lime_tplus14_all_samples_high.csv")

colnames(lime_python_high)
dim(lime_python_high)

lime_python_high <- lime_python_high %>%
  rename(Feature_Time_Value = Feature,
         LIME_Value = LIME.Value) %>%
  mutate(
    Feature = Feature_Time_Value %>%
      str_extract("[a-zA-Z].*") %>% # keeps only the text part starting from a letter 
      # (removes numbers at beginning)
      str_split("_[^_]*$") %>%      # splits at the last underscore + suffix (e.g. _t-1)
      map_chr(1), # after the split, extracts the first element
    Time_Step = Feature_Time_Value %>%
      str_extract("t-[0-13]*")
  ) %>%
  select(Feature_Time_Value, Feature, Time_Step, LIME_Value, Sample)

# standardised scale

lime_high <- lime_python_high %>%
  group_by(Feature) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  arrange(desc(abs(LIME_Value_mean)))

setwd(transformed_path)

write.csv2(lime_high, "lime_tplus14_allMean_standardised_high.csv", fileEncoding = "latin1")


# back to original scale

scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]
scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


# lime values per sample and time step

lime_high_orig<- lime_python_high %>%
  mutate(LIME_Value = LIME_Value * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value))) %>%
  left_join(feat_mean_high_df)

write.csv2(lime_high_orig, "lime_tplus14_orig_scale_high.csv", fileEncoding = "latin1")


# Mean over time steps and samples

lime_high_orig_scale <- lime_python_high %>%
  group_by(Feature) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value_mean))) %>%
  left_join(feat_mean_high_df)

write.csv2(lime_high_orig_scale, "lime_tplus14_allMean_orig_scale_high.csv", fileEncoding = "latin1")


# Mean over samples, per time step * feature

lime_high_sampleMean <- lime_python_high %>%
  group_by(Feature, Time_Step) %>%
  summarise(LIME_Value_mean = mean(LIME_Value)) %>%
  mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
  arrange(desc(abs(LIME_Value_mean))) %>%
  left_join(feat_mean_high_df) %>%
  left_join(feature_means_perTime_high_long)

write.csv2(lime_high_sampleMean, "lime_tplus14_sampleMean_orig_scale_high.csv", fileEncoding = "latin1")




