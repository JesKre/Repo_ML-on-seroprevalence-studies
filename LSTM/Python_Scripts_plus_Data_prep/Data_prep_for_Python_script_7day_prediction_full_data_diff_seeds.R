# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

library(missForest)
library(tidyverse)

# Dataset

output_root <- file.path(REPO_ROOT, "LSTM/Python_results")
# dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

setwd(DATA_DIR)
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

# Predict 7 days ahead

# Predict 7 days ahead

test_indices <- (dim(df_ts_all_inc)[1]-6):dim(df_ts_all_inc)[1]

val_indices <- (dim(df_ts_all_inc)[1]-13):(dim(df_ts_all_inc)[1]-7)


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


setwd(DATA_DIR)
write.csv(traindat, "MuSPAD_traindat_LSTM.csv")
write.csv(testdat, "MuSPAD_testdat_LSTM.csv")
write.csv(traindat_v, "MuSPAD_traindat_v_LSTM.csv")
write.csv(valdat, "MuSPAD_valdat_LSTM.csv")

result_path <- paste0(DATA_DIR, "/")
file <- paste0(result_path, "scaling_parameters_",Sys.Date(),".RData")

save(scale_center_train, scale_scale_train, scale_center_train_v, scale_scale_train_v,
     test_indices, val_indices,
     file = file)



#------------------------------------------------------------------------------

# For calculation of LIME and Shap per incidence cluster sample,
# build an imputed and fully scaled dataset

setwd(DATA_DIR)

df_imp_1 <- rbind(traindat_1, testdat_1)

setwd(DATA_DIR)
write.csv(df_imp_1, "MuSPAD_alldat_LSTM_imputed_unscaled.csv")

df_imp_scale <- scale(df_imp_1)

scale_center <- attr(df_imp_scale, "scaled:center")
scale_center_df <- data.frame(Feature = names(scale_center), scale_center = scale_center)

scale_scale <- attr(df_imp_scale, "scaled:scale")
scale_scale_df <- data.frame(Feature = names(scale_scale), scale_scale = scale_scale)

result_path <- paste0(DATA_DIR, "/")
file <- paste0(result_path, "scaling_parameters_fully_scaled_data_",Sys.Date(),".RData")

save(scale_center_df, scale_scale_df,
     file = file)

df_imp <- data.frame(df_imp_scale)

setwd(DATA_DIR)
write.csv(df_imp, "MuSPAD_alldat_LSTM_imputed_scaled.csv")


#------------------------------------------------------------------------------
# Create low and high cluster

df_imp_y <- df_imp_1[,1]


# Use elbow method to detect optimal number of clusters

wss <- sapply(1:10, function(k){
  kmeans(df_imp_y, centers = k, nstart = 10)$tot.withinss
})

plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")


#---

#     Feature means based on the samples of incidence clusters
#     = based on the output samples

library(missForest)
library(tidyverse)

# Get feature means (overall, low inc cluster, high inc cluster)

# feature means overall
mean_overall <-  apply(df_imp_1, 2, mean)
mean_overall_df <- data.frame(Feature = names(mean_overall), mean_overall = mean_overall)


setwd(file.path(REPO_ROOT, "LSTM/Python_results"))

write.csv2(mean_overall_df, "feature_means_overall_orig_scale.csv")

library(tidyr)
library(dplyr)
library(abind)
library(dplyr)
library(purrr)
library(tibble)


for(k in 3:5){

  # Perform k-means clustering
  set.seed(123)  # for reproducibility
  km <- kmeans(df_imp_y, centers = k)


  cl_list <- list()

  for(i in 1:k){
    cl_list[[i]] <- mean(df_imp_y[which(km$cluster == i)])
  }

  hcl <- which(unlist(cl_list) == max(unlist(cl_list)))
  lcl <- which(unlist(cl_list) == min(unlist(cl_list)))

  # Highest incidences: cluster hcl
  # Lowest incidences:  cluster lcl

  cluster_low <- which(km$cluster == lcl)
  cluster_high <- which(km$cluster == hcl)

  setwd(file.path(REPO_ROOT, "LSTM/Python_results"))

  write.csv2(cluster_low, paste0("cluster_low_",k,".csv"))
  write.csv2(cluster_high, paste0("cluster_high_",k,".csv"))

  # feature means low inc cluster
  feat_mean_low_vec <- apply(df_imp_1[cluster_low,], 2, mean)
  feat_mean_low_df <- data.frame(Feature = names(feat_mean_low_vec), mean_cluster = feat_mean_low_vec) %>%
    left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)

  # feature means high inc cluster
  feat_mean_high_vec <- apply(df_imp_1[cluster_high,], 2, mean)
  feat_mean_high_df <- data.frame(Feature = names(feat_mean_high_vec), mean_cluster = feat_mean_high_vec) %>%
    left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)


  setwd(file.path(REPO_ROOT, "LSTM/Python_results"))

  write.csv2(feat_mean_low_df, paste0("feature_means_low_",k,"overall_orig_scale.csv"))

  write.csv2(feat_mean_high_df, paste0("feature_means_high_",k,"overall_orig_scale.csv"))

  # -----------------------------------------------------------------------------
  # Feature means (over samples and time and over samples per time) for low cluster

  # Example: cluster_low$x is your vector of indices (like in cluster_low['x'])
  valid_indices_low <- cluster_low[cluster_low >= 14]

  # Feature names
  feature_names <- colnames(df_imp_1)

  # Create a list of 3D arrays (each: 7 time steps × n_features)
  test_low_x_list <- map(valid_indices_low, function(i) {
    df_imp_1[(i - 13):(i - 7), , drop = FALSE] |> as.matrix()
  })

  # Convert list of matrices to 3D array: (samples, time_steps, features)
  # We'll use abind for this
  test_low_x_array <- abind::abind(test_low_x_list, along = 1) # shape: (n_samples * 7, n_features)

  # Reshape to (samples, time_steps, features)
  n_samples <- length(test_low_x_list)
  time_steps <- 7
  n_features <- ncol(df_imp_1)

  # Reshape manually
  dim(test_low_x_array) <- c(n_samples, time_steps, n_features)

  # Calculate mean across samples and time steps
  # Equivalent to np.mean(array, axis=(0, 1)) → mean over sample and time
  feature_means_low <- apply(test_low_x_array, 3, mean)

  # Create data frame
  feature_means_low_df <- tibble(
    Feature = feature_names,
    `mean_cluster` = feature_means_low
  ) %>%
    left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)


  # Set labels for time steps (t-6 to t-0)
  time_step_labels <- paste0("t-", rev(0:6))  # gives: "t-6", ..., "t-0"

  # Calculate mean over samples (1st dim), keeping time and features
  # Result: time_steps x n_features matrix
  means_by_time_low <- apply(test_low_x_array, c(2, 3), mean)  # dim: 7 x n_features

  # Convert to long format

  feature_means_perTime_low <- as.data.frame(means_by_time_low)
  colnames(feature_means_perTime_low) <- feature_names
  feature_means_perTime_low$Time_Step <- time_step_labels

  # Pivot to long format: columns Feature, Time_Step, Mean_Value
  feature_means_perTime_low_long <- feature_means_perTime_low |>
    pivot_longer(-Time_Step, names_to = "Feature", values_to = "mean_cluster") |>
    select(Feature, Time_Step, mean_cluster) %>%
    left_join(mean_overall_df) %>% select(Feature, Time_Step, mean_overall, mean_cluster)


  # -----------------------------------------------------------------------------
  # Feature means (over samples and time and over samples per time) for high cluster

  # Example: cluster_high$x is your vector of indices (like in cluster_high['x'])
  valid_indices_high <- cluster_high[cluster_high >= 14]

  # Feature names
  feature_names <- colnames(df_imp_1)

  # Create a list of 3D arrays (each: 7 time steps × n_features)
  test_high_x_list <- map(valid_indices_high, function(i) {
    df_imp_1[(i - 13):(i - 7), , drop = FALSE] |> as.matrix()
  })

  # Convert list of matrices to 3D array: (samples, time_steps, features)
  # We'll use abind for this
  test_high_x_array <- abind::abind(test_high_x_list, along = 1) # shape: (n_samples * 7, n_features)

  # Reshape to (samples, time_steps, features)
  n_samples <- length(test_high_x_list)
  time_steps <- 7
  n_features <- ncol(df_imp_1)

  # Reshape manually
  dim(test_high_x_array) <- c(n_samples, time_steps, n_features)

  # Calculate mean across samples and time steps
  # Equivalent to np.mean(array, axis=(0, 1)) → mean over sample and time
  feature_means_high <- apply(test_high_x_array, 3, mean)

  # Create data frame
  feature_means_high_df <- tibble(
    Feature = feature_names,
    `mean_cluster` = feature_means_high
  ) %>%
    left_join(mean_overall_df) %>% select(Feature, mean_overall, mean_cluster)


  # Set labels for time steps (t-6 to t-0)
  time_step_labels <- paste0("t-", rev(0:6))  # gives: "t-6", ..., "t-0"

  # Calculate mean over samples (1st dim), keeping time and features
  # Result: time_steps x n_features matrix
  means_by_time_high <- apply(test_high_x_array, c(2, 3), mean)  # dim: 7 x n_features

  # Convert to long format

  feature_means_perTime_high <- as.data.frame(means_by_time_high)
  colnames(feature_means_perTime_high) <- feature_names
  feature_means_perTime_high$Time_Step <- time_step_labels

  # Pivot to long format: columns Feature, Time_Step, Mean_Value
  feature_means_perTime_high_long <- feature_means_perTime_high |>
    pivot_longer(-Time_Step, names_to = "Feature", values_to = "mean_cluster") |>
    select(Feature, Time_Step, mean_cluster) %>%
    left_join(mean_overall_df) %>% select(Feature, Time_Step, mean_overall, mean_cluster)

  # -----------
  # Save

  setwd(file.path(REPO_ROOT, "LSTM/Python_results"))

  write.csv2(feature_means_low_df, paste0("feature_means_",k,"_low_overall_orig_scale_inputs.csv"))

  write.csv2(feature_means_high_df, paste0("feature_means_",k,"_high_overall_orig_scale_inputs.csv"))


  write.csv2(feature_means_perTime_low_long, paste0("feature_means_perTime_",k,"_low_overall_orig_scale_inputs.csv"))

  write.csv2(feature_means_perTime_high_long, paste0("feature_means_perTime_",k,"_high_overall_orig_scale_inputs.csv"))


}

#------------------------------------------------------------------------------

# Run Python script

#------------------------------------------------------------------------------
# Feature importance tables

library(missForest)
library(tidyverse)

for(k in 3:5){

  setwd(file.path(REPO_ROOT, "LSTM/Python_results"))

  feat_mean_low_df <- read.csv2(paste0("feature_means_",k,"_low_overall_orig_scale_inputs.csv"))[,-1]

  feat_mean_high_df <- read.csv2(paste0("feature_means_",k,"_high_overall_orig_scale_inputs.csv"))[,-1]


  feature_means_perTime_low_long <- read.csv2(paste0("feature_means_perTime_",k,"_low_overall_orig_scale_inputs.csv"))[,-1]

  feature_means_perTime_high_long <- read.csv2(paste0("feature_means_perTime_",k,"_high_overall_orig_scale_inputs.csv"))[,-1]


  #------------------------------------------------------------------------------
  # Shap values: mean over input and output samples and rescaling

  #------------------------------------------------------------------------------
  # 1.) Low incidence cluster

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/Python_Original/results_h7"))

  shap_python_low <- read.csv(paste0("shap_tplus7_all_samples_low_",k,".csv"))

  head(shap_python_low)
  dim(shap_python_low)

  # standardised scale

  shapley_low <- shap_python_low %>%
    group_by(Feature) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    arrange(desc(abs(SHAP_Value_mean)))

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/transformed_h7"))

  write.csv2(shapley_low, paste0("shap_tplus7_",k,"_allMean_standardised_low.csv"), fileEncoding = "latin1")


  # back to original scale

  # scale_center_inc <- scale_center[which(names(scale_center) == "Inzidenz")]
  scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]

  scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


  # shapley values per sample and time step

  shapley_low_orig <- shap_python_low %>%
    mutate(SHAP_Value = SHAP_Value * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value))) %>%
    left_join(feat_mean_low_df)

  write.csv2(shapley_low_orig, paste0("shap_tplus7_orig_scale_",k,"_low.csv"), fileEncoding = "latin1")

  # Mean over time steps and samples

  shapley_low_orig_scale <- shap_python_low %>%
    group_by(Feature) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value_mean))) %>%
    left_join(feat_mean_low_df)

  write.csv2(shapley_low_orig_scale, paste0("shap_tplus7_allMean_orig_scale_",k,"_low.csv"), fileEncoding = "latin1")


  # Mean over samples, per time step * feature

  shapley_low_sampleMean <- shap_python_low %>%
    group_by(Feature, Time_Step) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value_mean))) %>%
    left_join(feature_means_perTime_low_long)

  write.csv2(shapley_low_sampleMean, paste0("shap_tplus7_sampleMean_orig_scale_",k,"_low.csv"), fileEncoding = "latin1")



  #------------------------------------------------------------------------------
  # 2.) High incidence cluster

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/Python_Original/results_h7"))

  shap_python_high <- read.csv(paste0("shap_tplus7_all_samples_high_",k,".csv"))

  head(shap_python_high)
  dim(shap_python_high)

  # -----------------------------------------------------------------------------

  # standardised scale

  shapley_high <- shap_python_high %>%
    group_by(Feature) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    arrange(desc(abs(SHAP_Value_mean)))

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/transformed_h7"))

  write.csv2(shapley_high, paste0("shap_tplus7_allMean_standardised_",k,"_high.csv"), fileEncoding = "latin1")


  # back to original scale

  # scale_center_inc <- scale_center[which(names(scale_center) == "Inzidenz")]
  scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]

  scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


  # shapley values per sample and time step

  shapley_high_orig <- shap_python_high %>%
    mutate(SHAP_Value = SHAP_Value * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value))) %>%
    left_join(feat_mean_high_df)

  write.csv2(shapley_high_orig, paste0("shap_tplus7_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")

  # Mean over time steps and samples

  shapley_high_orig_scale <- shap_python_high %>%
    group_by(Feature) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value_mean))) %>%
    left_join(feat_mean_high_df)

  write.csv2(shapley_high_orig_scale, paste0("shap_tplus7_allMean_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")


  # Mean over samples, per time step * feature

  shapley_high_sampleMean <- shap_python_high %>%
    group_by(Feature, Time_Step) %>%
    summarise(SHAP_Value_mean = mean(SHAP_Value)) %>%
    mutate(SHAP_Value_mean = SHAP_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(SHAP_Value_mean))) %>%
    left_join(feature_means_perTime_high_long)

  write.csv2(shapley_high_sampleMean, paste0("shap_tplus7_sampleMean_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")


  #------------------------------------------------------------------------------

  # LIME values: mean over input and output samples and rescaling

  #------------------------------------------------------------------------------
  # 1.) Low incidence cluster

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/Python_Original/results_h7"))

  lime_python_low <- read.csv(paste0("lime_tplus7_all_samples_low_",k,".csv"))

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
        str_extract("t-[0-6]*")
    ) %>%
    select(Feature_Time_Value, Feature, Time_Step, LIME_Value, Sample)

  # standardised scale

  lime_low <- lime_python_low %>%
    group_by(Feature) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    arrange(desc(abs(LIME_Value_mean)))

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/transformed_h7"))

  write.csv2(lime_low, paste0("lime_tplus7_allMean_standardised_",k,"_low.csv"), fileEncoding = "latin1")


  # back to original scale

  scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]
  scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


  # lime values per sample and time step

  lime_low_orig<- lime_python_low %>%
    mutate(LIME_Value = LIME_Value * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value))) %>%
    left_join(feat_mean_low_df)

  write.csv2(lime_low_orig, "lime_tplus7_orig_scale_low.csv", fileEncoding = "latin1")


  # Mean over time steps and samples

  lime_low_orig_scale <- lime_python_low %>%
    group_by(Feature) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value_mean))) %>%
    left_join(feat_mean_low_df)

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/transformed_h7"))

  write.csv2(lime_low_orig_scale, paste0("lime_tplus7_allMean_orig_scale_",k,"_low.csv"), fileEncoding = "latin1")


  # Mean over samples, per time step * feature

  lime_low_sampleMean <- lime_python_low %>%
    group_by(Feature, Time_Step) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value_mean))) %>%
    left_join(feature_means_perTime_high_long)

  write.csv2(lime_low_sampleMean, paste0("lime_tplus7_sampleMean_orig_scale_",k,"_low.csv"), fileEncoding = "latin1")


  #------------------------------------------------------------------------------
  # 2.) High incidence cluster

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/Python_Original/results_h7"))

  lime_python_high <- read.csv(paste0("lime_tplus7_all_samples_high_",k,".csv"))

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
        str_extract("t-[0-6]*")
    ) %>%
    select(Feature_Time_Value, Feature, Time_Step, LIME_Value, Sample)

  # standardised scale

  lime_high <- lime_python_high %>%
    group_by(Feature) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    arrange(desc(abs(LIME_Value_mean)))

  setwd(file.path(REPO_ROOT, "LSTM/Python_results/transformed_h7"))

  write.csv2(lime_high, paste0("lime_tplus7_allMean_standardised_",k,"_high.csv"), fileEncoding = "latin1")


  # back to original scale

  scale_center_inc <- scale_center_df$scale_center[which(scale_center_df$Feature == "Inzidenz")]
  scale_scale_inc <- scale_scale_df$scale_scale[which(scale_scale_df$Feature == "Inzidenz")]


  # lime values per sample and time step

  lime_high_orig<- lime_python_high %>%
    mutate(LIME_Value = LIME_Value * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value))) %>%
    left_join(feat_mean_high_df)

  write.csv2(lime_high_orig, paste0("lime_tplus7_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")


  # Mean over time steps and samples

  lime_high_orig_scale <- lime_python_high %>%
    group_by(Feature) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value_mean))) %>%
    left_join(feat_mean_high_df)

  write.csv2(lime_high_orig_scale, paste0("lime_tplus7_allMean_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")


  # Mean over samples, per time step * feature

  lime_high_sampleMean <- lime_python_high %>%
    group_by(Feature, Time_Step) %>%
    summarise(LIME_Value_mean = mean(LIME_Value)) %>%
    mutate(LIME_Value_mean = LIME_Value_mean * scale_scale_inc) %>%
    arrange(desc(abs(LIME_Value_mean))) %>%
    left_join(feat_mean_high_df) %>%
    left_join(feature_means_perTime_high_long)

  write.csv2(lime_high_sampleMean, paste0("lime_tplus7_sampleMean_orig_scale_",k,"_high.csv"), fileEncoding = "latin1")


}


#------------------------------------------------------------------------------
# Graphics

#------------------------------------------------------------------------------
# "Postprocessing":
# backtransforming real and pred values for metric calculation and graphics

library(magrittr)

# setwd(file.path(REPO_ROOT, "LSTM/Python_results"))
# traindat <- read.csv("MuSPAD_traindat_LSTM.csv")[,-1]
# testdat <- read.csv("MuSPAD_testdat_LSTM.csv")[,-1]
# traindat_v <- read.csv("MuSPAD_traindat_v_LSTM.csv")[,-1]
# valdat <- read.csv("MuSPAD_valdat_LSTM.csv")[,-1]
#

library(missForest)
library(tidyverse)
library(Metrics)

setwd(DATA_DIR)
(file <- tail(list.files(pattern = "scaling_parameters_[1-9]"), n=1))

load(file = file)


setwd(DATA_DIR)
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

prediction <- 7

make_Graph <- function(model){

  if(model == "MuSPAD"){
    f <- ""
  }else if(model == "onlyIncidence"){
    f <- "_only_incidence"
  }

  seeds <- c(seq(1, 10, 1))

  results <- list()

  for (i in seeds) {
    setwd(file.path(REPO_ROOT, "LSTM/Python_results/Python_Original/Different_seeds/keras_tuner/results_h7"))

    real_vs_pred_train <- read.csv(paste0("train_predictions",f,"_tplus7_seed",i,".csv"))

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

    t_train <- df_ts_all_inc[1:(test_indices[1]-1),2]

    real_vs_pred_train$time <-
      as.Date("2020-01-01") +
      t_train[(prediction+1):(length(t_train))]

    r_p_graph_train <- real_vs_pred_train %>%
      pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
      mutate(set = "Training",
             seed = i)

    real_vs_pred_test <- read.csv(paste0("test_predictions",f,"_tplus7_seed",i,".csv"))

    real_vs_pred_test$predicted %<>%
      str_remove_all("\\[") %<>%
      str_remove_all("\\]") %<>%
      as.numeric()

    #str(real_vs_pred_test)

    real_vs_pred_test <- data.frame(sapply(real_vs_pred_test, back_scaling))

    t_test <- df_ts_all_inc[test_indices,2]

    real_vs_pred_test$time <-
      as.Date("2020-01-01") +
      t_test

    r_p_graph_test <- real_vs_pred_test %>%
      pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
      mutate(set = "Test",
             seed = i)

    r_p_graph <- rbind(r_p_graph_train, r_p_graph_test)

    results[[i]] <- r_p_graph

  }

  results <- do.call(rbind, results)


  return(results)

}


graph_LSTM_7day_M <- make_Graph(model = "MuSPAD")

model = "MuSPAD"

if(model == "MuSPAD"){
  h <- "_muspad_varsubsetbasic"
}else if(model == "onlyIncidence"){
  h <- "_onlyIncidence"
}

setwd(file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/fullData"))
save(graph_LSTM_7day_M, file = "LSTM_7day_MuSPAD_seeds.RData")

# # Graphics
#
# setwd(file.path(REPO_ROOT, "LSTM/Python_results"))
#
# png(file=paste0("ggplot_forecast_lstm_muspad_varsubsetbasic_ep20_no_reset_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)
#
# ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) +
#   geom_line(aes(color=real_vs_pred)) +
#   geom_point(aes(color=real_vs_pred, shape = set)) +
#   theme_bw() +
#   scale_shape_manual(values=c(4, 16)) +
#   scale_color_discrete(labels=c('Predicted', 'Real')) +
#   theme(legend.position = "bottom") +
#   guides(color=guide_legend(title="LSTM"))
#
# dev.off()
#
#
# setwd(file.path(REPO_ROOT, "Compare_models/Graphics/Single_graphics"))
#
# png(file=paste0("ggplot_forecast_lstm_muspad_varsubsetbasic_ep20_no_reset_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)
#
# ggplot(rbind(r_p_graph_train, r_p_graph_test), aes(y=Incidence, x=time)) +
#   geom_line(aes(color=real_vs_pred)) +
#   geom_point(aes(color=real_vs_pred, shape = set)) +
#   theme_bw() +
#   scale_shape_manual(values=c(4, 16)) +
#   scale_color_discrete(labels=c('Predicted', 'Real')) +
#   theme(legend.position = "bottom") +
#   guides(color=guide_legend(title="LSTM"))
#
# dev.off()
#


#------------------------------------------------------------------------------

graph_LSTM_7day_BL <- make_Graph(model = "onlyIncidence")

model = "onlyIncidence"

if(model == "MuSPAD"){
  h <- "_muspad_varsubsetbasic"
}else if(model == "onlyIncidence"){
  h <- "_onlyIncidence"
}

setwd(file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/fullData"))
save(graph_LSTM_7day_BL, file = "LSTM_7day_baseline_seeds.RData")





