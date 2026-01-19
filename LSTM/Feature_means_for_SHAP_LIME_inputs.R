

library(dplyr)
library(purrr)
library(tibble)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")

mean_overall_df <- read.csv2("feature_means_overall_orig_scale.csv")[,-1]

# -----------------------------------------------------------------------------
# Feature means (over samples and time and over samples per time) for low cluster

# Example: clus_low$x is your vector of indices (like in clus_low['x'])
valid_indices_low <- clus_low[clus_low >= 14]

# Feature names
feature_names <- colnames(df_imp_1)

# Create a list of 3D arrays (each: 7 time steps × n_features)
test_low_x_list <- map(valid_indices_low, function(i) {
  df_imp_1[(i - 13):(i - 7), , drop = FALSE] |> as.matrix()
})

# Convert list of matrices to 3D array: (samples, time_steps, features)
# We'll use abind for this
library(abind)
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
library(tidyr)
library(dplyr)

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

# Example: clus_high$x is your vector of indices (like in clus_high['x'])
valid_indices_high <- clus_high[clus_high >= 14]

# Feature names
feature_names <- colnames(df_imp_1)

# Create a list of 3D arrays (each: 7 time steps × n_features)
test_high_x_list <- map(valid_indices_high, function(i) {
  df_imp_1[(i - 13):(i - 7), , drop = FALSE] |> as.matrix()
})

# Convert list of matrices to 3D array: (samples, time_steps, features)
# We'll use abind for this
library(abind)
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
library(tidyr)
library(dplyr)

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

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")

write.csv2(feature_means_low_df, "feature_means_low_overall_orig_scale_inputs.csv")

write.csv2(feature_means_high_df, "feature_means_high_overall_orig_scale_inputs.csv")


write.csv2(feature_means_perTime_low_long, "feature_means_perTime_low_overall_orig_scale_inputs.csv")

write.csv2(feature_means_perTime_high_long, "feature_means_perTime_high_overall_orig_scale_inputs.csv")




