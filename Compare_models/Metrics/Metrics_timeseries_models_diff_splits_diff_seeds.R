
library(dplyr)
library(tidyverse)
library(lubridate)
library(Metrics)
library(tidyr)
library(purrr)

#---------------------------------------------------------------------------------------------------------

# 1.) Compute metrics for models without random seeds 
#     (because these do not have random elements and results stay the same)


folder <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values"

# list all .RData files
files <- list.files(folder, pattern = glob2rx("VAR*.RData$") , full.names = TRUE)

# create a named list to store data
real_vs_pred <- list()

for (f in files) {
  # create a temporary environment to avoid overwriting
  env <- new.env()
  load(f, envir = env)
  
  # extract objects
  objs <- as.list(env)
  
  # give them unique names based on file name
  names(objs) <- tools::file_path_sans_ext(basename(f))
  
  # add to results
  real_vs_pred <- c(real_vs_pred, objs)
}

# Convert all time columns to Date before binding
real_vs_pred_fixed <- lapply(real_vs_pred, function(df) {
  df %>%
    mutate(time = as.Date(time),
           real_vs_pred = case_when(real_vs_pred == "Predicted" ~ "predicted",
                                    real_vs_pred ==  "Real" ~ "real",
                                    .default = real_vs_pred))
})


results <- lapply(real_vs_pred_fixed, function(df) {
  df %>%
    group_by(set) %>%
    summarise(
      x = list(pivot_wider(cur_data(), names_from = real_vs_pred, values_from = Incidence)),
      mape =  map_dbl(x, ~ mape(.x$real, .x$predicted)),
      smape = map_dbl(x, ~ smape(.x$real, .x$predicted)),
      rmsle = map_dbl(x, ~ rmsle(.x$real, .x$predicted)),
      rmse = map_dbl(x, ~ rmse(.x$real, .x$predicted)),
      .groups = "drop"
    ) %>%
    select(-x)
})

# results

metrics_df <- bind_rows(results, .id = "Model") %>%
  mutate(Model = gsub("_7day", "", Model))

metrics_df_seedless <- metrics_df %>%
  mutate(
    # define suffix rules
    suffix = case_when(
      str_detect(Model, "_MuSPAD")              ~ "MuSPAD",
      str_detect(Model, "_baseline|_onlyIncidence") ~ "baseline",
      TRUE                                           ~ ""
    ),
    # remove the keyword from Model
    Model = str_remove_all(Model, "_MuSPAD|_baseline|_onlyIncidence"),
    # add suffix to set if needed
    set   = if_else(suffix != "", paste0(set, "_", suffix), set),
    mape = round(mape, 2),
    smape = round(smape, 2),
    rmsle = round(rmsle, 2),
    rmse = round(rmse, 2)
  ) %>%
  select(-suffix)


metrics <- c("mape", "smape", "rmsle", "rmse")

metric_tables_seedless <- lapply(metrics, function(metric) {
  
  metrics_df_seedless %>%
    select(Model, set, all_of(metric)) %>%
    pivot_wider(
      values_from = all_of(metric),
      names_from = set
    ) %>%
    select(
      Model,
      Training_MuSPAD,
      Test_MuSPAD,
      Training_baseline,
      Test_baseline
    )
})

names(metric_tables_seedless) <- toupper(metrics)


#---------------------------------------------------------------------------------------------------------

# 2.) Compute metrics for models with random seeds 


folder <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data/Diff_seeds"

# list all .RData files
files <- list.files(folder, pattern = "\\.RData$", full.names = TRUE)

# create a named list to store data
real_vs_pred <- list()

for (f in files) {
  # create a temporary environment to avoid overwriting
  env <- new.env()
  load(f, envir = env)
  
  # extract objects
  objs <- as.list(env)
  
  # give them unique names based on file name
  names(objs) <- tools::file_path_sans_ext(basename(f))
  
  # add to results
  real_vs_pred <- c(real_vs_pred, objs)
}

# Convert all time columns to Date before binding
real_vs_pred_fixed <- lapply(real_vs_pred, function(df) {
  df %>%
    mutate(time = as.Date(time),
           real_vs_pred = case_when(real_vs_pred == "Predicted" ~ "predicted",
                                    real_vs_pred ==  "Real" ~ "real",
                                    .default = real_vs_pred))
})


results_per_seed <- lapply(real_vs_pred_fixed, function(df) {
  df %>%
    group_by(set, seed) %>%
    summarise(
      x = list(
        pivot_wider(
          cur_data(),
          names_from = real_vs_pred,
          values_from = Incidence
        )
      ),
      mape  = map_dbl(x, ~ mape(.x$real, .x$predicted)),
      smape = map_dbl(x, ~ smape(.x$real, .x$predicted)),
      rmsle = map_dbl(x, ~ rmsle(.x$real, .x$predicted)),
      rmse  = map_dbl(x, ~ rmse(.x$real, .x$predicted)),
      .groups = "drop"
    ) %>%
    select(-x)
})

# results per seed

metrics_df <- bind_rows(results_per_seed, .id = "Model") %>%
  mutate(Model = gsub("_7day", "", Model))

metrics_df_per_seed <- metrics_df %>%
  mutate(
    # define suffix rules
    suffix = case_when(
      str_detect(Model, "_MuSPAD")              ~ "MuSPAD",
      str_detect(Model, "_baseline|_onlyIncidence") ~ "baseline",
      TRUE                                           ~ ""
    ),
    # remove the keyword from Model
    Model = str_remove_all(Model, "_MuSPAD|_baseline|_onlyIncidence|_seeds"),
    # add suffix to set if needed
    set   = if_else(suffix != "", paste0(set, "_", suffix), set),
    
    mape = round(mape, 2),
    smape = round(smape, 2),
    rmsle = round(rmsle, 2),
    rmse = round(rmse, 2)
  ) %>%
  select(-suffix)

#----

# results mean sd

results_summary <- lapply(results_per_seed, function(df) {
  df %>%
    group_by(set) %>%
    summarise(
      across(
        c(mape, smape, rmsle, rmse),
        list(
          mean = ~ mean(.x, na.rm = TRUE),
          sd   = ~ sd(.x, na.rm = TRUE)
        )
      ),
      .groups = "drop"
    )
})


metrics_df <- bind_rows(results_summary, .id = "Model") %>%
  mutate(Model = gsub("_7day", "", Model))

metrics_df_mean_sd <- metrics_df %>%
  mutate(
    # define suffix rules
    suffix = case_when(
      str_detect(Model, "_MuSPAD")              ~ "MuSPAD",
      str_detect(Model, "_baseline|_onlyIncidence") ~ "baseline",
      TRUE                                           ~ ""
    ),
    # remove the keyword from Model
    Model = str_remove_all(Model, "_MuSPAD|_baseline|_onlyIncidence|_seeds"),
    # add suffix to set if needed
    set   = if_else(suffix != "", paste0(set, "_", suffix), set),
    
    mape_mean = round(mape_mean, 2),
    smape_mean = round(smape_mean, 2),
    rmsle_mean = round(rmsle_mean, 2),
    rmse_mean = round(rmse_mean, 2),
    
    mape_sd = round(mape_sd, 2),
    smape_sd = round(smape_sd, 2),
    rmsle_sd = round(rmsle_sd, 2),
    rmse_sd = round(rmse_sd, 2)
  ) %>%
  select(-suffix)


#------------------------------------------------------------------------------


metrics <- c("mape", "smape", "rmsle", "rmse")

metric_tables_mean_sd <- lapply(metrics, function(metric) {
  
  mean_col <- paste0(metric, "_mean")
  sd_col   <- paste0(metric, "_sd")
  
  metrics_df_mean_sd %>%
    mutate(
      value = sprintf(
        "%.2f (± %.2f)",
        .data[[mean_col]],
        .data[[sd_col]]
      )
    ) %>%
    select(Model, set, value) %>%
    pivot_wider(
      values_from = value,
      names_from = set
    ) %>%
    select(
      Model,
      Training_MuSPAD,
      Test_MuSPAD,
      Training_baseline,
      Test_baseline
    )
})

names(metric_tables_mean_sd) <- toupper(metrics)


# Combine results of seedless and seed models

combined_tables <- Map(
  function(original, seed) {
    bind_rows(
      original %>% mutate(across(-Model, as.character)),
      seed
    )
  },
  metric_tables_seedless,
  metric_tables_mean_sd
)

# Save all tables

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Metric_tables")

for (metric in metrics) {
  
  write.csv(
    combined_tables[[toupper(metric)]],
    paste0(toupper(metric), "_Table_three_fourth_Data_7dayPred_seeds.csv"),
    row.names = FALSE
  )
}

# Tables to latex code

cat(toString(knitr::kable(combined_tables$MAPE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$SMAPE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$RMSLE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$RMSE, format = "latex", booktabs = TRUE)))

# \begin{tabular}{lllll}
# \toprule
# Model & Training\_MuSPAD & Test\_MuSPAD & Training\_baseline & Test\_baseline\\
# \midrule
# VAR\_p14 & 0.89 & 0.43 & 1.82 & 0.03\\
# VAR\_p21 & 0.72 & 0.38 & 2.09 & 0.21\\
# VAR\_p7 & 1.81 & 0.53 & 2.76 & 0.03\\
# LSTM\_seeds & 1.51 (± 0.28) & 0.35 (± 0.12) & 8.55 (± 2.83) & 0.50 (± 0.07)\\
# \bottomrule
# \end{tabular}

# \begin{tabular}{lllll}
# \toprule
# Model & Training\_MuSPAD & Test\_MuSPAD & Training\_baseline & Test\_baseline\\
# \midrule
# VAR\_p14 & 0.44 & 0.35 & 0.58 & 0.03\\
# VAR\_p21 & 0.43 & 0.32 & 0.63 & 0.19\\
# VAR\_p7 & 0.47 & 0.42 & 0.61 & 0.03\\
# LSTM\_seeds & 0.38 (± 0.07) & 0.30 (± 0.10) & 0.85 (± 0.21) & 0.40 (± 0.05)\\
# \bottomrule
# \end{tabular}

# \begin{tabular}{lllll}
# \toprule
# Model & Training\_MuSPAD & Test\_MuSPAD & Training\_baseline & Test\_baseline\\
# \midrule
# VAR\_p14 & 0.75 & 0.35 & 0.97 & 0.04\\
# VAR\_p21 & 0.67 & 0.32 & 1.06 & 0.19\\
# VAR\_p7 & 0.88 & 0.42 & 1.06 & 0.04\\
# LSTM\_seeds & 0.70 (± 0.09) & 0.30 (± 0.10) & 1.55 (± 0.33) & 0.40 (± 0.05)\\
# \bottomrule
# \end{tabular}

# \begin{tabular}{lllll}
# \toprule
# Model & Training\_MuSPAD & Test\_MuSPAD & Training\_baseline & Test\_baseline\\
# \midrule
# VAR\_p14 & 31.19 & 29.47 & 53.31 & 2.84\\
# VAR\_p21 & 32.2 & 26.06 & 67.21 & 14.89\\
# VAR\_p7 & 33.8 & 36.58 & 53.04 & 2.51\\
# LSTM\_seeds & 21.87 (± 4.01) & 24.23 (± 8.43) & 104.30 (± 41.47) & 34.82 (± 5.09)\\
# \bottomrule
# \end{tabular}
