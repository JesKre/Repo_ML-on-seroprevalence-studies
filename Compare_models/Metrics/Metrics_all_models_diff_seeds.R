# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

library(dplyr)
library(tidyverse)
library(lubridate)
library(Metrics)
library(tidyr)
library(purrr)

#---------------------------------------------------------------------------------------------------------

# 1.) Compute metrics for models without random seeds
#     (because these do not have random elements and results stay the same)


folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/fullData")

# list all .RData files
files <- list.files(
  folder,
  pattern = "^(VAR|LASSO|MLP).*\\.RData$",
  full.names = TRUE
)

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


folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/fullData")

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

setwd(file.path(REPO_ROOT, "Compare_models/Metric_tables/keras_tuner"))

for (metric in metrics) {

  write.csv(
    combined_tables[[toupper(metric)]],
    paste0(toupper(metric), "_Table_7dayPred_seeds.csv"),
    row.names = FALSE
  )
}

# Tables to latex code

cat(toString(knitr::kable(combined_tables$MAPE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$SMAPE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$RMSLE, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(combined_tables$RMSE, format = "latex", booktabs = TRUE)))

