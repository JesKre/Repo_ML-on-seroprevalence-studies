
library(dplyr)
library(tidyverse)
library(lubridate)
library(Metrics)
library(tidyr)
library(purrr)



folder <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values"

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
# x <- pivot_wider(real_vs_pred_fixed[[1]], names_from = real_vs_pred, values_from = Incidence)
# mape(x$real, x$predicted)

# results <- lapply(real_vs_pred_fixed, function(df) {
#   df %>%
#     group_by(set) %>%
#     summarise(
#       mape = {
#         x <- pivot_wider(cur_data(), names_from = real_vs_pred, values_from = Incidence)
#         mape(x$real, x$predicted)
#       },      
#       .groups = "drop"
#     )
# })

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

metrics_df_new <- metrics_df %>%
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

mape_df_new <- metrics_df_new %>%
  select(Model, set, mape) %>%
  pivot_wider(values_from = mape, names_from = set) %>%
  select("Model","Training_MuSPAD","Test_MuSPAD","Training_baseline","Test_baseline")

smape_df_new <- metrics_df_new %>%
  select(Model, set, smape) %>%
  pivot_wider(values_from = smape, names_from = set) %>%
  select("Model","Training_MuSPAD","Test_MuSPAD","Training_baseline","Test_baseline")

rmsle_df_new <- metrics_df_new %>%
  select(Model, set, rmsle) %>%
  pivot_wider(values_from = rmsle, names_from = set) %>%
  select("Model","Training_MuSPAD","Test_MuSPAD","Training_baseline","Test_baseline")

rmse_df_new <- metrics_df_new %>%
  select(Model, set, rmse) %>%
  pivot_wider(values_from = rmse, names_from = set) %>%
  select("Model","Training_MuSPAD","Test_MuSPAD","Training_baseline","Test_baseline")


# Why is rmsle in VAR p7 h7, MuSPAD training NAN?
# Compute sle:

xnew <- pivot_wider(real_vs_pred[["VAR_MuSPAD_p7_h7"]], names_from = real_vs_pred, values_from = Incidence)

(log(xnew["predicted"]+1)-log(xnew["real"]+1))^2

xnew[67,]

# Save all tables

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Compare_models\\Metric_tables")
write.csv(mape_df_new, "MAPE_Table_full_Data_7dayPred.csv")
write.csv(smape_df_new, "SMAPE_Table_full_Data_7dayPred.csv")
write.csv(rmsle_df_new, "RMSLE_Table_full_Data_7dayPred.csv")
write.csv(rmse_df_new, "RMSE_Table_full_Data_7dayPred.csv")

# Tables to latex code

cat(toString(knitr::kable(mape_df_new, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(smape_df_new, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(rmsle_df_new, format = "latex", booktabs = TRUE)))
cat(toString(knitr::kable(rmse_df_new, format = "latex", booktabs = TRUE)))


# Check for smape in LSTM baseline:

LSTM_smape <- pivot_wider(real_vs_pred_fixed[["LSTM_7day_baseline"]], names_from = real_vs_pred, values_from = Incidence) %>% mutate(smape = 2*(abs(real-predicted))/(abs(real)+abs(predicted)))
View(LSTM_smape)
mean(subset(LSTM_smape, set == "Training")$smape)
mean(subset(LSTM_smape, set == "Test")$smape)


