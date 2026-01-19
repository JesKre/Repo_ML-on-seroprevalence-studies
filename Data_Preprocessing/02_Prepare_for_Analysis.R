

# -----------------------------------------------------------
#  Feature removal helper (used for BOTH datasets)
# -----------------------------------------------------------
load("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/Datasets/20percent_missings.RData")

features_to_remove_1 <- c(
  "kurzfragen_cov19_tested_febr2020_result_Ich.weiß.das.Ergebnis.meines.Tests.nicht",
  "kurzfragen_cov19_sero_tested_febr2020_result_Ich.weiß.das.Ergebnis.meines.Testes.nicht",
  "kurzfragen_cov19_household_tested_febr2020_yn_Ja..jemand.wurde.getestet..aber.ich.weiß.das.Testergebnis.nicht",
  "kurzfragen_cov19_household_tested_febr2020_yn_Weiß.nicht",
  "langfragen_hand_cleaning_cope_Nein..überhaupt.nicht",
  "quarantine_do.not.know.no.information",
  "employ_retraining.voluntary.service",
  "household_PCR_someone.tested..result.unclear",
  "tested_PCR_do.not.know.no.information",
  "tested_PCR_result.unknown"
)

all_features_to_remove <- unique(c(vars_with_high_na_days, features_to_remove_1))

filter_features <- function(df) {
  df[, !sapply(names(df), function(col) 
    any(str_detect(col, paste(all_features_to_remove, collapse = "|"))))]
}


#------------------------------------------------------------------------------
# 1) Dataset for Non-time-aware Data Analysis (MLP, LASSO)
#------------------------------------------------------------------------------


library(tidyverse)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

(file <- tail(list.files(pattern="Time_series_data_all_cities_levelcount_varsubset_basic_binary"),
             1))
df_ts <- read.csv2(file)

#------------------------------------------------------------------------------
# Create a variable for each city, indicating if this city was sampled at the day

df_ts$present <- 1

# Reshape the data to the desired format
result_table <- reshape2::dcast(df_ts, time_day ~ standort, value.var = "present", fill = 0, fun.aggregate = any)

result_table <- result_table %>%
  mutate(time_day = as.Date(time_day))

df_ts <- df_ts %>%
  select(-present)

#------------------------------------------------------------------------------
# Join RKI Incidence & Create Final Output

ID_LK <- data.frame(
  standort=c("reutlingen","osnabrueck","freiburg","aachen",
             "magdeburg","chemnitz","greifswald","hannover"),
  Landkreis_id=c(08415,03404,08311,05334,15003,14511,13075,03241)
)

Co_Inc <- read.csv("COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv")

Co_Inc_all <- Co_Inc %>%
  mutate(Fit_to_date = as.Date(Meldedatum)-7) %>%
  rename(Inzidenz = Inzidenz_7.Tage) %>%
  right_join(ID_LK, by="Landkreis_id") %>%
  select(Fit_to_date, Inzidenz, standort)

Co_Inc_all_wide <- pivot_wider(Co_Inc_all,
                               names_from = standort,
                               values_from = Inzidenz)

df_ts_inc_1 <- df_ts %>%
  mutate(time_day = as.Date(time_day)) %>%
  left_join(Co_Inc_all_wide,
            by = c("time_day" = "Fit_to_date")) %>%
  mutate(Inzidenz = case_when(
    standort == "aachen" ~ aachen,
    standort == "chemnitz" ~ chemnitz,
    standort == "chemnitz;osnabrueck" ~ (chemnitz + osnabrueck)/2,
    standort == "freiburg" ~ freiburg,
    standort == "freiburg;magdeburg" ~ (freiburg + magdeburg)/2,
    standort == "greifswald" ~ greifswald,
    standort == "hannover" ~ hannover,
    standort == "magdeburg" ~ magdeburg,
    standort == "osnabrueck" ~ osnabrueck,
    standort == "osnabrueck;reutlingen" ~ (osnabrueck + reutlingen)/2,
    standort == "reutlingen" ~ reutlingen,
    TRUE ~ NA_real_
  )) %>%
  select(Inzidenz, everything()) %>%
  select(-X, -standort) %>%
  select(!(hannover:magdeburg)) # %>%
  # full_join(result_table)


df_ts_inc <- cbind(df_ts_inc_1[,1:2], 
                   df_ts_inc_1[,3:length(df_ts_inc_1)] %>%
                     mutate_all(~ifelse(is.nan(.), NA, .)) %>% 
                     select(where(~ any(!is.na(.))))
) %>%
  # set 2020 as intercept
  mutate(time_day = difftime(time_day, as.Date("2020-01-01"))) %>%
  
  # first and last rows contain implausible dates
  slice(-(1:2), -((n() - 1):n()))


# Filter features with too much missings or that contain too little information
df_ts_inc <- filter_features(df_ts_inc)

save(df_ts_inc,
     file=paste0("Time_series_data_all_levelcount_plus_incidence_1W_var_subset_basic_binary_classnumber_",
                 Sys.Date(),".RData"))



#------------------------------------------------------------------------------
# 2) Dataset for Time-series Data Analysis (LSTM, VAR)
#------------------------------------------------------------------------------


library(tidyverse)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
(file <- tail(list.files(pattern = "Time_series_data_all_cities_levelcount_varsubset_basic_binary"), n = 1))
df_ts <- read.csv2(file)[,-1]

# some dates seem to be wrong (no sampling at that times), delete them 
df_ts <- df_ts[-c(1,2,3,283,284),]

df_ts$time_day <- as.Date(df_ts$time_day) 

#------------------------------------------------------------------------------
# add NA-rows for missing dates by adding a vector containing all dates

dates <- data.frame(time_day = seq(as.Date("2020-07-01"), as.Date("2021-08-17"), by="days"))

df_ts_all <- df_ts %>%
  full_join(dates) %>%
  arrange(time_day)

for (i in seq_along(df_ts_all$standort)) {
  if(is.na(df_ts_all$standort[i])){
    df_ts_all$standort[i] <- df_ts_all$standort[i-1]
  }
}

#------------------------------------------------------------------------------
# Create a variable for each city, indicating if this city was sampled at the day

df_ts_all$present <- 1

# Reshape the data to the desired format
result_table <- reshape2::dcast(df_ts_all, time_day ~ standort, value.var = "present", fill = 0, fun.aggregate = any)

result_table <- result_table %>%
  mutate(time_day = as.Date(time_day))

# Print the result
print(result_table)

df_ts_all <- df_ts_all %>%
  select(-present)

#------------------------------------------------------------------------------
# Add 7-day-incidence data from RKI to all city data


# IDs of cities:
# Reutlingen = 08415
# Osnabrueck = 03404
# Freiburg = 08311
# Aachen = 05334
# Magdeburg = 15003
# Chemnitz = 14511
# Greifswald = 13075
# Hannover = 03241 

ID_LK <-  data.frame(standort = c("reutlingen","osnabrueck","freiburg",
                                  "aachen","magdeburg","chemnitz","greifswald","hannover"),
                     Landkreis_id = c(08415,03404,08311,05334,15003,14511,13075,03241))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
Co_Inc <- read.csv("COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv")
str(Co_Inc)

library(tidyverse)

Co_Inc_all <- Co_Inc %>%
  mutate(Fit_to_date = as.Date(Meldedatum)) %>%
  rename(Inzidenz = Inzidenz_7.Tage) %>%
  right_join(ID_LK, by = join_by(Landkreis_id)) %>%
  select(1:3, 7:9)

str(Co_Inc_all)

Co_Inc_all_wide <- pivot_wider(id_cols = Fit_to_date, id_expand = F, Co_Inc_all, names_from = standort, values_from = Inzidenz)

df_ts_all$time_day <- as.Date(df_ts_all$time_day)

table(df_ts_all$standort)

str(Co_Inc_all_wide$Fit_to_date)
str(df_ts_all$time_day)

# Put incidences to MuSPAD data of one week before (-> predict one week later)
df_ts_all_inc_1 <- df_ts_all %>%
  mutate(time_day = as.Date(time_day)) %>% 
  left_join(Co_Inc_all_wide, by = c("time_day" = "Fit_to_date")) %>%
  mutate(Inzidenz = case_when(
    standort == "aachen" ~ aachen,
    standort == "chemnitz" ~ chemnitz,
    standort == "chemnitz;osnabrueck" ~ (chemnitz + osnabrueck) / 2,
    standort == "freiburg" ~ freiburg,
    standort == "freiburg;magdeburg" ~ (freiburg + magdeburg) / 2,
    standort == "greifswald" ~ greifswald,
    standort == "hannover" ~ hannover,
    standort == "magdeburg" ~ magdeburg,
    standort == "osnabrueck" ~ osnabrueck,
    standort == "osnabrueck;reutlingen" ~ (osnabrueck + reutlingen) / 2,
    standort == "reutlingen" ~ reutlingen,
    TRUE ~ NA_real_  # default case, set to NA if none of the conditions are met
  )) %>%
  select(Inzidenz, everything()) %>%
  select(-standort) %>%
  select(!(hannover:magdeburg)) # %>%
  # full_join(result_table)

# replace NaN with NA and remove columns with all NA

not_all_na <- function(x) any(!is.na(x))

df_ts_all_inc <- cbind(df_ts_all_inc_1[,1:2], 
                       df_ts_all_inc_1[,3:length(df_ts_all_inc_1)] %>%
                         mutate_all(~ifelse(is.nan(.), NA, .)) %>% 
                         select(where(not_all_na))
) %>%
  # set 2020 as intercept
  mutate(time_day = difftime(time_day, as.Date("2020-01-01"))) %>%
  
  # first and last rows contain implausible dates
  slice(-(1:2), -((n() - 2):n()))



# Filter features with too much missings or that contain too little information
df_ts_all_inc <- filter_features(df_ts_all_inc)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
save(df_ts_all_inc, file = paste0("Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_",Sys.Date(),".RData"))

