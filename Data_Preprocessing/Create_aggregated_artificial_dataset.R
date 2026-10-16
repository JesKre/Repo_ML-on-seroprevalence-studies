

# Generate artificial Dataset

# With synthpop
# Time dependencies are not preserved

# 1) Time-agnostic

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n=1)
file
load(file)

str(df_ts_inc)

df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day=="168"),]

library(missForest)

df_ts_inc_imp <- missForest(df_ts_inc)$ximp

# setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
# file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missForest"), n=1)
# load(file)
# df_ts_all_inc_imp

#install.packages("synthpop")
library(synthpop)
# ?syn

muspad_aggr_art <- syn(df_ts_inc_imp[,-c(1:2)], seed = 2024)

df_muspad_aggr_art <- cbind(df_ts_inc_imp[,c(1:2)], muspad_aggr_art["syn"][["syn"]])

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
write.csv2(df_muspad_aggr_art, "Artificial_data_MuSPAD_time_agnostic.csv")

#------------------------------------------------------------------------------

# 2) Time-aware

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

df_ts_all_inc$time_day <- as.numeric(df_ts_all_inc$time_day)

df_ts_all_inc_imp <- missForest(df_ts_all_inc)$ximp

muspad_aggr_art_ts <- syn(df_ts_all_inc_imp[,-c(1:2)], seed = 2024)

df_muspad_aggr_art_ts <- cbind(df_ts_all_inc_imp[,c(1:2)], muspad_aggr_art_ts["syn"][["syn"]])

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
write.csv2(df_muspad_aggr_art_ts, "Artificial_data_MuSPAD_time_aware.csv")
