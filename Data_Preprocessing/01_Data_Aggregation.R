#------------------------------------------------------------------------------
#------------------- ML in MuSPAD: Time-Series Summary Creation ----------------
#------------------------------------------------------------------------------

# This is a more concise version of the first main part of 
# "ML_LSTM_time_series_data_formating_varsubset_basic_3_binary_classnumber.R"

library(tidyverse)
library(reshape2)

#------------------------------------------------------------------------------
# 0) Load Data & Select Variables
#------------------------------------------------------------------------------

load("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/export_review/cleaned_files/20221220_merge_results_clean.RData")

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/Datasets")
var <- read.csv2("Variable_selection_basic.csv")

df <- df_merge_clean[, c("check_in_time", var$ID_merge), drop = FALSE]


#------------------------------------------------------------------------------
# 1) Basic Cleaning / Fixing Known Issues
#------------------------------------------------------------------------------

# Fix typo
df$kurzfragen_cov19_household_tested_febr2020_yn[
  df$kurzfragen_cov19_household_tested_febr2020_yn ==
    "Ja, jemand wurde getestet, dass Ergebnis war positiv"
] <- "Ja, jemand wurde getestet, das Ergebnis war positiv"

# Convert '<3.80' to numeric 0
df$diasorin_result_quantitativ[df$diasorin_result_quantitativ == "<3.80"] <- "0"

#------------------------------------------------------------------------------
# 2) Create day variable & drop exact timestamp
#------------------------------------------------------------------------------

df <- df %>%
  mutate(time_day = as.factor(str_split_fixed(check_in_time, " ", 2)[,1])) %>%
  select(-check_in_time)

table(is.na(df$time_day))
length(levels(df$time_day))
sort(unique(df$time_day))

#------------------------------------------------------------------------------
# 3) Identify Variable Types
#------------------------------------------------------------------------------

# Binary 0/1 variables
var_01 <- names(which(sapply(df, function(col) all(col %in% c(0,1,NA)))))

# Date / POSIX → convert to numeric distance to participation day
date_vars <- which(sapply(df, inherits, what = "Date") |
                     sapply(df, inherits, what = "POSIXct"))

for (i in date_vars) {
  df[[i]] <- as.numeric(difftime(df$time_day, df[[i]], units = "days"))
}

# Ordinal variable example: health status (explicit mapping)
ordinal_levels <- c("Schlecht","Weniger gut","Gut","Sehr gut","Ausgezeichnet")
if("kurzfragen_healthstatus" %in% names(df)){
  df$kurzfragen_healthstatus <-
    as.numeric(factor(df$kurzfragen_healthstatus, levels = ordinal_levels)) - 1
}

# Convert numeric-like variables to numeric where possible
is_numeric_like <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  suppressWarnings(!all(is.na(as.numeric(x))))
}

df <- df %>%
  mutate(across(where(is_numeric_like),
                ~ suppressWarnings(as.numeric(.))))

#------------------------------------------------------------------------------
# 4) Character Variables → Binary / Categorical Handling
#------------------------------------------------------------------------------

ch_var <- names(which(sapply(df, is.character)))

# Helper: convert to binary with 2 allowed values
bin_fun <- function(v, lev1, lev2){
  v[!(v %in% c(lev1, lev2))] <- NA
  v
}

# Binary groups
var_ja <- unique(which(df == "Ja", arr.ind = TRUE)[,2])
var_mw <- unique(which(df == "Weiblich", arr.ind = TRUE)[,2])
var_np <- unique(which(df == "negativ", arr.ind = TRUE)[,2])

df[,var_ja] <- lapply(df[,var_ja], bin_fun, "Ja","Nein")
df[,var_mw] <- lapply(df[,var_mw], bin_fun, "Männlich","Weiblich")
df[,var_np] <- lapply(df[,var_np], bin_fun, "negativ","positiv")

Bin_var <- unique(c(var_ja,var_mw,var_np))

standardize_binary <- function(var) {
  if (is.character(var)) {
    var <- gsub("^Nein$", "No", var)
    var <- gsub("^Ja$", "Yes", var)
  }
  return(var)
}

df[, Bin_var] <- lapply(df[, Bin_var], standardize_binary)


# Categorical variables
Cat_var <- setdiff(which(sapply(df, is.character)), Bin_var)

# Keep categorical < 11 levels

level_num <- sapply(
  Cat_var,
  function(x)
    length(levels(as.factor(na.omit(df[[x]]))))
)

names(level_num) <- names(df)[Cat_var]

table(level_num)

Cat_var_3 <- names(which(level_num < 11))[-1]

#------------------------------------------------------------------------------
# 5) Identify Variables Missing Entire Days (>20% of days)
#------------------------------------------------------------------------------

total_days <- n_distinct(df$time_day)
threshold <- 0.2

vars_with_high_na_days <- df %>%
  group_by(time_day) %>%
  summarise(across(everything(), ~ all(is.na(.)), .names = "only_na_{.col}")) %>%
  summarise(across(starts_with("only_na_"), ~ sum(.x) / total_days >= threshold)) %>%
  select(where(~ any(.))) %>%
  names() %>%
  sub("^only_na_", "", .)  # Remove prefix to get original variable names

save(vars_with_high_na_days, file="20percent_missings.RData")


#------------------------------------------------------------------------------
# 6) Create Time-Series Datasets
#------------------------------------------------------------------------------

valid_days <- sort(unique(df$time_day[!is.na(df$time_day)]))

## Numeric
# dat_num <- data.frame(time_day = valid_days)
# for(v in num_vars){
#   x <- tapply(df[[v]], df$time_day, mean, na.rm=TRUE)
#   dat_num <- merge(dat_num,
#                    data.frame(time_day=names(x), val=as.vector(x)),
#                    by="time_day", all.x=TRUE)
#   colnames(dat_num)[ncol(dat_num)] <- v
# }

dat_num <- data.frame(time_day = valid_days)

for(v in num_vars){
  x <- tapply(df[[v]], df$time_day, mean, na.rm=TRUE)  # exactly like original
  dat_num <- merge(dat_num,
                   data.frame(time_day=names(x), val=as.vector(x)),
                   by="time_day", all.x=TRUE)
  colnames(dat_num)[ncol(dat_num)] <- v
}

## Binary
Bin_var_all <- c(names(df)[Bin_var], var_01)

# Ensure 0/1 order matches original
for(v in var_01){
  df[[v]] <- factor(df[[v]], levels = c(0,1))   # preserves 0 then 1
}

dat_Bv <- data.frame(time_day = valid_days)

# Aggregate per day and per level
for(v in Bin_var_all){
  # Preserve level order: 0 then 1 for numeric, "No" then "Yes" for character
  levs <- sort(unique(na.omit(df[[v]])), decreasing = FALSE)
  
  x <- df %>%
    filter(!is.na(.data[[v]]), time_day %in% valid_days) %>%
    mutate(!!v := factor(.data[[v]], levels = levs)) %>%
    group_by(time_day, .data[[v]]) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(
      names_from = !!sym(v),
      values_from = count,
      values_fill = 0
    ) %>%
    rename_with(~paste(v, ., sep="_"), -time_day)
  
  # Ensure columns are ordered according to levels
  colnames_order <- paste0(v, "_", levs)
  colnames_order <- colnames_order[colnames_order %in% colnames(x)]
  x <- x[, c("time_day", colnames_order)]
  
  # Merge preserving only the original rows
  dat_Bv <- merge(dat_Bv, x, by="time_day", all.x=TRUE)
}


## Categorical < 11 levels

dat_Cat <- data.frame(time_day = valid_days)

for(v in Cat_var_3){
  # preserve the order of levels as they appear in the original data
  levs <- sort(unique(na.omit(df[[v]])), decreasing = FALSE)
  
  # aggregate counts per day and per level
  x <- df %>%
    filter(!is.na(.data[[v]]), time_day %in% valid_days) %>%
    mutate(!!v := factor(.data[[v]], levels = levs)) %>%
    group_by(time_day, .data[[v]]) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(
      names_from = !!sym(v),
      values_from = count,
      values_fill = 0
    ) %>%
    rename_with(~paste(v, ., sep="_"), -time_day)
  
  # ensure columns appear in the correct order according to levels
  colnames_order <- paste0(v, "_", levs)
  colnames_order <- colnames_order[colnames_order %in% colnames(x)]
  x <- x[, c("time_day", colnames_order)]
  
  # merge with dat_Cat preserving only the original rows
  dat_Cat <- merge(dat_Cat, x, by="time_day", all.x=TRUE)
}

# Replace NAs after merge
dat_Bv[is.na(dat_Bv)] <- 0
dat_Cat[is.na(dat_Cat)] <- 0

#------------------------------------------------------------------------------
# 7) Standort Variables
#------------------------------------------------------------------------------

# Collapsed Standort string
st <- by(df$standort, df$time_day,
         function(x) paste0(names(table(x)), collapse=";"))
df_st <- data.frame(time_day = names(st), standort = unlist(st))


# Presence matrix
df$present <- 1
result_table <- dcast(df, time_day ~ standort,
                      value.var="present", fill=0, fun.aggregate=any)
df$present <- NULL


#------------------------------------------------------------------------------
# 8) Final TS Dataset + Save
#------------------------------------------------------------------------------

df_ts_new <- df_st %>%
  full_join(dat_num, by="time_day") %>%
  full_join(dat_Bv, by="time_day") %>%
  full_join(dat_Cat, by="time_day")

write.csv2(result_table,
           paste0("City_variables_TS_data_",Sys.Date(),".csv"))

write.csv2(df_ts,
           paste0("Time_series_data_all_cities_levelcount_varsubset_basic_",
                  Sys.Date(),".csv"))

