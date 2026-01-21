
###############################################################################
#--------- Comparison of variable importance table ----------------------------
###############################################################################


library(tidyverse)

# MLP

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_varimp_comp_other_methods")

(file <- tail(list.files(pattern = "Var_Imp_shapley_low_inc"), n=1))
shapley_low <- read.csv2(file)[,-1]

MLP_shapley_low <- shapley_low %>%
  rename(MLP_low_Feature = feature,
         MLP_low_Shapley_Value_mean = phi_mean,
         MLP_low_mean_overall = mean_overall,
         MLP_low_mean_cluster = mean_cluster) %>%
  select(-phi.var_mean)


(file <- tail(list.files(pattern = "Var_Imp_shapley_high_inc"), n=1))
shapley_high <- read.csv2(file)[,-1]

MLP_shapley_high <- shapley_high %>%
  rename(MLP_high_Feature = feature,
         MLP_high_Shapley_Value_mean = phi_mean,
         MLP_high_mean_overall = mean_overall,
         MLP_high_mean_cluster = mean_cluster) %>%
  select(-phi.var_mean)



# LSTM 7-day prediction

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")

setwd(".\\transformed_h7")

LSTM_7day_shap_low <- read.csv2("shap_tplus7_allMean_orig_scale_low.csv", fileEncoding = "latin1")[,-1]

LSTM_7day_shap_low <- LSTM_7day_shap_low %>%
  rename(LSTM_7day_low_Feature = Feature,
         LSTM_7day_low_Shap_Value_mean = SHAP_Value_mean,
         LSTM_7day_low_mean_overall = mean_overall,
         LSTM_7day_low_mean_cluster = mean_cluster)


LSTM_7day_shap_high <- read.csv2("shap_tplus7_allMean_orig_scale_high.csv", fileEncoding = "latin1")[,-1]

LSTM_7day_shap_high <- LSTM_7day_shap_high %>%
  rename(LSTM_7day_high_Feature = Feature,
         LSTM_7day_high_Shap_Value_mean = SHAP_Value_mean,
         LSTM_7day_high_mean_overall = mean_overall,
         LSTM_7day_high_mean_cluster = mean_cluster)

which(duplicated(LSTM_7day_shap_high$LSTM_7day_high_Feature))

# LSTM 14-day prediction

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LSTM\\Python_results")

setwd(".\\transformed_h14")

LSTM_14day_shap_low <- read.csv2("shap_tplus14_allMean_orig_scale_low.csv", fileEncoding = "latin1")[,-1]

LSTM_14day_shap_low <- LSTM_14day_shap_low %>%
  rename(LSTM_14day_low_Feature = Feature,
         LSTM_14day_low_Shap_Value_mean = SHAP_Value_mean,
         LSTM_14day_low_mean_overall = mean_overall,
         LSTM_14day_low_mean_cluster = mean_cluster)


LSTM_14day_shap_high <- read.csv2("shap_tplus14_allMean_orig_scale_high.csv", fileEncoding = "latin1")[,-1]

LSTM_14day_shap_high <- LSTM_14day_shap_high %>%
  rename(LSTM_14day_high_Feature = Feature,
         LSTM_14day_high_Shap_Value_mean = SHAP_Value_mean,
         LSTM_14day_high_mean_overall = mean_overall,
         LSTM_14day_high_mean_cluster = mean_cluster)

which(duplicated(LSTM_14day_shap_high$LSTM_14day_high_Feature))


# LASSO

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\LASSO")
(file <- tail(list.files(pattern = "Variablelist_LASSO_varsubsetbasic_"), n=1))
load(file)

LASSO_Coef <- var_list_ordered %>%
  rename(LASSO_Feature = Var,
         LASSO_Coefficient = Coef)

which(duplicated(LASSO_Coef$LASSO_Feature))


# VAR model

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\VAR")
VAR_Coef_p7 <- read.csv2(paste0("Coefficients_mean_over_time_steps_p7_h7.csv"))[,-1]
VAR_Coef_p14 <- read.csv2(paste0("Coefficients_mean_over_time_steps_p14_h7.csv"))[,-1]
VAR_Coef_p21 <- read.csv2(paste0("Coefficients_mean_over_time_steps_p21_h7.csv"))[,-1]

#------------------------------------------------------------------------------

Important_Var_Coef <- cbind(MLP_shapley_low[1:50,], MLP_shapley_high[1:50,], 
                            LASSO_Coef[1:50,], LSTM_7day_shap_low[1:50,], 
                            LSTM_7day_shap_high[1:50,], LSTM_14day_shap_low[1:50,], 
                            LSTM_14day_shap_high[1:50,], VAR_Coef_p7[1:50,],
                            VAR_Coef_p14[1:50,], VAR_Coef_p21[1:50,])

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Variable_importance")
save(Important_Var_Coef, file = paste0("Variable_importance_shapley_comparison_",Sys.Date(),".RData"))
write.csv2(Important_Var_Coef, file = paste0("Variable_importance_shapley_comparison_",Sys.Date(),".csv"), fileEncoding = "latin1")



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Create table with variable titles 


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Variable_importance")
load(file = tail(list.files(pattern = glob2rx("Variable_importance_shapley_comparison_*.RData")), n=1))
# Important_Var_Coef

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
var_sub <- read.csv2("Variable_selection_basic.csv")

# Add the variable "time_day" into the dictionary:
var_sub[nrow(var_sub) + 1,] = c("time_day",NA , "Number of days since 01st Jan 2020", "Number of days since 01st Jan 2020", "Integer", "metric", "Number of days since 01st Jan 2020", "days", NA)

var_sub[nrow(var_sub) + 1,] = c("Inzidenz",NA , "7-day-Incidence of COVID-19 per 100,000 based on RKI-Notifications", "7-day-Incidence of COVID-19 per 100,000 based on RKI-Notifications", "Integer", "metric", "7-day-Incidence of COVID-19 per 100,000 based on RKI-Notifications", "count/100,000", NA)

# grep(sub("_[^_]*$", "", Important_Var_Coef$MLP_low_Feature[19]), var_sub$ID_merge)



#---------------------------------------------------------

# Function to create title column from a given feature column
make_feature_titles <- function(df, feature_col, var_sub) {
  
  print(feature_col)
  
  feature_vals <- df[[feature_col]]
  out <- character(length(feature_vals))
  
  for (i in seq_along(feature_vals)) {
    feat <- feature_vals[i]
    if (is.na(feat)) {
      out[i] <- NA_character_
      next
    }
    
    # try exact match
    match <- var_sub[grep(paste0("^", feat, "$"), var_sub$ID_merge), "variable_title_2"]
    
    if (length(match) > 0) {
      out[i] <- match
    } else {
      last_part <- sub(".*_", "", feat)
      base_id   <- sub("_[^_]*$", "", feat)
      
      if (last_part == "1") {
        out[i] <- paste0(
          var_sub[grep(base_id, var_sub$ID_merge), "variable_title_2"],
          "; Answer: Yes / Positive"
        )
      } else if (last_part == "0") {
        out[i] <- paste0(
          var_sub[grep(base_id, var_sub$ID_merge), "variable_title_2"],
          "; Answer: No / Negative"
        )
      } else {
        out[i] <- paste0(
          var_sub[grep(base_id, var_sub$ID_merge), "variable_title_2"],
          "; Answer: ", last_part
        )
      }
    }
  }
  
  # return vector with same length as input
  out
}

# Loop over all feature columns to create their feature titles

feature_cols <- c("MLP_low_Feature", "MLP_high_Feature", "LASSO_Feature", 
                  "VAR_p7_Feature", "VAR_p14_Feature", "VAR_p21_Feature",
                  "LSTM_7day_low_Feature", "LSTM_7day_high_Feature",
                  "LSTM_14day_low_Feature", "LSTM_14day_high_Feature") 

for (col in feature_cols) {
  new_col <- paste0(col, "_Title")
  Important_Var_Coef[[new_col]] <- make_feature_titles(Important_Var_Coef, col, var_sub)
}



#---------------------------------------------------------

Important_Var_Coef_Shapley <- Important_Var_Coef %>%
  
  select(starts_with("MLP_low_Feat"), starts_with("MLP_low_"),
         starts_with("MLP_high_Feat"), starts_with("MLP_high_"),
         starts_with("LASSO_Feat"), starts_with("LASSO_"),
         starts_with("VAR_p7_Feat"), starts_with("VAR_p7_"),
         starts_with("VAR_p14_Feat"), starts_with("VAR_p14_"),
         starts_with("VAR_p21_Feat"), starts_with("VAR_p21_"),
         starts_with("LSTM_7day_low_Feat"), starts_with("LSTM_7day_low_"),
         starts_with("LSTM_7day_high_Feat"), starts_with("LSTM_7day_high_"),
         starts_with("LSTM_14day_low_Feat"), starts_with("LSTM_14day_low_"),
         starts_with("LSTM_14day_high_Feat"), starts_with("LSTM_14day_high_")
  ) %>%
  mutate(MLP_low_Shapley_Value_mean = round(MLP_low_Shapley_Value_mean, digits = 2),
         MLP_high_Shapley_Value_mean = round(MLP_high_Shapley_Value_mean, digits = 2),
         LASSO_Coefficient = round(LASSO_Coefficient, digits = 2),
         VAR_p7_Coefficient = round(VAR_p7_Coefficient, digits = 5),
         VAR_p14_Coefficient = round(VAR_p14_Coefficient, digits = 5),
         VAR_p21_Coefficient = round(VAR_p21_Coefficient, digits = 5),
         LSTM_7day_low_Shap_Value_mean = round(LSTM_7day_low_Shap_Value_mean, digits = 2),
         LSTM_7day_high_Shap_Value_mean = round(LSTM_7day_high_Shap_Value_mean, digits = 2),
         LSTM_14day_low_Shap_Value_mean = round(LSTM_14day_low_Shap_Value_mean, digits = 2),
         LSTM_14day_high_Shap_Value_mean = round(LSTM_14day_high_Shap_Value_mean, digits = 2)
  )

cols <- c(grep("mean_overall", colnames(Important_Var_Coef_Shapley)),
grep("mean_cluster", colnames(Important_Var_Coef_Shapley)))

for(i in cols){
  Important_Var_Coef_Shapley[,i] <- round(Important_Var_Coef_Shapley[,i], digits = 2)
}


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Variable_importance")
save(Important_Var_Coef_Shapley, file = paste0("Variable_importance_shapley_comparison_with_value_and_title_",Sys.Date(),".RData"))
write.csv2(Important_Var_Coef_Shapley, file = paste0("Variable_importance_shapley_comparison_with_value_and_title_",Sys.Date(),".csv"), fileEncoding = "latin1")



