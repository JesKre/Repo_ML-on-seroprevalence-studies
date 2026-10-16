


# Different train-val-test-splits

#------------------------------------------------------------------------------
# Preprocessing: 
# - split into train, val, test
# - normalization
# - imputation


library(missForest)
library(tidyverse)

# Dataset

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")
file <- tail(list.files(pattern = "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"), n=1)
load(file)
# df_ts_all_inc

# Predict 7 days ahead 

make_split_data <- function(p, suffix) {
  
  n <- nrow(df_ts_all_inc)
  cut <- floor(n * p)
  
  test_indices <- (cut + 1):(cut + 7)
  val_indices <- (cut - 6):cut
  train_indices <- 1:cut
  train_indices_v <- 1:(cut - 7)
  
  # -----------------------------
  # Impute training data
  # -----------------------------
  
  traindat_1 <- missForest(
    df_ts_all_inc[train_indices, -2]
  )$ximp
  
  
  # -----------------------------
  # Scale training data
  # -----------------------------
  
  traindat_scale <- scale(traindat_1)
  traindat <- as.data.frame(traindat_scale)
  
  which_na <- apply(traindat, 2, function(x) sum(is.na(x)))
  traindat <- traindat[, which_na == 0]
  
  scale_center_train <- attr(
    traindat_scale, "scaled:center"
  )
  
  scale_scale_train <- attr(
    traindat_scale, "scaled:scale"
  )
  
  
  # -----------------------------
  # Train subset for validation
  # -----------------------------
  
  traindat_scale_v <- scale(
    traindat_1[train_indices_v, ]
  )
  
  traindat_v <- as.data.frame(traindat_scale_v)
  
  which_na <- apply(
    traindat_v, 2, function(x) sum(is.na(x))
  )
  
  traindat_v <- traindat_v[, which_na == 0]
  
  scale_center_train_v <- attr(
    traindat_scale_v, "scaled:center"
  )
  
  scale_scale_train_v <- attr(
    traindat_scale_v, "scaled:scale"
  )
  
  
  # -----------------------------
  # Validation data
  # -----------------------------
  
  valdat_scale <- scale(
    traindat_1[val_indices, ],
    center = scale_center_train_v,
    scale = scale_scale_train_v
  )
  
  valdat <- as.data.frame(valdat_scale)
  
  which_na <- apply(
    valdat, 2, function(x) sum(is.na(x))
  )
  
  valdat <- valdat[, which_na == 0]
  
  
  # -----------------------------
  # Test data
  # -----------------------------
  
  testdat_1 <- missForest(
    df_ts_all_inc[test_indices, -2]
  )$ximp
  
  testdat_scale <- scale(
    testdat_1,
    center = scale_center_train,
    scale = scale_scale_train
  )
  
  testdat <- as.data.frame(testdat_scale)
  
  which_na <- apply(
    testdat, 2, function(x) sum(is.na(x))
  )
  
  testdat <- testdat[, which_na == 0]
  
  
  # -----------------------------
  # Save data
  # -----------------------------
  
  output_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets\\LSTM_data_subsets"
  
  write.csv(
    traindat,
    file.path(output_path,
              paste0("MuSPAD_traindat_LSTM_", suffix, ".csv"))
  )
  
  write.csv(
    testdat,
    file.path(output_path,
              paste0("MuSPAD_testdat_LSTM_", suffix, ".csv"))
  )
  
  write.csv(
    traindat_v,
    file.path(output_path,
              paste0("MuSPAD_traindat_v_LSTM_", suffix, ".csv"))
  )
  
  write.csv(
    valdat,
    file.path(output_path,
              paste0("MuSPAD_valdat_LSTM_", suffix, ".csv")) #,
    #row.names = FALSE
  )
  
  
  # -----------------------------
  # Save scaling parameters
  # -----------------------------
  
  save(
    scale_center_train,
    scale_scale_train,
    scale_center_train_v,
    scale_scale_train_v,
    test_indices,
    val_indices,
    train_indices,
    train_indices_v,
    file = file.path(
      output_path,
      paste0(
        "scaling_parameters_", suffix, "_",
        Sys.Date(), ".RData"
      )
    )
  )
  
  
  # Return useful information
  invisible(list(
    train_indices = train_indices,
    train_indices_v = train_indices_v,
    val_indices = val_indices,
    test_indices = test_indices
  ))
}


make_split_data(0.65,  "65percentData")
make_split_data(0.675, "675percentData")
make_split_data(0.75,  "threefourthData")
make_split_data(0.80,  "80percentData")



#------------------------------------------------------------------------------
# Run Python scripts



#------------------------------------------------------------------------------
# "Postprocessing": 
# backtransforming real and pred values for metric calculation and graphics
#
# Verarbeitet automatisch: threefourthData, 65percentData, 80percentData, 675percentData
#------------------------------------------------------------------------------

# ============================================================================
# CONFIGURATION
# ============================================================================

# Alle Datensätze, die verarbeitet werden sollen
data_subsets <- c("threefourthData", "65percentData", "80percentData", "675percentData")

# Pfade (anpassen falls nötig)
base_path <- "S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML"
datasets_path <- file.path(base_path, "Datasets\\LSTM_data_subsets")
results_base_path <- file.path(base_path, "muspad-ml-loki-sp2.1\\LSTM\\Python_results\\Python_Original\\Different_seeds\\keras_tuner")
output_base_path <- file.path(base_path, "muspad-ml-loki-sp2.1\\Compare_models\\Graphics\\Predicted_values\\Diff_seeds")

prediction <- 7
seeds <- c(seq(1, 10, 1))

# ============================================================================
# LOAD REQUIRED LIBRARIES
# ============================================================================

library(missForest)
library(tidyverse)
library(Metrics)
library(magrittr)

# ============================================================================
# LOAD TIME SERIES DATA (gemeinsam für alle Datensätze)
# ============================================================================

setwd(file.path(base_path, "Datasets"))
ts_pattern <- "Time_series_data_all_levelcount_plus_current_incidence_var_subset_basic_all_dates_missing_zero_"
ts_files <- list.files(pattern = ts_pattern)

if (length(ts_files) == 0) {
  stop("Keine Time-Series-Datendatei gefunden")
}

ts_file <- tail(ts_files, n = 1)
cat("Lade Time-Series-Daten:", ts_file, "\n")
load(file = ts_file)
# df_ts_all_inc

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

back_scaling <- function(x, scale_center, scale_scale) {
  return(x * scale_scale + scale_center)
}

clean_predictions <- function(pred_vector) {
  pred_vector %>%
    str_remove_all("\\[") %>%
    str_remove_all("\\]") %>%
    as.numeric()
}

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

process_dataset <- function(data_subset, model) {
  
  cat("\n", strrep("=", 70), "\n")
  cat("Verarbeite", model, "Modell für", data_subset, "\n")
  cat(strrep("=", 70), "\n")
  
  # Determine output directory based on data_subset
  results_dir <- file.path(results_base_path, paste0("results_h7_", data_subset))
  
  if (!dir.exists(results_dir)) {
    warning(paste("Results-Verzeichnis nicht gefunden:", results_dir))
    return(NULL)
  }
  
  setwd(results_dir)
  
  if(model == "MuSPAD"){
    f <- ""
  } else if(model == "onlyIncidence"){
    f <- "_only_incidence"
  } else {
    stop("Unknown model:", model)
  }
  
  # Load scaling parameters for this data subset
  setwd(file.path(base_path, "Datasets"))
  pattern <- paste0("scaling_parameters_", data_subset, "_")
  scaling_files <- list.files(pattern = pattern)
  
  if (length(scaling_files) == 0) {
    warning(paste("Keine Scaling-Parameter-Dateien gefunden für:", data_subset))
    return(NULL)
  }
  
  file <- tail(scaling_files, n = 1)
  cat("Lade Scaling-Parameter:", file, "\n")
  load(file = file)
  
  results <- list()
  
  for (i in seeds) {
    
    setwd(results_dir)
    
    # Read training predictions
    train_file <- paste0("train_predictions", f, "_tplus7_seed", i, ".csv")
    if (!file.exists(train_file)) {
      warning(paste("Trainings-Datei nicht gefunden:", train_file))
      next
    }
    
    real_vs_pred_train <- read.csv(train_file)
    
    # Get scaling parameters for Inzidenz
    scale_center_inc <- scale_center_train[which(names(scale_center_train) == "Inzidenz")]
    scale_scale_inc <- scale_scale_train[which(names(scale_scale_train) == "Inzidenz")]
    
    # Clean and back-transform predictions
    real_vs_pred_train$predicted <- clean_predictions(real_vs_pred_train$predicted)
    real_vs_pred_train <- data.frame(sapply(real_vs_pred_train, 
                                            function(x) back_scaling(x, scale_center_inc, scale_scale_inc)))
    
    # Add time information
    t_train <- df_ts_all_inc[1:(test_indices[1]-1), 2]
    real_vs_pred_train$time <- 
      as.Date("2020-01-01") + t_train[(prediction+1):(length(t_train))]
    
    # Format for plotting
    r_p_graph_train <- real_vs_pred_train %>%
      pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
      mutate(set = "Training",
             seed = i)
    
    # Read test predictions
    test_file <- paste0("test_predictions", f, "_tplus7_seed", i, ".csv")
    if (!file.exists(test_file)) {
      warning(paste("Test-Datei nicht gefunden:", test_file))
      next
    }
    
    real_vs_pred_test <- read.csv(test_file)
    
    # Clean and back-transform predictions
    real_vs_pred_test$predicted <- clean_predictions(real_vs_pred_test$predicted)
    real_vs_pred_test <- data.frame(sapply(real_vs_pred_test, 
                                           function(x) back_scaling(x, scale_center_inc, scale_scale_inc)))
    
    # Add time information
    t_test <- df_ts_all_inc[test_indices, 2]
    real_vs_pred_test$time <- as.Date("2020-01-01") + t_test
    
    # Format for plotting
    r_p_graph_test <- real_vs_pred_test %>%
      pivot_longer(cols = real:predicted, names_to = "real_vs_pred", values_to = "Incidence") %>%
      mutate(set = "Test",
             seed = i)
    
    # Combine training and test results
    r_p_graph <- rbind(r_p_graph_train, r_p_graph_test)
    results[[i]] <- r_p_graph
    
    cat("  Seed", i, "✓\n")
  }
  
  if (length(results) == 0) {
    warning(paste("Keine Ergebnisse für", model, "-", data_subset))
    return(NULL)
  }
  
  # Combine all results
  results <- do.call(rbind, results)
  rownames(results) <- NULL
  
  return(results)
}

# ============================================================================
# MAIN LOOP: Process all data subsets
# ============================================================================

cat("\n", strrep("#", 70), "\n")
cat("# STARTE VERARBEITUNG ALLER DATENSÄTZE", "\n")
cat(strrep("#", 70), "\n")

for (data_subset in data_subsets) {
  
  # Process MuSPAD model
  graph_LSTM_M <- process_dataset(data_subset, "MuSPAD")
  
  if (!is.null(graph_LSTM_M)) {
    # Determine output directory
    output_subdir <- tolower(gsub("Data", "_data", data_subset))
    output_dir <- file.path(output_base_path, paste0(output_subdir))
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    setwd(output_dir)
    output_file <- paste0("LSTM_7day_MuSPAD_seeds_", data_subset, ".RData")
    save(graph_LSTM_M, file = output_file)
    cat("✓ Gespeichert:", file.path(output_dir, output_file), "\n")
  }
  
  # Process baseline model (onlyIncidence)
  graph_LSTM_BL <- process_dataset(data_subset, "onlyIncidence")
  
  if (!is.null(graph_LSTM_BL)) {
    # Use same output directory
    output_subdir <- tolower(gsub("Data", "_data", data_subset))
    output_dir <- file.path(output_base_path, paste0(output_subdir))
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    setwd(output_dir)
    output_file <- paste0("LSTM_7day_baseline_seeds_", data_subset, ".RData")
    save(graph_LSTM_BL, file = output_file)
    cat("✓ Gespeichert:", file.path(output_dir, output_file), "\n")
  }
}

cat("\n", strrep("#", 70), "\n")
cat("# VERARBEITUNG ABGESCHLOSSEN", "\n")
cat(strrep("#", 70), "\n\n")

# ============================================================================
# OPTIONAL: GENERATE PLOTS (uncomment to use for all datasets)
# ============================================================================

# for (data_subset in data_subsets) {
#   
#   output_subdir <- tolower(gsub("Data", "_data", data_subset))
#   output_dir <- file.path(output_base_path, paste0(output_subdir, "/Diff_seeds"))
#   setwd(output_dir)
#   
#   # Load saved data
#   load(paste0("LSTM_7day_MuSPAD_seeds_", data_subset, ".RData"))
#   load(paste0("LSTM_7day_baseline_seeds_", data_subset, ".RData"))
#   
#   # Plot MuSPAD
#   png(file = paste0("ggplot_forecast_lstm_muspad_varsubsetbasic_", data_subset, "_", Sys.Date(), ".png"), 
#       width = 28, height = 24, unit = "cm", res = 800)
#   
#   print(ggplot(graph_LSTM_M, aes(y = Incidence, x = time)) + 
#     geom_line(aes(color = real_vs_pred)) +
#     geom_point(aes(color = real_vs_pred, shape = set)) +
#     theme_bw() +
#     scale_shape_manual(values = c(4, 16)) +
#     scale_color_discrete(labels = c('Predicted', 'Real')) +
#     theme(legend.position = "bottom") +
#     guides(color = guide_legend(title = "LSTM")))
#   
#   dev.off()
#   
#   # Plot Baseline
#   png(file = paste0("ggplot_forecast_lstm_onlyincidence_", data_subset, "_", Sys.Date(), ".png"), 
#       width = 28, height = 24, unit = "cm", res = 800)
#   
#   print(ggplot(graph_LSTM_BL, aes(y = Incidence, x = time)) + 
#     geom_line(aes(color = real_vs_pred)) +
#     geom_point(aes(color = real_vs_pred, shape = set)) +
#     theme_bw() +
#     scale_shape_manual(values = c(4, 16)) +
#     scale_color_discrete(labels = c('Predicted', 'Real')) +
#     theme(legend.position = "bottom") +
#     guides(color = guide_legend(title = "LSTM")))
#   
#   dev.off()
# }


