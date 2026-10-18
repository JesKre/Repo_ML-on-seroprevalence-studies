


#------------------------------------------------------------------------------
# Fit ANN model to predict incidence 7 days afterwards on MuSPAD data

library(tidyverse)
library(Metrics)
library(missForest)
library(glmnet)
library(tensorflow)
library(keras)



setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n=1)
load(file)

df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day=="168"),]


# Hyperparameter-Kombinationen
layers_list <- c(1, 2, 3, 4)
epochs_list <- c(30, 45, 60)
# number of folds
f <- 10
repeats <- 10
  

# Update results data frame to include model type
results <- data.frame(Model_Type = character(),
                      Layers = integer(),
                      Epochs = integer(),
                      Mean_Train_MSE = numeric(),
                      SD_Train_MSE = numeric(),
                      Mean_Val_MSE = numeric(),
                      SD_Val_MSE = numeric(),
                      Mean_Test_MSE = numeric(),
                      SD_Test_MSE = numeric())

# Loop over model types: meanImp dataset vs. single variable

for (model_type in c("meanImp", "missForest", "onlyTime")) {
  
  # Define split data based on model type
  
  if (model_type == "meanImp") {

    # Training/Validation data
    train_val_dat <- as.data.frame(makeX(train = df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),-1], 
                                         test = df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),-1], na.impute = T)$x)
    
    # Test data
    testdat <- as.data.frame(makeX(train = df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),-1], 
                                   test = df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),-1], na.impute = T)$xtest)
    
  } else if (model_type == "missForest") {

    # Training/Validation data
    train_val_dat_1 <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),-1]
    train_val_dat <- missForest(train_val_dat_1)$ximp

    # Test data
    testdat_1 <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),-1]
    testdat <- missForest(testdat_1)$ximp
    
  } else {
    
    # Training/Validation data
    train_val_dat <- as.data.frame(df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),2])
    colnames(train_val_dat) = colnames(df_ts_inc)[2]

    # Test data
    testdat <- as.data.frame(df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),2])
    colnames(testdat) = colnames(df_ts_inc)[2]
    
  }

  # y data is the same for all
  
  # Training/Validation data
  train_val_dat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),1]
  
  # Test data
  testdat_y <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),1]
  
  #------------------------------------------------------------------------------
  # Split training/validation dataset 
  # randomly into training (9/10) and validation (1/10)
  
  # Use always the same random sample:
  set.seed(10)
  folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = f, labels = FALSE))
  
  # Let fold 10 be the validation set
  
  # Training data
  traindat_1 <- train_val_dat[-which(folds == 10, arr.ind = TRUE),]
  # Scale the training data
  traindat_scale <- scale(traindat_1)
  traindat <- as.matrix(traindat_scale)
  traindat_y <- train_val_dat_y[-which(folds == 10, arr.ind = TRUE)]
  
  # Validation data 
  valdat_1 <- train_val_dat[which(folds == 10, arr.ind = TRUE),]
  # Use the scaling parameters from the training data to scale the validation data
  valdat <- as.matrix(scale(valdat_1, center = attr(traindat_scale, "scaled:center"), 
                        scale = attr(traindat_scale, "scaled:scale")))
  valdat_y <- train_val_dat_y[which(folds == 10, arr.ind = TRUE)]
  
  # Training/Validation data
  # Scale the training data
  train_val_dat_scale <- scale(train_val_dat)
  train_val_dat_s <- as.matrix(train_val_dat_scale)
  
  # Test data
  # Use the scaling parameters from the training data to scale the test data
  testdat_s <- as.matrix(scale(testdat, center = attr(train_val_dat_scale, "scaled:center"), 
                               scale = attr(train_val_dat_scale, "scaled:scale")))
  
  
  
  ### Keras ####
  

  # Schleife über die Kombinationen
  for (layers in layers_list) {
    for (epochs in epochs_list) {
      
      train_mse_list <- c()
      val_mse_list <- c()
      test_mse_list <- c()
      
      cat("\nTraining", model_type, "Model: Layers =", layers, ", Epochs =", epochs, "\n")
      
      for (r in 1:repeats) {
        
        
  #------------------------------------------------------------------------------
  # 3.) Fit ANN to training data and validate on validation set
  #     repeatedly while changing the hyper parameters
  
        set.seed(r)  # Unterschiedliche Seeds für Replikationen
        tensorflow::tf$random$set_seed(r)
  
        # Modell initialisieren
        model <- keras_model_sequential()
        
        # First layer
        model %>% layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu")
        
        # Add hidden layers only if layers > 1
        if (layers > 1) {
          for (i in 2:layers) {
            model %>% layer_dense(units = 10, activation = "relu")
          }
        }   
        
        # Output Layer
        model %>% layer_dense(units = 1)
        
        # Modell kompilieren
        model %>% compile(
          optimizer = 'rmsprop',
          loss = 'mse',
          metrics = list('mean_squared_error')
        )
        
        # Modell trainieren
        history <- model %>% fit(
          traindat, traindat_y,
          epochs = epochs,
          validation_data = list(valdat, valdat_y),
          batch_size = round((dim(traindat)[1])/10, 0),
          verbose = 0
        )
        
        # Evaluierung
        eval_train <- model %>% evaluate(traindat, traindat_y, verbose = 0)
        eval_val <- model %>% evaluate(valdat, valdat_y, verbose = 0)
    
        # Ergebnisse speichern
        train_mse_list <- c(train_mse_list, eval_train[2])
        val_mse_list <- c(val_mse_list, eval_val[2])
        
        
        #------------------------------------------------------------------------------
        # 4.) Fit ANN to all training and validation data and test it on test set
  
        set.seed(r)  # Unterschiedliche Seeds für Replikationen
        tensorflow::tf$random$set_seed(r)
        
        # Modell initialisieren
        model <- keras_model_sequential()
        
        # First layer
        model %>% layer_dense(units = 10, input_shape = dim(train_val_dat_s)[2], activation = "relu")
        
        # Add hidden layers only if layers > 1
        if (layers > 1) {
          for (i in 2:layers) {
            model %>% layer_dense(units = 10, activation = "relu")
          }
        }   
        
        # Output Layer
        model %>% layer_dense(units = 1)
        
        # Modell kompilieren
        model %>% compile(
          optimizer = 'rmsprop',
          loss = 'mse',
          metrics = list('mean_squared_error')
        )
        
        # Modell trainieren
        history <- model %>% fit(
          train_val_dat_s, train_val_dat_y,
          epochs = epochs,
          batch_size = round((dim(traindat)[1])/10, 0),
          verbose = 0
        )
        
        # Evaluierung
        eval_test <- model %>% evaluate(testdat_s, testdat_y, verbose = 0)
        
        # Ergebnisse speichern
        test_mse_list <- c(test_mse_list, eval_test[2])
        
      }
      
      # Mittelwerte und Standardabweichungen berechnen
      results <- rbind(results, data.frame(
        Model_Type = model_type,
        Layers = layers, 
        Epochs = epochs,
        Mean_Train_MSE = mean(train_mse_list), SD_Train_MSE = sd(train_mse_list),
        Mean_Val_MSE = mean(val_mse_list), SD_Val_MSE = sd(val_mse_list),
        Mean_Test_MSE = mean(test_mse_list), SD_Test_MSE = sd(test_mse_list)
      ))
      
    }
  }

}


results[,4:9] <- round(results[,4:9], digits = 1)

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/ANN_MLP/Newest_results/New_Preprop_vs_time")
write.csv2(results, "MLP_Results_meanImp_vs_missForest_vs_onlyTime.csv", row.names = FALSE)



