

#------------------------------------------------------------------------------
# Functions


train_with_dp <- function(model, x_train, y_train, x_val, y_val, epochs, batch_size, 
                          num_batches, optimizer, C, sigma, patience, min_delta, phase) {
  # Get the number of training samples
  num_samples <- nrow(x_train)
  sampling_probability <- batch_size / num_samples # Sampling probability based on the size of the batch
  # Initialize a data frame to store the training history
  history <- data.frame(epoch = integer(), loss = numeric(), val_loss = numeric())
  best_loss <- Inf # Set the initial best loss
  counter <- 0 # Initialize a counter for early stopping
  
  for (epoch in 1:epochs) { # Loop over each epoch
    cat("Epoch", epoch, "\n")
    epoch_loss <- 0 # Initialize the epoch loss
    
    i <- 0
    # Loop over each batch before reaching the maximum num_batches
    while (i <= num_batches) {
      i <- i + 1  # Increment loop counter
      
      # At each iteration select batch randomly using poisson sub-sampling
      flags <- sample(c(FALSE, TRUE), size = num_samples, replace = TRUE, prob = c(1 - sampling_probability, sampling_probability)) # replace = TRUE applies to (TRUE, FALSE)
      
      x_batch <- x_train[flags, , drop = FALSE] # Mask training data to get a subset
      y_batch <- y_train[flags] # Mask y data
      
      # Initialize the aggregated gradients for the batch
      batch_gradients <- lapply(model$trainable_variables, function(var) {
        tf$zeros_like(var)
      })
      
      # Loop over each sample in the batch
      for (sample_idx in 1:nrow(x_batch)) {
        x_sample <- x_batch[sample_idx, , drop = FALSE]
        y_sample <- y_batch[sample_idx]
        # Record the operations for gradient calculation
        with(tf$GradientTape() %as% tape, {
          predictions <- model(x_sample)
          loss <- mse_loss(y_sample, predictions)
        })
        # Compute the gradients for the sample
        sample_gradients <- tape$gradient(loss, model$trainable_variables)
        # Clip the gradients
        clipped_sample_gradients <- clip_gradients(sample_gradients, C)
        # Aggregate the gradients for the batch
        batch_gradients <- mapply("+", batch_gradients, clipped_sample_gradients, SIMPLIFY = FALSE) # SIMPLIFY = FALSE: output should not be simplified to a list, it should be returned in an original structure.
        
        # Accumulating the total loss for all samples processed so far in the current epoch
        epoch_loss <- epoch_loss + as.numeric(loss$numpy())
      }
      
      # Generate noise and add it to the aggregated gradients
      noisy_gradients <- lapply(batch_gradients, function(grad) {
        grad + tf$random$normal(shape = tf$shape(grad), mean = 0.0, stddev = sigma * C)
      })
      
      # Average gradients by the batch size
      averaged_gradients <- lapply(noisy_gradients, function(grad) grad / batch_size)
      
      # Apply noisy gradients to the optimizer
      optimizer$apply_gradients(zip_lists(averaged_gradients, model$trainable_variables))
    }
    
    # Save the weights of epochs 1, 15, 30
    
    if(epoch %in% c(1,15,30)){
      
      setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Models_diffOpt_diffSigma")
      # save_model_weights_tf(model, paste0("Model_", optimizer_name, "_sigma", sigma, "_epoch", epoch, "_fold", k))
      save_model_weights_hdf5(model, paste0("Model_", optimizer_name, "_sigma", sigma, "_epoch", epoch, "_fold", k, phase, ".h5"))
      
    }
    
    # Calculate the average epoch loss
    epoch_loss <- epoch_loss / num_samples
    val_predictions <- model(x_val)
    val_loss <- mse_loss(y_val, val_predictions)
    
    cat("Training Loss:", epoch_loss, "Validation Loss:", as.numeric(val_loss$numpy()), "\n")
    
    history <- rbind(history, data.frame(epoch = epoch, loss = epoch_loss, val_loss = as.numeric(val_loss$numpy())))
    
    # if (epoch_loss < best_loss - min_delta) {
    #   best_loss <- epoch_loss
    #   counter <- 0
    # } else {
    #   counter <- counter + 1
    #   if (counter >= patience) {
    #     cat("Early stopping: Validation loss did not improve for", patience, "epochs. Stopping training.\n")
    #     break
    #   }
    # }
  }
  
  return(history)
}



# Clip gradients function
# Function to calculate the L2 norm (Euclidean norm) of a tensor
clip_gradients <- function(gradients, C) {
  l2_norm <- function(x) { 
    tf$sqrt(tf$reduce_sum(tf$square(x))) # Calculate the square root of the sum of the squares of the elements
  }
  clipped_gradients <- lapply(gradients, function(grad) {
    grad_norm <- l2_norm(grad) # Compute the L2 norm of the gradient
    factor <- tf$minimum(1.0, C / grad_norm) # Calculate the clipping factor
    grad * factor # Scale the gradient by the clipping facto
  })
  return(clipped_gradients)
}


#------------------------------------------------------------------------------


library(keras)
library(tensorflow)
library(tidyverse)
library(glmnet)
library(missForest)
library(iml)



setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\Datasets")

file <- tail(list.files(pattern = "binary_classnumber_var_subset_missing_zero"), n=1)
file
load(file)

str(df_ts_inc)

# Dependent variable: incidence
# Independent variables: MuSPAD-variables

head(colnames(df_ts_inc))
tail(colnames(df_ts_inc))

dim(df_ts_inc)


df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day=="168"),]

#------------------------------------------------------------------------------
# 1.) Split data into training/validation (first 9/10) and test set (last 1/10)

# Training/Validation data

train_val_dat <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),-1]

train_val_dat <- missForest(train_val_dat)$ximp

train_val_dat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),1]

# Test data

testdat <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),-1]

testdat <- missForest(testdat)$ximp

testdat_y <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),1]

#------------------------------------------------------------------------------
# 2.) Split training/validation dataset 
#     randomly into training (9/10) and validation (1/10)

# number of folds
n_fold <- 10
# Use always the same random sample:
set.seed(10)
folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = n_fold, labels = FALSE))

#------------------------------------------------------------------------------
# 10-fold-crossvalidation

traindat_list <- list()
traindat_y_list <- list()
valdat_list <- list()
valdat_y_list <- list()


for (k in 1:n_fold) {
  
  # training data
  traindat_1 <- train_val_dat[-which(folds == k, arr.ind = TRUE),]
  # Scale the training data
  traindat_scale <- scale(traindat_1)
  
  not_all_na <- function(x) any(!is.na(x))
  not_all_nan <- function(x) any(!is.nan(x))
  
  traindat <- as.data.frame(traindat_scale) %>% 
    select(where(not_all_na)) %>%
    select(where(not_all_nan))
  
  traindat_list[[k]] <- as.matrix(traindat)
  traindat_y_list[[k]] <- train_val_dat_y[-which(folds == k, arr.ind = TRUE)]
  
  valdat_1 <- train_val_dat[which(folds == k, arr.ind = TRUE),]
  
  # Use the scaling parameters from the training data to scale the validation data
  valdat_scale <- scale(valdat_1, center = attr(traindat_scale, "scaled:center"), 
                        scale = attr(traindat_scale, "scaled:scale"))
  
  valdat <- as.data.frame(valdat_scale) %>% 
    select(colnames(traindat))
  
  valdat_list[[k]] <- as.matrix(valdat)
  
  valdat_y_list[[k]] <- train_val_dat_y[which(folds == k, arr.ind = TRUE)]
  
}

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Validation phase


# Loop over different sigmas

sigma_loop <- c(1.627197265625, 2.7294921875, 9.0625)


# Define optimizers

optimizers <- c("RMSprop", "Adam")


library(doParallel)
library(foreach)

stopImplicitCluster()  # Stop any existing cluster


# Set up parallel backend
cl <- makeCluster(detectCores() - 1)  # Use available cores minus one
registerDoParallel(cl)


# Run cross-validation in parallel
results <- foreach(j = 1:length(optimizers), .combine = 'rbind') %:% 
  
  foreach(i = 1:length(sigma_loop), .combine = 'rbind') %:% 
  
  foreach(k = 1:n_fold, .combine = 'rbind', .packages = c("keras", "tensorflow", "iml", "dplyr")) %dopar% {
    
    optimizer_name <- optimizers[j]
    
  
      # Trainings- und Validierungsdaten für den aktuellen Fold
      traindat <- traindat_list[[k]]
      traindat_y <- traindat_y_list[[k]]
      valdat <- valdat_list[[k]]
      valdat_y <- valdat_y_list[[k]]
      
      
      # Define model
      model <- keras_model_sequential()  
      
      model %>% 
        layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu") %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 1)
      
      #######DP########
      
      # Differential Privacy parameters
      sigma <- sigma_loop[i] #This parameter controls the standard deviation of the noise added to the gradients during training to ensure differential privacy. A larger sigma means more noise.
      C <- 1 #The clipping norm C (S) is used to clip the gradients during training. It limits the maximum norm of the gradients to CC to ensure that no single data point has too much influence on the model, which is important for maintaining privacy.
      batch_size <- 32 #has to be increase because of the noise
      learning_rate <- 0.01
      epochs <- 30
      patience <- 5
      min_delta <- 0
      num_batches <- 8
      
      # Custom loss function for MSE
      # Mean Squared Error (MSE) loss function
      mse_loss <- function(y_true, y_pred) {
        k_mean(k_square(y_pred - y_true))
      }
      
      if(j == 1){
        optimizer = tensorflow::tf$keras$optimizers$legacy$RMSprop(learning_rate = learning_rate)
        optimizer_name = "RMSprop"
      }else if(j == 2){
        optimizer = tensorflow::tf$keras$optimizers$legacy$Adam(learning_rate = learning_rate)
        optimizer_name = "Adam"
      }
      
      
      # Compile the model before training or evaluation
      model %>% compile(
        optimizer = optimizer, #optimizer[[j]],
        loss = mse_loss
      )
      
      # Train the model with DP-SGD
      history_df <- train_with_dp(model, traindat, traindat_y, valdat, valdat_y, epochs, batch_size, num_batches, optimizer, C, sigma, patience, min_delta, phase = "validation")
      
      
      train_loss_1_15_30 <- paste(round(history_df$loss[c(1, 15, 30)], 2), collapse = " ; ")
      val_loss_1_15_30 <- paste(round(history_df$val_loss[c(1, 15, 30)], 2), collapse = " ; ")
      
      # Compute Evaluation Metrics
      train_mse <- (model %>% evaluate(traindat, traindat_y))[1]
      val_mse <- (model %>% evaluate(valdat, valdat_y))[1]
      
      # -----------------------
      # SHAPLEY VALUES ANALYSIS
      # -----------------------
      predictor <- Predictor$new(model, data = data.frame(traindat), batch.size = 1)
      
      shapley_list <- lapply(1:nrow(valdat), function(g) {
        Shapley$new(predictor, x.interest = data.frame(valdat)[g,], sample.size = 80)$results[, -4]
      })
      
      # Aggregate Shapley results
      shapley_df <- do.call(rbind, shapley_list) %>%
        group_by(feature) %>%
        summarise(across(everything(), ~ mean(abs(.), na.rm = TRUE), .names = "{.col}_mean"))
      
      
      shapley_df$Optimizer <- optimizer_name
      shapley_df$Fold <- k
      shapley_df$DP_level <- sigma_loop[i]

      
      setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Variable_Importance_diffOpt_diffSigma\\Shapley")
      write.csv2(shapley_df, paste0("Var_Imp_Shapley_", optimizer_name, "_sigma", sigma_loop[i], "_fold", k, ".csv"))
      
      # -----------------
      # LIME ANALYSIS
      # -----------------
      lime_list <- lapply(1:nrow(valdat), function(g) {
        LocalModel$new(predictor, x.interest = data.frame(valdat)[g,], k = ncol(valdat))$results[, -6]
      })
      
      # Aggregate LIME results
      lime_df <- do.call(rbind, lime_list) %>%
        mutate(x.original = as.numeric(x.original)) %>%
        group_by(feature) %>%
        summarise(across(everything(), ~ mean(abs(.), na.rm = TRUE), .names = "{.col}_mean"))
      
      lime_df$Optimizer <- optimizer_name
      lime_df$Fold <- k
      lime_df$DP_level <- sigma_loop[i]

      
      setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Variable_Importance_diffOpt_diffSigma\\LIME")
      write.csv2(lime_df, paste0("Var_Imp_LIME_", optimizer_name, "_sigma", sigma_loop[i], "_fold", k, ".csv"))
      
      
      # Save final model
      
      setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Models_diffOpt_diffSigma")
      save_model_weights_hdf5(model, paste0("Model_", optimizer_name, "_sigma", sigma, "_fold", k, "_final.h5"))

      # ---------------------
      # COLLECT ALL RESULTS
      # ---------------------
      list(
        results_table = data.frame(
          Optimizer = optimizer_name,
          Sigma = sigma_loop[i],
          Fold = k,
          train_loss_Epochs_1_15_30 = train_loss_1_15_30,
          val_loss_Epochs_1_15_30 = val_loss_1_15_30,
          num_epochs = 30,
          train_mse = round(train_mse, 2),
          val_mse = round(val_mse, 2)
        ),
        shapley_results = shapley_df,
        lime_results = lime_df
      )
      
    }

# Stop Parallel Cluster
stopCluster(cl)

# Extract Dataframes from Nested List
Results_table <- do.call(rbind, results[1:60])
all_shapley_results <- do.call(rbind, results[61:120])
all_lime_results <- do.call(rbind, results[121:180])



setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Variable_Importance_diffOpt_diffSigma\\Shapley")
write.csv2(all_shapley_results, paste0("Var_Imp_shapley_all_abs.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Variable_Importance_diffOpt_diffSigma\\LIME")
write.csv2(all_lime_results, paste0("Var_Imp_lime_all_abs.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
write.csv2(Results_table, paste0("Results_diffOpt_diffSigma_10foldCV_DP_abs_", Sys.Date(), ".csv"))



#------------------------------------------------------------------------------
# Testing phase


train_with_dp_test <- function(model, x_train, y_train, x_val, y_val, epochs, batch_size, 
                          num_batches, optimizer, C, sigma, patience, min_delta, phase) {
  # Get the number of training samples
  num_samples <- nrow(x_train)
  sampling_probability <- batch_size / num_samples # Sampling probability based on the size of the batch
  # Initialize a data frame to store the training history
  history <- data.frame(epoch = integer(), loss = numeric(), val_loss = numeric())
  best_loss <- Inf # Set the initial best loss
  counter <- 0 # Initialize a counter for early stopping
  
  for (epoch in 1:epochs) { # Loop over each epoch
    cat("Epoch", epoch, "\n")
    epoch_loss <- 0 # Initialize the epoch loss
    
    i <- 0
    # Loop over each batch before reaching the maximum num_batches
    while (i <= num_batches) {
      i <- i + 1  # Increment loop counter
      
      # At each iteration select batch randomly using poisson sub-sampling
      flags <- sample(c(FALSE, TRUE), size = num_samples, replace = TRUE, prob = c(1 - sampling_probability, sampling_probability)) # replace = TRUE applies to (TRUE, FALSE)
      
      x_batch <- x_train[flags, , drop = FALSE] # Mask training data to get a subset
      y_batch <- y_train[flags] # Mask y data
      
      # Initialize the aggregated gradients for the batch
      batch_gradients <- lapply(model$trainable_variables, function(var) {
        tf$zeros_like(var)
      })
      
      # Loop over each sample in the batch
      for (sample_idx in 1:nrow(x_batch)) {
        x_sample <- x_batch[sample_idx, , drop = FALSE]
        y_sample <- y_batch[sample_idx]
        # Record the operations for gradient calculation
        with(tf$GradientTape() %as% tape, {
          predictions <- model(x_sample)
          loss <- mse_loss(y_sample, predictions)
        })
        # Compute the gradients for the sample
        sample_gradients <- tape$gradient(loss, model$trainable_variables)
        # Clip the gradients
        clipped_sample_gradients <- clip_gradients(sample_gradients, C)
        # Aggregate the gradients for the batch
        batch_gradients <- mapply("+", batch_gradients, clipped_sample_gradients, SIMPLIFY = FALSE) # SIMPLIFY = FALSE: output should not be simplified to a list, it should be returned in an original structure.
        
        # Accumulating the total loss for all samples processed so far in the current epoch
        epoch_loss <- epoch_loss + as.numeric(loss$numpy())
      }
      
      # Generate noise and add it to the aggregated gradients
      noisy_gradients <- lapply(batch_gradients, function(grad) {
        grad + tf$random$normal(shape = tf$shape(grad), mean = 0.0, stddev = sigma * C)
      })
      
      # Average gradients by the batch size
      averaged_gradients <- lapply(noisy_gradients, function(grad) grad / batch_size)
      
      # Apply noisy gradients to the optimizer
      optimizer$apply_gradients(zip_lists(averaged_gradients, model$trainable_variables))
    }
    
    # Save the weights of epochs 1, 15, 30
    
    if(epoch %in% c(1,15,30)){
      
      setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Models_diffOpt_diffSigma")
      # save_model_weights_tf(model, paste0("Model_", optimizer_name, "_sigma", sigma, "_epoch", epoch))
      save_model_weights_hdf5(model, paste0("Model_", optimizer_name, "_sigma", sigma, "_epoch", epoch, phase, ".h5"))
      
    }
    
    # Calculate the average epoch loss
    epoch_loss <- epoch_loss / num_samples
    val_predictions <- model(x_val)
    val_loss <- mse_loss(y_val, val_predictions)
    
    cat("Training Loss:", epoch_loss, "Validation Loss:", as.numeric(val_loss$numpy()), "\n")
    
    history <- rbind(history, data.frame(epoch = epoch, loss = epoch_loss, val_loss = as.numeric(val_loss$numpy())))
  }
  
  return(history)
}


# Training/Validation data

train_val_dat <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),-1]

train_val_dat <- missForest(train_val_dat)$ximp

# Scale the input training data
traindat_scale <- scale(train_val_dat)

# target training data
train_val_dat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),1]


# Test data

testdat <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),-1]

testdat <- missForest(testdat)$ximp

testdat_y <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),1]

# Use the scaling parameters from the training data to scale the test data
testdat_scale <- scale(testdat, center = attr(traindat_scale, "scaled:center"), 
                      scale = attr(traindat_scale, "scaled:scale"))



# Loop over different sigmas

sigma_loop <- c(1.627197265625, 2.7294921875, 9.0625)


library(doParallel)
library(foreach)

stopImplicitCluster()  # Stop any existing cluster


# Set up parallel backend
cl <- makeCluster(detectCores() - 1)  # Use available cores minus one
registerDoParallel(cl)

# Run cross-validation in parallel
results <- foreach(i = 1:length(sigma_loop), .combine = 'rbind', .packages = c("keras", "tensorflow", "iml", "dplyr")) %dopar% {

  # Define model
  model <- keras_model_sequential()  
  
  model %>% 
    layer_dense(units = 10, input_shape = dim(traindat_scale)[2], activation = "relu") %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 1)
  
  #######DP########
  
  # Differential Privacy parameters
  sigma <- sigma_loop[i] #This parameter controls the standard deviation of the noise added to the gradients during training to ensure differential privacy. A larger sigma means more noise.
  C <- 1 #The clipping norm C (S) is used to clip the gradients during training. It limits the maximum norm of the gradients to CC to ensure that no single data point has too much influence on the model, which is important for maintaining privacy.
  batch_size <- 32 #has to be increase because of the noise
  learning_rate <- 0.01
  epochs <- 30
  patience <- 5
  min_delta <- 0
  num_batches <- 8
  
  # Custom loss function for MSE
  # Mean Squared Error (MSE) loss function
  mse_loss <- function(y_true, y_pred) {
    k_mean(k_square(y_pred - y_true))
  }
  
  # Compile the model before training or evaluation
  model %>% compile(
    optimizer = "adam",
    loss = mse_loss
  )
  
  optimizer = tensorflow::tf$keras$optimizers$legacy$Adam(learning_rate = learning_rate)
  optimizer_name = "Adam"
  
  # Train the model with DP-SGD
  history_df <- train_with_dp_test(model, traindat_scale, train_val_dat_y, testdat_scale, testdat_y, 
                              epochs, batch_size, num_batches, optimizer, C, sigma, patience, 
                              min_delta, phase = "test")
  
  
  train_loss_1_15_30 <- paste(round(history_df$loss[c(1, 15, 30)], 2), collapse = " ; ")
  test_loss_1_15_30 <- paste(round(history_df$val_loss[c(1, 15, 30)], 2), collapse = " ; ")

  # Save final model
  setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP\\Models_diffOpt_diffSigma")
  save_model_weights_hdf5(model, paste0("test_model_final_sigma_",sigma,".h5"))
  
  # predict on training set
  
  pred_t <- model %>% predict(traindat_scale)
  
  res_training <- data.frame(Predicted = pred_t, Real = train_val_dat_y, 
                             time = as.Date("2020-01-01") + train_val_dat[,1],
                             Sigma = sigma_loop[i])
  
  
  # predict on test set

  pred_val <- model %>% predict(testdat_scale)
  
  (res_test <- data.frame(Predicted = pred_val, Real = testdat_y, 
                          time = as.Date("2020-01-01") + testdat[,1],
                          Sigma = sigma_loop[i]))

  # ---------------------
  # COLLECT ALL RESULTS
  # ---------------------
  list(
    results_table = data.frame(
      # Optimizer = optimizer_name,
      Sigma = sigma_loop[i],
      train_loss_Epochs_1_15_30 = train_loss_1_15_30,
      test_loss_Epochs_1_15_30 = test_loss_1_15_30,
      num_epochs = 30
    ),
    training_results = res_training,
    test_results = res_test
  )

}

# Stop Parallel Cluster
stopCluster(cl)

# Extract Dataframes from Nested List
Results_table <- do.call(rbind, results[1:3])
all_training_results <- do.call(rbind, results[4:6])
# all_training_results$Sigma <- c(rep(sigma_loop, each = 251))
all_test_results <- do.call(rbind, results[7:9])
# all_test_results$Sigma <- c(rep(sigma_loop, each = 27))

library(Metrics)

library(dplyr)
library(purrr)

# Define your metrics
metrics <- list(
  mape = mape,
  smape = smape,
  rmsle = rmsle,
  rmse = rmse
)

# Helper function to compute all metrics for a given subset
compute_metrics <- function(Real, Predicted) {
  map_dbl(metrics, ~ .x(Real, Predicted))
}

# Apply to training data
train_metrics <- all_training_results %>%
  group_by(Sigma) %>%
  summarise(across(c(Real, Predicted),
                   list),  # keep grouped data as lists
            .groups = "drop") %>%
  mutate(results = map2(Real, Predicted, compute_metrics)) %>%
  select(Sigma, results) %>%
  unnest_wider(results)

# Apply to test data
test_metrics <- all_test_results %>%
  group_by(Sigma) %>%
  summarise(across(c(Real, Predicted),
                   list),
            .groups = "drop") %>%
  mutate(results = map2(Real, Predicted, compute_metrics)) %>%
  select(Sigma, results) %>%
  unnest_wider(results)

train_metrics <- train_metrics %>% mutate(dataset = "training")
test_metrics  <- test_metrics  %>% mutate(dataset = "test")

all_metrics <- bind_rows(train_metrics, test_metrics)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
write.csv(all_training_results, paste0("Predicted_training_2.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
write.csv(all_test_results, paste0("Predicted_test_2.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
write.csv(Results_table, paste0("Results_diffSigma_test_DP_abs_", Sys.Date(), ".csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
write.csv(all_metrics, paste0("Metrics_diffSigma_test_DP_abs_", Sys.Date(), ".csv"))

