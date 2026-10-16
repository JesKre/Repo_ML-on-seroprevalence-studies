


#------------------------------------------------------------------------------
# Fit ANN model to predict incidence 7 days afterwards on MuSPAD data

library(tidyverse)
library(Metrics)
library(glmnet)
library(missForest)
library(tensorflow)
library(keras)
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
  
  traindat <- traindat_scale
  
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
# Validation phase

#------------------------------------------------------------------------------
# 3.) Fit ANN to training data and validate on validation set
#     repeatedly while changing the hyper parameters



# Custom callback to save weights only at specific epochs
CustomCheckpoint <- R6::R6Class("CustomCheckpoint",
                                inherit = KerasCallback,
                                public = list(
                                  filepath = NULL,
                                  epochs_to_save = NULL,
                                  
                                  initialize = function(filepath, epochs_to_save) {
                                    self$filepath <- filepath
                                    self$epochs_to_save <- epochs_to_save
                                    #self$optimizer_name <- optimizer_name
                                    #self$fold <- k
                                  },
                                  
                                  on_epoch_end = function(epoch, logs = NULL) {
                                    if ((epoch + 1) %in% self$epochs_to_save) {
                                      save_path <- sprintf(self$filepath, epoch + 1)
                                      self$model$save_weights(save_path)
                                      cat(sprintf("Saved weights at epoch %d to %s\n", epoch + 1, save_path))
                                    }
                                  }
                                )
)



library(doParallel)
library(foreach)


# Set up parallel backend
cl <- makeCluster(detectCores() - 1)  # Use available cores minus one
registerDoParallel(cl)

# Define optimizers
optimizers <- c("RMSprop", "Adam")

# Run cross-validation in parallel
results <- foreach(j = 1:length(optimizers), .combine = 'rbind') %:% 

  foreach(k = 1:n_fold, .combine = 'rbind', .packages = c("keras", "iml", "dplyr")) %dopar% {
    
    optimizer_name <- optimizers[j]
    
    # Get training & validation data
    traindat <- traindat_list[[k]]
    traindat_y <- traindat_y_list[[k]]
    valdat <- valdat_list[[k]]
    valdat_y <- valdat_y_list[[k]]
    
    set.seed(10)
    tensorflow::tf$random$set_seed(10)
    
    # Define Model
    model <- keras_model_sequential() %>%
      layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu") %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 1)
    
    # Compile Model
    if(j == 1){
      model %>% compile(
        optimizer = 'rmsprop',
        loss = 'mse',
        metrics = 'mean_squared_error'
      )
      optimizer_name = 'RMSprop'
      
    }else if(j == 2){
      model %>% compile(
        optimizer = 'adam',
        loss = 'mse',
        metrics = 'mean_squared_error'
      )
      optimizer_name = 'Adam'
      
    }
    
    # Train Model
    
    # Use the callback in model training
    epochs_to_save <- c(1, 15, 30)
    setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_without_DP/Models_diffOpt")
    filepath <- sprintf("model_test_weights_%s_fold%d_epoch_%%d.h5", optimizer_name, k)
    checkpoint_callback <- CustomCheckpoint$new(filepath = filepath, epochs_to_save = epochs_to_save)
    
    num_epochs = 30
    
    history <- model %>% fit(as.matrix(traindat), as.matrix(traindat_y), epochs = num_epochs,
                             validation_data = list(valdat, valdat_y),
                             batch_size = (dim(traindat)[1])/10,
                             # batch_size = (1),
                             # callbacks = callback_early_stopping(monitor = "val_loss", 
                             #                                     min_delta = 0.01, 
                             #                                     mode = "min", 
                             #                                     patience = 5, 
                             #                                     restore_best_weights = TRUE),
                             verbose = 1,
                             callbacks = list(checkpoint_callback)
    )
    
    
    # Extract Loss Values at Epochs 1, 15, 30
    history_df <- data.frame(loss = history$metrics$loss, 
                             val_loss = history$metrics$val_loss, 
                             epoch = c(1:length(history$metrics$loss)))
    
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
    
    # ---------------------
    # COLLECT ALL RESULTS
    # ---------------------
    list(
      results_table = data.frame(
        Optimizer = optimizer_name,
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
Results_table <- do.call(rbind, results[1:20])
all_shapley_results <- do.call(rbind, results[21:40])
all_lime_results <- do.call(rbind, results[41:60])



setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP\\Variable_Importance_diffOpt\\Shapley")
write.csv2(all_shapley_results, paste0("Var_Imp_shapley_all_abs.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP\\Variable_Importance_diffOpt\\LIME")
write.csv2(all_lime_results, paste0("Var_Imp_lime_all_abs.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP")
write.csv2(Results_table, paste0("Results_diffOpt_10fold_abs_preFold_", Sys.Date(), ".csv"))


#------------------------------------------------------------------------------
# Test and save model and create graphics

# Fit ANN to all training and validation data and test it on test set


# Training/Validation data

# Scale the training data
train_val_dat_scale <- scale(train_val_dat)

not_all_na <- function(x) any(!is.na(x))
not_all_nan <- function(x) any(!is.nan(x))

train_val_dat_s <- as.data.frame(train_val_dat_scale) %>% 
  select(where(not_all_na)) %>%
  select(where(not_all_nan))

train_val_dat_s <- as.matrix(train_val_dat_s)

# Test data

# Use the scaling parameters from the training data to scale the test data
testdat_scale <- scale(testdat, center = attr(train_val_dat_scale, "scaled:center"), 
                       scale = attr(train_val_dat_scale, "scaled:scale"))

testdat_s <- as.data.frame(testdat_scale) %>% 
  select(colnames(train_val_dat_s))

testdat_s <- as.matrix(testdat_s)


set.seed(10)  # Unterschiedliche Seeds für Replikationen
tensorflow::tf$random$set_seed(10)

model <-  keras_model_sequential()

model %>% 
  layer_dense(units = 10, input_shape = dim(train_val_dat_s)[2], activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list('mean_squared_error')
)

epochs_to_save <- c(1, 15, 30)
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_without_DP/Models_diffOpt")
filepath <- sprintf("test_model_weights_epoch_%%d.h5")
checkpoint_callback <- CustomCheckpoint$new(filepath = filepath, epochs_to_save = epochs_to_save)



history <- model %>% fit(train_val_dat_s, train_val_dat_y, epochs = 30,
                         batch_size = round((dim(train_val_dat_s)[1])/10, 0),
                         # batch_size = (1),
                         #                         callbacks = callback_early_stopping(monitor = "val_loss", 
                         #                                                             min_delta = 0.01, 
                         #                                                             mode = "min", 
                         #                                                             patience = 5, 
                         #                                                             restore_best_weights = TRUE),
                         callbacks = list(checkpoint_callback),
                         verbose = 1)

# pred <- model %>% predict_classes(testdat_sub_s, batch_size = dim(traindat)[1], verbose = 0)

history

# Save final model
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP\\Models_diffOpt")
save_model_weights_hdf5(model, "test_model_final.h5")


# result on training set
(eval_train <- model %>% evaluate(train_val_dat_s, train_val_dat_y))


pred_val <- model %>% predict(train_val_dat_s)

res_training <- data.frame(Predicted = pred_val, Real = train_val_dat_y, 
                           time = as.Date("2020-01-01") + train_val_dat[,1])


# result on test set
(eval_test <- model %>% evaluate(testdat_s, testdat_y))

pred_val <- model %>% predict(testdat_s)

(res_test <- data.frame(Predicted = pred_val, Real = testdat_y, 
                        time = as.Date("2020-01-01") + testdat[,1]))


plot(history)

eval <- data.frame(train=eval_train, test=eval_test)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP")
write.csv2(eval, paste0("Results_testing_diffOpt_10fold_", Sys.Date(), ".csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP")
(file <- tail(list.files(pattern = "Results_testing_diffOpt_10fold_"), n=1))
eval <- read.csv2(file = file)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP")

write.csv(res_training, paste0("Predicted_training.csv"))

write.csv(res_test, paste0("Predicted_test.csv"))

compute_metrics <- function(data, real_col = "Real", pred_col = "Predicted") {
  metrics <- list(mape = mape, smape = smape, rmsle = rmsle, rmse = rmse)
  x <- map_dbl(metrics, ~ .x(data[[real_col]], data[[pred_col]]))
  y <- cbind(as.data.frame(x), name = names(train_metrics))
  y %>% pivot_wider(names_from = name, values_from = x)
}

train_metrics <- compute_metrics(res_training) %>% mutate(dataset = "training")
test_metrics  <- compute_metrics(res_test) %>% mutate(dataset = "test")

all_metrics <- bind_rows(train_metrics, test_metrics)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_with_DP")
file <- tail(list.files(pattern = "Metrics_diffSigma_test_DP_abs_"), n=1)
all_metrics_DP <- read.csv(file = file)[,-1]

des_order <- c("training", "test")

all_metrics_DP_noDP <- rbind(
  cbind(Sigma = "-", all_metrics),
  all_metrics_DP
) %>%
  mutate(dataset = factor(dataset, levels = des_order)) %>%
  arrange(dataset)

all_metrics_DP_noDP

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP")

write.csv(all_metrics_DP_noDP, paste0("Metrics_diffSigma_test_DP_noDP_",Sys.Date(),".csv"))

paper_tab <- all_metrics_DP_noDP[, c(1,3,5,6)]
paper_tab$smape <- round(paper_tab$smape, digits = 3)
paper_tab$rmse <- round(paper_tab$rmse, digits = 2)

cat(toString(knitr::kable(paper_tab, format = "latex", booktabs = TRUE)))

#------------------------------------------------------------------------------
# Graphic

r_p_graph_train <- res_training %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- res_test %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")

r_p_graph <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\MLP\\MLP_without_DP")
write.csv2(r_p_graph, paste0("Graph_input_testing_diffOpt_10fold_", Sys.Date(), ".csv"))

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/ANN_MLP/Newest_results\\MLP_without_DP")

p <- ggplot(r_p_graph, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw(base_size = 18) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="MLP")) +
  theme(legend.title=element_text(size=22), 
        legend.text=element_text(size=18))

ggsave(
  filename = paste0("comp_DP_savemod_binary_classnumber_missing_zero_missForest_ggplot_MLP_4layer_30epochs_MuSPAD_",Sys.Date(),".png"), # Dateiname
  plot = p,                                 # Grafik-Objekt
  width=12, height=8, dpi = 300             # Grafikgröße und Auflösung
)




