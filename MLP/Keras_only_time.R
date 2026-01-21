

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

# str(df_ts_inc)

# Dependent variable: incidence
# Independent variables: MuSPAD-variables

# head(colnames(df_ts_inc))
# tail(colnames(df_ts_inc))

dim(df_ts_inc)


df_ts_inc$time_day <- as.numeric(df_ts_inc$time_day)
df_ts_inc <- df_ts_inc[-which(df_ts_inc$time_day=="168"),]

#------------------------------------------------------------------------------
# 1.) Split data into training/validation (first 9/10) and test set (last 1/10)

# Training/Validation data
train_val_dat <- as.data.frame(df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),2])
colnames(train_val_dat) = colnames(df_ts_inc)[2]

train_val_dat_y <- df_ts_inc[1:(dim(df_ts_inc)[1]/10*9),1]

# Test data
testdat <- as.data.frame(df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),2])
colnames(testdat) = colnames(df_ts_inc)[2]

testdat_y <- df_ts_inc[(dim(df_ts_inc)[1]/10*9+1):(dim(df_ts_inc)[1]),1]


#------------------------------------------------------------------------------
# 2.) Split training/validation dataset 
#     randomly into training (9/10) and validation (1/10)

# number of folds
n_fold <- 10
# Use always the same random sample:
set.seed(10)

folds <- sample(cut(1:(dim(train_val_dat)[1]), breaks = n_fold, labels = FALSE))

# Let fold 10 be the validation set

# training data

traindat_1 <- train_val_dat[-which(folds == 10, arr.ind = TRUE),]
# Scale the training data
traindat_scale <- scale(traindat_1)

traindat <- as.matrix(traindat_scale)

traindat_y <- train_val_dat_y[-which(folds == 10, arr.ind = TRUE)]

# validation data

valdat_1 <- train_val_dat[which(folds == 10, arr.ind = TRUE),]

# Use the scaling parameters from the training data to scale the validation data
valdat <- scale(valdat_1, center = attr(traindat_scale, "scaled:center"), 
                scale = attr(traindat_scale, "scaled:scale"))


valdat_y <- train_val_dat_y[which(folds == 10, arr.ind = TRUE)]


#------------------------------------------------------------------------------
# Validation phase

#------------------------------------------------------------------------------
# 3.) Fit ANN to training data and validate on validation set
#     repeatedly while changing the hyper parameters

library(tensorflow)
library(keras)
library(iml)
library(tidyverse)


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

set.seed(10)
tensorflow::tf$random$set_seed(10)

# Define Model
model <- keras_model_sequential() %>%
  layer_dense(units = 10, input_shape = dim(traindat)[2], activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1)

# Load the saved weights
# load_model_weights_hdf5(model, filepath = "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods/validation_model_final_oT.h5")

# Compile Model
model %>% compile(
  optimizer = 'adam',
  loss = 'mse',
  metrics = 'mean_squared_error'
)

# Train Model

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
                         verbose = 0,
                         #callbacks = list(checkpoint_callback)
)

# Compute Evaluation Metrics
(train_mse <- (model %>% evaluate(traindat, traindat_y))[1])
(val_mse <- (model %>% evaluate(valdat, valdat_y))[1])

# Save final model
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
save_model_weights_hdf5(model, "validation_model_final_oT.h5")


#------------------------------------------------------------------------------
# Test and save model and create graphics

# Fit ANN to all training and validation data and test it on test set


# Training/Validation data

# Scale the training data
train_val_dat_s <- scale(train_val_dat)

# Test data

# Use the scaling parameters from the training data to scale the test data
testdat_s <- scale(testdat, center = attr(train_val_dat_s, "scaled:center"), 
                       scale = attr(train_val_dat_s, "scaled:scale"))


set.seed(10)  # Unterschiedliche Seeds für Replikationen
tensorflow::tf$random$set_seed(10)

model <-  keras_model_sequential()

model %>% 
  layer_dense(units = 10, input_shape = dim(train_val_dat_s)[2], activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1)

# Load the saved weights
# load_model_weights_hdf5(model, filepath = "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods/test_model_final_oT.h5")

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list('mean_squared_error')
)

history <- model %>% fit(train_val_dat_s, train_val_dat_y, epochs = 30,
                         batch_size = round((dim(traindat)[1])/10, 0),
                         # batch_size = (1),
                         #                         callbacks = callback_early_stopping(monitor = "val_loss", 
                         #                                                             min_delta = 0.01, 
                         #                                                             mode = "min", 
                         #                                                             patience = 5, 
                         #                                                             restore_best_weights = TRUE),
                         # callbacks = list(checkpoint_callback),
                         verbose = 1)



# Save final model
setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
save_model_weights_hdf5(model, "test_model_final_oT.h5")


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

mse_MLP_only_time <- data.frame(train_mse = eval_train[[2]], test_mse = eval_test[[2]])

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
write.csv2(mse_MLP_only_time, "mse_MLP_only_time.csv", row.names = FALSE)

#------------------------------------------------------------------------------
# Graphic

r_p_graph_train <- res_training %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- res_test %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")

r_p_graph <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
write.csv2(r_p_graph, paste0("results_MLP_only_time_",Sys.Date(),".csv"))

#setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
#file <- tail(list.files(pattern = "results_MLP_only_time_"), n=1)
#r_p_graph <- read.csv2(file = file)[,-1]
#r_p_graph$time <- as.Date(r_p_graph$time)


graph_MLP_7day_BL <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values")
save(graph_MLP_7day_BL, file = "MLP_7day_baseline.RData")

ggplot(r_p_graph, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="MLP"))


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")

png(file=paste0("ggplot_MLP_only_time_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(r_p_graph, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw(base_size = 18) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="MLP")) +
  theme(legend.title=element_text(size=22), 
        legend.text=element_text(size=18))

dev.off()



setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Single_graphics")

png(file=paste0("ggplot_MLP_only_time_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(r_p_graph, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw(base_size = 18) +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="MLP")) +
  theme(legend.title=element_text(size=22), 
        legend.text=element_text(size=18))

dev.off()




