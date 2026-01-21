

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
# df_ts_inc <- df_ts_inc[,c(1,2,match(selected_vars,colnames(df_ts_inc)))]

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
load_model_weights_hdf5(model, filepath = "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods/validation_model_final.h5")

# Compile Model
model %>% compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = 'mean_squared_error'
  )

# Train Model

# Use the callback in model training
#epochs_to_save <- c(1, 15, 30)
#setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
#filepath <- sprintf("validation_model_weights_epoch_%%d.h5")
#checkpoint_callback <- CustomCheckpoint$new(filepath = filepath, epochs_to_save = epochs_to_save)

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
# setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
# save_model_weights_hdf5(model, "validation_model_final.h5")


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

# Load the saved weights
load_model_weights_hdf5(model, filepath = "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods/test_model_final.h5")

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list('mean_squared_error')
)

#epochs_to_save <- c(1, 15, 30)
#setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_without_DP/Models_diffOpt")
#filepath <- sprintf("test_model_weights_epoch_%%d.h5")
#checkpoint_callback <- CustomCheckpoint$new(filepath = filepath, epochs_to_save = epochs_to_save)


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
# setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
# save_model_weights_hdf5(model, "test_model_final.h5")


# result on training set
(eval_train <- model %>% evaluate(train_val_dat_s, train_val_dat_y))

# mse: 931.7


pred_val <- model %>% predict(train_val_dat_s)

res_training <- data.frame(Predicted = pred_val, Real = train_val_dat_y, 
                           time = as.Date("2020-01-01") + train_val_dat[,1])


# result on test set
(eval_test <- model %>% evaluate(testdat_s, testdat_y))

# mse: 228.8

pred_val <- model %>% predict(testdat_s)

(res_test <- data.frame(Predicted = pred_val, Real = testdat_y, 
                        time = as.Date("2020-01-01") + testdat[,1]))

plot(history)

mse_MLP_MuSPAD <- data.frame(train_mse = eval_train[[2]], test_mse = eval_test[[2]])

# Save results
setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
write.csv2(mse_MLP_MuSPAD, "mse_MLP_MuSPAD.csv", row.names = FALSE)

#------------------------------------------------------------------------------
# Graphic

r_p_graph_train <- res_training %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Training")

r_p_graph_test <- res_test %>%
  pivot_longer(cols = Predicted:Real, names_to = "real_vs_pred", values_to = "Incidence") %>%
  mutate(set = "Test")



#setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
#write.csv2(r_p_graph, paste0("results_MLP_MuSPAD_varsubsetbasic_no_cities_",Sys.Date(),".csv"))

#setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")
#file <- tail(list.files(pattern = "results_MLP_MuSPAD_varsubsetbasic_no_cities_"), n=1)
#r_p_graph <- read.csv2(file = file)[,-1]

graph_MLP_7day_M <- rbind(r_p_graph_train, r_p_graph_test)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values")
save(graph_MLP_7day_M, file = "MLP_7day_MuSPAD.RData")

ggplot(graph_MLP_7day_M, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Real')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="MLP"))


setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/MLP/MLP_varimp_comp_other_methods")

png(file=paste0("ggplot_MLP_MuSPAD_varsubsetbasic_no_cities_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(graph_MLP_7day_M, aes(y=Incidence, x=time)) + 
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

png(file=paste0("ggplot_MLP_MuSPAD_varsubsetbasic_no_cities_",Sys.Date(),".png"), width=28, height=24, unit="cm", res=800)

ggplot(graph_MLP_7day_M, aes(y=Incidence, x=time)) + 
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



# -----------------------------
# Cluster incidence of ALL data
# -----------------------------

# Use train_val_dat, train_val_dat_y, testdat, testdat_y

df_imp_feat <- rbind(train_val_dat, testdat)
df_imp_y <- c(train_val_dat_y, testdat_y)


# Use elbow method to detect optimal number of clusters

wss <- sapply(1:10, function(k){
  kmeans(df_imp_y, centers = k, nstart = 10)$tot.withinss
})

plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")


#---

# Use 4 clusters
k <- 4

# Perform k-means clustering
set.seed(123)  # for reproducibility
km <- kmeans(df_imp_y, centers = k)

# cluster labels
km$cluster

# Optional: view summary
table(km$cluster)

#  1   2   3   4 
# 15 141  57  65 

df_imp_y[which(km$cluster == 1)]

df_imp_y[which(km$cluster == 2)]

df_imp_y[which(km$cluster == 3)]

df_imp_y[which(km$cluster == 4)]

# Highest incidences: cluster 1
# Lowest incidences:  cluster 2

cluster_low <- which(km$cluster == 2)
cluster_high <- which(km$cluster == 1)

write.csv2(cluster_low, "cluster_low.csv")
write.csv2(cluster_high, "cluster_high.csv")

# -----------------------
# SHAPLEY VALUES ANALYSIS
# -----------------------

# 1.) For low incidences (cluster 2)

features <- data.frame(scale(df_imp_feat))

feat_clus <- data.frame(features[which(km$cluster == 2),])

predictor <- Predictor$new(model, data = features, batch.size = 1)

shapley_list <- lapply(1:nrow(feat_clus), function(g) {
  Shapley$new(predictor, x.interest = feat_clus[g,], sample.size = 80)$results[, -4]
})

# Aggregate Shapley results
shapley_low <- do.call(rbind, shapley_list) %>%
  group_by(feature) %>%
  summarise(across(everything(), ~ mean(., na.rm = TRUE), .names = "{.col}_mean")) %>%
  arrange(desc(abs(phi_mean)))


mean(df_ts_inc[,"mask_restaurant_0"])
# 41.8
mean(df_ts_inc[which(km$cluster == 2),"mask_restaurant_0"])
# 26.7

# -> A decrease in the number of people NOT wearing marks in restaurants
#    let to a lower predicted incidence

mean(df_ts_inc[,"mask_restaurant_1"])
# 50.7
mean(df_ts_inc[which(km$cluster == 2),"mask_restaurant_1"])
# 38.8

# -> A decrease in the number of people wearing marks in restaurants
#    let to a lower predicted incidence


cor(df_ts_inc$Inzidenz, df_ts_inc$mask_restaurant_0)
# 0.71
cor(df_ts_inc$Inzidenz, df_ts_inc$mask_restaurant_1)
# 0.22

plot(df_ts_inc$Inzidenz, df_ts_inc$mask_restaurant_0)

plot(df_ts_inc$Inzidenz, df_ts_inc$mask_restaurant_1)

# Highest number of people wearing mask at restaurants is at middle incidences
# (people are aware of rising incidences)

# Less people wearing masks in restaurants
# 1.) at low incidences (before beginning of mask wearing, no masks available / 
#                        no need anymore, people are immune, incidences are low)

# 2.) at high incidences (restaurants are closed)

plot(df_ts_inc$time_day, df_ts_inc$Inzidenz)

plot(df_ts_inc$time_day, df_ts_inc$mask_restaurant_1)

df_ts_inc %>%
  ggplot(aes(y = scale(Inzidenz), x = time_day)) +
  geom_point() +
  geom_point(aes(y = scale(mask_restaurant_1)), color = "blue") +
  theme_bw() +
  ylab("Scaled incidence (black) & mask_restaurant_1 (blue)")

# Most people were wearing masks at restaurants at the first wave with middle incidences
# when the incidences reduced again

# The model does not predict if the incidence reduces, 
# it only predicts what is the actual incidence



#------------------------------------
# 2.) For high incidences (cluster 1)

feat_clus <- data.frame(features[which(km$cluster == 1),])

predictor <- Predictor$new(model, data = features, batch.size = 1)

shapley_list <- lapply(1:nrow(feat_clus), function(g) {
  Shapley$new(predictor, x.interest = feat_clus[g,], sample.size = 80)$results[, -4]
})

# Aggregate Shapley results
shapley_high <- do.call(rbind, shapley_list) %>%
  group_by(feature) %>%
  summarise(across(everything(), ~ mean(., na.rm = TRUE), .names = "{.col}_mean")) %>%
  arrange(desc(abs(phi_mean)))


mean(df_ts_inc[,"mask_restaurant_0"])
# 41.8
mean(df_ts_inc[which(km$cluster == 1),"mask_restaurant_0"])
# 108.6

# -> An increase in the number of people NOT wearing marks in restaurants
#    let to a higher predicted incidence



# 3.) Compute values for low and high incidences with function

compute_shapley_cluster <- function(model, data, cluster_vector, cluster_id, sample_size = 80) {

  # Scale the features
  features <- data.frame(scale(df_imp_feat))
  
  # Filter rows belonging to the specified cluster
  feat_clus <- data.frame(features[which(cluster_vector == cluster_id), ])
  
  # Create iml Predictor object once
  predictor <- Predictor$new(model, data = features, batch.size = 1)
  
  # Compute Shapley values for each observation in the cluster
  shapley_list <- lapply(1:nrow(feat_clus), function(g) {
    Shapley$new(predictor, x.interest = feat_clus[g, ], sample.size = sample_size)$results[, -4]
  })
  
  # Aggregate results: mean absolute Shapley values per feature
  shapley_agg <- do.call(rbind, shapley_list) %>%
    group_by(feature) %>%
    summarise(across(everything(), ~ mean(., na.rm = TRUE), .names = "{.col}_mean")) %>%
    arrange(desc(abs(phi_mean)))
  
  return(shapley_agg)
}

# Run for low incidence cluster (cluster 2)
shapley_low <- compute_shapley_cluster(model, df_ts_inc, km$cluster, cluster_id = 2)

feat_mean_low <- data.frame(feature = colnames(df_imp_feat), 
                             mean_overall = apply(df_imp_feat, 2, mean), 
                             mean_cluster = apply(df_imp_feat[which(km$cluster == 2),], 2, mean))

shapley_low <- shapley_low %>% left_join(feat_mean_low)

# Run for high incidence cluster (cluster 1)
shapley_high <- compute_shapley_cluster(model, df_ts_inc, km$cluster, cluster_id = 1)

feat_mean_high <- data.frame(feature = colnames(df_imp_feat), 
                             mean_overall = apply(df_imp_feat, 2, mean), 
                             mean_cluster = apply(df_imp_feat[which(km$cluster == 1),], 2, mean))

shapley_high <- shapley_high %>% left_join(feat_mean_high)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
write.csv2(shapley_low, paste0("Var_Imp_shapley_low_inc.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
write.csv2(shapley_high, paste0("Var_Imp_shapley_high_inc.csv"))



# -----------------
# LIME ANALYSIS
# -----------------

compute_lime_cluster <- function(model, data, cluster_vector, cluster_id, sample_size = 80) {
  
  # Scale the features
  features <- data.frame(scale(df_imp_feat))
  
  # Filter rows belonging to the specified cluster
  feat_clus <- data.frame(features[which(cluster_vector == cluster_id), ])
  
  # Create iml Predictor object once
  predictor <- Predictor$new(model, data = features, batch.size = 1)

  # Compute lime values for each observation in the cluster
  lime_list <- lapply(1:nrow(feat_clus), function(g) {
    LocalModel$new(predictor, x.interest = feat_clus[g,], k = ncol(feat_clus))$results[, -6]
  })

  # Aggregate results: mean absolute LIME values per feature
  lime_agg <- do.call(rbind, lime_list) %>%
    group_by(feature) %>%
    summarise(across(everything(), ~ mean(., na.rm = TRUE), .names = "{.col}_mean")) %>%
    arrange(desc(abs(effect_mean)))
  
  return(lime_agg)
}

#data = df_ts_inc
#cluster_vector = km$cluster
#cluster_id = 2
#sample_size = 80

# Run for low incidence cluster (cluster 2)
lime_low <- compute_lime_cluster(model, df_ts_inc, km$cluster, cluster_id = 2)

lime_low <- lime_low %>% select(-x.original_mean) %>% left_join(feat_mean_low)

# Run for high incidence cluster (cluster 1)
lime_high <- compute_lime_cluster(model, df_ts_inc, km$cluster, cluster_id = 1)

lime_high <- lime_high %>% select(-x.original_mean) %>% left_join(feat_mean_high)


setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
write.csv2(lime_low, paste0("Var_Imp_lime_low_inc.csv"))

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\ANN_MLP\\Newest_results\\MLP_varimp_comp_other_methods")
write.csv2(lime_high, paste0("Var_Imp_lime_high_inc.csv"))





