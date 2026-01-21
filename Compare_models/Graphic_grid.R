
Create_overview_graphic <- function(folder, model_order){
  
  # list all .RData files
files <- list.files(folder, pattern = "\\.RData$", full.names = TRUE)

# create a named list to store data
real_vs_pred <- list()

for (f in files) {
  # create a temporary environment to avoid overwriting
  env <- new.env()
  load(f, envir = env)
  
  # extract objects
  objs <- as.list(env)
  
  # give them unique names based on file name
  names(objs) <- tools::file_path_sans_ext(basename(f))
  
  # add to results
  real_vs_pred <- c(real_vs_pred, objs)
}


# Convert all time columns to Date before binding
real_vs_pred_fixed <- lapply(real_vs_pred, function(df) {
  df %>%
    mutate(time = as.Date(time),
           real_vs_pred = case_when(real_vs_pred == "Predicted" ~ "predicted",
                                    real_vs_pred ==  "Real" ~ "real",
                                    .default = real_vs_pred))
})

real_vs_pred_df <- bind_rows(real_vs_pred_fixed, .id = "Model") %>%
  mutate(Model = gsub("_7day", "", Model))

levels(as.factor(real_vs_pred_df$Model))


real_vs_pred_df$Model <- factor(real_vs_pred_df$Model, levels = model_order)

p <- ggplot(real_vs_pred_df, aes(y=Incidence, x=time)) + 
  geom_line(aes(color=real_vs_pred)) +
  geom_point(aes(color=real_vs_pred, shape = set)) +
  theme_bw() +
  ylim(0, 375) +
  facet_wrap(~ Model, ncol = 2) +  # arrange in grid
  scale_shape_manual(values=c(4, 16)) +
  scale_color_discrete(labels=c('Predicted', 'Observed')) +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(title="Observed vs. Predicted")) +
  # scale_x_date(date_breaks = "3 months", date_labels =  "%b %Y")
  scale_x_date(
    breaks = seq(
      from = as.Date("2020-07-01"),
      to   = max(real_vs_pred_df$time),
      by   = "3 months"
    ),
    date_labels = "%b %Y"
  )

return(p)

}

#------------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(lubridate)

# Run separately for 1 (full data) and 2 (three fourth train) 

# For 1: 
folder <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values"

model_order <- c("LASSO_MuSPAD","LASSO_baseline","MLP_MuSPAD","MLP_baseline" , "LSTM_MuSPAD","LSTM_baseline",
                 "VAR_MuSPAD_p07_h7","VAR_onlyIncidence_p07_h7","VAR_MuSPAD_p14_h7", "VAR_onlyIncidence_p14_h7",
                 "VAR_MuSPAD_p21_h7", "VAR_onlyIncidence_p21_h7")

p <- Create_overview_graphic(folder, model_order)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics")

png(file=paste0("ggplot_forecast_all_models_",Sys.Date(),".png"), width=28, height=40, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_all_models_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 28,
  height = 40,
  units = "cm",
  dpi = 800
)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Figures_for_paper")

png(file=paste0("ggplot_forecast_all_models_",Sys.Date(),".png"), width=28, height=40, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_all_models_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 28,
  height = 40,
  units = "cm",
  dpi = 800
)


# For 2: 
folder <- "S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics/Predicted_values/Three_fourth_data"

model_order <- c("LSTM_MuSPAD","LSTM_baseline",
                 "VAR_MuSPAD_p07_h7","VAR_onlyIncidence_p07_h7","VAR_MuSPAD_p14_h7", "VAR_onlyIncidence_p14_h7",
                 "VAR_MuSPAD_p21_h7", "VAR_onlyIncidence_p21_h7")

p <- Create_overview_graphic(folder, model_order)

setwd("S:/PROJACTIVE/MUSPAD-LABORERGEBNISSE/LOKI_ML/muspad-ml-loki-sp2.1/Compare_models/Graphics")

png(file=paste0("ggplot_forecast_three_fourth_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_three_fourth_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 28,
  height = 27,
  units = "cm",
  dpi = 800
)

setwd("S:\\PROJACTIVE\\MUSPAD-LABORERGEBNISSE\\LOKI_ML\\muspad-ml-loki-sp2.1\\Figures_for_paper")

png(file=paste0("ggplot_forecast_three_fourth_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_three_fourth_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 28,
  height = 27,
  units = "cm",
  dpi = 800
)
