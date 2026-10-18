# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

Create_overview_graphic <- function(folder, model_order, show_range = TRUE){

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

  # convert all time columns to Date before binding
  real_vs_pred_fixed <- lapply(real_vs_pred, function(df) {
    df %>%
      mutate(time = as.Date(time),
             real_vs_pred = case_when(real_vs_pred == "Predicted" ~ "predicted",
                                      real_vs_pred ==  "Real" ~ "real",
                                      .default = real_vs_pred))
  })

  # identify which original file names contain "seeds"
  seeds_models <- names(real_vs_pred)[str_detect(names(real_vs_pred), "seeds")]

  real_vs_pred_df <- bind_rows(real_vs_pred_fixed, .id = "Model") %>%
    mutate(has_seeds = str_detect(Model, "seeds")) %>%
    mutate(Model = str_replace_all(Model, c("_7day" = "",
                                            "_\\d+percentData" = "",
                                            "_threefourthData" = "",
                                            "_seeds" = "",
                                            "\\_" = " ",
                                            "p" = "p="))) %>%
    rename(Set = set)

  real_vs_pred_df$Model <- factor(real_vs_pred_df$Model, levels = model_order)

  # separate models with seeds and without seeds
  seeds_data <- real_vs_pred_df %>%
    filter(has_seeds == TRUE)

  no_seeds_data <- real_vs_pred_df %>%
    filter(has_seeds == FALSE)


  # check if folder contains 675percentData
  use_expand_limits <- str_detect(folder, "675percentData")


  if(show_range) {
    # aggregate models with seeds: calculate mean, min, max for each time point and set
    seeds_agg <- seeds_data %>%
      filter(!is.na(Incidence)) %>%
      group_by(Model, time, real_vs_pred, Set) %>%
      summarise(
        Incidence_mean = mean(Incidence, na.rm = TRUE),
        Incidence_min = min(Incidence, na.rm = TRUE),
        Incidence_max = max(Incidence, na.rm = TRUE),
        .groups = 'drop'
      ) %>%
      # ensure min is not below 0
      mutate(Incidence_min = pmax(Incidence_min, 0))

    # combine both datasets
    plot_data <- bind_rows(
      seeds_agg %>% mutate(data_type = "seeds"),
      no_seeds_data %>% mutate(Incidence_mean = Incidence,
                               Incidence_min = NA,
                               Incidence_max = NA,
                               data_type = "no_seeds")
    )

    p <- ggplot(plot_data, aes(y = Incidence_mean, x = time)) +
      # ribbon only for models with seeds (min/max range)
      geom_ribbon(data = filter(plot_data, data_type == "seeds"),
                  aes(ymin = Incidence_min,
                      ymax = Incidence_max,
                      fill = real_vs_pred), alpha = 0.3, show.legend = FALSE) +
      geom_line(aes(color = real_vs_pred)) +
      geom_point(aes(color = real_vs_pred, shape = Set)) +
      theme_bw() +
      {if(use_expand_limits) expand_limits(y = 0) else ylim(0, 375)} +
      xlab("Time") +
      facet_wrap(~ Model, ncol = 2) +
      scale_shape_manual(values = c(4, 16)) +
      scale_color_discrete(labels = c('Predicted', 'Observed')) +
      theme(legend.position = "bottom",
            axis.text = element_text(size = 12),
            axis.title = element_text(size = 13, face = "bold"),
            axis.text.x = element_text(angle = 330, vjust = 0.5),
            legend.text = element_text(size = 12),
            legend.title = element_text(size = 13, face = "bold")) +
      guides(color = guide_legend(title = "Observed vs. Predicted")) +
      scale_x_date(
        breaks = seq(
          from = as.Date("2020-07-01"),
          to = max(plot_data$time),
          by = "3 months"
        ),
        date_labels = "%b %Y"
      )
  } else {
    # version without range: only mean Incidence for models with seeds
    seeds_agg <- seeds_data %>%
      filter(!is.na(Incidence)) %>%
      group_by(Model, time, real_vs_pred, Set) %>%
      summarise(
        Incidence = mean(Incidence, na.rm = TRUE),
        .groups = 'drop'
      )

    # combine both datasets
    plot_data <- bind_rows(
      seeds_agg %>% mutate(data_type = "seeds"),
      no_seeds_data %>% mutate(data_type = "no_seeds")
    )

    p <- ggplot(plot_data, aes(y = Incidence, x = time)) +
      geom_line(aes(color = real_vs_pred)) +
      geom_point(aes(color = real_vs_pred, shape = Set)) +
      theme_bw() +
      ylim(0, 375) +
      xlab("Time") +
      ylab("COVID-19 incidence") +
      facet_wrap(~ Model, ncol = 2) +
      scale_shape_manual(values = c(4, 16)) +
      scale_color_discrete(labels = c('Predicted', 'Observed')) +
      theme(legend.position = "bottom",
            axis.text = element_text(size = 12),
            axis.title = element_text(size = 13, face = "bold"),
            axis.text.x = element_text(angle = 330, vjust = 0.5),
            legend.text = element_text(size = 12),
            legend.title = element_text(size = 13, face = "bold")) +
      guides(color = guide_legend(title = "Observed vs. Predicted")) +
      scale_x_date(
        breaks = seq(
          from = as.Date("2020-07-01"),
          to = max(plot_data$time),
          by = "3 months"
        ),
        date_labels = "%b %Y"
      )
  }

  return(p)
}

#------------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(lubridate)
library(stringr)

# Run separately


# For 65 % data:

folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/65percentData")

model_order <- c("LSTM MuSPAD","LSTM baseline",
                 "VAR MuSPAD p=7","VAR baseline p=7","VAR MuSPAD p=14", "VAR baseline p=14",
                 "VAR MuSPAD p=21", "VAR baseline p=21")

# Version with min/max range
p_with_range <- Create_overview_graphic(folder, model_order, show_range = TRUE)

# Version with mean only
p_mean_only <- Create_overview_graphic(folder, model_order, show_range = FALSE)


# Save

setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

png(file=paste0("ggplot_forecast_65percentData_with_range_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_with_range
dev.off()

png(file=paste0("ggplot_forecast_65percentData_mean_only_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_mean_only
dev.off()

setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

ggsave(
  filename = paste0("ggplot_forecast_65percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_65percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

ggsave(
  filename = paste0("ggplot_forecast_65percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_65percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)


# For 3/4 data (threefourthData)
folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/threefourthData")

model_order <- c("LSTM MuSPAD","LSTM baseline",
                 "VAR MuSPAD p=7","VAR baseline p=7","VAR MuSPAD p=14", "VAR baseline p=14",
                 "VAR MuSPAD p=21", "VAR baseline p=21")

# Version with min/max range
p_with_range <- Create_overview_graphic(folder, model_order, show_range = TRUE)

# Version with mean only
p_mean_only <- Create_overview_graphic(folder, model_order, show_range = FALSE)

# Save
setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

png(file=paste0("ggplot_forecast_threefourthData_with_range_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_with_range
dev.off()

png(file=paste0("ggplot_forecast_threefourthData_mean_only_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_mean_only
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_threefourthData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_threefourthData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

ggsave(
  filename = paste0("ggplot_forecast_threefourthData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_threefourthData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

# ============================================================================

# For 80% data
folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/80percentData")

model_order <- c("LSTM MuSPAD","LSTM baseline",
                 "VAR MuSPAD p=7","VAR baseline p=7","VAR MuSPAD p=14", "VAR baseline p=14",
                 "VAR MuSPAD p=21", "VAR baseline p=21")

# Version with min/max range
p_with_range <- Create_overview_graphic(folder, model_order, show_range = TRUE)

# Version with mean only
p_mean_only <- Create_overview_graphic(folder, model_order, show_range = FALSE)

# Save
setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

png(file=paste0("ggplot_forecast_80percentData_with_range_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_with_range
dev.off()

png(file=paste0("ggplot_forecast_80percentData_mean_only_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_mean_only
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_80percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_80percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

ggsave(
  filename = paste0("ggplot_forecast_80percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_80percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

# ============================================================================

# For full data
folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/fullData")

model_order <- str_replace_all(
  c("LASSO_MuSPAD","LASSO_baseline","MLP_MuSPAD","MLP_baseline" , "LSTM_MuSPAD","LSTM_baseline",
    "VAR_MuSPAD_p7","VAR_baseline_p7","VAR_MuSPAD_p14", "VAR_baseline_p14",
    "VAR_MuSPAD_p21", "VAR_baseline_p21"),
  c("\\_" = " ", "p" = "p="))

# Version with min/max range
p_with_range <- Create_overview_graphic(folder, model_order, show_range = TRUE)

# Version with mean only
p_mean_only <- Create_overview_graphic(folder, model_order, show_range = FALSE)

# Save
setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

png(file=paste0("ggplot_forecast_fullData_with_range_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_with_range
dev.off()

png(file=paste0("ggplot_forecast_fullData_mean_only_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_mean_only
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_fullData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_fullData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

ggsave(
  filename = paste0("ggplot_forecast_fullData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_fullData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)


# For 67.5% data
folder <- file.path(REPO_ROOT, "Compare_models/Graphics/Predicted_values/675percentData")

model_order <- c("LSTM MuSPAD","LSTM baseline",
                 "VAR MuSPAD p=7","VAR baseline p=7","VAR MuSPAD p=14", "VAR baseline p=14",
                 "VAR MuSPAD p=21", "VAR baseline p=21")

# Version with min/max range
p_with_range <- Create_overview_graphic(folder, model_order, show_range = TRUE)

# Version with mean only
p_mean_only <- Create_overview_graphic(folder, model_order, show_range = FALSE)

# Save
setwd(file.path(REPO_ROOT, "Compare_models/Graphics"))

png(file=paste0("ggplot_forecast_675percentData_with_range_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_with_range
dev.off()

png(file=paste0("ggplot_forecast_675percentData_mean_only_",Sys.Date(),".png"), width=28, height=27, unit="cm", res=800)
p_mean_only
dev.off()

ggsave(
  filename = paste0("ggplot_forecast_675percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_675percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

ggsave(
  filename = paste0("ggplot_forecast_675percentData_with_range_",Sys.Date(),".eps"),
  plot = p_with_range,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

ggsave(
  filename = paste0("ggplot_forecast_675percentData_mean_only_",Sys.Date(),".eps"),
  plot = p_mean_only,
  device = cairo_ps,
  width = 20,
  height = 19.8,
  units = "cm",
  dpi = 800
)

