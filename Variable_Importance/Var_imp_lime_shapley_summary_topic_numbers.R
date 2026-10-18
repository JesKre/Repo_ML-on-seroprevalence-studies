# Run from the repository root; keep that base when later sections change directory.
REPO_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!dir.exists(file.path(REPO_ROOT, "VAR")) || !file.exists(file.path(REPO_ROOT, "README.md")))
  stop("Run this script from the repository root (the directory containing README.md and VAR/).")
DATA_DIR <- Sys.getenv("MUSPAD_DATA_DIR", file.path(REPO_ROOT, "Datasets"))

library(readxl)

library(stringr)
library(dplyr)

library(ggplot2)

#----------------------------------
# LIME
#----------------------------------

setwd(file.path(REPO_ROOT, "Variable_Importance"))
topics <- read_xlsx("Variable_importance_lime_summary_topic_numbers_colored.xlsx")
overall_topics <- read_xlsx("Topic_numbers_overall.xlsx", range = "D11:E19")

topics <- topics %>%
  mutate(across(MLP_low_Feature:last_col(), ~ str_extract(., "\\d")))

setwd(file.path(REPO_ROOT, "Variable_Importance"))
write_csv2(topics, "Variable_importance_lime_summary_topic_numbers.csv")

# topics_long <- topics %>%
#   pivot_longer(cols = MLP_low_Feature:last_col(),
#                names_to = "Model",
#                values_to = "Number_of_Variables") %>%
#   mutate(Model = factor(Model, levels = colnames(topics)[-1]),
#     Model_group = ifelse(as.numeric(factor(Model)) <= 5, "Row 1", "Row 2"))

topics_long <- topics %>%
  mutate(
    across(
      MLP_low_Feature:LSTM_14day_high_Feature, as.numeric
    )) %>%
  left_join(overall_topics) %>%
  mutate(
    across(
      MLP_low_Feature:LSTM_14day_high_Feature,
      ~ .x / Number #,
      #.names = "{.col}_ratio"
    )) %>%
  pivot_longer(cols = MLP_low_Feature:LSTM_14day_high_Feature,
               names_to = "Model",
               values_to = "Ratio_top50_to_overall_variable_number") %>%
  mutate(Model = factor(Model, levels = colnames(topics)[-1]),
         Model_group = ifelse(as.numeric(factor(Model)) <= 5, "Row 1", "Row 2"))


# ggplot(data=topics_long, aes(y=Number_of_Variables, x=Model, group=Topic)) +
#   geom_bar(data=topics_long, aes(y=Number_of_Variables, fill=Topic), stat="identity",
#            position=position_dodge(width=0.9)) +
#   facet_wrap(~ Model_group, nrow = 2, scales = "free_x") +
#   theme_bw() + ylab("Number of features per topic (in 50 most important features according to LIME)") +
#   theme(strip.text = element_blank())

p <- ggplot(data=topics_long, aes(y=Ratio_top50_to_overall_variable_number, x=Model, group=Topic)) +
  geom_bar(data=topics_long, aes(y=Ratio_top50_to_overall_variable_number, fill=Topic), stat="identity",
           position=position_dodge(width=0.9)) +
  facet_wrap(~ Model_group, nrow = 2, scales = "free_x") +
  theme_bw() + ylab("Ratio of feature number in 50 most important features (LIME) to overall feature number per topic") +
  theme(strip.text = element_blank())


setwd(file.path(REPO_ROOT, "Variable_Importance"))

png(file=paste0("Explainability_lime_topic_barchart_",Sys.Date(),".png"), width=32, height=24, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("Explainability_lime_topic_barchart_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 32,
  height = 24
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

png(file=paste0("Explainability_lime_topic_barchart_",Sys.Date(),".png"), width=32, height=24, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("Explainability_lime_topic_barchart_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 32,
  height = 24
)


#----------------------------------
# Shapley
#----------------------------------

setwd(file.path(REPO_ROOT, "Variable_Importance"))
topics <- read_xlsx("Variable_importance_shapley_summary_topic_numbers_colored.xlsx")

topics <- topics %>%
  mutate(across(MLP_low_Feature:last_col(), ~ str_extract(., "\\d")))

setwd(file.path(REPO_ROOT, "Variable_Importance"))
write_csv2(topics, "Variable_importance_shapley_summary_topic_numbers.csv")

# topics_long <- topics %>%
#   pivot_longer(cols = MLP_low_Feature:last_col(),
#                names_to = "Model",
#                values_to = "Number_of_Variables") %>%
#   mutate(Model = factor(Model, levels = colnames(topics)[-1]),
#          Model_group = ifelse(as.numeric(factor(Model)) <= 5, "Row 1", "Row 2"))

topics_long <- topics %>%
  mutate(
    across(
      MLP_low_Feature:LSTM_14day_high_Feature, as.numeric
    )) %>%
  left_join(overall_topics) %>%
  mutate(
    across(
      MLP_low_Feature:LSTM_14day_high_Feature,
      ~ .x / Number #,
      #.names = "{.col}_ratio"
    )) %>%
  pivot_longer(cols = MLP_low_Feature:LSTM_14day_high_Feature,
               names_to = "Model",
               values_to = "Ratio_top50_to_overall_variable_number") %>%
  mutate(Model = factor(Model, levels = colnames(topics)[-1]),
         Model_group = ifelse(as.numeric(factor(Model)) <= 5, "Row 1", "Row 2"))


# ggplot(data=topics_long, aes(y=Number_of_Variables, x=Model, group=Topic)) +
#   geom_bar(data=topics_long, aes(y=Number_of_Variables, fill=Topic), stat="identity",
#            position=position_dodge(width=0.9)) +
#   facet_wrap(~ Model_group, nrow = 2, scales = "free_x") +
#   theme_bw() + ylab("Number of features per topic (in 50 most important features according to SHAP/Shapley)") +
#   theme(strip.text = element_blank())


p <- ggplot(data=topics_long, aes(y=Ratio_top50_to_overall_variable_number, x=Model, group=Topic)) +
  geom_bar(data=topics_long, aes(y=Ratio_top50_to_overall_variable_number, fill=Topic), stat="identity",
           position=position_dodge(width=0.9)) +
  facet_wrap(~ Model_group, nrow = 2, scales = "free_x") +
  theme_bw() + ylab("Ratio of feature number in 50 most important features (SHAP) to overall feature number per topic") +
  theme(strip.text = element_blank())

setwd(file.path(REPO_ROOT, "Variable_Importance"))

png(file=paste0("Explainability_shapley_topic_barchart_",Sys.Date(),".png"), width=32, height=24, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("Explainability_shapley_topic_barchart_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 32,
  height = 24
)

setwd(file.path(REPO_ROOT, "Figures_for_paper"))

png(file=paste0("Explainability_shapley_topic_barchart_",Sys.Date(),".png"), width=32, height=24, unit="cm", res=800)
p
dev.off()

ggsave(
  filename = paste0("Explainability_shapley_topic_barchart_",Sys.Date(),".eps"),
  plot = p,
  device = cairo_ps,
  width = 32,
  height = 24
)

