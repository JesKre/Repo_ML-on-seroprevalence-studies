## Reproducing supplementary Tables S12 and S13

This directory contains the scripts and input data used for the sensitivity analysis of SHAP and LIME explanations across alternative numbers of incidence clusters (k = 3, 4, and 5).

- `explainability_cluster_sensitivity.R` calculates Spearman rank correlations, top-feature overlap, Jaccard similarity, and coefficients of variation for the MLP and LSTM models. These results correspond to **Table S12**.
- `lstm_lime_category_sensitivity_from_raw.R` calculates the distribution of the 50 highest-ranked LSTM LIME features across predictor categories for low- and high-incidence clusters. These results correspond to **Table S13**.

All required merged feature-attribution CSV files and the LSTM LIME feature-category mapping are included in this directory. The analyses can therefore be repeated without retraining the predictive models or recomputing the original SHAP/LIME attributions.