# Reproduce the LSTM LIME top-50 category sensitivity table from original
# merged attribution CSVs and an explicit, auditable category mapping.
# Run from the directory containing all three input CSVs.
# Install once if needed: install.packages(c("dplyr", "tidyr", "readr"))
library(dplyr)
library(tidyr)
library(readr)

mapping <- read_csv("LSTM_LIME_feature_category_mapping.csv", show_col_types = FALSE)
stopifnot(!anyDuplicated(mapping[c("regime", "Feature")]))

load_regime <- function(reg) {
  file <- paste0("LSTM_lime_", reg, "_k3_k4_k5_merged.csv")
  raw <- read_csv(file, show_col_types = FALSE)
  expected <- c("Feature", "k=3", "k=4", "k=5")
  stopifnot(all(expected %in% names(raw)), !anyDuplicated(raw$Feature))
  # Rank by absolute signed mean LIME attribution for each k.
  # Ties are resolved alphabetically for reproducibility.
  ranked <- raw %>%
    pivot_longer(cols = all_of(c("k=3", "k=4", "k=5")),
                 names_to = "k", values_to = "attribution") %>%
    group_by(k) %>%
    arrange(desc(abs(attribution)), Feature, .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    ungroup() %>%
    filter(rank <= 50) %>%
    mutate(regime = reg)
  ranked
}

top50 <- bind_rows(load_regime("low"), load_regime("high"))
classified <- top50 %>%
  left_join(mapping, by = c("regime", "Feature"))
if (anyNA(classified$category)) {
  print(classified %>% filter(is.na(category)) %>% select(regime, Feature) %>% distinct())
  stop("Unmapped top-50 features: update LSTM_LIME_feature_category_mapping.csv")
}
stopifnot(nrow(classified) == 300L)

categories <- c("Contacts and NPIs", "Employment", "Immunity",
                "Masks", "Testing", "Vaccination", "Other")
counts <- classified %>%
  count(regime, category, k, name = "n") %>%
  complete(regime = c("low", "high"), category = categories,
           k = c("k=3", "k=4", "k=5"), fill = list(n = 0L)) %>%
  pivot_wider(names_from = k, values_from = n) %>%
  arrange(factor(regime, levels = c("low", "high")),
          match(category, categories))
print(counts, n = Inf)
write_csv(counts, "LSTM_LIME_category_counts_from_raw.csv")
write_csv(classified, "LSTM_LIME_top50_classified_from_raw.csv")

# Optional category-set Jaccard similarity across clustering solutions.
for (reg in c("low", "high")) {
  for (pair in list(c("k=3", "k=4"), c("k=3", "k=5"), c("k=4", "k=5"))) {
    a <- unique(classified$category[classified$regime == reg & classified$k == pair[1]])
    b <- unique(classified$category[classified$regime == reg & classified$k == pair[2]])
    cat(sprintf("%s %s vs %s: category Jaccard = %.3f\n",
                reg, pair[1], pair[2], length(intersect(a,b))/length(union(a,b))))
  }
}
