# Artificial datasets

Synthetic versions of the two MuSPAD modeling datasets, generated with [`synthpop`](https://cran.r-project.org/package=synthpop) for external reproduction and method development without access to the restricted individual-level MuSPAD data. The `Inzidenz` time series, the dates, and `time_day` are the real observed values, the rest are synthetic.

## Files

| File | Rows | Structure |
| --- | --- | --- |
| `Artificial_data_MuSPAD_time_agnostic.csv` / `.RData` | 279 | One row per MuSPAD survey day, `2020-07-01` to `2021-08-17` (`time_day` 182–594, i.e. days since 2020-01-01). `Inzidenz` is the RKI 7-day incidence at **t + 7 days** (prediction target one week ahead). |
| `Artificial_data_MuSPAD_time_aware.csv` / `.RData` | 408 | Complete daily calendar,one row per day, no gaps. `Inzidenz` is the RKI 7-day incidence on the **same day** (`feature_date` = `incidence_date`). |

Each dataset is available twice: as a `.RData` file (R object `df_ts_inc`, the modeling format) and as a `.csv` export that adds the auxiliary columns described below. The two versions contain the same data.

## Auxiliary columns (CSV exports)

The CSV files contain auxiliary columns that are **not** part of the `.RData` modeling files. They help understand the structure of the data — which dates the values refer to and whether the features had to be imputed:

| Column | In | Meaning |
| --- | --- | --- |
| `feature_date` | both | Date to which the feature (questionnaire / serological) values of the row refer, i.e. the day the participant data was collected (`DD.MM.YYYY`). |
| `incidence_date` | both | Date to which the `Inzidenz` value refers. For the time-agnostic dataset this is always `feature_date + 7 days`; for the time-aware dataset it equals `feature_date`. |
| `features_imputed` | time-aware only | `0`: the row is an actual survey day and the feature values were measured. `1`: the row is a non-survey day of the filled daily calendar and the feature values were statistically imputed (`missForest`) to bridge the gaps (275 measured days, 133 imputed days). |

The time-aware CSV additionally drops `time_day` (it can be recovered from `feature_date`); the time-agnostic CSV keeps it.

CSV files are written with R's `write.csv2` and are therefore semicolon-delimited with decimal commas.

## Generation

The datasets are created by [`../Data_Preprocessing/Create_aggregated_artificial_dataset.R`](../Data_Preprocessing/Create_aggregated_artificial_dataset.R):

1. load the original modeling dataset (time-agnostic: `df_ts_inc`, row `time_day = 168` dropped; time-aware: `df_ts_all_inc`),
2. impute missing feature values with `missForest` (this is what fills the feature values of the non-survey days marked by `features_imputed = 1`),
3. synthesize feature columns with `synthpop` (seed 2024) while keeping the real `Inzidenz` and `time_day`.
