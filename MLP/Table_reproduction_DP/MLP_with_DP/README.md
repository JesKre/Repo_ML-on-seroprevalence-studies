# Reproduce Tables 7 and S8

This folder extends the existing DP analysis in to the ten-seed Adam
study reported in the Table 7 and supplementary Table S8.

## Connection to the original scripts

The reference implementation is
[`DP\_SGD\_MuSPAD\_diffOpt\_diffSigma\_xfold\_Expl.R`](../MLP/DP_SGD_MuSPAD_diffOpt_diffSigma_xfold_Expl.R).
The new workers adapt its **validation phase**, rather than calling the whole
original script, which also runs RMSprop optimizer, SHAP/LIME, model saving and a separate
test phase with fixed local paths. Those stages are not needed for these tables.

|Existing reference component|New script|Relationship|
|-|-|-|
|Data loading and preprocessing (`df\_ts\_inc`, numeric `time\_day`, remove day 168)|`01\_make\_data\_cache.R`|Saves the same input state for portable runs|
|Adam validation arm; `train\_with\_dp`, `clip\_gradients`, architecture and fold preprocessing|`02\_run\_adam\_dp\_grid\_shard.R`|Repeats that analysis over ten explicitly seeded runs; retains the training calculations|
|Same Adam training loop|`03\_run\_epsinf\_shard.R`|Adds a sigma=0 control retaining clipping and subsampling; this control is new|
|Fold-level training/validation MSE output|`04\_make\_tables.R`|Converts MSE to fold RMSE and generates both revised tables|

## Included files: only the essentials

* Four R scripts listed above.
* `data/dataset\_local\_cache.rds`: the supplied post-preprocessing input (279 x 123), before imputation/scaling; needed for retraining.
* `results/adam10seed\_perfold\_master.csv`: 300 trained-model records (10 seeds x 3 settings x 10 folds).
* `results/epsinf10seed\_perfold\_master.csv`: 100 baseline records (10 seeds x 10 folds).

There are no shipped worker shards, intermediate summaries, paired-delta
results, RMSprop tables, working copies, duplicate reference scripts, model
weights or generated table copies. Table outputs are generated when needed.

## Reproduce the tables immediately

From this folder, run:

```sh
Rscript scripts/04\_make\_tables.R
```

This requires only base R and the two supplied master CSVs. It checks that all
expected seed/setting/fold combinations occur exactly once, with the expected
noise multipliers and finite nonnegative MSEs. It writes `results/table7.csv`
and `results/tableS8.csv` and prints the tables.

For each seed/setting, take sqrt of each fold MSE, then calculate the mean and
sample SD over ten folds. Table 7 reports the mean and sample SD of the ten
seed-specific mean RMSEs. Table S8 adds the mean of the ten within-seed fold
SDs. Round final statistics to one decimal; all SDs use n-1. Do not take sqrt
of the mean MSE.

## Retrain from the cached input

Install R packages `reticulate`, `tensorflow`, `keras`, `tidyverse`, `missForest`.
The supplied training environment was R 4.6.1, Python 3.12.14, TensorFlow 2.21.0
and compatible legacy `tf-keras`. Set `DP10SEED\_PYTHON` to that environment's
Python executable. Workers set `TF\_USE\_LEGACY\_KERAS=True` before initialization.

For example, in PowerShell, from the current folder:

```powershell
$env:DP10SEED\_PYTHON = 'C:/path/to/venv/Scripts/python.exe'
foreach ($seed in 1..10) {
    Rscript scripts/02\_run\_adam\_dp\_grid\_shard.R $seed
    if ($LASTEXITCODE -ne 0) { throw "DP worker failed: seed $seed" }
    Rscript scripts/03\_run\_epsinf\_shard.R $seed
    if ($LASTEXITCODE -ne 0) { throw "Baseline worker failed: seed $seed" }
}
Rscript scripts/04\_make\_tables.R --from-shards
if ($LASTEXITCODE -ne 0) { throw 'Table generation failed' }
```

Workers save their progress and resume completed runs. After all ten seeds finish, run 04\_make\_tables.R --from-shards to combine the results and generate Tables 7 and S8. Use a separate folder copy for retraining to preserve the supplied results.

The new scripts retain the original Adam validation procedure and add explicit random seeds and a zero-noise baseline. The baseline retains gradient clipping and subsampling.

The preserved implementation uses tensor-wise gradient clipping and nine updates per epoch, giving 270 updates in total. The ε labels are inherited from the original analysis; this package does not recalculate privacy budgets.

Exact table regeneration uses the supplied per-fold CSVs. Retraining results may vary across software environments.

