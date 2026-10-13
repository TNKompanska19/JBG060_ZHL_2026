# Model outputs

`main/` = full run. `sensitivity_3years/` = same strict h=1 run with flood-prone scope >= 3 flood years instead of 2 (subset of the files below).
Written by `explain_models.py` (predictions, EBM, susceptibility, map) and `analysis/summarise.py` (scores, tables).

Naming: `h` = months ahead (1, 2, 3). Validation: `strict` = held-out year AND spatial block group, `year` = leave one year out, `forward` = train only on earlier years. `physical` = M1 with standardised anomalies (M1-physical). Targets: `occurrence` = dry square at t-1 becomes water, `persistence` = wet square stays water.

| File | Contains |
|---|---|
| `results.md` | All result tables in one file. Start here. |
| `scores.csv` | Metrics for every config x model x subset (all, cropland, harvest, cropland_harvest). |
| `year_scores.csv` | PR-AUC per test year, every config x model x subset. |
| `ablation.csv` | PR-AUC change from dropping each input group from M1 / adding it to state-only model S, with noise band and `informative` flag. |
| `intervals.csv` | Bootstrap 95% intervals for the main comparisons (strict validation). |
| `cutoff_sweep.csv` | Threshold metrics at fixed alarm budgets and probability cutoffs (strict validation). |
| `preds_<cv>_h<h>[_physical].parquet` | Row-level predictions per square x issue month for every config and model, plus the observed label. |
| `preds_strict_h1_ebm[_physical].parquet` | Same, for the EBM models. |
| `columns_h<h>[_physical].json` | Feature names and which input group each belongs to. |
| `blocks.json` | Spatial blocks and groups used for strict validation. |
| `ebm_<model>_importance.csv` | EBM term importances. |
| `ebm_<model>_curves.csv` / `.png` | EBM shape curves: effect of each input value on the prediction (png: top 8). |
| `susceptibility.csv` | Static model of per-square flood frequency vs lat/lon, spatial-block CV. |
| `susceptibility_drop_one.csv` | Susceptibility model without one static covariate at a time. |
| `map_2022.npz` / `.png` | Per square, 2022 harvest months: mean predicted probability, observed flooded share, crop %, lat, lon. |
| `run_log.txt` | Console log of the run. |
