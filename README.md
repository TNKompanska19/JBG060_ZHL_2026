# JBG060-2026: Flood drivers in South Sudan (Group 8)

## Overview

Code for the JBG060 Capstone Data Challenge (TU/e, with Zero Hunger Lab and ZOA). It studies which public weather,
river and land data explain riverine flood **occurrence** (a dry square becomes flooded) and **persistence** (a
flooded square stays flooded) in South Sudan, with a focus on cropland in the harvest months, to support
anticipatory action (early harvesting, temporary flood defences).

What the code does, in order:

1. Downloads the data (CYGNSS flood maps, GloFAS river flow, ERA5 weather, soil, terrain, climate indices, ...).
2. Builds the flood target: CYGNSS water maps with permanent water removed, averaged to 0.02 degree squares, monthly.
3. Builds the model inputs on the same grid and monthly clock.
4. Trains gradient-boosted trees (and logistic regression for some models) under three validation schemes and three
   lead times, compares them with two simple rules (climatology, persistence) and runs add/remove tests per input
   group.
5. Summarises every model with all metrics and bootstrap intervals over test years.

The `2_eda/` folder also holds the data exploration, including the analyses that led to rejecting the
MODIS/VIIRS flood masks as a target.

## Repository structure

```text
JBG060_ZHL_2026/
|-- 1_data_processing/
|   |-- 1_download/          get_cygnss.py, get_glofas.py, get_covariates.py (all other sources)
|   |-- 2_loaders/           loading.py, loading_impact_data.py (course loaders, used by build_inputs.py and the EDA)
|   `-- 3_build_dataset/     build_target.py (flood target), build_inputs.py (model inputs)
|-- 2_eda/
|   |-- 1_data_overview/     first look at every course dataset
|   |-- 2_modis_rejection/   checks showing why the MODIS/VIIRS flood masks were not used as target
|   |-- 3_exposure/          flood exposure of cropland, rangeland and food insecurity (IPC), per county
|   `-- 4_target_scope/      cropland under CYGNSS flood water; checks of the flood-frequency thresholds
|-- 3_model/
|   |-- 1_run/               run_all.py (runs everything), explain_models.py (data, features, models, validation)
|   `-- 2_analysis/          summarise.py (tables), shap_gbm.py (SHAP), county_performance.py (skill per county)
|-- processed_data/          panel_0.02/ and cygnss/ are committed; everything else is rebuilt from raw_data/
|-- raw_data/                downloaded data, not in Git
|-- results/                 one subfolder per script, named after the script (see "Results")
|-- requirements.txt
`-- README.md
```

## Installation

Tested with Python 3.14.0 on Windows. All packages and versions are pinned in `requirements.txt`.

```powershell
git clone https://github.com/<your-user>/JBG060_ZHL_2026.git
cd JBG060_ZHL_2026
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux: `python3 -m venv .venv`, `source .venv/bin/activate`, then the same `pip` commands.

Run every script **from the repository root**: several scripts use paths relative to it (`./raw_data/...`).

## Usage

### Option A: rerun the models without the raw data

`processed_data/panel_0.02/` (target and all inputs) is committed, so the models and tables can be rerun without
downloading anything:

```powershell
python 3_model/1_run/run_all.py --no-build
```

Finished stages are skipped (an existing `preds_*.parquet` is not recomputed). To redo a stage, delete its file.

### Option B: full pipeline from the raw data

1. **Course data.** Download it from SURFdrive (link and password on Canvas) into `raw_data/`.
2. **Accounts.** Downloads need a NASA Earthdata account (CYGNSS, GRACE), a Copernicus CDS key and an EWDS key in
   `~/.cdsapirc` (ERA5, GloFAS; accept each dataset licence once on the website). Hydroweb needs
   `HYDROWEB_API_KEY`, but Hydroweb is switched off in the model (`USE_HYDROWEB = False` in `explain_models.py`).
3. **Manual downloads.** MERIT Hydro (`hnd`, `upa`, `elv`, tiles n00e000 and n00e030) into `raw_data/MERITHydro/`;
   HydroBASINS Africa (standard) into `raw_data/hydrobasins/`. See the docstring of `get_covariates.py`.
4. **Run in this order:**

```powershell
python 1_data_processing/1_download/get_cygnss.py
python 1_data_processing/1_download/get_glofas.py
python 1_data_processing/1_download/get_covariates.py
python 1_data_processing/3_build_dataset/build_target.py
python 1_data_processing/3_build_dataset/build_inputs.py --res 0.02
python 3_model/1_run/run_all.py
python 3_model/2_analysis/shap_gbm.py
python 3_model/2_analysis/county_performance.py
```

`run_all.py` also builds any input missing from the panel unless `--no-build` is given.

### `run_all.py` options

| Option | Meaning |
|---|---|
| `--no-build` | do not build missing inputs; use the committed panel |
| `--stages ...` | run only these stages (default: all) |
| `--min-years N` | scope: a square is in scope if flooded in at least N different years (default 2) |
| `--out NAME` | output folder `results/NAME` (default `run_all`) |
| `--res` | grid resolution in degrees (default 0.02) |

Stage keys: `s`, `y`, `f` (strict, year-only, forward validation) + `1`, `2`, `3` (months ahead), e.g. `s1`, `f3`.
An `a` in front (`as1`, `ay2`, `af3`) runs the anomaly-input models. `susc` = susceptibility, `map` = 2022 harvest map.

Scope check with squares flooded in at least 3 years:

```powershell
python 3_model/1_run/run_all.py --no-build --min-years 3 --out sensitivity_3years --stages s1 as1
```

The EDA scripts take no arguments: `python 2_eda/<folder>/<script>.py`.

## Models and validation

| Name in code | What it is |
|---|---|
| `climatology` (`p_clim`) | rule: share of training years in which the square was flooded in that calendar month |
| `persistence` (`p_pers`) | rule: the square's flood share last month |
| `H` | trained model on flood history (last 3 months, neighbourhood, last 12 months) + place (climatology, lat, lon, month); no drivers |
| `M1`, `M2`, `M3` | raw driver inputs; M1 = drivers only, M2 = drivers + flood history, M3 = drivers + flood history + place |
| `M1_anomaly`, `M2_anomaly`, `M3_anomaly` | same ladder with anomaly drivers (value minus its calendar-month normal over training years, divided by its usual variation); M1_anomaly is the explanation model |
| `adrop_<group>` | M1_anomaly without one input group |
| `fdrop_<family>` | M1_anomaly without one family (weather, rivers_storage, land, climate_indices) |
| `hadd_<family>` | H plus one family |
| `M1a_s1..4`, `H_s1..4` | M1_anomaly and H refit on a random 80% of the training rows (run-to-run noise) |
| `M1_anomaly_additive`, `_small_trees`, `_large_trees` | M1_anomaly without interactions / with smaller / larger trees |

Model type is the second part of each prediction column: `gbm` (HistGradientBoostingClassifier) or `logreg`.

Validation schemes: **strict** (test = one year and one group of spatial blocks, with a buffer ring removed from
training), **year** (leave one year out), **forward** (train only on years before the test year, test years
2021-2025). Ablations, noise runs and tree settings run only under strict validation. All design choices and their
constants are documented at the top of `3_model/1_run/explain_models.py`.

## Results

Each script writes to `results/<script name>/`.

| Folder | Made by | Contents |
|---|---|---|
| `run_all/` | `run_all.py`, `summarise.py` | see below |
| `shap_gbm/` | `shap_gbm.py` | SHAP of the M1_anomaly GBM (strict, 1 month ahead): `shap_features.csv`, `shap_families.csv`, `shap_families.png`, `shap_occurrence.png`, `shap_persistence.png`, `shap_values_<target>.npz` |
| `county_performance/` | `county_performance.py` | skill and input coverage per county: `county_performance.md` |
| `build_target/` | `build_target.py` | `cygnss_national.png`, `cygnss_2022_anomaly.png` (target checks) |
| `get_cygnss/` | `get_cygnss.py` | `cygnss_sudd_monthly.csv` (the two CYGNSS plausibility tests) |
| `cropland_flood_overlap/` | `cropland_flood_overlap.py` | cropland under flood water per year (`.csv`, `.md`), map `cropland_flood_2022.png` |
| `flood_freq_check/` | `flood_freq_check.py` | `flood_freq_check.txt` (flood-frequency and permanent-water thresholds) |
| `run_eda/`, `2021_shift_investigation/`, `data_quality_2021_shift/`, `discharge_*`, `negative_correlation_south_sudan_station/`, `rainfall_*`, `recurring_vs_unusual_drivers/`, `exposure_analysis/`, `exposure_ipc_analysis/` | the EDA script of the same name | plots and tables of that script |

### `results/run_all/`

| File | Contents |
|---|---|
| `preds_<cv>_h<h>.parquet` | held-out predictions of the raw-input models and H; one row per sample: `cv`, `h`, `target`, `test_year`, `block_group`, `issue` (month the forecast is made), `tmonth` (target calendar month), `row`, `col` (square), `y` (flooded or not), `p_clim`, `p_pers`, `crop` (% cropland), and one `p|<model>|<gbm or logreg>` column per model |
| `preds_<cv>_h<h>_anomaly.parquet` | same for the anomaly-input models and the ablations; rows match the raw file one to one |
| `scores.csv` | every model x subset: PR-AUC, ROC-AUC, Gini, Brier, log loss; for the main models also accuracy, precision, recall, F1, balanced accuracy, false-alarm rate (cutoff chosen on the other test years) |
| `year_scores.csv` | PR-AUC per test year |
| `ablation.csv` | per group / family: PR-AUC and Brier change, noise SD, years with a positive effect, `informative` flag (rule in the `summarise.py` docstring) |
| `intervals.csv` | PR-AUC and PR-AUC differences with 95% bootstrap intervals over test years (strict) |
| `cutoff_sweep.csv` | precision / recall / F1 / false-alarm rate at fixed alarm budgets and probability cutoffs (strict) |
| `susceptibility.csv`, `susceptibility_drop_one.csv` | per-square flood frequency predicted from static terrain/soil inputs vs lat/lon, and the loss when one input is removed |
| `map_2022.npz`, `map_2022.png` | predicted vs observed flooding in the 2022 harvest months |
| `blocks.json` | measured correlation distance and the spatial block and buffer sizes |
| `columns_h<h>[_anomaly].json` | input names and their groups |
| `results.md` | all tables above in one file |
| `run_log.txt` | log of every run |

Subsets in the tables: `all`, `cropland` (square >= 10% cropland), `harvest` (target month Aug-Dec),
`cropland_harvest`.

## Troubleshooting

- **FileNotFoundError on `raw_data/...`**: run from the repository root.
- **`ssd_admin*.geojson` missing**: `loading_impact_data.py` loads the admin boundaries when it is imported, so
  `raw_data/Administrative boundaries/` must exist even for unrelated functions.
- **cfgrib / ecCodes on Windows**: only `2_eda/1_data_overview/eda_rainfall_evo_health.py` imports `cfgrib`. GloFAS
  is downloaded as NetCDF, so the pipeline itself does not need ecCodes.
- **`.cdsapirc` not found on Windows**: Notepad may save it as `.cdsapirc.txt`; check the exact file name.
- **Tables in `results/run_all/` look outdated**: `summarise.py` reuses `scores.csv`, `year_scores.csv` and
  `intervals.csv` when they are newer than all prediction files. Delete them and rerun
  `python 3_model/2_analysis/summarise.py`. A `run_all.py` process uses the code as it was when it started, so
  rerun the summary after editing `summarise.py` during a run.
- **Running stages in parallel windows**: start one window first so it writes `blocks.json` before the others read it.

## Credits

- Course repository and loaders (`1_data_processing/2_loaders/`): https://github.com/eerandi/JBG060_ZHL_2026
- Flood target: UC Berkeley CYGNSS L3 monthly RWAWC watermask (NASA PO.DAAC; Pu et al., 2024).
- Permanent water: JRC Global Surface Water (Pekel et al., 2016).
- Inputs: ERA5 and ERA5 monthly (Copernicus CDS), AgERA5 evapotranspiration, GloFAS-ERA5 river discharge (Copernicus
  EWDS), GRACE/GRACE-FO JPL mascons, SoilGrids, MERIT Hydro, HydroBASINS, NOAA CPC climate indices (Nino3.4, DMI),
  Dartmouth Flood Observatory discharge and satellite lake levels (course data), ASAP cropland mask (scoring only).
- Libraries: scikit-learn, SHAP, NumPy, pandas, xarray, rasterio, GeoPandas, SciPy, Matplotlib, earthaccess, cdsapi.

Full references are in the report.

## Legal and ethical considerations

- Educational and research use only. The models are not validated for operational humanitarian decisions.
- The target is a satellite estimate. CYGNSS monthly maps are partly inferred where satellite tracks are sparse,
  and a square counts as flooded at >= 50% water. It is not a measure of damage or of who is affected.
- Only flood-prone squares are modelled (flooded in at least 2 different years). Areas outside that scope get no
  prediction. A missing prediction must not be read as "safe".
- Some ERA5 cells show isolated, implausibly high rainfall, consistent with a known ERA5 artefact. They were not
  masked and are reported as a limitation.
- Respect each data provider's licence and citation terms. Access through SURFdrive does not replace them.
