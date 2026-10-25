# JBG060-2026: Flood drivers in South Sudan (Group 8)

## Overview

Code for the JBG060 Capstone Data Challenge (TU/e, with Zero Hunger Lab and ZOA). It tests which public weather,
river and land data relate to riverine flood **occurrence** (a dry square becomes flooded) and **persistence** (a
flooded square stays flooded) in South Sudan one to three months ahead, with a focus on cropland in the harvest
months, to support early harvesting and temporary flood defences.

The pipeline:

1. downloads the data (CYGNSS flood maps, GloFAS river discharge, ERA5 weather, soil, terrain, climate indices);
2. builds the flood target: monthly CYGNSS water maps, permanent water removed, averaged to 0.02 degree squares;
3. builds the model inputs on the same grid and months;
4. trains gradient-boosted trees under three validation schemes and three lead times, compares them with two
   baselines (climatology, latest state) and runs add/remove tests per input group;
5. scores every model with bootstrap intervals over test years, and explains the main model with TreeSHAP.

`2_eda/` holds the data exploration, including the analyses that led to rejecting the MODIS/VIIRS and GFM flood
products as target.

## Repository structure

```text
JBG060_ZHL_2026/
|-- 1_data_processing/
|   |-- 1_download/          get_cygnss.py, get_glofas.py, get_covariates.py (other sources), gfm_inventory.py
|   |-- 2_loaders/           loading.py, loading_impact_data.py (course loaders)
|   `-- 3_build_dataset/     build_target.py (flood target), build_inputs.py (model inputs)
|-- 2_eda/
|   |-- 1_data_overview/     first look at the course datasets
|   |-- 2_modis_rejection/   why the MODIS/VIIRS and GFM flood products were not used as target
|   |-- 3_exposure/          flood exposure of cropland, rangeland and food insecurity (IPC) per county
|   `-- 4_target_scope/      flood products compared, cropland under flood water, flood-frequency thresholds
|-- 3_model/
|   |-- 1_run/               run_all.py (runs the models), explain_models.py (data, inputs, models, validation)
|   `-- 2_analysis/          tables, TreeSHAP, county results and figures
|-- literature/              papers consulted; not used by the code
|-- processed_data/          panel_0.02/ and cygnss/ are committed; the rest is rebuilt from raw_data/
|-- raw_data/                downloaded data, not in Git
|-- results/                 one folder per script, named after it
|-- requirements.txt
`-- README.md
```

## Installation

Tested with Python 3.14.0 on Windows. Package versions are pinned in `requirements.txt`.

```powershell
git clone https://github.com/<your-user>/JBG060_ZHL_2026.git
cd JBG060_ZHL_2026
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux: `python3 -m venv .venv` and `source .venv/bin/activate`, then the same `pip` commands.

Run every script from the repository root: several scripts use paths relative to it.

## Usage

### Rerun the models without the raw data

`processed_data/panel_0.02/` (target and all inputs) is committed:

```powershell
python 3_model/1_run/run_all.py --no-build
```

Finished stages are skipped: a stage whose `preds_*.parquet` exists is not recomputed. Delete the file to redo it.

### Full pipeline from the raw data

1. **Course data**: download from SURFdrive (link and password on Canvas) into `raw_data/`.
2. **Accounts**: NASA Earthdata (CYGNSS, GRACE); a Copernicus CDS key (ERA5) and an EWDS key (GloFAS) in
   `~/.cdsapirc`, with each dataset licence accepted once on the website. Hydroweb needs `HYDROWEB_API_KEY`, but its
   river levels are not used by the model (`USE_HYDROWEB = False` in `explain_models.py`).
3. **Manual downloads**: MERIT Hydro (`hnd`, `upa`, `elv`, tiles n00e000 and n00e030) into `raw_data/MERITHydro/`;
   HydroBASINS Africa (standard) into `raw_data/hydrobasins/`. See the docstring of `get_covariates.py`.
4. **Run**:

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

`run_all.py` builds any input missing from the panel unless `--no-build` is given, and ends with `summarise.py`.

### Figures and county results

Each needs the outputs named in brackets.

| Script (`3_model/2_analysis/`) | Output in `results/` | Needs |
|---|---|---|
| `county_map.py` | `county_map/county_map.png` (report) | `county_performance.py` |
| `county_appendix.py` | `county_appendix/` (SLE essay table and figure) | `county_performance.py` |
| `cropland_skill_map.py` | `cropland_skill_map/cropland_skill_map.png` (poster) | `county_performance.py` |
| `group_importance.py` | `group_importance/group_importance.png` (poster) | `shap_gbm.py`, `run_all.py` |
| `skill_gain.py` | `skill_gain/skill_gain.png` (poster) | `run_all.py` |
| `method_flow.py` | `method_flow/method_flow.png` (poster) | nothing |

`shap_gbm.py --plot-only` redraws the TreeSHAP figures from the saved values without refitting.

The report's flood-product figure needs the GFM metadata first:

```powershell
python 1_data_processing/1_download/gfm_inventory.py fetch
python 2_eda/4_target_scope/flood_products.py
```

The EDA scripts take no arguments (`python 2_eda/<folder>/<script>.py`), except `gfm_check.py` (`meta` or `sites`,
see its docstring) and `flood_freq_check.py`, which prints to the screen.

### `run_all.py` options

| Option | Meaning |
|---|---|
| `--no-build` | use the committed panel; do not build missing inputs |
| `--stages ...` | run only these stages (default: all) |
| `--min-years N` | a square is in scope if flooded in at least N different years (default 2) |
| `--out NAME` | output folder `results/NAME` (default `run_all`) |
| `--res` | grid resolution in degrees (default 0.02) |

Stage keys: `s`, `y`, `f` (strict, leave-one-year-out, forward validation) + `1`, `2`, `3` (months ahead), e.g. `s1`,
`f3`. An `a` in front (`as1`, `af3`) runs the anomaly-input models. `susc` = susceptibility, `map` = 2022 harvest map.

Scope check with squares flooded in at least 3 years:

```powershell
python 3_model/1_run/run_all.py --no-build --min-years 3 --out sensitivity_3years --stages s1 as1
```

## Models and validation

| Name in code | Meaning |
|---|---|
| `p_clim` | climatology baseline: share of training years in which the square was flooded in that calendar month |
| `p_pers` | latest-state baseline: the square's flood share in the last observed month |
| `H` | flood history (last 3 months, neighbourhood, last 12 months) and place (climatology, latitude, longitude, month) |
| `M1`, `M2`, `M3` | raw environmental inputs; M2 adds flood history, M3 adds place |
| `M1_anomaly`, `M2_anomaly`, `M3_anomaly` | the same ladder on standardised anomalies (deviation from the calendar-month mean of the training years, divided by its spread) |
| `adrop_<group>`, `fdrop_<family>` | M1_anomaly without one input group / family |
| `hadd_<family>` | H plus one family |
| `M1a_s1..4`, `H_s1..4` | M1_anomaly and H refit on a random 80% of the training rows (run-to-run noise) |
| `M1_anomaly_additive`, `_small_trees`, `_large_trees` | M1_anomaly without interactions, with smaller or with larger trees |

Families: weather, rivers_storage, land, climate_indices. The second part of each prediction column is the model
type: `gbm` (HistGradientBoostingClassifier) or `logreg` (logistic regression: M1_anomaly, and H and M1-M3 under strict validation).

Validation: **strict** (test = one year and one group of spatial blocks; a buffer ring is removed from training),
**leave-one-year-out**, **forward** (training on earlier years only; test years 2021-2025). Ablations, noise runs
and tree settings run under strict validation only. Every constant and its reason is at the top of
`3_model/1_run/explain_models.py`.

## Results

Every script writes to `results/<script name>/`. The repository holds the outputs used in the report and poster;
the other EDA folders appear when their script is run.

### `results/run_all/`

| File | Contents |
|---|---|
| `preds_<cv>_h<h>.parquet` | held-out predictions of the raw-input models and H, one row per sample: `cv`, `h`, `target`, `test_year`, `block_group`, `issue` (month the prediction is made), `tmonth` (target calendar month), `row`, `col` (square), `y` (flooded or not), `p_clim`, `p_pers`, `crop` (% cropland), and one `p\|<model>\|<gbm or logreg>` column per model |
| `preds_<cv>_h<h>_anomaly.parquet` | the same for the anomaly-input models and ablations; rows match the raw file |
| `scores.csv` | per model and subset: PR-AUC, ROC-AUC, Gini, Brier, log loss; for the main models also accuracy, precision, recall, F1, balanced accuracy, false-alarm rate (cutoff chosen on the other test years) |
| `year_scores.csv` | PR-AUC per test year |
| `ablation.csv` | per group / family: PR-AUC and Brier change, noise SD, test years with a positive effect, `informative` (rule in the `summarise.py` docstring) |
| `intervals.csv` | PR-AUC and PR-AUC differences with 95% bootstrap intervals over test years (strict) |
| `cutoff_sweep.csv` | precision, recall, F1 and false-alarm rate at fixed alarm budgets and cutoffs (strict) |
| `susceptibility.csv`, `susceptibility_drop_one.csv` | per-square flood frequency predicted from static terrain and soil inputs vs latitude/longitude, and the loss when one input is removed |
| `map_2022.npz`, `map_2022.png` | predicted vs observed flooding in the 2022 harvest months |
| `blocks.json` | measured correlation distance, spatial block and buffer sizes |
| `columns_h<h>[_anomaly].json` | input names and their groups |
| `results.md` | all tables in one file |
| `run_log.txt` | log of every run |

Subsets: `all`, `cropland` (square >= 10% cropland), `harvest` (target month August-December), `cropland_harvest`.

### Other folders

| Folder | Contents |
|---|---|
| `shap_gbm/` | TreeSHAP of the M1_anomaly GBM (strict, 1 month ahead): per input and per family (`.csv`), figures, raw values (`.npz`) |
| `county_performance/` | skill and input coverage per county |
| `build_target/` | target checks: national CYGNSS series, 2022 anomaly map |
| `get_cygnss/` | monthly Sudd water share behind the two CYGNSS plausibility tests |
| `flood_products/` | MODIS/VIIRS, GFM and CYGNSS over South Sudan (report figure) |
| `gfm_check/` | GFM scene table |
| `cropland_flood_overlap/` | cropland under flood water per year, 2022 map |
| `flood_freq_check/` | flood-frequency and permanent-water thresholds |
| `run_eda/` | first look at the course data |

## Troubleshooting

- **FileNotFoundError on `raw_data/...`**: run from the repository root.
- **`ssd_admin*.geojson` missing**: importing `loading_impact_data.py` loads the admin boundaries, so
  `raw_data/Administrative boundaries/` must exist even for unrelated functions.
- **`.cdsapirc` not found on Windows**: Notepad may save it as `.cdsapirc.txt`.
- **Outdated tables in `results/run_all/`**: `summarise.py` reuses `scores.csv`, `year_scores.csv` and
  `intervals.csv` when they are newer than every prediction file. Delete them and run
  `python 3_model/2_analysis/summarise.py`.
- **Stages in parallel windows**: start one first so it writes `blocks.json` before the others read it.

## Credits

- Course repository and loaders (`1_data_processing/2_loaders/`): https://github.com/eerandi/JBG060_ZHL_2026
- Flood target: UC Berkeley CYGNSS L3 monthly RWAWC watermask (NASA PO.DAAC; Pu et al., 2024).
- Permanent water: JRC Global Surface Water (Pekel et al., 2016).
- Inputs: ERA5 and ERA5 monthly (Copernicus CDS), AgERA5 evapotranspiration, GloFAS-ERA5 river discharge (Copernicus
  EWDS), GRACE/GRACE-FO JPL mascons, SoilGrids, MERIT Hydro, HydroBASINS, NOAA CPC climate indices (Nino3.4, DMI);
  course data: Dartmouth Flood Observatory discharge, lake levels from Hydroweb (Victoria, Kyoga) and DAHITI
  (Albert), ASAP cropland mask (scoring only).
- Rejected targets: MODIS/VIIRS flood masks (course data, NASA LANCE), GFM (Copernicus, EODC STAC).
- Libraries: scikit-learn, SHAP, NumPy, pandas, xarray, rasterio, GeoPandas, SciPy, Matplotlib, earthaccess, cdsapi,
  pystac-client.

Full references are in the report.

## Legal and ethical considerations

- Educational and research use only. The models are not validated for operational humanitarian decisions.
- The target is a satellite estimate. CYGNSS monthly maps are partly inferred where satellite tracks are sparse,
  and a square counts as flooded at >= 50% water. It does not measure damage or who is affected.
- Only flood-prone squares (flooded in at least 2 different years) are modelled. A missing prediction does not mean
  "safe".
- Skill differs strongly between counties (`results/county_performance/`); some counties are not scored.
- Respect each data provider's licence and citation terms. Access through SURFdrive does not replace them.
