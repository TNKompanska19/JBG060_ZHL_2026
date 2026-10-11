"""Which inputs explain flood occurrence and persistence? Model ladder, two-way ablation, EBM, susceptibility.

Run through run_all.py. Everything below is our design; the choices are stated in the constants.

ONE SAMPLE = one flood-prone square (flooded in >= MIN_FLOOD_YEARS different years, < NEAR_PERMANENT of months;
permanent water removed per 1 km cell in build_inputs.target) x one issue month t.
We stand at the end of month t: drivers are known up to t, the CYGNSS map only up to t-1.
TARGETS (discrete-time hazard split by the state at t-1):
  occurrence   square dry at t-1 (share < 0.5)  -> is it water at t+h?
  persistence  square wet at t-1 (share >= 0.5) -> is it still water at t+h?
FEATURES
  drivers, per series: raw1 (month t), raw3 (mean t-2..t), anom3 / anom12 / anom24 (mean anomaly over the
    last 3/12/24 months; anomaly = value minus that series' calendar-month mean over TRAINING years only).
    Windows need at least half their months, else NaN.
  static: terrain (HAND, slope, wetness index, upstream area, river distance and size), soil (clay, sand, Vertisol)
  state:  y at t-1, t-2, t-3, neighbourhood at t-1, mean y over t-12..t-1
  place:  climatology (training years, target calendar month), lat, lon, target month (sin, cos)
MODEL LADDER: M1 = drivers + static (the explanation model); S = state only; M2 = M1 + state; M3 = M2 + place.
ABLATION: drop each group from M1; add each group to S. Noise band: 4 runs of M1 and of S, each on a random
  80% of the training rows. Sensitivity: GBM with smaller / larger trees; GBM without interactions.
VALIDATION
  strict: test = one year AND one group of spatial blocks; training drops that year, that block group and a
    buffer ring around it. Block size from the measured correlation distance of flood anomalies.
  year:   leave one year out only (for comparison with the earlier results).
  forward: train only on target years before the test year (from FORWARD_START); climatology and anomaly
    normals from those years only. Nothing from the future is used.
  Training also drops samples whose flood-state window (t-12..t-1) touches the test year.
BASELINES: climatology, persistence (share at t-1).
"""
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

ROOT = Path(__file__).resolve().parents[2]
TEST_YEARS = list(range(2019, 2026))
RARE, PERMANENT = 0.05, 0.90   # old scope rule (frequency of water share), kept for the susceptibility map only
MIN_FLOOD_YEARS = 2            # scope: square flooded (share >= 0.5) in at least 2 different calendar years
NEAR_PERMANENT = 0.95          # scope: and flooded in < 95% of its months (permanent water is already removed per 1 km cell)
CROP_MIN_PCT = 10.0            # cropland square: mean ASAP crop >= 10 % (our choice)
HARVEST = (8, 9, 10, 11, 12)   # sorghum harvest: early Aug, middle Oct-Nov, late Nov-Dec (FAO/SIFSIA Sudan Seasonal
                               # Monitor 2009); GIEWS: harvest starts Aug/Sep. Used for scoring only.
G = 4                          # spatial block groups
CORR_LIMIT = 0.2               # correlation distance = where mean anomaly correlation falls below this
RANGE_CAP_KM = 100.0           # cap, so blocks stay small enough to leave training data
WINDOWS = (3, 12, 24)
NOISE_SEEDS = (1, 2, 3, 4)
ABL_HORIZONS = (1, 3)
LOGREG_C = 0.1                 # L2 strength (our choice); C=1 overfit to the training blocks in the first full run
FORWARD_START = 2021           # forward validation: first test year (CYGNSS starts Aug 2018, so 2019-20 are the first training years)

LOCAL = {"tp": "era5_idx", "ro": "era5_idx", "r20": "era5_idx", "rx1": "era5_idx", "et": "et_idx",
         "sm1": "sm_idx", "sm3": "sm_idx", "grace": "grace_idx", "q": "q_idx", "hw": "hw_idx",
         "t2m": "atm_idx", "d2m": "atm_idx", "si10": "atm_idx", "ssrd": "atm_idx",
         "wetd": "era5_idx", "wspell": "era5_idx", "conc5": "era5_idx", "dsd": "era5_idx", "uprain": "uprain_idx"}
STATIC = ("hnd_mean", "hnd_min", "low_2m", "upa_max", "slope", "twi", "q_dist", "q_size", "clay", "sand", "vertisol",
          "hw_dist", "hnd_q1", "hnd_sd", "slope_sd", "depress")
NO_RAW = ("hw",)               # series whose raw value is a location ID (river altitude): anomalies only
USE_HYDROWEB = False           # tested 2026-10-10: measured river levels added nothing (strict h=1); kept out of the main model
STATE = ("y_t-1", "y_t-2", "y_t-3", "nbr_t-1", "y_m12")
PLACE = ("climatology", "lat", "lon", "sin_month", "cos_month")
GBM_BASE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)


def group_of(name):
    """Driver group of a series or static input."""
    if name in ("hw", "hw_dist"):
        return "river_levels"
    if name in ("t2m", "d2m", "si10", "ssrd"):
        return "atmosphere"
    if name in ("wetd", "wspell", "conc5", "dsd"):
        return "rain_shape"
    if name == "uprain":
        return "upstream_catchment_rain"
    if name in ("hnd_q1", "hnd_sd", "slope_sd", "depress"):
        return "heterogeneity"
    if name in ("tp", "ro", "r20", "rx1"):
        return "local_rain"
    if name.startswith("rain_"):
        return "upstream_rain"
    if name == "et":
        return "evapotranspiration"
    if name in ("sm1", "sm3"):
        return "soil_moisture"
    if name == "grace":
        return "water_storage"
    if name.startswith("lake_"):
        return "lakes"
    if name in ("q_dist", "q_size", "hnd_mean", "hnd_min", "low_2m", "upa_max", "slope", "twi"):
        return "terrain"
    if name == "q" or name.startswith("qk") or name.startswith("q_"):
        return "glofas"
    if name.startswith("dfo_"):
        return "dartmouth"
    if name in ("nino34", "dmi"):
        return "climate_indices"
    if name in ("clay", "sand", "vertisol"):
        return "soil"
    if name.startswith("overbank_") or name == "lowland_x_q":
        return "overbank"
    raise KeyError(name)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


BASE_LABELS = {
    "tp": "local rainfall (mm/month)", "ro": "local runoff (mm/month)",
    "r20": "heavy-rain days (>=20 mm)", "rx1": "wettest day (mm)",
    "et": "evapotranspiration (mm/day)", "sm1": "topsoil moisture 0-7 cm", "sm3": "subsoil moisture 28-100 cm",
    "grace": "water storage anomaly, GRACE (cm)", "q": "flow of nearest large river, GloFAS (log)",
    "qk10": "river flow within ~10 km, GloFAS (log)", "qk30": "river flow within ~30 km, GloFAS (log)",
    "qk100": "river flow within ~100 km, GloFAS (log)",
    "nino34": "El Nino index (Nino3.4)", "dmi": "Indian Ocean Dipole index",
    "hnd_mean": "height above nearest drainage, mean (m)", "hnd_min": "height above nearest drainage, min (m)",
    "low_2m": "share of land < 2 m above drainage", "upa_max": "upstream drainage area (log km2)",
    "slope": "slope (degrees)", "twi": "topographic wetness index", "q_dist": "distance to nearest large river (km)",
    "q_size": "size of nearest large river (log flow)", "clay": "clay 0-30 cm (%)", "sand": "sand 0-30 cm (%)",
    "vertisol": "Vertisol (swelling clay) probability (%)",
    "overbank_q": "overbank: nearest-river anomaly / (1 + HAND)", "overbank_qk30": "overbank: 30-km river anomaly / (1 + HAND)",
    "lowland_x_q": "low-lying share x nearest-river anomaly",
    "hw": "measured river level, nearest Hydroweb station (m)", "hw_dist": "distance to that Hydroweb station (km)",
    "overbank_hw": "overbank: measured river-level anomaly / (1 + HAND)",
    "t2m": "air temperature (C)", "d2m": "dewpoint (humidity, C)", "si10": "wind speed (m/s)",
    "ssrd": "solar radiation (MJ/m2/day)", "wetd": "rain days (>= 1 mm)", "wspell": "longest wet spell (days)",
    "conc5": "share of month's rain in its 5 wettest days", "dsd": "day-to-day rain variability (SD, mm)",
    "uprain": "rain over the square's upstream catchment (mm/month)",
    "hnd_q1": "height above drainage, 25th percentile (m)", "hnd_sd": "height above drainage, spread (SD, m)",
    "slope_sd": "slope spread within square (SD, deg)", "depress": "share of square in local depressions"}
SUFFIX_LABELS = {"raw1": "this month", "raw3": "mean of last 3 months", "anom3": "anomaly, last 3 months",
                 "anom12": "anomaly, last 12 months", "anom24": "anomaly, last 24 months",
                 "z3": "standardised anomaly, last 3 months (SDs from normal)",
                 "z12": "standardised anomaly, last 12 months (SDs from normal)",
                 "z24": "standardised anomaly, last 24 months (SDs from normal)"}


def pretty(term):
    """Plain-language name of a model input (or an EBM pair 'a & b')."""
    if " & " in term:
        return " x ".join(pretty(t) for t in term.split(" & "))
    base, _, suf = term.partition("__")
    if base in BASE_LABELS:
        b = BASE_LABELS[base]
    elif base.startswith("q_"):
        b = f"GloFAS flow at {base[2:].replace('_', ' ').title()} (log)"
    elif base.startswith("lake_"):
        b = f"Lake {base[5:].title()} level"
    elif base.startswith("rain_"):
        b = f"rainfall over {base[5:].replace('_', ' ')}"
    elif base.startswith("dfo_"):
        b = f"Dartmouth station {base[4:]} discharge"
    else:
        b = base
    return f"{b}, {SUFFIX_LABELS.get(suf, suf)}" if suf else b


ERRORS = []          # every caught error, so run_all can fail the smoke test on them


# ---------------------------------------------------------------- panel

class Panel:
    def __init__(self, res=0.02):
        p = ROOT / "processed_data" / f"panel_{res:.2f}"
        z = np.load(p / "inputs.npz")
        self.a = {k: z[k] for k in z.files}
        sh = pd.read_csv(p / "shared.csv", index_col=0)
        self.shared, self.shared_names = sh.to_numpy(np.float32), list(sh.columns)
        self.months = pd.PeriodIndex(self.a["months"], freq="M")
        self.cal = self.months.month.to_numpy() - 1
        self.year = self.months.year.to_numpy()
        self.y = self.a["y"].astype(np.float32)
        self.nbr = self.a["nbr"].astype(np.float32)
        self.lat, self.lon = self.a["lat"].astype(np.float32), self.a["lon"].astype(np.float32)
        self.res = float(abs(self.lat[1] - self.lat[0]))
        self.local = [k for k in LOCAL if k in self.a and LOCAL[k] in self.a and (USE_HYDROWEB or k != "hw")]
        self.cubes = sorted(k for k in self.a if k.startswith("qk"))
        self.static = [k for k in STATIC if k in self.a and (USE_HYDROWEB or k != "hw_dist")]
        self.series = self.local + self.cubes + self.shared_names
        missing = [k for k in LOCAL if k not in self.local] + [k for k in STATIC if k not in self.static]
        if not USE_HYDROWEB:
            missing = [k for k in missing if k not in ("hw", "hw_dist")]
        if missing:
            log(f"!!! INPUTS NOT IN THE PANEL: {missing}")
        self.freq = np.nanmean(self.y, 0)                       # scope and reporting only
        wet = np.nan_to_num(self.y) >= 0.5
        nval = np.isfinite(self.y).sum(0)
        self.wetfreq = wet.sum(0) / np.maximum(nval, 1)
        self.flood_years = sum(wet[self.year == yy].any(0) for yy in np.unique(self.year)).astype(np.int16)
        self.scope = (self.flood_years >= MIN_FLOOD_YEARS) & (self.wetfreq < NEAR_PERMANENT) & (nval >= 24)
        log(f"scope: {self.scope.sum():,} squares flooded in >= {MIN_FLOOD_YEARS} different years "
            f"(>= 3 years: {((self.flood_years >= 3) & (self.wetfreq < NEAR_PERMANENT) & (nval >= 24)).sum():,}; "
            f"old 5-90% rule: {((self.freq >= RARE) & (self.freq < PERMANENT)).sum():,})")
        self.crop = self.a["crop_pct"].astype(np.float32) if "crop_pct" in self.a else np.zeros_like(self.freq)
        if "crop_pct" not in self.a:
            log("!!! NO CROPLAND MASK IN THE PANEL: cropland scores will be empty")
        self.era_ok = ~np.isnan(self.a["tp"]).all(1)
        self.T = len(self.months)


def sample(P, per_month, h_max, seed):
    """Per issue month: half cropland seasonal squares, half other seasonal squares."""
    rng = np.random.default_rng(seed)
    seasonal = P.scope
    crop = seasonal & (P.crop >= CROP_MIN_PCT)
    other = seasonal & ~crop
    rows = []
    for t in range(3, P.T - h_max):
        if not P.era_ok[t]:
            continue
        need = ~(np.isnan(P.y[t - 3:t]).any(0) | np.isnan(P.y[t + 1:t + h_max + 1]).any(0))
        got = 0
        for pool, k in ((crop, per_month // 2), (other, None)):
            ok = np.flatnonzero((pool & need).ravel())
            k = per_month - got if k is None else k
            if ok.size and k > 0:
                pick = rng.choice(ok, size=min(k, ok.size), replace=False)
                got += pick.size
                rows.append(np.column_stack([np.full(pick.size, t), *np.unravel_index(pick, P.y.shape[1:])]))
    s = np.concatenate(rows).astype(np.int64)
    log(f"{len(s):,} samples, issue months {P.months[s[:, 0].min()]} to {P.months[s[:, 0].max()]}, "
        f"cropland share {np.mean(P.crop[s[:, 1], s[:, 2]] >= CROP_MIN_PCT):.0%}")
    return s


# ---------------------------------------------------------------- features

def wmean(x, w):
    """Mean of the first w columns (lag 0 = column 0); NaN if fewer than half are present."""
    v = x[:, :w]
    n = np.isfinite(v).sum(1)
    m = np.nansum(v, 1) / np.maximum(n, 1)
    m[n < max(1, w // 2)] = np.nan
    return m.astype(np.float32)


def train_months(P, test_year, forward=False):
    if test_year is None:
        return np.arange(P.T)
    return np.flatnonzero(P.year < test_year) if forward else np.flatnonzero(P.year != test_year)


def calendar_mean(arr, P, months):
    """[12, ...] mean of arr over the given months, per calendar month."""
    out = np.full((12,) + arr.shape[1:], np.nan, np.float32)
    for cm in range(12):
        idx = months[P.cal[months] == cm]
        if idx.size:
            out[cm] = np.nanmean(arr[idx].astype(np.float32), 0)
    return out


def series_values(P, name, t, r, c, lags):
    tt = t[:, None] - lags[None, :]
    bad = tt < 0
    tc = np.clip(tt, 0, None)
    if name in P.local:
        cell = P.a[LOCAL[name]][r, c].astype(np.int64)
        v = P.a[name][tc, cell[:, None]].astype(np.float32)
        key = cell
    elif name in P.cubes:
        v = P.a[name][tc, r[:, None], c[:, None]].astype(np.float32)
        key = (r, c)
    else:
        j = P.shared_names.index(name)
        v = P.shared[tc, j].astype(np.float32)
        key = j
    v[bad] = np.nan
    return v, tc, key


def anomaly(P, name, v, tc, key, norm):
    return v - lookup(P, name, tc, key, norm)


def lookup(P, name, tc, key, arr):
    """Per-calendar-month array (mean or SD) of one series, picked for every sample and lag."""
    cal = P.cal[tc]
    if name in P.local:
        return arr[cal, key[:, None]]
    if name in P.cubes:
        return arr[cal, key[0][:, None], key[1][:, None]]
    return arr[cal]


def calendar_sd(arr, P, months):
    """[12, ...] standard deviation of arr over the given months, per calendar month."""
    out = np.full((12,) + arr.shape[1:], np.nan, np.float32)
    for cm in range(12):
        idx = months[P.cal[months] == cm]
        if idx.size > 1:
            out[cm] = np.nanstd(arr[idx].astype(np.float32), 0)
    return out


def normals(P, test_year, forward=False, sd=False):
    """Calendar-month means (and with sd=True standard deviations) of every series and of y, over training
    years only (forward: years before the test year)."""
    m = train_months(P, test_year, forward)
    out = {n: calendar_mean(P.a[n], P, m) for n in P.local + P.cubes}
    sh = calendar_mean(P.shared, P, m)
    out.update({n: sh[:, j] for j, n in enumerate(P.shared_names)})
    if sd:
        out.update({f"{n}__sd": calendar_sd(P.a[n], P, m) for n in P.local + P.cubes})
        shs = calendar_sd(P.shared, P, m)
        out.update({f"{n}__sd": shs[:, j] for j, n in enumerate(P.shared_names)})
    out["_y"] = calendar_mean(P.y, P, m)
    return out


def features(P, s, h, norm, physical=False):
    """Full feature matrix for samples s at horizon h, with this fold's normals. Returns X, names.
    physical=True adds standardised anomalies (anomaly / that series' calendar-month SD, so 'unusually high'
    means the same on every river) and the overbank inputs (needs normals(..., sd=True))."""
    t, r, c = s[:, 0], s[:, 1], s[:, 2]
    lags = np.arange(max(WINDOWS))
    cols, names = [], []
    for n in P.series:
        v, tc, key = series_values(P, n, t, r, c, lags)
        an = anomaly(P, n, v, tc, key, norm[n])
        cols += [v[:, 0], wmean(v, 3)] + [wmean(an, w) for w in WINDOWS]
        names += [f"{n}__raw1", f"{n}__raw3"] + [f"{n}__anom{w}" for w in WINDOWS]
        if physical:
            sdv = lookup(P, n, tc, key, norm[f"{n}__sd"])
            z = an / np.where(sdv > 0, sdv, np.nan)
            cols += [wmean(z, w) for w in WINDOWS]
            names += [f"{n}__z{w}" for w in WINDOWS]
    for n in P.static:
        cols.append(P.a[n][r, c].astype(np.float32))
        names.append(n)
    if physical and "hnd_mean" in P.static:
        hnd = np.clip(P.a["hnd_mean"][r, c].astype(np.float32), 0, None)
        for qn in ("q", "qk30", "hw"):
            if f"{qn}__z3" in names:
                cols.append(cols[names.index(f"{qn}__z3")] / (1 + hnd))
                names.append(f"overbank_{qn}")
        if "low_2m" in P.static and "q__z3" in names:
            cols.append(P.a["low_2m"][r, c].astype(np.float32) * cols[names.index("q__z3")])
            names.append("lowland_x_q")
    yl = np.stack([P.y[np.clip(t - k, 0, None), r, c] for k in range(1, 13)], 1)
    cols += [yl[:, 0], yl[:, 1], yl[:, 2], P.nbr[t - 1, r, c], wmean(yl, 12)]
    names += list(STATE)
    tcal = P.cal[t + h]
    cols += [norm["_y"][tcal, r, c], P.lat[r], P.lon[c],
             np.sin(2 * np.pi * (tcal + 1) / 12).astype(np.float32), np.cos(2 * np.pi * (tcal + 1) / 12).astype(np.float32)]
    names += list(PLACE)
    return np.column_stack(cols).astype(np.float32), names


def column_sets(names, P):
    """Column indices per role and per driver group."""
    idx = {n: i for i, n in enumerate(names)}
    groups = {}
    for n in P.series:
        keys = ([] if n in NO_RAW else ["raw1", "raw3"]) + [f"anom{w}" for w in WINDOWS]
        groups.setdefault(group_of(n), []).extend(idx[f"{n}__{k}"] for k in keys)
    for n in P.static:
        groups.setdefault(group_of(n), []).append(idx[n])
    drivers = sorted(i for g in groups.values() for i in g)
    state = [idx[n] for n in STATE]
    place = [idx[n] for n in PLACE]
    return groups, drivers, state, place


PHYS_RAW = ("local_rain", "upstream_rain", "evapotranspiration", "soil_moisture",   # keep raw3: they carry the season
            "atmosphere", "rain_shape", "upstream_catchment_rain")
PHYS_DROP_STATIC = ("slope", "q_size")   # slope is mostly DEM noise in the Sudd; river size acts as a river ID


def physical_sets(names, P):
    """M1-physical: standardised anomalies for every series, raw 3-month means only for rain-type series,
    static inputs without slope and river size, plus the overbank inputs. Returns groups, columns."""
    idx = {n: i for i, n in enumerate(names)}
    groups = {}
    for n in P.series:
        g = group_of(n)
        keys = (["raw3"] if g in PHYS_RAW else []) + [f"z{w}" for w in WINDOWS]
        groups.setdefault(g, []).extend(idx[f"{n}__{k}"] for k in keys if f"{n}__{k}" in idx)
    for n in P.static:
        if n not in PHYS_DROP_STATIC:
            groups.setdefault(group_of(n), []).append(idx[n])
    for n in names:
        if n == "overbank_hw":
            groups.setdefault("river_levels", []).append(idx[n])     # keep everything Hydroweb in one group
        elif n.startswith("overbank_") or n == "lowland_x_q":
            groups.setdefault("overbank", []).append(idx[n])
    return groups, sorted(i for g in groups.values() for i in g)


FAMILIES = {   # families of related groups: dropping a whole family leaves no correlated substitute behind
    "weather": ("local_rain", "rain_shape", "upstream_rain", "evapotranspiration", "atmosphere", "soil_moisture",
                "upstream_catchment_rain"),
    "rivers_storage": ("glofas", "dartmouth", "lakes", "water_storage", "overbank", "river_levels"),
    "land": ("terrain", "soil", "heterogeneity"),
    "climate_indices": ("climate_indices",)}


def physical_configs(groups, cols, state):
    gbm = ("gbm",)
    C = {"M1_physical": dict(cols=cols, models=("gbm", "logreg")),
         "M2_physical": dict(cols=cols + state, models=gbm)}
    for g, gc in groups.items():
        C[f"pdrop_{g}"] = dict(cols=[i for i in cols if i not in set(gc)], models=gbm)
    for fam, members in FAMILIES.items():
        gone = {i for g in members for i in groups.get(g, [])}
        if gone:
            C[f"fdrop_{fam}"] = dict(cols=[i for i in cols if i not in gone], models=gbm)
    for k in NOISE_SEEDS:
        C[f"M1p_s{k}"] = dict(cols=cols, models=gbm, sub=k)
    return C


def configs(h, groups, drivers, state, place, cv):
    """name -> dict(cols, models, subsample seed, gbm overrides)."""
    both, gbm = ("gbm", "logreg"), ("gbm",)
    C = {"M1": dict(cols=drivers, models=both), "S": dict(cols=state, models=both),
         "M2": dict(cols=drivers + state, models=both), "M3": dict(cols=drivers + state + place, models=both)}
    if cv in ("year", "forward"):
        return {k: dict(v, models=gbm) for k, v in C.items() if k in ("M1", "M3")}
    if h not in ABL_HORIZONS:
        return C
    for g, gc in groups.items():
        C[f"drop_{g}"] = dict(cols=[i for i in drivers if i not in set(gc)], models=both)
        C[f"add_{g}"] = dict(cols=state + gc, models=both)
    for k in NOISE_SEEDS:
        C[f"M1_s{k}"] = dict(cols=drivers, models=both, sub=k)
        C[f"S_s{k}"] = dict(cols=state, models=both, sub=k)
    if h == 1:
        C["M1_additive"] = dict(cols=drivers, models=gbm, gbm=dict(interaction_cst="no_interactions"))
        C["M1_small_trees"] = dict(cols=drivers, models=gbm, gbm=dict(max_leaf_nodes=15))
        C["M1_large_trees"] = dict(cols=drivers, models=gbm, gbm=dict(max_leaf_nodes=63, learning_rate=0.1))
    return C


# ---------------------------------------------------------------- validation blocks

def blocks(P, out_dir, seed=0):
    """Spatial blocks and their groups. Block side = 2 x correlation distance, buffer = 0.5 x distance."""
    f = out_dir / "blocks.json"
    seasonal = P.scope
    if f.exists():
        info = json.loads(f.read_text())
    else:
        rng = np.random.default_rng(seed)
        rr, cc = np.nonzero(seasonal)
        pick = rng.choice(rr.size, size=min(3000, rr.size), replace=False)
        r, c = rr[pick], cc[pick]
        Y = P.y[:, r, c]
        cm = calendar_mean(P.y, P, np.arange(P.T))[:, r, c]
        an = Y - cm[P.cal]
        sd = np.nanstd(an, 0)
        Z = np.nan_to_num((an - np.nanmean(an, 0)) / np.where(sd > 0, sd, 1))
        corr = (Z.T @ Z) / max(1, np.isfinite(an).sum(0).mean())
        la, lo = P.lat[r], P.lon[c]
        d = np.hypot((la[:, None] - la[None, :]) * 111.0,
                     (lo[:, None] - lo[None, :]) * 111.0 * np.cos(np.radians(la.mean())))
        edges = np.arange(0, 420, 20.0)
        b = np.digitize(d, edges) - 1
        off = ~np.eye(d.shape[0], dtype=bool)
        curve = [float(np.mean(corr[(b == i) & off])) if ((b == i) & off).any() else None for i in range(len(edges) - 1)]
        below = [edges[i + 1] for i, v in enumerate(curve) if v is not None and v < CORR_LIMIT]
        rng_km = float(below[0]) if below else RANGE_CAP_KM
        capped = rng_km > RANGE_CAP_KM
        rng_km = min(rng_km, RANGE_CAP_KM)
        info = dict(corr_limit=CORR_LIMIT, range_km=rng_km, capped=capped, block_km=2 * rng_km,
                    buffer_km=0.5 * rng_km, groups=G, curve_km=edges[:-1].tolist(), curve_corr=curve)
        f.write_text(json.dumps(info, indent=1))
    blk = info["block_km"] / 111.0
    bi = np.floor((P.lat.max() - P.lat) / blk).astype(int)
    bj = np.floor((P.lon - P.lon.min()) / blk).astype(int)
    bid = bi[:, None] * (bj.max() + 1) + bj[None, :]
    perm = np.random.default_rng(seed).permutation(bid.max() + 1)
    grp = (perm[bid] % G).astype(np.int8)
    rad = max(1, int(np.ceil(info["buffer_km"] / 111.0 / P.res)))
    st = np.ones((2 * rad + 1, 2 * rad + 1), bool)
    excl = np.stack([binary_dilation(grp == g, st) & (grp != g) for g in range(G)])
    log(f"blocks: correlation distance {info['range_km']:.0f} km{' (capped)' if info['capped'] else ''}, "
        f"block {info['block_km']:.0f} km, buffer {info['buffer_km']:.0f} km, {G} groups")
    return grp, excl


def fold_masks(P, s, h, Y, cv, grp, excl):
    t, r, c = s[:, 0], s[:, 1], s[:, 2]
    tgt_year = P.year[t + h]
    touch = (P.year[np.clip(t - 12, 0, None)] <= Y) & (P.year[t - 1] >= Y)
    train = (tgt_year != Y) & ~touch
    test = tgt_year == Y
    g = None
    if cv == "forward":
        train = (tgt_year < Y) & (P.year[t - 1] < Y)
    if cv == "strict":
        g = TEST_YEARS.index(Y) % G
        test &= grp[r, c] == g
        train &= (grp[r, c] != g) & ~excl[g][r, c]
    return train, test, g


# ---------------------------------------------------------------- models

def fit_predict(model, Xtr, ytr, Xte, opts, seed):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), np.nan, np.float32)
    if model == "gbm":
        m = HistGradientBoostingClassifier(**{**GBM_BASE, **opts.get("gbm", {}), "random_state": seed})
        return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    mu = np.where(np.isfinite(mu), mu, 0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1)
    # clip: held-out blocks can lie far outside the training range, and a linear model extrapolates
    # them to confident, wrong probabilities (seen in the first full run: Brier worse than the base rate)
    z = lambda X: np.clip(np.nan_to_num((X - mu) / sd), -5, 5)       # noqa: E731
    m = LogisticRegression(C=LOGREG_C, max_iter=2000, tol=1e-4).fit(z(Xtr), ytr)
    return m.predict_proba(z(Xte))[:, 1]


def run_pass(P, cfg, out_dir, h, cv, grp, excl, years=None, variant=""):
    """All configs for one horizon and one validation scheme. Writes preds_<cv>_h<h>[_variant].parquet.
    variant='physical': only the M1-physical configs."""
    phys = variant == "physical"
    out = out_dir / f"preds_{cv}_h{h}{'_' + variant if variant else ''}.parquet"
    if out.exists():
        log(f"skip {out.name} (exists)")
        return
    s = sample(P, cfg["per_month"], max(cfg["horizons"]), cfg["sample_seed"])
    lab = P.y[s[:, 0] + h, s[:, 1], s[:, 2]] >= 0.5
    state = P.y[s[:, 0] - 1, s[:, 1], s[:, 2]] >= 0.5
    parts, t0 = [], time.time()
    yrs = years or TEST_YEARS
    if cv == "forward":
        yrs = [Y for Y in yrs if Y >= FORWARD_START]
    for Y in yrs:
        norm = normals(P, Y, forward=cv == "forward", sd=phys)
        X, names = features(P, s, h, norm, physical=phys)
        groups, drivers, st_cols, place = column_sets(names, P)
        C = configs(h, groups, drivers, st_cols, place, cv)
        if phys:
            groups, phys_cols = physical_sets(names, P)
            C = physical_configs(groups, phys_cols, st_cols)
        if cfg.get("max_configs"):
            C = dict(list(C.items())[:cfg["max_configs"]])
        train, test, g = fold_masks(P, s, h, Y, cv, grp, excl)
        for target, sel in (("occurrence", ~state), ("persistence", state)):
            tr, te = np.flatnonzero(train & sel), np.flatnonzero(test & sel)
            if te.size < 20 or tr.size < 200:
                log(f"  h={h} {cv} {Y} {target}: too few rows (train {tr.size}, test {te.size}), skipped")
                continue
            base = pd.DataFrame({"cv": cv, "h": h, "target": target, "test_year": Y, "block_group": g,
                                 "issue": P.months[s[te, 0]].astype(str), "tmonth": P.cal[s[te, 0] + h] + 1,
                                 "row": s[te, 1], "col": s[te, 2], "y": lab[te].astype(np.int8),
                                 "p_clim": np.nan_to_num(X[te, names.index("climatology")]),
                                 "p_pers": X[te, names.index("y_t-1")],
                                 "crop": P.crop[s[te, 1], s[te, 2]]})
            pcols = {}
            for name, cf in C.items():
                for model in cf["models"]:
                    try:
                        trr = tr
                        if cf.get("sub"):
                            rng = np.random.default_rng(cf["sub"])
                            trr = np.sort(rng.choice(tr, size=int(0.8 * tr.size), replace=False))
                        cols = np.array(cf["cols"])
                        p = fit_predict(model, X[np.ix_(trr, cols)], lab[trr], X[np.ix_(te, cols)], cf, cfg["model_seed"])
                        pcols[f"p|{name}|{model}"] = p.astype(np.float32)
                    except Exception as e:
                        ERRORS.append(f"{cv} h={h} {Y} {target} {name} {model}: {e}")
                        log(f"  !! {cv} h={h} {Y} {target} {name} {model}: {type(e).__name__}: {e}")
            parts.append(pd.concat([base, pd.DataFrame(pcols)], axis=1))
            log(f"  h={h} {cv} test {Y} {target}: train {tr.size:,} test {te.size:,}, {len(C)} configs "
                f"({(time.time() - t0) / 60:.1f} min so far)")
        if Y == yrs[0]:
            (out_dir / f"columns_h{h}{'_' + variant if variant else ''}.json").write_text(json.dumps(
                {"names": names, "groups": {k: [names[i] for i in v] for k, v in groups.items()}}, indent=1))
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        log(f"wrote {out.name} ({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------- EBM

def informative_groups(out_dir, target, direction="drop", k_fallback=4):
    """Inputs for the EBM. 'pdrop' (M1-physical): every group of each FAMILY whose removal costs PR-AUC with a 95%
    interval above zero (strict, h=1, all rows; from intervals.csv). Families, not single groups, because correlated
    groups substitute for each other in single-group tests. 'drop' (M1): single groups, as before."""
    if direction == "pdrop" and (out_dir / "intervals.csv").exists():
        iv = pd.read_csv(out_dir / "intervals.csv")
        iv = iv[(iv.h == 1) & (iv.target == target) & (iv.subset == "all") & iv.what.str.contains("fdrop_")]
        fams = [f for f in FAMILIES if not iv[iv.what.str.contains(f"fdrop_{f}:") & (iv.lo > 0)].empty]
        if fams:
            return [g for f in fams for g in FAMILIES[f]]
    from summarise import ablation_table
    abl = ablation_table(out_dir)
    if abl.empty:
        return []
    a = abl[(abl.cv == "strict") & (abl.h == 1) & (abl.target == target) & (abl.model == "gbm")
            & (abl.direction == direction) & (abl.subset == "all")].sort_values("delta_pr_auc", ascending=False)
    sel = a[a.informative].group.tolist()
    return sel if sel else a.group.head(k_fallback).tolist()


def run_ebm(P, cfg, out_dir, grp, excl, variant=""):
    """EBM on the informative groups. variant='physical': groups and inputs of M1-physical."""
    from interpret.glassbox import ExplainableBoostingClassifier
    phys = variant == "physical"
    tag = "_physical" if phys else ""
    out = out_dir / f"preds_strict_h1_ebm{tag}.parquet"
    if out.exists():
        log(f"skip EBM{tag} (exists)")
        return
    h = 1
    s = sample(P, cfg["per_month"], max(cfg["horizons"]), cfg["sample_seed"])
    lab = P.y[s[:, 0] + h, s[:, 1], s[:, 2]] >= 0.5
    state = P.y[s[:, 0] - 1, s[:, 1], s[:, 2]] >= 0.5
    parts, rng = [], np.random.default_rng(0)
    groups_of = (lambda names: physical_sets(names, P)[0]) if phys else (lambda names: column_sets(names, P)[0])  # noqa: E731
    for target, sel in (("occurrence", ~state), ("persistence", state)):
        gsel = informative_groups(out_dir, target, "pdrop" if phys else "drop")
        log(f"EBM{tag} {target}: groups {gsel}")
        if not gsel:
            continue
        for Y in cfg.get("years") or TEST_YEARS:
            X, names = features(P, s, h, normals(P, Y, sd=phys), physical=phys)
            groups = groups_of(names)
            cols = sorted(i for g in gsel for i in groups.get(g, []))
            train, test, g = fold_masks(P, s, h, Y, "strict", grp, excl)
            tr, te = np.flatnonzero(train & sel), np.flatnonzero(test & sel)
            if te.size < 20 or tr.size < 200:
                continue
            if tr.size > cfg["ebm_rows"]:
                tr = np.sort(rng.choice(tr, cfg["ebm_rows"], replace=False))
            m = ExplainableBoostingClassifier(feature_names=[names[i] for i in cols], interactions=10,
                                              outer_bags=4, n_jobs=-1, random_state=0)
            m.fit(X[np.ix_(tr, cols)], lab[tr])
            p = m.predict_proba(X[np.ix_(te, cols)])[:, 1]
            parts.append(pd.DataFrame({"cv": "strict", "h": h, "target": target, "test_year": Y, "block_group": g,
                                       "issue": P.months[s[te, 0]].astype(str), "tmonth": P.cal[s[te, 0] + h] + 1,
                                       "row": s[te, 1], "col": s[te, 2], "y": lab[te].astype(np.int8),
                                       "p_clim": np.nan_to_num(X[te, names.index("climatology")]),
                                       "p_pers": X[te, names.index("y_t-1")], "crop": P.crop[s[te, 1], s[te, 2]],
                                       f"p|EBM{tag}|ebm" if phys else "p|EBM_selected|ebm": p.astype(np.float32)}))
            log(f"  EBM{tag} {target} test {Y}: train {tr.size:,} test {te.size:,}")
        # final fit on all rows of this target (normals from all years) for the curves
        X, names = features(P, s, h, normals(P, None, sd=phys), physical=phys)
        groups = groups_of(names)
        cols = sorted(i for g in gsel for i in groups.get(g, []))
        idx = np.flatnonzero(sel)
        if idx.size > cfg["ebm_rows"]:
            idx = np.sort(rng.choice(idx, cfg["ebm_rows"], replace=False))
        m = ExplainableBoostingClassifier(feature_names=[names[i] for i in cols], interactions=10,
                                          outer_bags=4, n_jobs=-1, random_state=0)
        m.fit(X[np.ix_(idx, cols)], lab[idx])
        export_ebm(m, out_dir / f"ebm{tag}_{target}")
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)


def export_ebm(m, stem):
    """Term importances (csv) and main-effect curves (csv + png of the top 8)."""
    imp = pd.DataFrame({"term": m.term_names_, "importance": m.term_importances()}).sort_values("importance", ascending=False)
    imp.to_csv(f"{stem}_importance.csv", index=False)
    rows = []
    g = m.explain_global()
    for i, term in enumerate(m.term_names_):
        if " & " in term:
            continue
        try:
            d = g.data(i)
            x, sc = np.asarray(d["names"]), np.asarray(d["scores"], float)
            if x.size == sc.size + 1:
                x = (x[:-1].astype(float) + x[1:].astype(float)) / 2
            rows += [{"term": term, "x": float(a), "score": float(b)} for a, b in zip(x, sc)]
        except Exception as e:
            ERRORS.append(f"EBM curve {term}: {e}")
            log(f"  !! EBM curve {term}: {e}")
    cur = pd.DataFrame(rows)
    cur.to_csv(f"{stem}_curves.csv", index=False)
    plot_ebm(stem)


def plot_ebm(stem):
    """Top 8 main-effect curves from the saved csv files, with plain-language titles."""
    stem = str(stem)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        imp, cur = pd.read_csv(f"{stem}_importance.csv"), pd.read_csv(f"{stem}_curves.csv")
        top = [t for t in imp.term if " & " not in t and t in set(cur.term)][:8]
        fig, axs = plt.subplots(2, 4, figsize=(18, 8))
        for ax, t in zip(axs.ravel(), top):
            d = cur[cur.term == t]
            ax.step(d.x, d.score, where="mid")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(pretty(t), fontsize=9, wrap=True)
            ax.set_xlabel("input value", fontsize=8)
            ax.set_ylabel("effect on flood chance (log-odds)", fontsize=8)
        for ax in axs.ravel()[len(top):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"{stem}_curves.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        ERRORS.append(f"EBM plot {stem}: {e}")
        log(f"  !! EBM plot {stem}: {e}")


# ---------------------------------------------------------------- susceptibility

def susceptibility(P, cfg, out_dir, grp, excl):
    """Where does land flood? Per-square flood frequency from static covariates vs lat/lon, spatial-block CV."""
    from scipy.stats import spearmanr
    out = out_dir / "susceptibility.csv"
    if out.exists():
        log("skip susceptibility (exists)")
        return
    ok = np.isfinite(P.freq) & (P.freq < PERMANENT)
    for n in P.static:
        ok &= np.isfinite(P.a[n])
    rr, cc = np.nonzero(ok)
    rng = np.random.default_rng(0)
    pick = rng.choice(rr.size, size=min(cfg["susc_n"], rr.size), replace=False)
    r, c = rr[pick], cc[pick]
    y = P.freq[r, c]
    stat = np.column_stack([P.a[n][r, c] for n in P.static]).astype(np.float32)
    ll = np.column_stack([P.lat[r], P.lon[c]]).astype(np.float32)
    variants = {"static": stat, "lat_lon": ll, "static+lat_lon": np.column_stack([stat, ll])}
    crop = P.crop[r, c] >= CROP_MIN_PCT
    rows, preds = [], {}
    for v, X in variants.items():
        p = np.full(y.size, np.nan)
        for g in range(G):
            te = grp[r, c] == g
            tr = (grp[r, c] != g) & ~excl[g][r, c]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0)
            p[te] = m.fit(X[tr], y[tr]).predict(X[te])
        preds[v] = p
        for sub, sm in (("all", np.ones_like(crop)), ("cropland", crop)):
            k = sm & np.isfinite(p)
            if k.sum() < 20:
                continue
            ss = ((y[k] - p[k]) ** 2).sum() / ((y[k] - y[k].mean()) ** 2).sum()
            rows.append({"variant": v, "subset": sub, "n_squares": int(k.sum()), "r2": 1 - ss,
                         "spearman": float(spearmanr(y[k], p[k])[0])})
    pd.DataFrame(rows).to_csv(out, index=False)
    # drop-one importance for the static model, same CV
    base = {sub: next(rw["r2"] for rw in rows if rw["variant"] == "static" and rw["subset"] == sub)
            for sub in ("all",)}
    imp = []
    for j, n in enumerate(P.static):
        X = np.delete(stat, j, 1)
        p = np.full(y.size, np.nan)
        for g in range(G):
            te, tr = grp[r, c] == g, (grp[r, c] != g) & ~excl[g][r, c]
            p[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True,
                                                  random_state=0).fit(X[tr], y[tr]).predict(X[te])
        k = np.isfinite(p)
        r2 = 1 - ((y[k] - p[k]) ** 2).sum() / ((y[k] - y[k].mean()) ** 2).sum()
        imp.append({"dropped": n, "r2": r2, "r2_loss": base["all"] - r2})
    pd.DataFrame(imp).sort_values("r2_loss", ascending=False).to_csv(out_dir / "susceptibility_drop_one.csv", index=False)
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        m = ExplainableBoostingRegressor(feature_names=list(P.static), interactions=5, outer_bags=4,
                                         n_jobs=-1, random_state=0).fit(stat, y)
        export_ebm(m, out_dir / "ebm_susceptibility")
    except Exception as e:
        ERRORS.append(f"susceptibility EBM: {e}")
        log(f"  !! susceptibility EBM: {type(e).__name__}: {e}")
    log("wrote susceptibility.csv")


def season_map(P, cfg, out_dir, year=2022, h=1, configs_=("M1", "M3")):
    """Predicted vs observed flooding for every seasonal square in the harvest months of one year.
    Models: GBM, trained on all other years (leave-one-year-out: a map needs every place predicted, so no
    spatial holdout here). Dry squares at t-1 get the occurrence model, wet squares the persistence model.
    Output: map_<year>.npz (per square: mean predicted probability and observed flooded share over those
    target months, crop %), map_<year>.png."""
    out = out_dir / f"map_{year}.npz"
    if out.exists():
        log("skip map (exists)")
        return
    s = sample(P, cfg["per_month"], max(cfg["horizons"]), cfg["sample_seed"])
    lab = P.y[s[:, 0] + h, s[:, 1], s[:, 2]] >= 0.5
    wet = P.y[s[:, 0] - 1, s[:, 1], s[:, 2]] >= 0.5
    train, _, _ = fold_masks(P, s, h, year, "year", None, None)
    norm = normals(P, year)
    X, names = features(P, s, h, norm)
    groups, drivers, st_cols, place = column_sets(names, P)
    cols = {"M1": drivers, "M3": drivers + st_cols + place}
    seasonal = P.scope
    ts = [t for t in range(3, P.T - h) if P.year[t + h] == year and P.cal[t + h] + 1 in HARVEST and P.era_ok[t]]
    rows = []
    for t in ts:
        ok = seasonal & ~(np.isnan(P.y[t - 3:t]).any(0) | np.isnan(P.y[t + h]))
        rr, cc = np.nonzero(ok)
        rows.append(np.column_stack([np.full(rr.size, t), rr, cc]))
    if not rows:
        log(f"map: no target months of {year} in HARVEST with data")
        return
    sa = np.concatenate(rows).astype(np.int64)
    Xa, _ = features(P, sa, h, norm)
    wa = P.y[sa[:, 0] - 1, sa[:, 1], sa[:, 2]] >= 0.5
    obs = (P.y[sa[:, 0] + h, sa[:, 1], sa[:, 2]] >= 0.5).astype(np.float32)
    res = {}
    for cf in configs_:
        p = np.full(len(sa), np.nan, np.float32)
        c = np.array(cols[cf])
        for sel_tr, sel_a in ((train & ~wet, ~wa), (train & wet, wa)):
            tr = np.flatnonzero(sel_tr)
            m = HistGradientBoostingClassifier(**{**GBM_BASE, "random_state": cfg["model_seed"]})
            m.fit(X[np.ix_(tr, c)], lab[tr])
            ia = np.flatnonzero(sel_a)
            for i in range(0, ia.size, 200000):
                j = ia[i:i + 200000]
                p[j] = m.predict_proba(Xa[np.ix_(j, c)])[:, 1]
        res[cf] = p
    H, W = P.y.shape[1:]
    def grid(v):
        s_ = np.zeros((H, W)); n_ = np.zeros((H, W))
        np.add.at(s_, (sa[:, 1], sa[:, 2]), v)
        np.add.at(n_, (sa[:, 1], sa[:, 2]), 1)
        return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan).astype(np.float32)
    maps = {f"pred_{k}": grid(v) for k, v in res.items()}
    maps["observed"] = grid(obs)
    np.savez_compressed(out, **maps, crop_pct=P.crop, lat=P.lat, lon=P.lon,
                        months=np.array([str(P.months[t + h]) for t in ts]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        crop = P.crop >= CROP_MIN_PCT
        panels = [("observed", "observed flooded share")] + [(f"pred_{k}", f"predicted, {k} GBM") for k in res]
        fig, axs = plt.subplots(2, len(panels), figsize=(5 * len(panels), 9))
        ext = [P.lon.min(), P.lon.max(), P.lat.min(), P.lat.max()]
        cmap = matplotlib.colormaps["Blues"].copy()
        cmap.set_bad("lightgrey")                     # grey = not in this panel; white = 0
        for j, (k, title) in enumerate(panels):
            for i, (mask, lab_) in enumerate(((np.ones_like(crop), "seasonal squares"), (crop, "cropland only"))):
                im = axs[i, j].imshow(np.where(mask, maps[k], np.nan), extent=ext, vmin=0, vmax=1, cmap=cmap)
                axs[i, j].set_title(f"{title}, {lab_}", fontsize=9)
        fig.colorbar(im, ax=axs, shrink=0.6, label="share of target months")
        fig.suptitle(f"{year}, target months {', '.join(str(P.months[t + h]) for t in ts)}, lead {h} month(s)")
        fig.savefig(out_dir / f"map_{year}.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        ERRORS.append(f"map plot: {e}")
        log(f"  !! map plot: {e}")
    log(f"wrote map_{year} ({len(sa):,} square-months)")


def safe(f, *a, **k):
    try:
        f(*a, **k)
        return True
    except Exception:
        ERRORS.append(f.__name__)
        log(f"!! {f.__name__} FAILED\n{traceback.format_exc()}")
        return False
