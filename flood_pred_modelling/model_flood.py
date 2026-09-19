"""
Flood occurrence and severity models, and how far ahead either one works.

    python flood_pred_modelling/model_flood.py

Three questions:
  1. how much lead time do we actually have before the model stops beating
     climatology and persistence,
  2. can severity be predicted at all once occurrence is known,
  3. which data streams are carrying the skill.

The target is satellite-detected inundation, not inundation. See README.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 mean_absolute_error, roc_auc_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PANEL = HERE / "output" / "panel.parquet"
ADM2 = REPO / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = HERE / "output"

STATES = ["Unity", "Jonglei"]

AGG_DAYS = 10                              # length of one time block
BLOCKS_PER_YEAR = int(np.ceil(366 / AGG_DAYS))
LEAD_DAYS = [0, 10, 20, 30, 60, 90]   # forecast horizons to test, in days

# Occurrence and onset use the whole record. all_years.py compared the two eras
# on lift, which is the only fair comparison when the base rates differ, and the
# extra six years matched or beat the short record in five of six cases. There
# is no case for discarding them.
START_YEAR = 2015
COMPARE_START_YEAR = 2021  # the single-instrument era, kept as the contrast

# Severity is a different matter. Mean flooded fraction runs about 35x higher
# after 2020 while rainfall ran 0.84x, so the scale is not comparable across the
# break and pooling it would be modelling the sensor. Occurrence survives the
# break because a threshold crossing is more robust than the fraction itself.
SEVERITY_START_YEAR = 2021
FLOOD_THRESHOLD = 0.01     # share of the cell inundated before we call it a flood

ABLATION_LEADS = [10, 30]  # days; one short horizon and one operationally useful

# Leave-one-year-out only controls time. Neighbouring 0.25 degree cells in one
# floodplain are close to duplicates, so a year-fold still trains on near-copies
# of what it is tested on. These reruns hold out whole counties instead.
SPATIAL_LEADS = [10, 30, 90]
COMBINED_LEAD = 30         # strictest scheme, one lead only, it is expensive

# A block where nothing at all was detected anywhere in the AOI is almost
# certainly a block the satellite could not see, not a block with no water in
# it. Scoring those as dry is what teaches the model that rain means no flood.
MASK_UNOBSERVED = False   # measured at 0.2% of blocks, see README section 13

# Onset: cells that are dry now and flood later. Persistence is structurally
# weak for these cases, so this gives the meteorological features a clearer test.
ONSET_LEADS = [10, 20, 30, 60, 90]
ONSET_ABLATION_LEAD = 30

# Rolling sums throw away the shape inside the window: 50 mm in one day and
# 50 mm spread over a week give the same tp_sum7 and flood very differently.
# These recover it cheaply and test whether the rainfall pattern adds information
# that a tree using only totals would miss.
SHAPE_WINDOWS = [7, 30]
WET_DAY_MM = 1.0

# Discharge gauges in Sudan. All downstream of the Sudd, so they cannot lead our
# floods hydrologically, but those at and above Khartoum carry the Blue Nile and
# therefore Ethiopian highland rainfall, which is the same monsoon that rains on
# South Sudan. They are also the only cloud independent predictor available.
SUDAN_STATIONS = [1541, 1542, 1543, 1544, 1545, 1547, 1548, 11808, 11842, 28546]
DISCHARGE_DIR = REPO / "raw_data" / "Darthmouth Flood Observatory"

# Which columns belong to which stream. Prefixes are resolved against the real
# frame at run time and an empty stream is an error, not a skipped row: a data
# stream that never made it into the panel used to look exactly like a stream
# that was tested and found useless.
STREAM_PREFIXES = {
    "location": ("cell_lat", "cell_lon"),
    "seasonality": ("doy_", "block_of_year"),
    "current_state": ("state_",),
    "rainfall": ("tp_sum", "tp_basin_sum"),
    "runoff": ("ro_sum", "ro_basin_sum"),
    "evapotranspiration": ("et_mean",),
    "lakes": ("lake_",),
    "observability": ("obs_",),
    "rain_shape": ("shape_",),
    "sudan_discharge": ("sudan_",),
}


def add_discharge(d: pd.DataFrame) -> pd.DataFrame:
    """Merge the Sudan gauges in as anomalies.

    Raw level would be one national series per station with no spatial
    variation, which under a year fold is a route to memorising the year rather
    than learning hydrology. The deviation from that station's normal flow for
    that day of the year keeps "the river is unusually high" and drops "it is
    2022". Same treatment as the lakes.
    """
    series = {}
    for station in SUDAN_STATIONS:
        path = DISCHARGE_DIR / f"{station}_discharge.csv"
        if not path.exists():
            print(f"  no discharge file for station {station}, skipped")
            continue
        q = pd.read_csv(path)
        q["date"] = pd.to_datetime(q["Date"])
        q = q.dropna(subset=["Discharge (m3/s)"])
        series[f"sudan_{station}"] = q.groupby("date")["Discharge (m3/s)"].mean()

    if not series:
        raise RuntimeError(f"no discharge files found under {DISCHARGE_DIR}")

    wide = pd.concat(series, axis=1)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full).interpolate(method="time", limit_direction="both")
    wide.index.name = "date"

    doy = wide.index.dayofyear
    anom = wide.sub(wide.groupby(doy).transform("mean"))
    anom.columns = [f"{c}_anom" for c in anom.columns]

    # one regional index, so the model has a summary as well as the detail
    anom["sudan_index_anom"] = anom.mean(axis=1)
    anom["sudan_index_d10"] = anom["sudan_index_anom"].diff(10)

    print(f"  discharge: {len(series)} Sudan stations, "
          f"{wide.index.min().date()} -> {wide.index.max().date()}")
    return d.merge(anom.reset_index(), on="date", how="left")


def to_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily panel to (cell, time block) taking the peak inundation.

    The flood product is optical, so a cloudy day reads as a dry day. Taking the
    maximum over a block means one clear pass is enough to register the water.
    """
    d = df.copy()
    d["year"] = d.date.dt.year
    # Blocks are anchored to the calendar rather than counted from an epoch, so
    # block 3 is the same part of the year in every year and the seasonality
    # feature means something. The final block of a year is short.
    d["block_of_year"] = (d.date.dt.dayofyear - 1) // AGG_DAYS
    d["t"] = d.year * BLOCKS_PER_YEAR + d.block_of_year

    # Observability: how much of the AOI reported anything that day. Leave the
    # cell's own detection out of its own covariate, otherwise the feature
    # partly contains the target.
    day_sum = d.groupby("date").flood_any.transform("sum")
    day_n = d.groupby("date").flood_any.transform("count")
    d["obs_day_cov"] = (day_sum - d.flood_any) / (day_n - 1)

    # The masks record flooded pixels only, so there is no explicit "clear sky,
    # no water" record. A day on which not one of the 206 cells reported
    # anything is far more likely a day with no usable observation than a day on
    # which a floodplain with a 16% base rate was entirely dry.
    d["observed"] = (day_sum > 0).astype("int8")

    # Lake levels are one national series: every cell sees the same number on a
    # given day, so the raw level carries no spatial information, only "which
    # day is it", and because the levels trend it effectively encodes the year.
    # Under leave-one-year-out that is a route to memorising the held-out year.
    # The anomaly keeps "the lake is unusually high" and drops "it is 2022".
    doy = d.date.dt.dayofyear
    for lake in ("victoria", "kyoga", "albert"):
        col = f"lake_{lake}"
        if col in d.columns:
            normal = d.groupby(doy)[col].transform("mean")
            d[f"{col}_anom"] = d[col] - normal
            d = d.drop(columns=col)

    d = d.sort_values(["cell_lat", "cell_lon", "date"])

    # Rainfall shape. tp_sum already carries the total, so what is added here is
    # how the total arrived: the biggest single day, and how many days it fell
    # over. Intensity is derived from the two after aggregation.
    d["_wet"] = (d.tp > WET_DAY_MM).astype("int8")
    gtp = d.groupby(["cell_lat", "cell_lon"], sort=False)
    for w in SHAPE_WINDOWS:
        d[f"shape_max{w}"] = gtp.tp.transform(
            lambda s: s.rolling(w, min_periods=w).max())
        d[f"shape_wetdays{w}"] = gtp._wet.transform(
            lambda s: s.rolling(w, min_periods=w).sum())
    d = d.drop(columns="_wet")

    d = add_discharge(d)

    met_cols = [c for c in d.columns if c.startswith(
        ("tp_sum", "ro_sum", "et_mean", "tp_basin_sum", "ro_basin_sum", "lake_",
         "doy_", "shape_", "sudan_"))]

    # Features are read off the LAST day of the block: that is what a forecaster
    # would have in hand at decision time.
    agg = {c: "last" for c in met_cols}
    agg.update({
        "flood_frac": "max",
        "unusual_frac": "max",
        "flood_any": "max",
        "obs_day_cov": "mean",
        "observed": "sum",
        "date": "last",
    })

    blk = (d.groupby(["cell_lat", "cell_lon", "t", "year", "block_of_year"])
             .agg(agg).reset_index())

    for w in SHAPE_WINDOWS:
        # A window with no rain has an intensity of zero, not an unknown one.
        # Dividing by a NaN made it missing on 40% of blocks, and because
        # detection anti-correlates with rainfall those were disproportionately
        # the flooded ones, so the dropna downstream was silently deleting the
        # positives. The genuine warm-up NaN, where the window is not full yet,
        # is preserved.
        wd = blk[f"shape_wetdays{w}"]
        blk[f"shape_intensity{w}"] = (
            blk[f"tp_sum{w}"] / wd.replace(0, np.nan)).mask(wd == 0, 0.0)

    # float rather than int so the column can carry NaN when masking is on.
    blk["flood"] = (blk.flood_frac >= FLOOD_THRESHOLD).astype(float)
    if MASK_UNOBSERVED:
        blind = blk.observed == 0
        blk.loc[blind, ["flood", "flood_frac", "unusual_frac"]] = np.nan
        print(f"  masked {blind.sum():,} of {len(blk):,} blocks "
              f"({blind.mean():.1%}) with no observation anywhere in the AOI")
    # The cell's own current state, which is what the persistence baseline uses.
    # Withholding it from the model while handing it to the baseline is not a
    # fair comparison, and that is how the first version of this was set up.
    blk["state_flood"] = blk.flood
    blk["state_flood_frac"] = blk.flood_frac
    blk["state_unusual_frac"] = blk.unusual_frac

    return blk.sort_values(["cell_lat", "cell_lon", "t"])


def assign_counties(blk):
    """Tag every cell with the county it falls in, for the spatial folds.

    build_panel.py dissolves the AOI and loses county identity, so this joins
    against the ungrouped adm2 frame again.
    """
    import geopandas as gpd

    adm2 = gpd.read_file(ADM2)
    adm2 = adm2[adm2.adm1_name.isin(STATES)][["adm2_name", "geometry"]]

    cells = blk[["cell_lat", "cell_lon"]].drop_duplicates()
    pts = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells.cell_lon, cells.cell_lat),
        crs="EPSG:4326",
    )
    joined = (gpd.sjoin(pts, adm2, predicate="within", how="left")
                 .drop_duplicates(subset=["cell_lat", "cell_lon"]))

    missing = int(joined.adm2_name.isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} of {len(cells)} cells fell outside every county polygon. "
            f"Do not carry on, the spatial folds would silently drop them."
        )

    out = blk.merge(
        joined[["cell_lat", "cell_lon", "adm2_name"]].rename(
            columns={"adm2_name": "county"}),
        on=["cell_lat", "cell_lon"], how="left",
    )
    print(f"  {out.county.nunique()} counties over "
          f"{len(cells)} cells, smallest has "
          f"{out.groupby('county').size().div(out.t.nunique()).min():.0f} cells")
    return out


def add_lead_targets(blk, lead_blocks):
    """Attach the target columns, one per forecast horizon.

    Shifting by -L within each cell puts the state L blocks ahead onto the
    current row, so a row holds features known now and the answer for later.
    The shift relies on each cell having a contiguous run of blocks, which the
    cell-by-day cross join in build_panel.py guarantees.
    """
    g = blk.groupby(["cell_lat", "cell_lon"], sort=False)
    for L in lead_blocks:
        blk[f"y_bin_{L}"] = g.flood.shift(-L)
        blk[f"y_sev_{L}"] = g.flood_frac.shift(-L)
    return blk


def resolve_streams(blk):
    """Map each stream to the columns it owns, and complain if one is empty."""
    streams, seen = {}, set()
    for name, prefixes in STREAM_PREFIXES.items():
        cols = [c for c in blk.columns if c.startswith(prefixes)]
        if not cols:
            raise RuntimeError(
                f"stream '{name}' matched no columns (prefixes {prefixes}). "
                f"Rebuild the panel, do not carry on with the stream missing."
            )
        streams[name] = cols
        seen.update(cols)
    return streams, sorted(seen)


def climatology(train, test, target):
    """Per cell and block-of-year mean of the same lead-shifted target.

    Keyed on the target column rather than on the current state, so the baseline
    and the model are scored on the same quantity.
    """
    # Keyed on cell and time of year. A cell the training folds never saw has no
    # entry, falls back to the global mean below, and scores at the base rate.
    # This is expected because climatology has no cell-specific value for an
    # unseen county, while the model can still use the other features.
    key = ["cell_lat", "cell_lon", "block_of_year"]
    clim = train.groupby(key)[target].mean().rename("clim").reset_index()
    merged = test[key].merge(clim, on=key, how="left")
    return merged.clim.fillna(train[target].mean()).to_numpy()


def eval_binary(y, p):
    """PR-AUC, ROC-AUC and Brier for one fold.

    PR-AUC leads because the classes are imbalanced and it ignores the true
    negatives that dominate the panel. Brier is there because a trigger needs
    calibrated probabilities, not just a good ranking. Accuracy is deliberately
    absent: predicting no flood everywhere would score over 80%.

    Returns NaN scores rather than raising when a fold is single-class, so one
    degenerate year does not kill the whole sweep.
    """
    out = {"n": len(y), "n_pos": int(np.sum(y)), "base_rate": float(np.mean(y))}
    if len(np.unique(y)) < 2:
        return {**out, "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan}
    return {**out,
            "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, np.clip(p, 0, 1))}


def blocked_binary(blk, feats, lead_blocks):
    """Hold out a county AND a year, training on neither.

    Grouping folds by a county_year label is not this: there the county still
    appears in training in other years and the year still appears in other
    counties, so it is the loosest scheme rather than the tightest. Here the
    whole county row and the whole year column are removed from training, so
    neither the place nor the period has been seen.
    """
    tgt = f"y_bin_{lead_blocks}"
    data = blk.dropna(subset=[tgt, "flood"] + feats)
    rows = []

    for county in sorted(data.county.unique()):
        for year in sorted(data.year.unique()):
            tr = data[(data.county != county) & (data.year != year)]
            te = data[(data.county == county) & (data.year == year)]
            if te.empty or tr[tgt].nunique() < 2 or te[tgt].nunique() < 2:
                continue

            gb = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=0,
            ).fit(tr[feats].to_numpy(), tr[tgt].to_numpy())

            preds = {"gbm": gb.predict_proba(te[feats].to_numpy())[:, 1],
                     "climatology": climatology(tr, te, tgt),
                     "persistence": te["flood"].to_numpy().astype(float)}
            for name, pr in preds.items():
                rows.append({"lead_days": lead_blocks * AGG_DAYS,
                             "scheme": "blocked", "fold": f"{county}_{year}",
                             "model": name, **eval_binary(te[tgt].to_numpy(), pr)})

    return pd.DataFrame(rows)


def loyo_binary(blk, feats, lead_blocks, models=("logistic", "gbm"),
                fold_col="year", onset=False):
    """Held-out-fold scores for the models plus both baselines.

    fold_col picks the validation scheme. "year" holds out a year and controls
    temporal autocorrelation. "county" holds out a whole county and controls
    spatial autocorrelation, which is the one leave-one-year-out cannot see:
    adjacent 0.25 degree cells in one floodplain are near duplicates, so a year
    fold still trains on copies of its own test set.
    """
    tgt = f"y_bin_{lead_blocks}"
    if lead_blocks == 0:
        # At lead 0 the cell's current state is the target, so handing it to the
        # model would just be copying the answer across.
        feats = [f for f in feats if not f.startswith("state_")]
    # flood has to be here as well as the features. Persistence reads it
    # directly, and the masked blocks carry NaN, so an ablation run that drops
    # the current_state stream would otherwise let NaN through to the scorer.
    data = blk.dropna(subset=[tgt, "flood"] + feats)
    if onset:
        # Only cells that are dry right now. Persistence predicts "stays dry"
        # for every one of them and is wrong every time a flood starts, so any
        # skill here is skill at anticipating a new flood rather than at
        # noticing an old one.
        data = data[data.flood == 0]
    rows = []

    for fold in sorted(data[fold_col].unique()):
        tr = data[data[fold_col] != fold]
        te = data[data[fold_col] == fold]
        if te.empty or tr[tgt].nunique() < 2 or te[tgt].nunique() < 2:
            continue
        year = fold

        Xtr, ytr = tr[feats].to_numpy(), tr[tgt].to_numpy()
        Xte, yte = te[feats].to_numpy(), te[tgt].to_numpy()

        preds = {"climatology": climatology(tr, te, tgt)}
        if lead_blocks > 0:
            # Carry the cell's current state forward. For inundation this is a
            # hard baseline and it has to be reported next to everything else.
            preds["persistence"] = te["flood"].to_numpy().astype(float)

        if "logistic" in models:
            lr = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
            ).fit(Xtr, ytr)
            preds["logistic"] = lr.predict_proba(Xte)[:, 1]

        if "gbm" in models:
            gb = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=0,
            ).fit(Xtr, ytr)
            preds["gbm"] = gb.predict_proba(Xte)[:, 1]

        for name, p in preds.items():
            rows.append({"lead_days": lead_blocks * AGG_DAYS, "scheme": fold_col,
                         "fold": year, "model": name, **eval_binary(yte, p)})

    return pd.DataFrame(rows)


def summarise(res):
    """Collapse the per-year rows, weighting by positives rather than by year.

    A fold with nine positive cell-blocks should not count as much as a fold
    with nine hundred, which is what a plain mean does.
    """
    rows = []
    for (lead, model), g in res.groupby(["lead_days", "model"]):
        g = g.dropna(subset=["pr_auc"])
        if g.empty:
            continue
        w = g.n_pos.to_numpy(dtype=float)
        w = w / w.sum() if w.sum() else np.full(len(g), 1 / len(g))
        rows.append({
            "lead_days": lead, "model": model, "folds": len(g),
            "scheme": g.scheme.iloc[0] if "scheme" in g else "year",
            "pr_auc": float(np.dot(w, g.pr_auc)),
            "pr_auc_min": float(g.pr_auc.min()),
            "pr_auc_max": float(g.pr_auc.max()),
            "roc_auc": float(np.dot(w, g.roc_auc)),
            "brier": float(np.dot(w, g.brier)),
            "base_rate": float(np.dot(w, g.base_rate)),
        })
    return pd.DataFrame(rows).sort_values(["lead_days", "model"])


def skill_horizon(summary):
    """Longest lead at which the GBM still beats both baselines on PR-AUC."""
    best = None
    for lead in sorted(summary.lead_days.unique()):
        s = summary[summary.lead_days == lead].set_index("model").pr_auc
        if "gbm" not in s:
            continue
        rivals = [s[m] for m in ("climatology", "persistence") if m in s]
        if rivals and s["gbm"] > max(rivals):
            best = lead
    return best


def loyo_severity(blk, feats, lead_blocks):
    """Severity given that a flood happens, so the binary stage has to come first."""
    tgt = f"y_sev_{lead_blocks}"
    if lead_blocks == 0:
        feats = [f for f in feats if not f.startswith("state_")]
    data = blk.dropna(subset=[tgt, "flood_frac"] + feats)
    data = data[data[tgt] >= FLOOD_THRESHOLD]
    rows = []

    for year in sorted(data.year.unique()):
        tr, te = data[data.year != year], data[data.year == year]
        if len(tr) < 200 or te.empty:
            continue

        # Logit transform: the fraction is heavily right skewed and this keeps
        # predictions inside (0, 1).
        clipped = np.clip(tr[tgt], 1e-4, 0.999)
        ytr = np.log(clipped / (1 - clipped))

        gb = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=0,
        ).fit(tr[feats].to_numpy(), ytr)
        pred = 1 / (1 + np.exp(-gb.predict(te[feats].to_numpy())))

        clim = climatology(tr, te, tgt)
        rows.append({
            "lead_days": lead_blocks * AGG_DAYS, "year": year, "n": len(te),
            "mae_gbm": mean_absolute_error(te[tgt], pred),
            "mae_climatology": mean_absolute_error(te[tgt], clim),
            "mae_persistence": mean_absolute_error(te[tgt], te.flood_frac),
        })
    return pd.DataFrame(rows)


def feature_contribution(blk, streams, feats, lead_blocks, onset=False):
    """Drop-one and keep-one, per data stream.

    Drop-column rather than permutation importance, because runoff is largely a
    function of recent rainfall and permutation splits credit between correlated
    features more or less at random. Drop-one on its own still under-reports
    anything redundant, so keep-one is run alongside it: a stream can look
    worthless when dropped and still be perfectly good on its own.
    Only the GBM is fitted here, the logistic result is never read.
    """
    lead_days = lead_blocks * AGG_DAYS

    # Pin the row set once, on the full feature list. Each variant used to
    # re-derive its own dropna from whatever columns it was handed, so 'full',
    # 'dropped' and 'alone' were scored on different populations with different
    # base rates and the delta mixed the feature's real effect with a base-rate
    # shift. PR-AUC is bounded below by the base rate, so a variant that kept
    # more rows scored higher for no reason connected to the feature.
    tgt = f"y_bin_{lead_blocks}"
    keep = [c for c in ([tgt, "flood"] + feats) if c in blk.columns]
    blk = blk.dropna(subset=keep)
    print(f"    common row set: {len(blk):,} blocks")

    full = summarise(loyo_binary(blk, feats, lead_blocks, models=("gbm",),
                                 onset=onset))
    full_score = full.loc[full.model == "gbm", "pr_auc"].iloc[0]

    rows = [{"lead_days": lead_days, "stream": "all", "variant": "full",
             "pr_auc": full_score, "delta": 0.0, "n_features": len(feats)}]

    for name, cols in streams.items():
        kept = [f for f in feats if f not in cols]
        if kept:
            res = summarise(loyo_binary(blk, kept, lead_blocks, models=("gbm",),
                                        onset=onset))
            score = res.loc[res.model == "gbm", "pr_auc"]
            score = float(score.iloc[0]) if len(score) else np.nan
            rows.append({"lead_days": lead_days, "stream": name, "variant": "dropped",
                         "pr_auc": score, "delta": score - full_score,
                         "n_features": len(kept)})

        res = summarise(loyo_binary(blk, cols, lead_blocks, models=("gbm",),
                                    onset=onset))
        score = res.loc[res.model == "gbm", "pr_auc"]
        score = float(score.iloc[0]) if len(score) else np.nan
        rows.append({"lead_days": lead_days, "stream": name, "variant": "alone",
                     "pr_auc": score, "delta": score - full_score,
                     "n_features": len(cols)})
        print(f"    {name} done")

    return pd.DataFrame(rows)


def threshold_sweep(blk):
    """What the flood definition does to the problem.

    A low threshold triggers action on puddles, a high one misses real events.
    The choice needs defending in the report, not assuming.
    """
    rows = []
    blk = blk.dropna(subset=["flood_frac"])
    for thr in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        y = blk.flood_frac >= thr
        rows.append({
            "threshold": thr,
            "base_rate": y.mean(),
            "positive_blocks": int(y.sum()),
            "cells_ever_positive": blk[y].groupby(["cell_lat", "cell_lon"]).ngroups,
        })
    return pd.DataFrame(rows)


def run_sweep(blk, feats, lead_blocks_list, label, say):
    """Score every lead for one training period and report the skill horizon."""
    longest = max(lb * AGG_DAYS for lb in lead_blocks_list)
    per_year, summaries = [], []
    for lb in lead_blocks_list:
        res = loyo_binary(blk, feats, lb)
        if res.empty:
            continue
        res["start_year"] = label
        per_year.append(res)
        summaries.append(summarise(res))

    if not summaries:
        return pd.DataFrame(), pd.DataFrame()

    summary = pd.concat(summaries, ignore_index=True)
    summary["start_year"] = label

    wide = summary.pivot(index="lead_days", columns="model", values="pr_auc")
    say(wide.round(4).to_string())

    h = skill_horizon(summary)
    if h is None:
        say("\nSkill horizon: none. The model never beats both baselines, at any lead.")
    elif h == 0:
        say("\nSkill horizon: 0 days. It only wins on the nowcast, which is not "
            "actionable for ZOA.")
    elif h == longest:
        say(f"\nSkill horizon: at least {h} days, which is the longest lead tested. "
            f"The sweep never found the limit, so extend LEAD_DAYS before quoting "
            f"{h} days as the ceiling.")
    else:
        say(f"\nSkill horizon: {h} days. Beyond that the baselines win.")

    return pd.concat(per_year, ignore_index=True), summary


def main():
    if not PANEL.exists():
        sys.exit(f"panel not found at {PANEL}")

    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    print(f"daily panel: {df.shape}")

    blk_all = to_blocks(df)
    blk_all = assign_counties(blk_all)
    blk_all["county_year"] = blk_all.county + "_" + blk_all.year.astype(str)
    print(f"{AGG_DAYS}-day blocks: {blk_all.shape}  "
          f"({blk_all.t.nunique()} blocks, "
          f"{blk_all.groupby(['cell_lat','cell_lon']).ngroups} cells)")

    lead_blocks_list = sorted({d // AGG_DAYS for d in LEAD_DAYS})

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    def prepare(start_year):
        blk = blk_all[blk_all.year >= start_year].copy()
        blk = add_lead_targets(blk, lead_blocks_list)
        streams, feats = resolve_streams(blk)
        return blk, streams, feats

    blk, streams, feats = prepare(START_YEAR)
    print(f"{START_YEAR}+: {blk.shape}, {len(feats)} features in {len(streams)} streams")

    say("HOW A FLOOD IS DEFINED")
    say(threshold_sweep(blk).round(4).to_string(index=False))
    say(f"\nusing FLOOD_THRESHOLD = {FLOOD_THRESHOLD} "
        f"({int(FLOOD_THRESHOLD * 14400)} of 14400 pixels in a cell)")
    say(f"base rate at that threshold, {START_YEAR}+: {blk.flood.mean():.4f}")

    say("\n\nHOW FAR AHEAD CAN OCCURRENCE BE PREDICTED")
    say(f"PR-AUC by lead, leave-one-year-out, training from {START_YEAR}.")
    say("Every model here sees location, season and the cell's current state, so")
    say("the baselines no longer hold information the model was denied.\n")
    per_year_main, summary_main = run_sweep(blk, feats, lead_blocks_list,
                                            str(START_YEAR), say)

    say("\n\nSAME SWEEP TRAINED FROM %d" % COMPARE_START_YEAR)
    say("Detection volume before 2021 is a fraction of what it is after, so this")
    say("run mostly measures how much the observing system changed.\n")
    blk_early, _, feats_early = prepare(COMPARE_START_YEAR)
    per_year_early, summary_early = run_sweep(blk_early, feats_early,
                                              [d // AGG_DAYS for d in SPATIAL_LEADS],
                                              str(COMPARE_START_YEAR), say)

    all_summary = pd.concat([s for s in (summary_main, summary_early) if not s.empty],
                            ignore_index=True)
    all_summary.to_csv(OUT / "skill_by_lead.csv", index=False)
    pd.concat([s for s in (per_year_main, per_year_early) if not s.empty],
              ignore_index=True).to_csv(OUT / "skill_by_lead_per_year.csv", index=False)

    say()
    say()
    say("DOES IT HOLD UP ACROSS SPACE, NOT JUST ACROSS TIME")
    say("Leave-one-year-out only controls time. Neighbouring cells in one")
    say("floodplain are near duplicates, so a year fold still trains on copies of")
    say("its own test set. These schemes hold out whole counties instead, which")
    say("is the situation ZOA is actually in: deploying to a cell the model was")
    say("never fitted on.")
    say()

    scheme_rows = []
    for scheme, leads, label in (
        ("year", SPATIAL_LEADS, "leave-one-year-out"),
        ("county", SPATIAL_LEADS, "leave-one-county-out"),
    ):
        for lead_days in leads:
            lb = lead_days // AGG_DAYS
            print(f"  {label}, lead {lead_days} days")
            res = loyo_binary(blk, feats, lb, models=("gbm",), fold_col=scheme)
            if res.empty:
                continue
            sm = summarise(res)
            sm["scheme"] = label
            scheme_rows.append(sm)

    print(f"  blocked county+year, lead {COMBINED_LEAD} days")
    bres = blocked_binary(blk, feats, COMBINED_LEAD // AGG_DAYS)
    if not bres.empty:
        bsm = summarise(bres)
        bsm["scheme"] = "held out county AND year"
        scheme_rows.append(bsm)

    if scheme_rows:
        schemes = pd.concat(scheme_rows, ignore_index=True)
        schemes.to_csv(OUT / "validation_schemes.csv", index=False)
        say(schemes.pivot_table(index=["scheme", "lead_days"], columns="model",
                                values="pr_auc").round(4).to_string())
        say("")
        for lead_days in SPATIAL_LEADS:
            t = schemes[(schemes.scheme == "leave-one-year-out")
                        & (schemes.lead_days == lead_days)
                        & (schemes.model == "gbm")].pr_auc
            sp = schemes[(schemes.scheme == "leave-one-county-out")
                         & (schemes.lead_days == lead_days)
                         & (schemes.model == "gbm")].pr_auc
            if len(t) and len(sp):
                say(f"lead {lead_days:>3} days: temporal {t.iloc[0]:.4f} -> "
                    f"spatial {sp.iloc[0]:.4f}  ({sp.iloc[0] - t.iloc[0]:+.4f})")
        say("")
        say("Read these three together, they control different things.")
        say("Leave-one-county-out removes the test cell's own history but keeps")
        say(f"full temporal overlap: the model still saw {START_YEAR}-2025 in")
        say("the other counties, and floods here are regionally synchronous, so")
        say("knowing a year was wet everywhere carries it. That is why the GBM")
        say("does not fall. Climatology does fall hard, because a held-out cell")
        say("has no per-cell climatology to look up.")
        say("")
        say("Holding out a county AND a year is the strict test: neither the")
        say("place nor the period is in training. Quote that one. Note that a")
        say("county_year fold label is NOT this, it leaves the county in other")
        say("years and the year in other counties, which makes it the loosest")
        say("scheme rather than the tightest.")

    say("\n\nSEVERITY, GIVEN THAT A FLOOD HAPPENS")
    say("Mean absolute error on the inundated fraction, flooded blocks only.")
    say("This is stage two: it is only usable behind the occurrence model.\n")
    sev_rows = []
    sev_blk = blk[blk.year >= SEVERITY_START_YEAR]
    say(f"Restricted to {SEVERITY_START_YEAR}+ because the flooded-fraction scale")
    say("is not comparable across the 2020 instrument change.")
    say()
    for lb in lead_blocks_list:
        res = loyo_severity(sev_blk, feats, lb)
        if res.empty:
            continue
        sev_rows.append(res)
        w = res.n / res.n.sum()
        say(f"lead {lb * AGG_DAYS:>3} days   gbm {np.dot(w, res.mae_gbm):.4f}   "
            f"climatology {np.dot(w, res.mae_climatology):.4f}   "
            f"persistence {np.dot(w, res.mae_persistence):.4f}   "
            f"(n={int(res.n.sum())})")
    if sev_rows:
        sev = pd.concat(sev_rows, ignore_index=True)
        sev.to_csv(OUT / "severity_by_lead.csv", index=False)
        mean_mae = sev.groupby("lead_days")[["mae_gbm", "mae_climatology"]].mean()
        beat = mean_mae.index[mean_mae.mae_gbm < mean_mae.mae_climatology].tolist()
        say(f"\nLeads where the regression beats climatology: "
            f"{beat if beat else 'none'}")
    else:
        say("no severity folds had enough flooded blocks to fit")

    say()
    say()
    say("CAN WE SEE A FLOOD COMING, NOT JUST SEE ONE THAT IS ALREADY HERE")
    say("Restricted to cells that are DRY at decision time. Persistence predicts")
    say("'stays dry' for all of them, so it scores at the base rate. Skill here is")
    say("skill at anticipating a new flood, not at noticing an old one.")
    say()
    onset_rows = []
    for lead_days in ONSET_LEADS:
        lb = lead_days // AGG_DAYS
        print(f"  onset, lead {lead_days} days")
        res = loyo_binary(blk, feats, lb, onset=True)
        if not res.empty:
            onset_rows.append(summarise(res))
    if onset_rows:
        onset = pd.concat(onset_rows, ignore_index=True)
        onset.to_csv(OUT / "onset_by_lead.csv", index=False)
        say(onset.pivot(index="lead_days", columns="model",
                        values="pr_auc").round(4).to_string())
        say()
        say("base rate of onset by lead:")
        say(onset[onset.model == "gbm"][["lead_days", "base_rate"]]
            .round(4).to_string(index=False))
        say()
        say("Compare each PR-AUC against the base rate on its own row, not against")
        say("the occurrence numbers above. A model that only matches the base rate")
        say("has found nothing.")
    else:
        say("no onset folds had enough positives to fit")

    say("\n\nWHICH STREAMS CARRY THE SKILL")
    say("dropped = PR-AUC change when the stream is removed from the full model.")
    say("alone   = PR-AUC when the stream is the only thing the model sees.")
    say("A stream can be near zero on 'dropped' and still strong on 'alone', which")
    say("means it is real but redundant with something else.\n")
    contrib = []
    for lead_days in ABLATION_LEADS:
        lb = lead_days // AGG_DAYS
        print(f"  feature contribution at lead {lead_days} days")
        tab = feature_contribution(blk, streams, feats, lb)
        contrib.append(tab)
        say(f"lead {lead_days} days:")
        say(tab.pivot(index="stream", columns="variant", values="pr_auc")
               .round(4).to_string())
        say("")
    if contrib:
        pd.concat(contrib, ignore_index=True).to_csv(
            OUT / "feature_contribution.csv", index=False)

    say()
    say()
    say("DOES METEOROLOGY HELP WHEN THERE IS NO FLOOD TO PERSIST")
    say("The same drop-one and keep-one, restricted to cells that are dry now.")
    say("This is the fair test of the rainfall, runoff and ET streams: there is")
    say("no standing water for them to be redundant with.")
    say()
    lb = ONSET_ABLATION_LEAD // AGG_DAYS
    print(f"  onset feature contribution at lead {ONSET_ABLATION_LEAD} days")
    onset_tab = feature_contribution(blk, streams, feats, lb, onset=True)
    onset_tab["target"] = "onset"
    onset_tab.to_csv(OUT / "feature_contribution_onset.csv", index=False)
    say(f"lead {ONSET_ABLATION_LEAD} days, onset only:")
    say(onset_tab.pivot(index="stream", columns="variant", values="pr_auc")
        .round(4).to_string())
    say()

    say("\n\nBEFORE QUOTING ANY NUMBER ABOVE")
    say("The target is satellite-detected inundation, not inundation. The sensor")
    say("is optical and cannot see through cloud, which is exactly when it rains.")
    say("Block maxima and the observability covariate reduce that bias, they do")
    say("not remove it. Read every score as skill at predicting what the sensor")
    say("will see, which is a lower bound on skill at predicting real water.")
    say("")
    say("The observability stream is built from the detection field itself, so it")
    say("is partly autoregressive. Skill that rests on it is not a meteorological")
    say("forecast. Check its row in the contribution table before claiming one.")
    say("")
    say("Lead 0 is a reference row. It is a nowcast and ZOA cannot act on it.")

    (OUT / "model_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten to {OUT / 'model_report.txt'}")


if __name__ == "__main__":
    main()
