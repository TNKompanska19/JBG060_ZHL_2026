"""Simple, explainable flood model: will a county flood more than usual at harvest time?

    python flood_pred_modelling/flood_model.py        (run from the repo root)

Question, per county and season: will the Oct-Dec flooded share of the county's
normally dry land be ABOVE that county's normal? The forecast is issued on
1 August, i.e. 2-5 months before the flooding it predicts.

Target: CYGNSS county table from cygnss_target.py. Normal = the county's mean
over the TRAINING seasons only, recomputed in every fold.
Inputs, all known on 1 August:
  prev_season  last year's Oct-Dec flooding minus the county normal.
               Sudd floods carry over between years (Alfieri et al. 2024).
  dry_season   this year's Mar-May flooding minus the county's Mar-May normal.
               The newest CYGNSS maps at issue time: a month's map needs the
               following month's data (Pu et al. 2024).
  lake         Lake Victoria mean level in Jun-Jul minus its training mean.
               Sudd Oct-Dec extent follows Lake Victoria, R = 0.73 (Hardy et al. 2023).
Counties: flood-prone ones (Oct-Dec mean >= 5%). Seasons: 2019-2025.
Validation: leave one season out. Baselines: climatology (training base rate)
and persistence (above normal last season => above normal this season).
Model: logistic regression on standardised inputs, so coefficients compare.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "processed_data" / "cygnss" / "county_month.csv"
OUT = Path(__file__).resolve().parent / "output"
SEASONS = list(range(2019, 2026))
PRONE = 5.0                                   # % Oct-Dec mean to count as flood-prone
FEATURES = ["prev_season", "dry_season", "lake"]


def lake_jun_jul():
    """Lake Victoria mean level in Jun-Jul per year, via the course loader."""
    sys.path.insert(0, str(ROOT / "processing_data"))
    os.chdir(ROOT)                            # the loader uses paths relative to the repo root
    from loading import load_lake_stations
    v = load_lake_stations()["victoria"]["height_wrt_ref"]
    v = v[v.index.month.isin([6, 7])]
    return v.groupby(v.index.year).mean()


def frame(year, ond, mam, lake, norm_ond, norm_mam, lake_mean):
    """Inputs and target for one season, using the given (training) normals."""
    return pd.DataFrame({
        "county": norm_ond.index, "year": year,
        "prev_season": (ond[year - 1] - norm_ond).to_numpy(),
        "dry_season": (mam[year] - norm_mam).to_numpy(),
        "lake": lake[year] - lake_mean,
        "y": (ond[year] > norm_ond).astype(int).to_numpy(),
    })


def data(train, ond, mam, lake):
    norm_ond, norm_mam = ond[train].mean(axis=1), mam[train].mean(axis=1)
    lake_mean = lake[train].mean()
    return lambda years: pd.concat(
        [frame(y, ond, mam, lake, norm_ond, norm_mam, lake_mean) for y in years],
        ignore_index=True)


def fit(tr, cols):
    sc = StandardScaler().fit(tr[cols])
    return sc, LogisticRegression().fit(sc.transform(tr[cols]), tr.y)


def main(lake=None):
    d = pd.read_csv(TARGET)
    ond = d[d.cal_month >= 10].groupby(["county", "year"]).water_pct.mean().unstack()
    mam = d[d.cal_month.between(3, 5)].groupby(["county", "year"]).water_pct.mean().unstack()
    prone = ond.index[ond.mean(axis=1) >= PRONE]
    ond, mam = ond.loc[prone], mam.loc[prone]
    lake = lake_jun_jul() if lake is None else lake
    print(f"{len(prone)} flood-prone counties x {len(SEASONS)} seasons")

    preds, ablation = [], {}
    for test in SEASONS:
        train = [s for s in SEASONS if s != test]
        make = data(train, ond, mam, lake)
        tr, te = make(train), make([test])
        sc, m = fit(tr, FEATURES)
        te["p_model"] = m.predict_proba(sc.transform(te[FEATURES]))[:, 1]
        te["p_persistence"] = (te.prev_season > 0).astype(float)
        te["p_climatology"] = tr.y.mean()
        preds.append(te)
        for label, cols in ([(f"without {f}", [c for c in FEATURES if c != f]) for f in FEATURES]
                            + [(f"{f} alone", [f]) for f in FEATURES]):
            s, mm = fit(tr, cols)
            ablation.setdefault(label, []).append(
                pd.Series(mm.predict_proba(s.transform(te[cols]))[:, 1], index=te.index))
    p = pd.concat(preds, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    p.to_csv(OUT / "flood_model_predictions.csv", index=False)

    print(f"\nLeave-one-season-out, {len(p)} county-seasons, {p.y.mean():.0%} above normal")
    print("PR-AUC: higher is better, chance = the base rate. Brier: lower is better.")
    res = pd.DataFrame({k: {"PR-AUC": average_precision_score(p.y, p[f"p_{k}"]),
                            "Brier": brier_score_loss(p.y, p[f"p_{k}"])}
                        for k in ("model", "persistence", "climatology")}).T
    print(res.round(3).to_string())

    print("\nPer season: share of counties called right (model at 50%, persistence):")
    p["hit_model"] = (p.p_model >= 0.5) == p.y
    p["hit_persistence"] = (p.p_persistence >= 0.5) == p.y
    print(p.groupby("year")[["y", "hit_model", "hit_persistence"]].mean().round(2)
          .rename(columns={"y": "share_above_normal"}).to_string())

    print("\nWhich input carries the skill (PR-AUC, pooled over seasons):")
    print(pd.Series({k: average_precision_score(p.y, pd.concat(v).sort_index())
                     for k, v in ablation.items()}).round(3).to_string())

    sc, m = fit(data(SEASONS, ond, mam, lake)(SEASONS), FEATURES)
    print("\nFitted on all seasons: odds multiplier per 1 standard deviation of each input")
    print(pd.Series(np.exp(m.coef_[0]), index=FEATURES).round(2).to_string())


if __name__ == "__main__":
    main()
