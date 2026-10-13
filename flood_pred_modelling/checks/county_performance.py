"""Per-county skill of the strict h=1 predictions, and per-county / per-year input coverage (SLE essay).

    python flood_pred_modelling/checks/county_performance.py      (from the repo root)

Writes checks/output/county_performance.md and prints the same tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "flood_pred_modelling" / "3_model" / "output" / "main" / "preds_strict_h1.parquet"
PANEL = ROOT / "processed_data" / "panel_0.02"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = Path(__file__).resolve().parent / "output" / "county_performance.md"

MODEL = "p|M3|gbm"   # scored model; baselines: p_clim (climatology), p_pers (persistence)
FAR = 0.10           # alarm cutoff = model score exceeded by 10% of non-flood rows, set once nationally per target
MIN_N = 20           # a county is scored only with >= 20 flood rows and >= 20 non-flood rows
SKIP = {"lat", "lon", "months", "y", "nbr", "crop_pct", "hw", "hw_dist"}   # hw is off in the main model
IDX = {**dict.fromkeys(["tp", "ro", "r20", "rx1", "wetd", "wspell", "conc5", "dsd"], "era5_idx"),
       "et": "et_idx", "sm1": "sm_idx", "sm3": "sm_idx", "uprain": "uprain_idx", "grace": "grace_idx",
       **dict.fromkeys(["t2m", "d2m", "si10", "ssrd"], "atm_idx"),
       **dict.fromkeys(["q", "qk10", "qk30", "qk100"], "q_idx")}


def md(df):
    df = df.reset_index().round(3)
    df = df.astype(object).where(df.notna(), "")
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    return "\n".join(rows + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def score(g, cut):
    y = g.y.to_numpy()
    out = {"rows": len(g), "squares": g[["row", "col"]].drop_duplicates().shape[0],
           "floods": int(y.sum()), "base_rate": y.mean(), "crop_pct": g.crop.mean()}
    if y.sum() < MIN_N or (1 - y).sum() < MIN_N:
        return out
    for k, p in [("model", MODEL), ("clim", "p_clim"), ("pers", "p_pers")]:
        out[f"pr_auc_{k}"] = average_precision_score(y, g[p])
        out[f"roc_auc_{k}"] = roc_auc_score(y, g[p])
        out[f"brier_{k}"] = brier_score_loss(y, g[p])
    out["pr_auc_gain_vs_clim"] = out["pr_auc_model"] - out["pr_auc_clim"]
    a = g[MODEL].to_numpy() >= cut
    hit = (a & (y == 1)).sum()
    out["alarm_rate"], out["recall"], out["precision"] = a.mean(), hit / y.sum(), hit / max(a.sum(), 1)
    out["f1"] = 2 * out["recall"] * out["precision"] / max(out["recall"] + out["precision"], 1e-9)
    return out


def main():
    d = pd.read_parquet(PRED)
    z = np.load(PANEL / "inputs.npz")
    lat, lon, months = z["lat"], z["lon"], pd.PeriodIndex(z["months"], freq="M")
    res = round(float(lon[1] - lon[0]), 6)

    # County per square, rasterised exactly as in build_target.counties
    gj = json.loads(ADMIN.read_text(encoding="utf8"))
    names = [f["properties"]["adm2_name"] for f in gj["features"]]
    states = [f["properties"]["adm1_name"] for f in gj["features"]]
    cid = rasterize([(f["geometry"], i + 1) for i, f in enumerate(gj["features"])], out_shape=(lat.size, lon.size),
                    transform=from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res), fill=0, dtype="int32")
    r, c = d.row.to_numpy(), d.col.to_numpy()
    d["county"] = [names[i - 1] if i else "outside" for i in cid[r, c]]
    d["state"] = [states[i - 1] if i else "outside" for i in cid[r, c]]
    t = months.get_indexer(pd.PeriodIndex(d.issue, freq="M"))   # inputs are read at the issue month

    # 1. Skill per county
    parts = [f"Model {MODEL}; strict CV, h=1. Alarm cutoff: score exceeded by {FAR:.0%} of non-flood rows, "
             f"set nationally per target. Counties with < {MIN_N} flood or non-flood rows are not scored."]
    for tgt, g in d.groupby("target"):
        cut = np.quantile(g.loc[g.y == 0, MODEL], 1 - FAR)
        tab = pd.DataFrame({k: score(v, cut) for k, v in g.groupby(["state", "county"])}).T
        tab.index.names = ["state", "county"]
        tab[["rows", "squares", "floods"]] = tab[["rows", "squares", "floods"]].astype(int)
        tab = tab.sort_values("pr_auc_gain_vs_clim" if "pr_auc_gain_vs_clim" in tab else "rows")
        parts += [f"## Skill per county, target {tgt}", md(tab)]
        sc = tab.dropna(subset=["pr_auc_gain_vs_clim"]).astype(float)
        cor = sc[["floods", "squares", "base_rate", "crop_pct"]].corrwith(sc.pr_auc_gain_vs_clim, method="spearman")
        parts += [f"## Spearman correlation of county PR-AUC gain over climatology, target {tgt} "
                  f"({len(sc)} scored counties)", md(cor.rename("spearman").rename_axis("variable").to_frame())]
        if tgt == "occurrence":
            occ_tab = tab

    # 2. Input coverage: share of prediction rows with the input missing at the issue month
    miss = {}
    for k in z.files:
        if k in SKIP or k.endswith("_idx"):
            continue
        a, ix = z[k], z[IDX[k]][r, c] if k in IDX else None
        if a.ndim == 3:
            v = a[t, r, c]
        elif a.ndim == 2 and a.shape == (lat.size, lon.size):
            v = a[r, c]
        elif a.ndim == 2 and ix is not None:
            v = a[t, ix]
        elif a.ndim == 1 and ix is not None:
            v = a[ix]
        else:
            print(f"  skipped {k}: shape {a.shape}, no index mapping")
            continue
        miss[k] = np.isnan(v.astype(np.float32))
    shared = pd.read_csv(PANEL / "shared.csv", index_col=0)
    shared.index = pd.PeriodIndex(shared.index, freq="M")
    sh = shared.reindex(pd.PeriodIndex(d.issue, freq="M"))
    for k in shared.columns:
        miss[f"shared:{k}"] = sh[k].isna().to_numpy()
    m = pd.DataFrame(miss).astype(float)
    any_gap = m.columns[m.mean() > 0]
    parts += ["## Inputs with any missing values at prediction time (% of all prediction rows)",
              md((m[any_gap].mean() * 100).rename("missing_pct").rename_axis("input").to_frame())]
    if len(any_gap):
        by_year = (m[any_gap].groupby(d.test_year.to_numpy()).mean() * 100).rename_axis("test_year")
        by_cty = m[any_gap].groupby([d.state.to_numpy(), d.county.to_numpy()]).mean() * 100
        by_cty = by_cty.loc[:, by_cty.std() > 0].rename_axis(["state", "county"])
        parts += ["## Missing % per test year", md(by_year),
                  "## Missing % per county (inputs varying by county only)",
                  md(by_cty)]

    # 3. Target coverage per county: squares in the flood-prone scope, and target missing (permanent water / no CYGNSS)
    ty = (months.year >= d.test_year.min()) & (months <= pd.Period(d.issue.max(), "M"))
    y_nan = np.isnan(z["y"][ty].astype(np.float32)).mean(0)
    sq = pd.DataFrame({"cid": cid.ravel(), "y_missing_pct": y_nan.ravel() * 100})
    sq = sq[sq.cid > 0]
    inscope = pd.Series(1, index=pd.MultiIndex.from_frame(d[["row", "col"]].drop_duplicates()))
    sq["in_scope"] = inscope.reindex(pd.MultiIndex.from_arrays(np.unravel_index(sq.index, cid.shape))).notna().to_numpy()
    cov = sq.groupby("cid").agg(county_squares=("cid", "size"), in_scope_pct=("in_scope", "mean"),
                                y_missing_pct=("y_missing_pct", "mean"))
    cov["in_scope_pct"] *= 100
    cov.index = pd.MultiIndex.from_arrays([[states[i - 1] for i in cov.index], [names[i - 1] for i in cov.index]],
                                           names=["state", "county"])
    cov = cov.sort_values("in_scope_pct")
    parts += ["## Target coverage per county (test period): share of squares in scope, target missing %", md(cov)]
    j = occ_tab.join(cov).dropna(subset=["pr_auc_gain_vs_clim"]).astype(float)
    parts += ["## Spearman: occurrence PR-AUC gain vs target coverage",
              md(j[["in_scope_pct", "y_missing_pct"]].corrwith(j.pr_auc_gain_vs_clim, method="spearman")
                 .rename("spearman").rename_axis("variable").to_frame())]

    text = "\n\n".join(parts) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
