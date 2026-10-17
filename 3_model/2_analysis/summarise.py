"""Scores and tables from the explain_models.py outputs.

    python 3_model/2_analysis/summarise.py [--out results/run_all]

Writes scores.csv (every config x model x subset), ablation.csv, results.md (tables only) into the run folder.
Subsets: all, cropland (crop >= CROP_MIN_PCT), harvest (target month in HARVEST), cropland_harvest.
Ablation (anomaly inputs, strict): adrop = PR-AUC(M1_anomaly) - PR-AUC(M1_anomaly without group);
  fdrop = same without a whole family; hadd = PR-AUC(H + family) - PR-AUC(H).
  noise = SD of PR-AUC over the 80%-subsample runs of the base (M1a_s*, H_s*).
  years_positive = test years in which the effect is > 0 (per-year PR-AUC).
  informative = delta > 2 x noise AND positive in all but at most one test year AND the full model (M3_anomaly,
  same model type) beats both baselines in that slice (pooled PR-AUC).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_run"))   # for explain_models
from explain_models import CROP_MIN_PCT, HARVEST, pretty

OUT = Path(__file__).resolve().parents[2] / "results" / "run_all"
KEYS = ["cv", "h", "target"]


def subsets(df):
    crop, harv = df.crop.to_numpy() >= CROP_MIN_PCT, df.tmonth.isin(HARVEST).to_numpy()
    return {"all": np.ones(len(df), bool), "cropland": crop, "harvest": harv, "cropland_harvest": crop & harv}


def metrics(y, p, years=None):
    """Ranking (PR-AUC, ROC-AUC, Gini), probability (Brier, log loss) and, with years, threshold metrics.
    The cutoff for each test year maximises F1 on the OTHER test years, never on the rows it is scored on."""
    k = np.isfinite(p)
    y, p = y[k], p[k]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    roc = float(roc_auc_score(y, p))
    r = dict(n=len(y), base_rate=float(y.mean()), pr_auc=float(average_precision_score(y, p)), roc_auc=roc,
             gini=2 * roc - 1, brier=float(brier_score_loss(y, np.clip(p, 0, 1))),
             log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])))
    if years is not None:
        r.update(threshold_metrics(y, p, years[k]))
    return r


def threshold_metrics(y, p, yr):
    pred = np.zeros(len(y), bool)
    for Y in np.unique(yr):
        tr, te = yr != Y, yr == Y
        if len(np.unique(y[tr])) < 2:
            continue
        pr, rc, th = precision_recall_curve(y[tr], p[tr])
        f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
        pred[te] = p[te] >= th[np.argmax(f1)]
    return confusion(y, pred)


def confusion(y, pred):
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(accuracy=(tp + tn) / len(y), always_no_accuracy=1 - float(np.mean(y)), precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), balanced_accuracy=(rec + tn / max(tn + fp, 1)) / 2,
                false_alarm_rate=fp / max(fp + tn, 1))


FULL_METRIC_CONFIGS = {"climatology", "persistence", "H", "M1", "M2", "M3", "M1_anomaly", "M2_anomaly", "M3_anomaly"}


SWEEP_COLS = ["p_clim", "p_pers", "p|M1|gbm", "p|M1_anomaly|gbm", "p|M3|gbm", "p|M3_anomaly|gbm"]
BUDGETS = (0.05, 0.10, 0.20, 0.30)          # share of squares flagged (alarm budget)
PROB_CUTS = (0.1, 0.2, 0.3, 0.5, 0.7)       # fixed probability cutoffs


def cutoff_sweep(out_dir):
    """Threshold metrics at fixed alarm budgets (flag the riskiest X% of squares) and fixed probability cutoffs,
    strict validation, main models. A sweep shows the trade-off; it does not pick a cutoff."""
    rows = []
    for f in sorted(out_dir.glob("preds_strict_h*.parquet")):
        d = pd.read_parquet(f)
        for (h, target), g in d.groupby(["h", "target"]):
            for sub, m in subsets(g).items():
                if sub not in ("all", "cropland_harvest") or m.sum() < 50:
                    continue
                gg = g[m]
                y = gg.y.to_numpy()
                for col in SWEEP_COLS:
                    if col not in gg:
                        continue
                    p = gg[col].to_numpy(float)
                    ok = np.isfinite(p)
                    name = col[2:].replace("|", ": ") if col.startswith("p|") else col[2:]
                    for b in BUDGETS:
                        cut = np.quantile(p[ok], 1 - b)
                        rows.append(dict(h=h, target=target, subset=sub, model=name, rule=f"top {b:.0%} flagged",
                                         **confusion(y[ok], p[ok] >= cut)))
                    for c in PROB_CUTS:
                        rows.append(dict(h=h, target=target, subset=sub, model=name, rule=f"p >= {c}",
                                         **confusion(y[ok], p[ok] >= c)))
    sw = pd.DataFrame(rows).drop_duplicates(["h", "target", "subset", "model", "rule"])
    sw.to_csv(out_dir / "cutoff_sweep.csv", index=False)
    return sw


def recall_at(y, p, fpr_max):
    k = np.isfinite(p)
    if k.sum() < 20 or len(np.unique(y[k])) < 2:
        return np.nan
    f, t, _ = roc_curve(y[k], p[k])
    return float(t[f <= fpr_max].max())


def score_table(out_dir):
    files = sorted(out_dir.glob("preds_*.parquet"))
    cache = out_dir / "scores.csv"
    if cache.exists() and files and cache.stat().st_mtime > max(f.stat().st_mtime for f in files):
        c = pd.read_csv(cache)
        if "accuracy" in c:
            return c
    rows = []
    for f in files:
        d = pd.read_parquet(f)
        pc = [c for c in d.columns if c.startswith("p|")]
        for key, g in d.groupby(KEYS):
            for sub, m in subsets(g).items():
                gg = g[m]
                y = gg.y.to_numpy()
                yrs = gg.test_year.to_numpy()
                for col, cfg, model in [("p_clim", "climatology", "baseline"), ("p_pers", "persistence", "baseline")] + \
                        [(c, c.split("|")[1], c.split("|")[2]) for c in pc]:
                    r = metrics(y, gg[col].to_numpy(float), yrs if cfg in FULL_METRIC_CONFIGS else None)
                    if r:
                        rows.append(dict(zip(KEYS, key), subset=sub, config=cfg, model=model, **r))
    sc = pd.DataFrame(rows).drop_duplicates(KEYS + ["subset", "config", "model"])
    sc.to_csv(cache, index=False)
    return sc


def year_scores(out_dir):
    """PR-AUC per test year, for every config x model x subset (for the per-year consistency of effects)."""
    files = sorted(out_dir.glob("preds_*.parquet"))
    cache = out_dir / "year_scores.csv"
    if cache.exists() and files and cache.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return pd.read_csv(cache)
    rows = []
    for f in files:
        d = pd.read_parquet(f)
        pc = [c for c in d.columns if c.startswith("p|")]
        for key, g in d.groupby(KEYS + ["test_year"]):
            for sub, m in subsets(g).items():
                gg = g[m]
                y = gg.y.to_numpy()
                for c in pc:
                    r = metrics(y, gg[c].to_numpy(float))
                    if r:
                        rows.append(dict(zip(KEYS + ["test_year"], key), subset=sub, config=c.split("|")[1],
                                         model=c.split("|")[2], pr_auc=r["pr_auc"]))
    ys = pd.DataFrame(rows).drop_duplicates(KEYS + ["test_year", "subset", "config", "model"])
    ys.to_csv(cache, index=False)
    return ys


BASES = {"adrop": "M1_anomaly", "fdrop": "M1_anomaly", "hadd": "H"}   # ablation prefix -> base
SEEDS = {"M1_anomaly": r"M1a_s\d+", "H": r"H_s\d+"}                   # base -> its 80%-subsample runs


def ablation_table(out_dir):
    import re
    sc = score_table(out_dir)
    if sc.empty:
        return pd.DataFrame()
    ys = year_scores(out_dir)
    yr = {k: g.set_index("test_year").pr_auc for k, g in ys.groupby(KEYS + ["subset", "config", "model"])}
    rows = []
    for key, g in sc.groupby(KEYS + ["subset"]):
        pr = {(r.config, r.model): r.pr_auc for r in g.itertuples()}
        br = {(r.config, r.model): r.brier for r in g.itertuples()}
        best_base = max(pr.get(("climatology", "baseline"), np.nan), pr.get(("persistence", "baseline"), np.nan))
        for model in ("gbm", "logreg"):
            sd = {}
            for b, pat in SEEDS.items():
                v = [pr[(c, mm)] for (c, mm) in pr if mm == model and re.fullmatch(pat, c)]
                sd[b] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            full_beats = pr.get(("M3_anomaly", model), np.nan) > best_base
            for (c, mm) in pr:
                d, _, grp = c.partition("_")
                if mm != model or d not in BASES:
                    continue
                base = BASES[d]
                if (base, model) not in pr:
                    continue
                sign = 1 if d == "hadd" else -1
                delta = sign * (pr[(c, mm)] - pr[(base, model)])
                dbr = sign * (br[(base, model)] - br[(c, mm)])
                ya, yb = yr.get(key + (c, model)), yr.get(key + (base, model))
                if ya is not None and yb is not None:
                    dy = (sign * (ya - yb)).dropna()
                    n_pos, n_yr = int((dy > 0).sum()), int(dy.size)
                else:
                    n_pos, n_yr = 0, 0
                noise = sd[base]
                rows.append(dict(zip(KEYS + ["subset"], key), model=model, direction=d, group=grp,
                                 delta_pr_auc=delta, delta_brier=dbr, noise_sd=noise,
                                 years_positive=f"{n_pos}/{n_yr}",
                                 informative=bool(np.isfinite(noise) and delta > 2 * noise and n_yr >= 5
                                                  and n_pos >= n_yr - 1 and full_beats)))
    return pd.DataFrame(rows)


def _ap(y, p):
    k = np.isfinite(p)
    return average_precision_score(y[k], p[k]) if k.sum() >= 20 and len(np.unique(y[k])) == 2 else np.nan


def year_bootstrap(g, cols, pairs, B=200, seed=0):
    """95% intervals of PR-AUC (cols) and of PR-AUC differences (pairs a - b), resampling whole test years
    with replacement: years are the independent units, squares within a year are not."""
    rng = np.random.default_rng(seed)
    yrs = g.test_year.unique()
    parts = {Y: g[g.test_year == Y] for Y in yrs}
    need = sorted(set(cols) | {c for p in pairs for c in p})
    reps = {c: [] for c in need}
    for _ in range(B):
        d = pd.concat([parts[Y] for Y in rng.choice(yrs, len(yrs), replace=True)])
        y = d.y.to_numpy()
        for c in need:
            reps[c].append(_ap(y, d[c].to_numpy(float)))
    y = g.y.to_numpy()
    nm = lambda c: c[2:].replace("|", ": ") if c.startswith("p|") else (c[2:] if c.startswith("p_") else c)  # noqa: E731
    out = []
    for c in cols:
        r = np.array(reps[c])
        out.append(dict(what=nm(c), value=_ap(y, g[c].to_numpy(float)),
                        lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
    for a, b in pairs:
        r = np.array(reps[a]) - np.array(reps[b])
        out.append(dict(what=f"{nm(a)} - {nm(b)}", value=_ap(y, g[a].to_numpy(float)) - _ap(y, g[b].to_numpy(float)),
                        lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
    return out


ROW_KEYS = ["cv", "h", "target", "test_year", "issue", "row", "col"]


def strict_frames(out_dir):
    """Per lead month: the raw-input predictions with the anomaly-input model columns added. Both passes use the
    same sample and folds, so the rows match one to one (checked; on a mismatch the anomaly columns are left out)."""
    for f in sorted(out_dir.glob("preds_strict_h?.parquet")):
        d = pd.read_parquet(f)
        fa = f.with_name(f.stem + "_anomaly.parquet")
        if fa.exists():
            a = pd.read_parquet(fa)
            if len(a) == len(d) and (a[ROW_KEYS].astype(str).to_numpy() == d[ROW_KEYS].astype(str).to_numpy()).all():
                d = pd.concat([d, a[[c for c in a.columns if c.startswith("p|") and c not in d]]], axis=1)
            else:
                print(f"  {fa.name}: rows differ from {f.name}, anomaly models left out of the intervals")
        yield d


def intervals(out_dir):
    """Bootstrap intervals for the main comparisons (strict validation), cached in intervals.csv."""
    files = sorted(out_dir.glob("preds_strict_*.parquet"))
    cache = out_dir / "intervals.csv"
    if cache.exists() and files and cache.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return pd.read_csv(cache)
    rows = []
    for d in strict_frames(out_dir):
        pc = [c for c in d.columns if c.startswith("p|")]
        main = ["p_clim", "p_pers"] + [c for c in ("p|H|gbm", "p|M1|gbm", "p|M2|gbm", "p|M3|gbm", "p|M1_anomaly|gbm",
                                                  "p|M2_anomaly|gbm", "p|M3_anomaly|gbm") if c in pc]
        pairs = [(m, "p_clim") for m in ("p|M3|gbm", "p|M3_anomaly|gbm") if m in pc]
        pairs += [(m, "p|H|gbm") for m in ("p|M3|gbm", "p|M3_anomaly|gbm") if m in pc and "p|H|gbm" in pc]   # beyond flood history
        if "p|M1_anomaly|gbm" in pc:
            pairs += [("p|M1_anomaly|gbm", c) for c in pc if c.startswith("p|fdrop_")]
        for (h, target), g in d.groupby(["h", "target"]):
            for sub, m in subsets(g).items():
                if sub not in ("all", "cropland_harvest") or m.sum() < 50:
                    continue
                for r in year_bootstrap(g[m], main, pairs):
                    rows.append(dict(h=h, target=target, subset=sub, **r))
    iv = pd.DataFrame(rows).drop_duplicates(["h", "target", "subset", "what"])
    iv.to_csv(cache, index=False)
    return iv


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(out_dir=OUT):
    out_dir = Path(out_dir)
    sc = score_table(out_dir)
    if sc.empty:
        print("no predictions found")
        return
    abl = ablation_table(out_dir)
    abl.to_csv(out_dir / "ablation.csv", index=False)
    parts = []
    if (out_dir / "blocks.json").exists():
        b = json.loads((out_dir / "blocks.json").read_text())
        parts.append("## Spatial blocks\n\n" + md(pd.DataFrame([{k: b[k] for k in
                     ("corr_limit", "range_km", "capped", "block_km", "buffer_km", "groups")}])))

    ladder = [("climatology", "baseline"), ("persistence", "baseline"), ("H", "gbm"), ("M1", "logreg"), ("M1", "gbm"),
              ("M1_anomaly", "logreg"), ("M1_anomaly", "gbm"), ("M2", "gbm"), ("M2_anomaly", "gbm"),
              ("M3", "logreg"), ("M3", "gbm"), ("M3_anomaly", "gbm")]
    for metric in ("pr_auc", "brier"):
        t = sc.assign(cm=sc.config + ": " + sc.model)
        t = t[t.cm.isin([f"{a}: {b}" for a, b in ladder])]
        pv = t.pivot_table(index=["cv", "target", "h", "subset", "n", "base_rate"], columns="cm", values=metric).reset_index()
        order = [f"{a}: {b}" for a, b in ladder if f"{a}: {b}" in pv.columns]
        parts.append(f"## Model ladder, {metric}\n\n" + md(pv[["cv", "target", "h", "subset", "n", "base_rate"] + order]))

    if "accuracy" in sc:
        full = sc[sc.config.isin(FULL_METRIC_CONFIGS) & sc.subset.isin(["all", "cropland_harvest"])
                  & sc.accuracy.notna()].copy()
        cb = full[full.config == "climatology"][KEYS + ["subset", "brier"]].rename(columns={"brier": "brier_clim"})
        full = full.merge(cb, on=KEYS + ["subset"], how="left")
        full["brier_skill_vs_clim"] = 1 - full.brier / full.brier_clim
        cols = KEYS + ["subset", "config", "model", "n", "base_rate", "accuracy", "always_no_accuracy", "precision",
                       "recall", "f1", "balanced_accuracy", "false_alarm_rate", "roc_auc", "gini", "pr_auc", "brier",
                       "brier_skill_vs_clim", "log_loss"]
        parts.append("## All metrics (threshold metrics use a cutoff chosen on the other test years)\n\n"
                     + md(full[cols].sort_values(KEYS + ["subset", "config"])))
    sw = cutoff_sweep(out_dir)
    if not sw.empty:
        parts.append("## Cutoff sweep (strict): alarm budgets and fixed probability cutoffs\n\n"
                     + md(sw[["h", "target", "subset", "model", "rule", "precision", "recall", "f1",
                              "false_alarm_rate", "accuracy"]]))

    if not abl.empty:
        for sub in ("all", "cropland", "cropland_harvest"):
            a = abl[(abl.cv == "strict") & (abl.subset == sub)]
            if a.empty:
                continue
            pv = a.pivot_table(index=["target", "h", "direction", "group"], columns="model",
                               values=["delta_pr_auc", "noise_sd", "years_positive", "informative"], aggfunc="first")
            pv.columns = [f"{m}_{v}" for v, m in pv.columns]
            pv = pv.reset_index()
            keep = ["target", "h", "direction", "group"] + [c for m in ("gbm", "logreg") for c in
                    (f"{m}_delta_pr_auc", f"{m}_noise_sd", f"{m}_years_positive", f"{m}_informative") if c in pv.columns]
            parts.append(f"## Ablation (strict), subset {sub}\n\n" + md(pv[keep]))

    iv = intervals(out_dir)
    if not iv.empty:
        fam = iv[iv.what.str.contains("fdrop_")].copy()
        if not fam.empty:
            fam["family"] = fam.what.str.extract(r"fdrop_(\w+)")[0]
            fam = fam.rename(columns={"value": "pr_auc_loss_without_family"})
            parts.append("## Families of inputs removed from M1_anomaly (strict): PR-AUC loss, 95% interval over test years\n\n"
                         + md(fam[["h", "target", "subset", "family", "pr_auc_loss_without_family", "lo", "hi"]]))
        rest = iv[~iv.what.str.contains("fdrop_")]
        parts.append("## PR-AUC with 95% interval (bootstrap over test years, strict)\n\n"
                     + md(rest[["h", "target", "subset", "what", "value", "lo", "hi"]]))

    sens = sc[(sc.cv == "strict") & (sc.h == 1) & (sc.model == "gbm") & (sc.subset.isin(["all", "cropland"]))
              & sc.config.isin(["M1_anomaly", "M1_anomaly_additive", "M1_anomaly_small_trees", "M1_anomaly_large_trees"])]
    if not sens.empty:
        pv = sens.pivot_table(index=["target", "subset"], columns="config", values="pr_auc").reset_index()
        parts.append("## Interactions and tree size (GBM, strict, h=1), PR-AUC\n\n" + md(pv))

    rows = []
    for f in sorted(out_dir.glob("preds_*.parquet")):
        d = pd.read_parquet(f)
        for key, g in d.groupby(KEYS):
            for sub, m in subsets(g).items():
                if sub not in ("all", "cropland_harvest"):
                    continue
                gg = g[m]
                y = gg.y.to_numpy()
                for col in ["p_clim", "p_pers", "p|M1|gbm", "p|M1_anomaly|gbm", "p|M3|gbm", "p|M3_anomaly|gbm"]:
                    if col in gg:
                        rows.append(dict(zip(KEYS, key), subset=sub, model=col.replace("p|", "").replace("|", ": ").replace("p_", ""),
                                         recall_at_fpr05=recall_at(y, gg[col].to_numpy(float), 0.05),
                                         recall_at_fpr10=recall_at(y, gg[col].to_numpy(float), 0.10)))
        if f.name.startswith("preds_strict_h1.") or f.name == "preds_strict_h1.parquet":
            yr = []
            for (target, Y), g in d.groupby(["target", "test_year"]):
                for sub, m in subsets(g).items():
                    if sub not in ("all", "cropland"):
                        continue
                    gg = g[m]
                    r = {"target": target, "test_year": Y, "subset": sub, "n": len(gg)}
                    for col in ["p_clim", "p_pers", "p|M1|gbm", "p|M3|gbm"]:
                        mm = metrics(gg.y.to_numpy(), gg[col].to_numpy(float)) if col in gg else None
                        r[col.replace("p|", "").replace("|", ": ").replace("p_", "")] = mm["pr_auc"] if mm else np.nan
                    yr.append(r)
            parts.append("## Per test year (strict, h=1), PR-AUC\n\n" + md(pd.DataFrame(yr)))
    if rows:
        parts.append("## Operating points (strict): share of floods flagged at 5% / 10% false-alarm rate\n\n"
                     + md(pd.DataFrame(rows).query("cv == 'strict'").drop(columns="cv")
                          .drop_duplicates(["h", "target", "subset", "model"])))

    for name, title in (("susceptibility.csv", "Susceptibility (flood frequency per square, spatial-block CV)"),
                        ("susceptibility_drop_one.csv", "Susceptibility, static model without one covariate")):
        if (out_dir / name).exists():
            t = pd.read_csv(out_dir / name).head(20)
            for col in ("term", "dropped"):
                if col in t:
                    t.insert(1, "meaning", t[col].map(pretty))
            parts.append(f"## {title}\n\n" + md(t))
    (out_dir / "results.md").write_text("\n\n".join(parts) + "\n", encoding="utf8")
    print(f"wrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    main(ap.parse_args().out)
