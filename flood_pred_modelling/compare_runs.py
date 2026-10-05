"""Two decision-relevant checks on the saved predictions of every run.

    python flood_pred_modelling/compare_runs.py --res 0.02

Reads every output/models_<res>*/<run>_preds.parquet written by train_models.py,
including older runs, so runs with and without an input can be compared.
Scope: seasonal squares only (water in --rare to --permanent of all months), the
squares where flooding is neither trivial nor absent.

1. YEAR BY YEAR: in how many test years does the model's Brier error beat
   'usual for this month' (the square's normal from training years)?
2. UNUSUAL FLOODS: target months in which the square is usually dry
   (training-year normal < 0.5). That is the early-warning question.
   - PR-AUC: can the model rank which usually-dry squares WILL flood? Chance =
     the flood rate among these cases.
   - flagged: share of those unusual floods that got probability >= 0.5.
     'Usual for this month' flags none by definition.
   - false alarms: share of the usually-dry cases that stayed dry but were flagged.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

OUT = Path(__file__).resolve().parent / "output"


def pr_auc(y, p):
    return average_precision_score(y, p) if y.nunique() == 2 else np.nan


def year_by_year(q):
    rows = []
    for (h, yr), g in q.groupby(["horizon", "test_year"]):
        rows.append({"horizon": h, "year": yr,
                     "skill": 1 - brier_score_loss(g.y, g.p_model)
                     / brier_score_loss(g.y, g.p_climatology.clip(0, 1))})
    t = pd.DataFrame(rows)
    return t.groupby("horizon").skill.agg(
        years_better=lambda s: f"{(s > 0).sum()}/{len(s)}", worst="min", best="max").round(3)


def unusual(q):
    rows = []
    for h, g in q.groupby("horizon"):
        u = g[g.p_climatology < 0.5]
        fl, dry = u[u.y == 1], u[u.y == 0]
        rows.append({"horizon": h, "cases": len(u), "flood_rate": u.y.mean(),
                     "PR-AUC model": pr_auc(u.y, u.p_model),
                     "PR-AUC usual": pr_auc(u.y, u.p_climatology),
                     "PR-AUC latest map": pr_auc(u.y, u.p_persistence),
                     "flagged": (fl.p_model >= 0.5).mean(),
                     "flagged latest map": (fl.p_persistence >= 0.5).mean(),
                     "false alarms": (dry.p_model >= 0.5).mean(),
                     "false alarms latest map": (dry.p_persistence >= 0.5).mean()})
    return pd.DataFrame(rows).set_index("horizon").round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--permanent", type=float, default=0.9)
    ap.add_argument("--rare", type=float, default=0.05)
    a = ap.parse_args()
    files = sorted(OUT.glob(f"models_{a.res:.2f}*/*_preds.parquet"))
    if not files:
        raise SystemExit(f"no predictions under {OUT}/models_{a.res:.2f}*")
    pd.set_option("display.width", 200)
    for f in files:
        p = pd.read_parquet(f)
        q = p[(p.freq_all >= a.rare) & (p.freq_all < a.permanent)]
        print(f"\n=== {f.parent.name}/{f.name.removesuffix('_preds.parquet')}"
              f"  ({len(q):,} seasonal-square predictions)")
        print("1. Year by year, Brier skill vs 'usual for this month' (> 0 = model better):")
        print(year_by_year(q).to_string())
        print("2. Unusual floods (square usually dry that month):")
        print(unusual(q).to_string())


if __name__ == "__main__":
    main()
