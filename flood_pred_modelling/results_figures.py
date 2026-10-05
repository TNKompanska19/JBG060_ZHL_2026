"""Presentation figures from the saved runs of train_models.py.

    python flood_pred_modelling/results_figures.py --res 0.02 --best gbm_year_merit --nn nn_year_merit --month 2022-10

Writes to output/models_<res>/:
  fig_scores.png    seasonal squares: ranking score (PR-AUC) per run and lead time, vs baselines
  fig_unusual.png   unusual floods (square usually dry that month), 1 month ahead:
                    share caught vs false-alarm rate, per run
  fig_map.png       --month, forecast 1 month ahead by --best: what happened / usual / model
  fig_nn.png        --nn: average weight per input and per month of history (explanation)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def seasonal(p, rare=0.05, permanent=0.9):
    return p[(p.freq_all >= rare) & (p.freq_all < permanent)]


def fig_scores(out):
    rows = []
    for f in sorted(out.glob("*_scores.csv")):
        s = pd.read_csv(f)
        s = s[(s.permanent == 0.9) & (s.group == "seasonal")]
        run = f.name.removesuffix("_scores.csv")
        rows += [(run, r.horizon, r["PR-AUC"]) for _, r in s[s.who == "model"].iterrows()]
        base = s
    for who, label in (("climatology", "usual for this month"), ("persistence", "latest map")):
        rows += [(label, r.horizon, r["PR-AUC"]) for _, r in base[base.who == who].iterrows()]
    t = pd.DataFrame(rows, columns=["run", "horizon", "pr"]).pivot(index="horizon", columns="run", values="pr")
    ax = t.plot.bar(figsize=(11, 5), width=0.85)
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Ranking score, PR-AUC (higher = better)")
    ax.set_ylim(0.4, 0.8)
    ax.set_title("Seasonal floodplain squares, each year tested on models trained without it")
    ax.legend(fontsize=7, ncol=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "fig_scores.png", dpi=130)
    plt.close()


def fig_unusual(out):
    pts = []
    for f in sorted(out.glob("*_preds.parquet")):
        p = seasonal(pd.read_parquet(f, columns=["horizon", "y", "p_model", "p_persistence",
                                                 "p_climatology", "freq_all"]))
        u = p[(p.horizon == 1) & (p.p_climatology < 0.5)]
        fl, dry = u[u.y == 1], u[u.y == 0]
        pts.append((f.name.removesuffix("_preds.parquet"),
                    (dry.p_model >= 0.5).mean(), (fl.p_model >= 0.5).mean()))
    pts.append(("latest map", (dry.p_persistence >= 0.5).mean(), (fl.p_persistence >= 0.5).mean()))
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fa, caught in pts:
        ax.scatter(fa * 100, caught * 100, s=60)
        ax.annotate(name, (fa * 100, caught * 100), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("False alarms: usually-dry squares that stayed dry but were flagged (%)")
    ax.set_ylabel("Caught: unusual floods flagged in advance (%)")
    ax.set_title("Unusual floods, 1 month ahead (up-left is better; 'usual for this month' catches 0%)")
    plt.tight_layout()
    plt.savefig(out / "fig_unusual.png", dpi=130)
    plt.close()


def fig_map(out, res, best, month):
    z = np.load(ROOT / "processed_data" / f"panel_{res:.2f}" / "inputs.npz")
    lat, lon = z["lat"], z["lon"]
    p = pd.read_parquet(out / f"{best}_preds.parquet")
    p = p[p.horizon == 1]
    p = p[(pd.PeriodIndex(p.issue, freq="M") + 1).astype(str) == month]
    if p.empty:
        print(f"fig_map: no 1-month forecasts for {month} in {best}")
        return
    x, y = lon[p.col], lat[p.row]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, col, title in zip(axes, ("share", "p_climatology", "p_model"),
                              ("What happened (CYGNSS)", "Usual for this month", f"Model ({best})")):
        sc = ax.scatter(x, y, c=p[col], s=4, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc, ax=axes, label="Water share / flood probability")
    fig.suptitle(f"{month}, forecast issued one month earlier (sampled squares)")
    plt.savefig(out / "fig_map.png", dpi=130)
    plt.close()


def fig_nn(out, nn):
    f = out / f"{nn}_preds.parquet"
    if not f.exists():
        print(f"fig_nn: {f.name} not found")
        return
    p = seasonal(pd.read_parquet(f))
    p = p[p.horizon == 1]
    gates = p[[c for c in p.columns if c.startswith("gate_")]].mean().sort_values()
    att = p[sorted([c for c in p.columns if c.startswith("att_lag")], key=lambda c: int(c[7:]))].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
    a1.barh([g[5:] for g in gates.index], gates.to_numpy())
    a1.set_xlabel("Average weight the network gives this input")
    a1.set_title("Which inputs the NN uses (seasonal squares, 1 month ahead)")
    a2.bar(range(len(att)), att.to_numpy())
    a2.set_xticks(range(len(att)))
    a2.set_xlabel("Months before the forecast is issued (0 = issue month)")
    a2.set_ylabel("Average attention weight")
    a2.set_title("Which months of history the NN looks at")
    plt.tight_layout()
    plt.savefig(out / "fig_nn.png", dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--best", default="gbm_year_merit", help="run used for the map")
    ap.add_argument("--nn", default="nn_year_merit", help="NN run used for the explanation figure")
    ap.add_argument("--month", default="2022-10", help="target month for the map, YYYY-MM")
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "output" / f"models_{a.res:.2f}"
    fig_scores(out)
    fig_unusual(out)
    fig_map(out, a.res, a.best, a.month)
    fig_nn(out, a.nn)
    print(f"wrote fig_scores, fig_unusual, fig_map, fig_nn to {out}")


if __name__ == "__main__":
    main()
