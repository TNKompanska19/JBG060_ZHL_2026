"""PR-AUC gain of the full model (M3-anomaly GBM) per lead time, strict validation (poster figure).

Bars: gain over the climatology baseline, over the flood history + location model (H), and over climatology on
cropland in the harvest months. Whiskers: 95% interval across test years. Source: results/run_all/intervals.csv.

    python 3_model/2_analysis/skill_gain.py      (from the repo root)

Writes results/skill_gain/skill_gain.png, drawn at print size so text prints at >= 20 pt.
Colours are taken from the poster: navy title bars, orange accent.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "run_all" / "intervals.csv"
OUT = ROOT / "results" / "skill_gain"
NAVY, ORANGE, LIGHT, TEXT = "#253c67", "#faa111", "#9fb3d6", "#222222"
plt.rcParams.update({"font.size": 24, "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": TEXT, "axes.edgecolor": TEXT, "axes.spines.top": False, "axes.spines.right": False})

BARS = [("all", "M3_anomaly: gbm - clim", "vs climatology", NAVY),                      # (subset, row, label, colour)
        ("all", "M3_anomaly: gbm - H: gbm", "vs flood history + location", ORANGE),
        ("cropland_harvest", "M3_anomaly: gbm - clim", "vs climatology, cropland harvest months", LIGHT)]
TITLE = {"occurrence": "Flood occurrence\n(dry -> flooded)", "persistence": "Flood persistence\n(flooded -> still flooded)"}
W = 0.27   # bar width


def main():
    d = pd.read_csv(SRC).set_index(["target", "subset", "what", "h"]).sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    for ax, target in zip(axes, TITLE):
        for k, (subset, row, label, col) in enumerate(BARS):
            r = d.loc[(target, subset, row)]
            x = r.index.to_numpy() + (k - 1) * W
            ax.bar(x, r.value, W, color=col, label=label)
            ax.errorbar(x, r.value, yerr=[r.value - r.lo, r.hi - r.value], fmt="none", ecolor=TEXT, capsize=6, lw=2)
        ax.axhline(0, color=TEXT, lw=1.5)
        ax.set(xticks=[1, 2, 3], xlabel="Months ahead")
        ax.set_title(TITLE[target], fontweight="bold", color=NAVY)
    axes[0].set_ylabel("PR-AUC gain of full model\n(above 0: full model better)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=20)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "skill_gain.png", dpi=200)
    rows = [(t, s, l, h) + tuple(np.round(d.loc[(t, s, w, h), ["value", "lo", "hi"]], 3))
            for t in TITLE for s, w, l, _ in BARS for h in (1, 2, 3)]
    print(pd.DataFrame(rows, columns=["target", "subset", "comparison", "h", "gain", "lo", "hi"]).to_string(index=False))
    print("wrote", OUT / "skill_gain.png")


if __name__ == "__main__":
    main()
