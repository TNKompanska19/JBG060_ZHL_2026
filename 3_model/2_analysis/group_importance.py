"""Variable-group importance of the environment-only model (M1-anomaly GBM), 1 month ahead (poster figure).

Bar length: mean absolute TreeSHAP per group, in log-odds of flooding (results/shap_gbm/shap_families.csv).
Bar colour: navy when removing the group lowers PR-AUC with the 95% interval across test years above 0
(rows "M1_anomaly: gbm - fdrop_<group>: gbm" in results/run_all/intervals.csv), light otherwise.
Run shap_gbm.py and run_all.py first.

    python 3_model/2_analysis/group_importance.py      (from the repo root)

Writes results/group_importance/group_importance.png, drawn at print size so text prints at >= 20 pt.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHAP = ROOT / "results" / "shap_gbm" / "shap_families.csv"
INTERVALS = ROOT / "results" / "run_all" / "intervals.csv"
OUT = ROOT / "results" / "group_importance"
NAVY, LIGHT, TEXT = "#253c67", "#b8c4d9", "#222222"
plt.rcParams.update({"font.size": 24, "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": TEXT, "axes.edgecolor": TEXT, "axes.spines.top": False, "axes.spines.right": False})

H = 1
NAME = {"weather": "Weather", "land": "Land", "rivers_storage": "Rivers & storage", "climate_indices": "Climate indices"}
TITLE = {"occurrence": "Flood occurrence\n(dry -> flooded)", "persistence": "Flood persistence\n(flooded -> still flooded)"}


def main():
    shap = pd.read_csv(SHAP)
    iv = pd.read_csv(INTERVALS)
    iv = iv[(iv.h == H) & (iv.subset == "all") & iv.what.str.startswith("M1_anomaly: gbm - fdrop_")]
    iv = iv.assign(family=iv.what.str.extract(r"fdrop_(\w+): gbm")[0]).set_index(["target", "family"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharex=True)
    for ax, target in zip(axes, TITLE):
        s = shap[shap.target == target].sort_values("mean_abs_shap")
        passed = [iv.loc[(target, f), "lo"] > 0 for f in s.family]
        ax.barh([NAME[f] for f in s.family], s.mean_abs_shap, color=[NAVY if p else LIGHT for p in passed])
        ax.set_title(TITLE[target], fontweight="bold", color=NAVY)
        ax.set_xlabel("Mean |SHAP| (log-odds of flooding)")
        print(target, {NAME[f]: (round(v, 3), bool(p)) for f, v, p in zip(s.family, s.mean_abs_shap, passed)})
    fig.legend(handles=[Patch(color=NAVY, label="Removing it lowers PR-AUC (95% interval above 0)"),
                        Patch(color=LIGHT, label="No clear loss when removed")],
               loc="lower center", ncol=2, frameon=False, fontsize=20)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "group_importance.png", dpi=200)
    print("wrote", OUT / "group_importance.png")


if __name__ == "__main__":
    main()
