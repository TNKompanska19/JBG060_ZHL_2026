"""Flow diagram of the method (poster figure). No data needed.

    python 3_model/2_analysis/method_flow.py      (from the repo root)

Writes results/method_flow/method_flow.png, drawn at print size (axes units = inches) so text prints at >= 20 pt
when placed at >= 83% of its width (22 in = 56 cm). Colours are taken from the poster: navy title bars, orange accent.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "method_flow"
W, H = 22, 11                       # figure size in inches = axes units
FS, LINE = 24, 24 * 1.3 / 72        # font size (pt) and line height (inches)
GAP = 0.4                           # space between title and body, in lines
plt.rcParams.update({"font.size": FS})
NAVY, ORANGE, TEXT = "#253c67", "#faa111", "#222222"
STYLE = {"data": ("#e9edf4", NAVY), "model": ("#fdf0dc", ORANGE), "eval": ("#ffffff", NAVY)}   # (fill, outline)

L, M, R = 3.7, 11.1, 18.3           # column centres
BOXES = {   # name: (x centre, y centre, width, height, title, body, style)
    "target": (L, 9.2, 7.0, 2.4, "Target", "CYGNSS radar, monthly flooded\nshare, ~5 km^2 cells, 2018-2025", "data"),
    "land":   (L, 5.5, 7.0, 2.4, "Land aggregation", "flood-prone cells (>=2 flood years)\n50% cropland", "data"),
    "groups": (L, 1.8, 7.0, 2.4, "Independent variable groups", "weather, rivers & storage\nland, climate indices", "data"),
    "models": (M, 5.5, 5.8, 4.2, "Gradient-boosted decision\ntrees (GBM)",
               "M1: environment\nM2: M1 + flood history\nM3: M2 + location\nH: flood history + location", "model"),
    "base":   (R, 9.2, 6.6, 2.0, "Baselines", "latest flood map, climatology", "eval"),
    "valid":  (R, 5.5, 6.6, 2.9, "Validation", "leave-one-year-out\n+ 80 km spatial blocks\n1, 2, 3 months ahead", "eval"),
    "tests":  (R, 1.8, 6.6, 2.4, "Group tests", "remove / add group\nTreeSHAP for direction", "eval"),
}
ARROWS = [("target", "land"), ("land", "models"), ("groups", "models"),
          ("models", "base"), ("models", "valid"), ("models", "tests")]


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(0, H))
    ax.set_axis_off()
    for x, y, w, h, title, body, style in BOXES.values():
        fill, edge = STYLE[style]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=fill, ec=edge, lw=3))
        nt, nb = title.count("\n") + 1, body.count("\n") + 1
        top = y + (nt + GAP + nb) * LINE / 2                 # title, gap and body centred together in the box
        ax.text(x, top, title, ha="center", va="top", fontweight="bold", color=NAVY, linespacing=1.3)
        ax.text(x, top - (nt + GAP) * LINE, body, ha="center", va="top", color=TEXT, linespacing=1.3)
    for a, b in ARROWS:
        xa, ya, wa, ha, *_ = BOXES[a]
        xb, yb, wb, hb, *_ = BOXES[b]
        start, end = (((xa, ya - ha / 2), (xb, yb + hb / 2)) if xa == xb       # same column: bottom -> top
                      else ((xa + wa / 2, ya), (xb - wb / 2, yb)))             # next column: right edge -> left edge
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=30, lw=3, color=NAVY,
                                     shrinkA=6, shrinkB=6))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "method_flow.png", dpi=200)
    print("wrote", OUT / "method_flow.png")


if __name__ == "__main__":
    main()
