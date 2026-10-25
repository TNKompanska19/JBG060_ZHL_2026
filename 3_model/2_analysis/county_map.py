"""County map of the M3-anomaly GBM's skill over the climatology baseline for flood occurrence, 1 month ahead (report Fig. 3).

Colour = PR-AUC gain over the climatology baseline per county, read from the occurrence table in
results/county_performance/county_performance.md (run county_performance.py first). Counties without a score are grey.

    python 3_model/2_analysis/county_map.py      (from the repo root)

Named: the N_AREA largest counties by area, the N_BEST counties with the highest gain and the N_WORST with the lowest.
Writes results/county_map/county_map.png and prints how many counties are scored and which are named.
"""
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results" / "county_performance" / "county_performance.md"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = ROOT / "results" / "county_map"

TARGET, GAIN = "occurrence", "pr_auc_gain_vs_clim"
VMAX = 0.20          # colour range is symmetric, -VMAX to +VMAX; larger gains share the darkest blue (arrow on the colourbar)
N_AREA, N_BEST, N_WORST = 15, 5, 3   # counties named: largest by area, highest gain, lowest gain
NEAR_ZERO = 0.01     # hatched: |gain| <= this. A display tolerance, not a statistical test (the table has no per-county interval)
GREY, EDGE = "#b8b8b8", "0.4"
FONT = 7             # county name size (points)
# Candidate label offsets (points): the county's own spot first, then rings of growing radius around it
OFFSETS = [(0, 0)] + [(r * np.cos(a), r * np.sin(a)) for r in (10, 18, 26, 36, 48) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]


def read_table(text, target):
    """The '## Skill per county, target <target>' markdown table as a DataFrame; empty cells become NaN."""
    block = text.split(f"## Skill per county, target {target}")[1].split("\n## ")[0]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in block.splitlines() if ln.startswith("|")]
    return pd.DataFrame(rows[2:], columns=rows[0]).replace("", np.nan)


def name_counties(ax, counties):
    """Print each county name at its representative point. A name that would overlap the legend or an earlier name moves
    to the nearest free spot around it, with a leader line and a dot on its county. Earlier rows keep their own spot.
    Returns the names for which no free spot was found (printed at their own spot)."""
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    taken, crowded = [ax.get_legend().get_window_extent(rend)], []
    for name, pt in zip(counties.county, counties.geometry.representative_point()):
        for dx, dy in OFFSETS:
            t = ax.annotate(name, (pt.x, pt.y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                            fontsize=FONT, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2", shrinkA=0, shrinkB=0) if dx or dy else None,
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
            t.update_positions(rend)          # the annotation only moves to its offset when drawn or updated
            box = Text.get_window_extent(t, rend).expanded(1.1, 1.3)
            if not any(box.overlaps(b) for b in taken):
                taken.append(box)
                if dx or dy:
                    ax.plot(pt.x, pt.y, ".", ms=2, color="0.2")   # marks the county a moved name belongs to
                break
            t.remove()
        else:
            crowded.append(name)
            ax.annotate(name, (pt.x, pt.y), ha="center", va="center", fontsize=FONT)
    return crowded


def main():
    text = TABLE.read_text(encoding="utf8")
    model = re.match(r"Model p\|(\w+)\|", text).group(1).replace("_", "-")
    gain = read_table(text, TARGET).set_index(["state", "county"])[GAIN].astype(float)   # NaN = county not scored
    adm = gpd.read_file(ADMIN).rename(columns={"adm1_name": "state", "adm2_name": "county"})
    adm = adm[["state", "county", "area_sqkm", "geometry"]].join(gain, on=["state", "county"])
    if adm[GAIN].notna().sum() != gain.notna().sum():
        raise SystemExit("scored counties in the table are missing from the boundary file (name mismatch)")

    norm = Normalize(-VMAX, VMAX)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    adm[adm[GAIN].isna()].plot(ax=ax, color=GREY, edgecolor=EDGE, linewidth=0.3)
    adm[adm[GAIN].notna()].plot(ax=ax, column=GAIN, cmap="RdBu", norm=norm, edgecolor=EDGE, linewidth=0.3)
    near = adm[adm[GAIN].abs() <= NEAR_ZERO].sort_values(GAIN)
    near.plot(ax=ax, facecolor="none", edgecolor="0.25", hatch="////", linewidth=0)
    adm.dissolve(by="state").boundary.plot(ax=ax, color="black", linewidth=0.9)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu"), ax=ax, extend="max", shrink=0.75, pad=0.02)
    cb.set_label("PR-AUC gain over climatology baseline\n(above 0: model better)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.legend(handles=[Patch(facecolor=GREY, edgecolor=EDGE, label="Not scored (too few flood or non-flood rows)"),
                       Patch(facecolor="white", edgecolor="0.25", hatch="////",
                             label=f"Gain within ±{NEAR_ZERO} of 0"),
                       Line2D([], [], color="none", label=f"Named: {N_AREA} largest counties, {N_BEST} highest and {N_WORST} lowest gain")],
              loc="upper left", fontsize=7, frameon=False)
    best, worst, large = adm.nlargest(N_BEST, GAIN), adm.nsmallest(N_WORST, GAIN), adm.nlargest(N_AREA, "area_sqkm")
    named = pd.concat([best, worst, large])
    crowded = name_counties(ax, named[~named.index.duplicated()])   # best and lowest gain are placed first
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{model} GBM: skill gain over climatology baseline per county\n"
                 "Flood occurrence (dry → flooded), 1 month ahead", fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "county_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fmt = lambda d: ", ".join(f"{c} ({g:+.3f})" if g == g else f"{c} (not scored)" for c, g in zip(d.county, d[GAIN]))
    print(f"{gain.notna().sum()} of {len(adm)} counties scored")
    print(f"gain within ±{NEAR_ZERO} of 0 (hatched) in {len(near)}: {fmt(near)}")
    print(f"gain below -{NEAR_ZERO}: {fmt(adm[adm[GAIN] < -NEAR_ZERO].sort_values(GAIN)) or 'none'}")
    print(f"{N_BEST} highest gain: {fmt(best)}")
    print(f"{N_WORST} lowest gain: {fmt(worst)}")
    print(f"{N_AREA} largest by area: {fmt(large)}")
    print(f"names without a free spot (left on their county): {', '.join(crowded) or 'none'}")
    print(f"gain outside the colour range (|gain| > {VMAX}): {fmt(adm[adm[GAIN].abs() > VMAX]) or 'none'}")


if __name__ == "__main__":
    main()
