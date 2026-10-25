"""Cropland next to model skill per county (poster figure, Findings).

(a) cropland cover per grid cell (crop_pct in processed_data/panel_0.02/inputs.npz), where early harvesting and
temporary flood defences on cropland apply; (b, c) PR-AUC gain of the M3-anomaly GBM over the climatology baseline per
county, 1 month ahead, strict validation, for flood occurrence and flood persistence (read from
results/county_performance/county_performance.md; run county_performance.py first). Grey = county not scored.
The N_CROP counties with the highest cropland share are outlined in orange on all three maps and named on the first.

    python 3_model/2_analysis/cropland_skill_map.py      (from the repo root)

Writes results/cropland_skill_map/cropland_skill_map.png (drawn at print size, text >= 20 pt) and prints the
cropland share and both gains of the outlined counties.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from county_map import read_table  # noqa: E402   same table reader as the report map

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "processed_data" / "panel_0.02" / "inputs.npz"
TABLE = ROOT / "results" / "county_performance" / "county_performance.md"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = ROOT / "results" / "cropland_skill_map"

GAIN = "pr_auc_gain_vs_clim"
VMAX = 0.20          # gain colour range -VMAX..+VMAX, as in county_map.py
N_CROP = 10           # counties outlined and named: highest cropland share
NAVY, ORANGE, TEXT, GREY, EDGE = "#253c67", "#faa111", "#222222", "#b8b8b8", "0.45"
plt.rcParams.update({"font.size": 24, "text.color": TEXT, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": TEXT})
TITLES = ["Cropland\n(where actions apply)",
          "Flood occurrence\n(timing of early harvesting)",
          "Flood persistence\n(where defences are needed)"]


def main():
    z = np.load(PANEL)
    lat, lon, crop = z["lat"], z["lon"], z["crop_pct"].astype(float)
    if crop.shape != (lat.size, lon.size):
        raise SystemExit(f"crop_pct has shape {crop.shape}, expected {(lat.size, lon.size)}")
    if lat[0] < lat[-1]:                                   # rows must run north -> south for the raster transform
        lat, crop = lat[::-1], crop[::-1]
    if np.nanmax(crop) <= 1:                               # stored as a fraction: show as %
        crop = crop * 100
    res = round(float(lon[1] - lon[0]), 6)
    adm = gpd.read_file(ADMIN).rename(columns={"adm1_name": "state", "adm2_name": "county"}).reset_index(drop=True)
    cid = rasterize([(g, i + 1) for i, g in enumerate(adm.geometry)], out_shape=crop.shape,
                    transform=from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res), fill=0, dtype="int32")
    crop[cid == 0] = np.nan                                # only South Sudan
    adm["crop_share"] = pd.Series(crop.ravel()).groupby(cid.ravel()).mean().reindex(range(1, len(adm) + 1)).to_numpy()

    text = TABLE.read_text(encoding="utf8")
    for tgt in ("occurrence", "persistence"):
        g = read_table(text, tgt).set_index(["state", "county"])[GAIN].astype(float).rename(tgt)
        adm = adm.join(g, on=["state", "county"])
        if adm[tgt].notna().sum() != g.notna().sum():
            raise SystemExit(f"{tgt}: scored counties missing from the boundary file (name mismatch)")

    fig, axes = plt.subplots(1, 3, figsize=(27, 11))
    ext = (lon[0] - res / 2, lon[-1] + res / 2, lat[-1] - res / 2, lat[0] + res / 2)
    vmax_c = float(np.nanpercentile(crop[crop > 0], 99))
    im = axes[0].imshow(np.where(crop > 0, crop, np.nan), extent=ext, cmap="YlGn", vmin=0, vmax=vmax_c,
                        interpolation="nearest")
    adm.boundary.plot(ax=axes[0], color=EDGE, linewidth=0.4)
    cb = fig.colorbar(im, ax=axes[0], orientation="horizontal", extend="max", shrink=0.85, pad=0.03)
    cb.set_label("Cropland cover (% of grid cell)")
    norm = Normalize(-VMAX, VMAX)
    for ax, tgt in zip(axes[1:], ("occurrence", "persistence")):
        adm[adm[tgt].isna()].plot(ax=ax, color=GREY, edgecolor=EDGE, linewidth=0.4)
        adm[adm[tgt].notna()].plot(ax=ax, column=tgt, cmap="RdBu", norm=norm, edgecolor=EDGE, linewidth=0.4)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu"), ax=ax, orientation="horizontal", extend="both",
                          shrink=0.85, pad=0.03)
        cb.set_label("PR-AUC gain over climatology\n(blue: model better, red: worse)")

    states, top = adm.dissolve(by="state"), adm.nlargest(N_CROP, "crop_share")
    x0, y0, x1, y1 = adm.total_bounds
    for name, pt in zip(top.county, top.geometry.representative_point()):    # names on the cropland map only
        axes[0].annotate(name, (pt.x, pt.y), ha="center", va="center", fontsize=20, fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    for ax, title in zip(axes, TITLES):
        states.boundary.plot(ax=ax, color="black", linewidth=1.2)
        top.boundary.plot(ax=ax, color=ORANGE, linewidth=4)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontweight="bold", color=NAVY)
    fig.legend(handles=[Line2D([], [], color=ORANGE, lw=4, label=f"{N_CROP} counties with most cropland"),
                        Patch(facecolor=GREY, edgecolor=EDGE, label="Not scored (too few flood or non-flood rows)")],
               loc="lower center", ncol=2, frameon=False, fontsize=22)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cropland_skill_map.png", dpi=200)
    plt.close(fig)

    print(f"cropland cover: max {np.nanmax(crop):.1f}%, colour scale capped at the 99th percentile of cropland cells "
          f"({vmax_c:.1f}%)")
    print(top[["state", "county", "crop_share", "occurrence", "persistence"]].round(3).to_string(index=False))
    print("wrote", OUT / "cropland_skill_map.png")


if __name__ == "__main__":
    main()
