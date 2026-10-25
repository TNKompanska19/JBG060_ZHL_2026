"""Figure: flooding in all of South Sudan as seen by three flood products (report, Section 2).

    python 2_eda/4_target_scope/flood_products.py      (run from the repo root)

Every product is taken over the whole country (the county polygons of ssd_admin2.geojson), every month:
- MODIS/VIIRS: flood pixels flagged per day (3-day composite, recurring + unusual). Its two tiles stop at 10N,
  so the part of South Sudan north of 10N is missing for MODIS only.
- GFM: % of the observed land flagged flooded, from the STAC metadata of every Sentinel-1 pass over South Sudan
  (raw_data/gfm/stac, from gfm_inventory.py), one value per pass. These counts cover each pass's whole strip,
  so they include land outside South Sudan.
- CYGNSS: % of normally dry land marked water, permanent water removed with the rule of the model target
  (build_target.py: JRC permanent water, or CYGNSS water in >= 95% of months).
The units differ, so each line is divided by its own peak: 1 = that product's highest value in the panel.
(a) mean seasonal cycle 2019-2025; (b) October of every year, record flood years 2019-2022 shaded.
Output: results/flood_products/flood_products.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = ["1_data_processing/2_loaders", "1_data_processing/3_build_dataset"]
from loading import load_flood_masks  # noqa: E402
import build_target as bt  # noqa: E402   same country box and permanent-water rule as the model target

OUT = Path("results/flood_products")
ADMIN = Path("raw_data/Administrative boundaries/ssd_admin2.geojson")
STAC = Path("raw_data/gfm/stac")
YEARS = range(2016, 2026)
COMMON = range(2019, 2026)                  # years all three products cover (CYGNSS starts Aug 2018)
ACQ = r"^(.*?)_[EW]\d+[NS]\d+T\d+$"         # GFM item id minus its Equi7 tile suffix: one row per pass
LABEL = {"MODIS/VIIRS": "MODIS/VIIRS: flood pixels per day",
         "GFM": "GFM: % observed land flooded",
         "CYGNSS": "CYGNSS: % dry land marked water"}


def country(transform, shape):
    """True for grid cells inside South Sudan (union of the county polygons)."""
    gj = json.loads(ADMIN.read_text(encoding="utf8"))
    return rasterize([(f["geometry"], 1) for f in gj["features"]], out_shape=shape, transform=transform,
                     fill=0) == 1


def month_year(s):
    """Series indexed by (year, month) -> month x year table."""
    return s.unstack(0).reindex(index=range(1, 13), columns=YEARS)


def modis():
    """Flood pixels flagged per day inside South Sudan, per year x month."""
    res, (left, bottom, right, top) = 0.01, bt.BBOX
    inside = country(from_origin(left, top, res, res), (int(np.ceil((top - bottom) / res)), int(np.ceil((right - left) / res))))
    env = dict(lon_min=left, lat_min=bottom, lon_max=right, lat_max=top)
    out = {}
    for y in YEARS:                          # one year at a time to keep memory down
        d = load_flood_masks(np.array([y]), env)
        i = ((top - d.lat.to_numpy()) / res).astype(int).clip(0, inside.shape[0] - 1)
        j = ((d.lon.to_numpy() - left) / res).astype(int).clip(0, inside.shape[1] - 1)
        n = d.date[inside[i, j]].dt.month.value_counts()
        for m in range(1, 13):
            out[(y, m)] = n.get(m, 0) / pd.Period(f"{y}-{m:02d}").days_in_month
        print(f"  MODIS {y}: {inside[i, j].sum():,} of {len(d):,} flagged pixels inside South Sudan", flush=True)
    return month_year(pd.Series(out))


def gfm():
    """% of observed land flagged flooded, summed over all passes of the month (one row per pass)."""
    frames = [pd.read_parquet(f) for f in sorted(STAC.glob("*.parquet"))]
    g = pd.concat([f for f in frames if len(f)], ignore_index=True)
    g = g.assign(acq=g.item_id.str.replace(ACQ, r"\1", regex=True)).drop_duplicates("acq")
    t = pd.to_datetime(g.datetime, format="mixed", utc=True)
    s = g.assign(year=t.dt.year, month=t.dt.month).groupby(["year", "month"])[
        ["flooded_pixels", "floodable_pixels"]].sum()
    print(f"  GFM: {len(g):,} passes")
    return month_year(100 * s.flooded_pixels / s.floodable_pixels)


def cygnss():
    """% of normally dry land inside South Sudan marked water, per year x month."""
    z = np.load(bt.CYG)
    w, lat, lon, mon = z["w"], z["lat"], z["lon"], z["month"]
    valid, wet = w >= 0, w == 1
    perm = ((wet.sum(0) >= bt.CYG_PERMANENT * np.maximum(valid.sum(0), 1))
            | (bt.jrc_permanent(lat, lon) >= bt.JRC_CELL_SHARE))
    res = round(float(lon[1] - lon[0]), 6)
    keep = country(from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res), (lat.size, lon.size)) & ~perm
    pct = 100 * (wet & keep).sum((1, 2)) / np.maximum((valid & keep).sum((1, 2)), 1)
    p = pd.PeriodIndex(mon, freq="M")
    return month_year(pd.Series(pct, index=pd.MultiIndex.from_arrays([p.year, p.month])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"MODIS/VIIRS": modis(), "GFM": gfm(), "CYGNSS": cygnss()}
    fig, ax = plt.subplots(2, 1, figsize=(4.2, 4.8))
    seas, octo = {}, {}
    for name, df in data.items():
        seas[name] = df.reindex(columns=COMMON).mean(axis=1)
        octo[name] = df.loc[10]
        ax[0].plot(seas[name].index, seas[name] / seas[name].max(), label=LABEL[name])
        ax[1].plot(octo[name].index, octo[name] / octo[name].max(), "o-", ms=3)
    ax[0].axvspan(5.5, 9.5, color="0.9", zorder=0)
    ax[0].set(xticks=range(1, 13), xlabel="Month", ylabel="Share of the product's own peak",
              title="(a) Mean seasonal cycle 2019-2025, rainy season shaded")
    ax[1].axvspan(2018.5, 2022.5, color="0.9", zorder=0)
    ax[1].set(xlabel="Year", ylabel="Share of the product's own peak",
              title="(b) October of each year, record floods shaded")
    for a in ax:
        a.set_ylim(0, 1.05)
        a.title.set_size(8)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
        a.tick_params(labelsize=7)
    ax[1].legend(*ax[0].get_legend_handles_labels(), loc="lower right", fontsize=4, framealpha=0.9,
                 handlelength=1.2, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "flood_products.png", dpi=300)
    pd.set_option("display.width", 150)
    print("\nmean seasonal cycle 2019-2025, own units:")
    print(pd.DataFrame(seas).round(2).to_string())
    print("\nOctober per year, own units:")
    print(pd.DataFrame(octo).round(2).to_string())
    print("wrote", OUT / "flood_products.png")


if __name__ == "__main__":
    main()
