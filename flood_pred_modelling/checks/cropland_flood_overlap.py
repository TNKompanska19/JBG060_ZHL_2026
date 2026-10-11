"""Cropland under CYGNSS flood water, per year, South Sudan.

    python flood_pred_modelling/checks/cropland_flood_overlap.py

Input:  processed_data/cygnss/ssd_watermask.npz   (made by 2_build_dataset/build_target.py)
        raw_data/farmland/asap_mask_crop_v04.tif   (% of each ~500 m pixel that is cropland, 1-100)
        raw_data/Administrative boundaries/ssd_admin2.geojson
Output: flood_pred_modelling/checks/output/cropland_flood_overlap.csv / .md   one row per year + average
        flood_pred_modelling/checks/output/cropland_flood_<YEAR>.png         map + monthly curve

Definitions (our choices, state them in the report):
- Crop mask averaged onto the CYGNSS ~1 km squares: crop_km2 = crop % x square area.
- Permanent water is dropped with the same rule as the model target (build_target.py: JRC occurrence >= 95%
  or CYGNSS water in >= 95% of months), so "flooded" means water on normally dry land.
- flooded_any_km2 / _pct: cropland in squares marked water in at least one month of the year.
- mean_month_pct: average over the year's months of (flooded cropland / cropland with data).
- peak_pct, peak_month: the month with the largest share of cropland under water.
- data_pct: average share of cropland with a valid CYGNSS value (no -99) per month.
- Average row: full calendar years only (12 months).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_build_dataset"))
from build_target import CYG_PERMANENT, JRC_CELL_SHARE, jrc_permanent  # noqa: E402  (same permanent-water rule as the target)

YEAR = 2022          # year to map
CROP_MIN = 1.0       # map only: squares below this crop % are drawn as non-cropland

ROOT = Path(__file__).resolve().parents[2]
CYG = ROOT / "processed_data" / "cygnss" / "ssd_watermask.npz"
CROP = ROOT / "raw_data" / "farmland" / "asap_mask_crop_v04.tif"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = Path(__file__).resolve().parent / "output"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def grid_transform(lat, lon):
    res = round(float(lon[1] - lon[0]), 6)
    return from_origin(float(lon[0]) - res / 2, float(lat[0]) + res / 2, res, res)


def crop_on_grid(lat, lon):
    """Mean crop % of each CYGNSS square. Non-crop and no-data pixels count as 0 %."""
    with rasterio.open(CROP) as src:
        res = abs(float(lon[1] - lon[0]))
        win = from_bounds(lon[0] - res, lat[-1] - res, lon[-1] + res, lat[0] + res, src.transform)
        win = win.round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32)   # South Sudan only; the file is global
        bad = (a < 0) | (a > 100) | ~np.isfinite(a)
        if src.nodata is not None:
            bad |= a == src.nodata
        a[bad] = 0
        out = np.zeros((lat.size, lon.size), np.float32)
        reproject(a, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=-1,
                  dst_transform=grid_transform(lat, lon), dst_crs="EPSG:4326",
                  dst_nodata=-1, resampling=Resampling.average)
    return np.clip(out, 0, 100)


def counties(lat, lon):
    gj = json.loads(ADMIN.read_text(encoding="utf8"))
    cid = rasterize([(ft["geometry"], 1) for ft in gj["features"]],
                    out_shape=(lat.size, lon.size), transform=grid_transform(lat, lon),
                    fill=0, dtype="uint8")
    return cid > 0, gj


def main():
    z = np.load(CYG)
    w, lat, lon, months = z["w"], z["lat"], z["lon"], z["month"]
    valid, wet = w >= 0, w == 1
    perm = (wet.sum(0) / np.maximum(valid.sum(0), 1) >= CYG_PERMANENT) | (jrc_permanent(lat, lon) >= JRC_CELL_SHARE)
    inside, gj = counties(lat, lon)
    keep = inside & ~perm

    sq_km2 = (111.32 * abs(lon[1] - lon[0])) ** 2 * np.cos(np.deg2rad(lat))[:, None]
    crop_pct = crop_on_grid(lat, lon)
    crop_km2 = np.where(keep, crop_pct / 100 * sq_km2, 0)
    total = crop_km2.sum()
    print(f"cropland in South Sudan, permanent water excluded: {total:,.0f} km2 "
          f"({(crop_pct / 100 * sq_km2)[inside & perm].sum():,.0f} km2 in permanent-water squares dropped)")

    yr = np.array([int(m[:4]) for m in months])
    mo = np.array([int(m[5:]) for m in months])
    monthly = pd.DataFrame({
        "year": yr, "month": mo,
        "flooded_km2": [(crop_km2 * (wet[t] & valid[t])).sum() for t in range(len(months))],
        "data_km2": [(crop_km2 * valid[t]).sum() for t in range(len(months))]})
    monthly["pct"] = monthly.flooded_km2 / monthly.data_km2 * 100

    rows = []
    for y in np.unique(yr):
        sel = yr == y
        m = monthly[monthly.year == y]
        pk = m.loc[m.pct.idxmax()]
        any_km2 = (crop_km2 * (wet[sel] & valid[sel]).any(0)).sum()
        rows.append({"year": str(y), "months": int(sel.sum()),
                     "flooded_any_km2": any_km2, "flooded_any_pct": any_km2 / total * 100,
                     "mean_month_pct": m.pct.mean(), "peak_pct": pk.pct,
                     "peak_month": MONTHS[int(pk.month) - 1],
                     "data_pct": (m.data_km2 / total * 100).mean()})
    t = pd.DataFrame(rows)
    full = t[t.months == 12]
    avg = full.drop(columns=["year", "months", "peak_month"]).mean()
    t = pd.concat([t, pd.DataFrame([{**avg, "year": f"Average {full.year.iloc[0]}-{full.year.iloc[-1]}",
                                     "months": "", "peak_month": ""}])], ignore_index=True)
    t["cropland_km2"] = total
    t = t[["year", "months", "cropland_km2", "flooded_any_km2", "flooded_any_pct",
           "mean_month_pct", "peak_pct", "peak_month", "data_pct"]]

    OUT.mkdir(exist_ok=True)
    t.round(2).to_csv(OUT / "cropland_flood_overlap.csv", index=False)
    fmt = t.copy()
    for c in ["cropland_km2", "flooded_any_km2"]:
        fmt[c] = fmt[c].map("{:,.0f}".format)
    for c in ["flooded_any_pct", "mean_month_pct", "peak_pct", "data_pct"]:
        fmt[c] = fmt[c].map("{:.2f}".format)
    md = ["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in fmt.itertuples(index=False)]
    (OUT / "cropland_flood_overlap.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n" + "\n".join(md))

    # Figure for YEAR: months flooded per cropland square, and monthly share vs all-year mean.
    sel = yr == YEAR
    if not sel.any():
        raise SystemExit(f"YEAR {YEAR} not in data ({months[0]} to {months[-1]})")
    n_wet = (wet[sel] & valid[sel]).sum(0).astype(float)
    cropsq = keep & (crop_pct >= CROP_MIN)
    img = np.where(cropsq, n_wet, np.nan)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ext = (lon[0], lon[-1], lat[-1], lat[0])
    a1.imshow(np.where(inside & ~cropsq, 1, np.nan), cmap="Greys", vmin=0, vmax=5, extent=ext)
    a1.imshow(np.where(perm & inside, 1, np.nan), cmap="Blues", vmin=0, vmax=1.5, extent=ext)
    cmap = plt.get_cmap("YlOrRd", int(sel.sum()) + 1)
    im = a1.imshow(img, cmap=cmap, vmin=-0.5, vmax=sel.sum() + 0.5, extent=ext)
    for ft in gj["features"]:
        g = ft["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            x, y = np.array(poly[0]).T
            a1.plot(x, y, color="black", lw=0.3)
    fig.colorbar(im, ax=a1, label=f"Months in {YEAR} the square was marked water")
    a1.set_xlabel("Longitude")
    a1.set_ylabel("Latitude")
    a1.set_title(f"Cropland squares (crop >= {CROP_MIN:g}%) by months flooded, {YEAR}\n"
                 f"light grey = <{CROP_MIN:g}% cropland, blue = permanent water (excluded)")

    clim = monthly[np.isin(monthly.year, full.year.astype(int))].groupby("month").pct.mean()
    m = monthly[monthly.year == YEAR]
    a2.plot(clim.index, clim.to_numpy(), color="grey", ls="--",
            label=f"Average {full.year.iloc[0]}-{full.year.iloc[-1]}")
    a2.plot(m.month, m.pct, color="tab:red", marker="o", label=str(YEAR))
    a2.set_xticks(range(1, 13), MONTHS)
    a2.set_ylim(bottom=0)
    a2.set_xlabel("Month")
    a2.set_ylabel("Cropland under water (% of cropland with data)")
    a2.set_title(f"Share of South Sudan's cropland marked water per month, {YEAR}")
    a2.legend(loc="best")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"cropland_flood_{YEAR}.png", dpi=150)
    print(f"\nwrote cropland_flood_overlap.csv/.md and cropland_flood_{YEAR}.png to {OUT}")


if __name__ == "__main__":
    main()
