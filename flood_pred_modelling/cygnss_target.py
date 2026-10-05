"""CYGNSS flood target: water share per county per month, South Sudan.

    python flood_pred_modelling/cygnss_target.py

Input:  raw_data/cygnss/*.nc (global monthly maps, from get_cygnss.py)
        raw_data/Administrative boundaries/ssd_admin2.geojson
Output: processed_data/cygnss/ssd_watermask.npz   South Sudan part of every map.
                                                   Once it exists, the 21 GB in
                                                   raw_data/cygnss/ can be deleted.
        processed_data/cygnss/county_month.csv     one row per county and month
        flood_pred_modelling/output/cygnss_national.png, cygnss_2022_anomaly.png

Definitions. These are our choices; state them in the report:
- Permanent water = a ~1 km square marked water in >= 90% of months: rivers,
  lakes, the swamp core. The product pastes these in (Pu et al. 2024, 3.3).
  They are excluded, so what remains measures flooding of normally dry land.
- water_pct  = % of a county's non-permanent squares marked water that month.
- normal_pct = that county's mean water_pct for the same calendar month over
  all years. The record (Aug 2018 on) includes the 2019-22 record floods, so
  the normal is inflated.
- anomaly    = water_pct - normal_pct. Positive = more water than usual.
- A month's map uses the following month's data (Pu et al. 2024, 3), so at
  decision time the newest usable map is two months old.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "cygnss"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = ROOT / "processed_data" / "cygnss"
FIG = Path(__file__).resolve().parent / "output"
BBOX = (24.10, 3.45, 36.00, 12.30)       # lon_min, lat_min, lon_max, lat_max (South Sudan)
PERMANENT = 0.90
NAME_KEYS = ["adm2_name", "ADM2_EN", "admin2Name_en", "NAME_2", "shapeName"]


def crop_all():
    """South Sudan part of every monthly map, north row first. Cached after the first run."""
    f = OUT / "ssd_watermask.npz"
    if f.exists():
        z = np.load(f)
        return z["w"], z["lat"], z["lon"], z["month"]
    files = sorted(RAW.glob("*.nc"))
    if not files:
        raise SystemExit(f"no CYGNSS files in {RAW}")
    stack, months = [], []
    for p in files:
        m = re.search(r"\.(\d{4})-(\d{2})\.", p.name)
        if not m:
            raise SystemExit(f"cannot read the month from {p.name}")
        with xr.open_dataset(p, decode_times=False) as ds:
            la = ds["lat"].to_numpy()
            lat_sl = slice(BBOX[1], BBOX[3]) if la[0] < la[-1] else slice(BBOX[3], BBOX[1])
            v = ds["watermask"].sel(lat=lat_sl, lon=slice(BBOX[0], BBOX[2]))
            a = v.to_numpy().squeeze()
            lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
        if lat[0] < lat[-1]:                 # rasters run north to south
            a, lat = a[::-1], lat[::-1]
        stack.append(np.where(np.isnan(a), -99, a).astype(np.int8))   # NaN = the -99 fill
        months.append(f"{m[1]}-{m[2]}")
        print(f"  {p.name}", flush=True)
    w, months = np.stack(stack), np.array(months)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, w=w, lat=lat, lon=lon, month=months)
    return w, lat, lon, months


def counties(lat, lon):
    """County number per square (0 = outside every county), county names, and the geojson."""
    gj = json.loads(ADMIN.read_text(encoding="utf8"))
    props = gj["features"][0]["properties"]
    key = next((k for k in NAME_KEYS if k in props), None)
    if key is None:
        raise SystemExit(f"no county-name field among {list(props)}: add it to NAME_KEYS")
    res = round(float(lon[1] - lon[0]), 6)
    tr = from_origin(float(lon[0]) - res / 2, float(lat[0]) + res / 2, res, res)
    cid = rasterize([(ft["geometry"], i + 1) for i, ft in enumerate(gj["features"])],
                    out_shape=(lat.size, lon.size), transform=tr, fill=0, dtype="int32")
    return cid, [ft["properties"][key] for ft in gj["features"]], gj


def main():
    w, lat, lon, months = crop_all()
    cid, names, gj = counties(lat, lon)
    valid, wet = w >= 0, w == 1
    perm = wet.sum(0) / np.maximum(valid.sum(0), 1) >= PERMANENT
    keep = (cid > 0) & ~perm
    print(f"{len(months)} months ({months[0]} to {months[-1]}); {keep.sum():,} squares in "
          f"counties; {(perm & (cid > 0)).sum():,} permanent-water squares excluded")

    n = len(names) + 1
    rows = []
    for t, mo in enumerate(months):
        ok = keep & valid[t]
        tot = np.bincount(cid[ok], minlength=n)
        hit = np.bincount(cid[ok & wet[t]], minlength=n)
        rows += [(c, names[c - 1], mo, int(hit[c]), int(tot[c]))
                 for c in range(1, n) if tot[c]]
    d = pd.DataFrame(rows, columns=["county_id", "county", "month", "water_squares", "squares"])
    d["year"] = d.month.str[:4].astype(int)
    d["cal_month"] = d.month.str[5:].astype(int)
    d["water_pct"] = d.water_squares / d.squares * 100
    d["normal_pct"] = d.groupby(["county_id", "cal_month"]).water_pct.transform("mean")
    d["anomaly"] = d.water_pct - d.normal_pct
    d.to_csv(OUT / "county_month.csv", index=False)
    print(f"wrote {OUT / 'county_month.csv'}: {len(d)} rows, {d.county_id.nunique()} counties")

    # Checks: national series, and the counties furthest above normal in Oct-Dec 2022.
    nat = d.groupby("month")[["water_squares", "squares"]].sum()
    nat = (nat.water_squares / nat.squares * 100).rename("water_pct")
    nat.index = pd.to_datetime(nat.index)
    ond = nat[nat.index.month >= 10]
    print("\nSouth Sudan, % of normally dry land marked water, Oct-Dec mean per year:")
    print(ond.groupby(ond.index.year).mean().round(2).to_string())
    top = (d[(d.year == 2022) & (d.cal_month >= 10)].groupby("county").anomaly.mean()
           .sort_values(ascending=False).head(10))
    print("\nOct-Dec 2022, counties most above their normal (percentage points):")
    print(top.round(1).to_string())

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nat.index, nat.to_numpy(), color="steelblue")
    ax.set_xlabel("Month")
    ax.set_ylabel("Normally dry land marked water (%)")
    ax.set_title("CYGNSS (L-band): flooded share of South Sudan's normally dry land, per month\n"
                 f"permanent water (water in >= {PERMANENT:.0%} of months) excluded")
    fig.tight_layout()
    fig.savefig(FIG / "cygnss_national.png", dpi=120)

    # Map: Oct-Dec 2022 water frequency minus the Oct-Dec mean of all years, per square.
    mo = pd.to_datetime(months)
    ond_all = mo.month >= 10
    ond_22 = ond_all & (mo.year == 2022)
    freq = lambda sel: wet[sel].sum(0) / np.maximum(valid[sel].sum(0), 1)  # noqa: E731
    diff = np.where(keep, (freq(ond_22) - freq(ond_all)) * 100, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(diff, cmap=cmap, vmin=-60, vmax=60,
                   extent=(lon[0], lon[-1], lat[-1], lat[0]))
    for ft in gj["features"]:
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            x, y = np.array(poly[0]).T
            ax.plot(x, y, color="black", lw=0.3)
    fig.colorbar(im, ax=ax, label="Oct-Dec 2022 minus Oct-Dec average (% of months as water)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where Oct-Dec 2022 flooding exceeded normal (blue), CYGNSS\n"
                 "grey = permanent water or outside South Sudan")
    fig.tight_layout()
    fig.savefig(FIG / "cygnss_2022_anomaly.png", dpi=120)
    print(f"\nwrote cygnss_national.png and cygnss_2022_anomaly.png to {FIG}")


if __name__ == "__main__":
    main()
