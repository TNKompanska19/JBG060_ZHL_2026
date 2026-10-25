"""The flood target: CYGNSS water maps for South Sudan, permanent water removed, averaged to the model grid.

    python 1_data_processing/3_build_dataset/build_target.py        (run from the repo root)

Run once after downloading CYGNSS: it caches the South Sudan part of every monthly map and writes the county
table and two check figures. build_inputs.py then calls target() for the model grid.

Input:  raw_data/cygnss/*.nc (global monthly maps, from 1_download/get_cygnss.py)
        raw_data/Administrative boundaries/ssd_admin2.geojson
        raw_data/jrc_gsw/*.tif (JRC Global Surface Water occurrence, from 1_download/get_covariates.py --only jrc)
Output: processed_data/cygnss/ssd_watermask.npz   South Sudan part of every map (1 = water, 0 = land, -99 = no data).
                                                   Once it exists, the 21 GB in raw_data/cygnss/ can be deleted.
        processed_data/cygnss/jrc_perm_1km.npy     share of each 1 km cell that is JRC permanent water
        processed_data/cygnss/county_month.csv     one row per county and month
        results/build_target/cygnss_national.png, cygnss_2022_anomaly.png

Definitions (our choices):
- Permanent water, removed per 1 km cell before averaging: at least JRC_CELL_SHARE of the cell is JRC water with
  occurrence >= JRC_OCC (the layer and threshold the CYGNSS product pastes in; Pekel et al. 2016, Pu et al. 2024),
  or CYGNSS marks the cell water in >= CYG_PERMANENT of its months (swamp missing from JRC).
- target y = share of a square's remaining 1 km cells marked water; a square counts as flooded at y >= 0.5.
- County table: water_pct = % of a county's non-permanent cells marked water that month (CYGNSS rule only);
  normal_pct = that county's climatology (mean water_pct for the calendar month over all years);
  anomaly = water_pct - normal_pct.
- A month's map uses the following month's data (Pu et al. 2024, 3), so at decision time the newest usable
  map is two months old.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CYG_RAW = ROOT / "raw_data" / "cygnss"
ADMIN = ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = ROOT / "processed_data" / "cygnss"
CYG = OUT / "ssd_watermask.npz"
FIG = ROOT / "results" / "build_target"
BBOX = (24.10, 3.45, 36.00, 12.30)       # lon_min, lat_min, lon_max, lat_max (South Sudan)
NAME_KEYS = ["adm2_name", "ADM2_EN", "admin2Name_en", "NAME_2", "shapeName"]
MONTHS = pd.period_range("2017-08", "2026-07", freq="M")     # the monthly clock of every panel array
NBR_DEG = 0.05              # neighbourhood for nbr: mean of y over the surrounding ~5 km
JRC_OCC = 95                # JRC water occurrence (%) counted as permanent: the layer and threshold CYGNSS pastes in
JRC_CELL_SHARE = 0.5        # a 1 km cell is permanent water if at least half of it is JRC permanent water (our choice)
CYG_PERMANENT = 0.95        # ... or if CYGNSS marks it water in >= 95% of its months (swamp missing from JRC)
PERMANENT = CYG_PERMANENT   # county table: same CYGNSS rule


def crop_all():
    """South Sudan part of every monthly map, north row first. Cached after the first run."""
    f = OUT / "ssd_watermask.npz"
    if f.exists():
        z = np.load(f)
        return z["w"], z["lat"], z["lon"], z["month"]
    files = sorted(CYG_RAW.glob("*.nc"))
    if not files:
        raise SystemExit(f"no CYGNSS files in {CYG_RAW}")
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


def jrc_permanent(lat, lon):
    """Share of each 1 km CYGNSS cell that is JRC Global Surface Water with occurrence >= JRC_OCC (Pekel et al.
    2016; the same layer and threshold the CYGNSS product pastes in). Read from the 30 m tiles in
    raw_data/jrc_gsw/ in strips, block-averaged to 0.01 deg. Cached in processed_data/cygnss/jrc_perm_1km.npy."""
    cache = CYG.parent / "jrc_perm_1km.npy"
    if cache.exists():
        return np.load(cache)
    tiles = sorted((ROOT / "raw_data" / "jrc_gsw").glob("*.tif"))
    if not tiles:
        print("\n!!! JRC PERMANENT-WATER MASK NOT INCLUDED: no raw_data/jrc_gsw/*.tif\n")
        return np.zeros((lat.size, lon.size), np.float32)
    out = np.full((lat.size, lon.size), np.nan, np.float32)
    step = 0.01
    for f in tiles:
        with rasterio.open(f) as src:
            px = abs(src.res[0])
            k = int(round(step / px))                      # 30 m pixels per 1 km cell side (40)
            top, left = src.transform.f, src.transform.c
            r0 = max(0, int((top - (lat.max() + step)) / px) // k * k)
            r1 = min(src.height, int(np.ceil((top - (lat.min() - step)) / px / k)) * k)
            c0 = max(0, int((lon.min() - step - left) / px) // k * k)
            c1 = min(src.width, int(np.ceil((lon.max() + step - left) / px / k)) * k)
            if r1 <= r0 or c1 <= c0:
                continue
            c1 = c0 + (c1 - c0) // k * k
            for rs in range(r0, r1 - k + 1, 100 * k):
                re_ = min(rs + 100 * k, r1 - (r1 - r0) % k)
                a = src.read(1, window=((rs, re_), (c0, c1)))
                perm = ((a >= JRC_OCC) & (a <= 100)).astype(np.float32)
                blk = perm.reshape((re_ - rs) // k, k, (c1 - c0) // k, k).mean((1, 3))
                clat = top - (rs + (np.arange(blk.shape[0]) + 0.5) * k) * px
                clon = left + (c0 + (np.arange(blk.shape[1]) + 0.5) * k) * px
                i = np.rint((lat[0] - clat) / step).astype(int)
                j = np.rint((clon - lon[0]) / step).astype(int)
                oi, oj = (i >= 0) & (i < lat.size), (j >= 0) & (j < lon.size)
                out[np.ix_(i[oi], j[oj])] = blk[np.ix_(oi, oj)]
        print(f"  JRC {f.name}", flush=True)
    out = np.nan_to_num(out)
    np.save(cache, out)
    return out


def target(res):
    """CYGNSS labels averaged onto the --res grid; NaN outside South Sudan.
    Permanent water is removed per 1 km cell first, so a square's share counts land only: cells that are
    >= JRC_CELL_SHARE JRC permanent water, or CYGNSS water in >= CYG_PERMANENT of their months."""
    z = np.load(CYG)
    w, lat, lon, mon = z["w"], z["lat"], z["lon"], z["month"]
    nval = (w >= 0).sum(0)
    cyg_perm = (w == 1).sum(0) >= CYG_PERMANENT * np.maximum(nval, 1)
    jrc = jrc_permanent(lat, lon) >= JRC_CELL_SHARE
    perm = cyg_perm | jrc
    print(f"  permanent water removed: {perm.sum():,} cells of 1 km (JRC {jrc.sum():,}, CYGNSS >= "
          f"{CYG_PERMANENT:.0%} {cyg_perm.sum():,}, both {(jrc & cyg_perm).sum():,})", flush=True)
    w = np.where(perm[None], np.int8(-99), w)
    f = int(round(res / 0.01))
    H, W = w.shape[1] // f * f, w.shape[2] // f * f
    w = w[:, :H, :W]
    shape = (w.shape[0], H // f, f, W // f, f)
    wet = (w == 1).reshape(shape).sum((2, 4)).astype(np.float32)
    val = (w >= 0).reshape(shape).sum((2, 4)).astype(np.float32)
    y_c = np.where(val > 0, wet / np.maximum(val, 1), np.nan)
    lat2 = lat[:H].reshape(-1, f).mean(1)
    lon2 = lon[:W].reshape(-1, f).mean(1)

    gj = json.loads(ADMIN.read_text(encoding="utf8"))
    tr = from_origin(lon2[0] - res / 2, lat2[0] + res / 2, res, res)
    inside = rasterize([(ft["geometry"], 1) for ft in gj["features"]],
                       out_shape=(lat2.size, lon2.size), transform=tr, fill=0) == 1
    y = np.full((len(MONTHS), lat2.size, lon2.size), np.nan, np.float32)
    pos = MONTHS.get_indexer(pd.PeriodIndex(mon, freq="M"))
    y[pos] = np.where(inside, y_c, np.nan)

    r = max(1, int(round(NBR_DEG / res)))
    ok = ~np.isnan(y)
    s = uniform_filter(np.nan_to_num(y), size=(1, 2 * r + 1, 2 * r + 1), mode="constant")
    c = uniform_filter(ok.astype(np.float32), size=(1, 2 * r + 1, 2 * r + 1), mode="constant")
    nbr = np.where(ok, s / np.maximum(c, 1e-6), np.nan).astype(np.float32)
    print(f"target: {lat2.size} x {lon2.size} squares at {res} deg, {inside.sum():,} in South "
          f"Sudan, CYGNSS months {mon[0]} to {mon[-1]}")
    return y, nbr, lat2, lon2


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

    # Checks: national series, and the counties furthest above their climatology in Oct-Dec 2022.
    nat = d.groupby("month")[["water_squares", "squares"]].sum()
    nat = (nat.water_squares / nat.squares * 100).rename("water_pct")
    nat.index = pd.to_datetime(nat.index)
    ond = nat[nat.index.month >= 10]
    print("\nSouth Sudan, % of normally dry land marked water, Oct-Dec mean per year:")
    print(ond.groupby(ond.index.year).mean().round(2).to_string())
    top = (d[(d.year == 2022) & (d.cal_month >= 10)].groupby("county").anomaly.mean()
           .sort_values(ascending=False).head(10))
    print("\nOct-Dec 2022, counties most above their climatology (percentage points):")
    print(top.round(1).to_string())

    FIG.mkdir(parents=True, exist_ok=True)
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
    ax.set_title("Where Oct-Dec 2022 flooding exceeded its climatology (blue), CYGNSS\n"
                 "grey = permanent water or outside South Sudan")
    fig.tight_layout()
    fig.savefig(FIG / "cygnss_2022_anomaly.png", dpi=120)
    print(f"\nwrote cygnss_national.png and cygnss_2022_anomaly.png to {FIG}")


if __name__ == "__main__":
    main()
