"""Build all model inputs on one grid and one monthly clock.

    python flood_pred_modelling/build_inputs.py --res 0.02      (run from the repo root)

--res = square size in degrees, a multiple of 0.01 (0.01 ~ 1 km, 0.02 ~ 2 km).
Needs processed_data/cygnss/ssd_watermask.npz (from cygnss_target.py).
Output in processed_data/panel_<res>/:
  inputs.npz   arrays below          shared.csv   one row per month, one column per series

Every array uses the same monthly clock, `months` (Aug 2017 - Jul 2026); months a
source does not cover are NaN.
  y        [month, row, col]  TARGET. Share of the square's 1 km CYGNSS cells labelled
                              water (at 0.01 deg: 0 or 1). NaN outside South Sudan.
  nbr      [month, row, col]  mean of y over the surrounding ~5 km (NBR_DEG)
  tp, ro   [month, cell]      ERA5 monthly rainfall and runoff sums (mm), 0.25 deg cells
  era5_idx [row, col]         the ERA5 cell each square takes its weather from
  et       [month, cell]      AgERA5 monthly mean reference evapotranspiration (mm/day)
  et_idx   [row, col]         the AgERA5 cell each square takes it from
shared.csv: lake levels (monthly mean), Dartmouth stations (monthly mean discharge),
rainfall summed over three upstream areas (UPSTREAM, approximate boxes, our choice).
Nothing is averaged across years here, so nothing leaks between test folds; the
square's normal is computed per fold in train_models.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
CYG = ROOT / "processed_data" / "cygnss" / "ssd_watermask.npz"
ADMIN = RAW / "Administrative boundaries" / "ssd_admin2.geojson"
MONTHS = pd.period_range("2017-08", "2026-07", freq="M")
ERA5_YEARS = range(2017, 2026)
NBR_DEG = 0.05
# Upstream areas (lon_min, lat_min, lon_max, lat_max). Approximate boxes: the Lake
# Victoria basin, the Uganda lakes (Kyoga, Albert), and the Ethiopian highlands that
# feed the Sobat. The ERA5 files stop at 3 S, which cuts the far south of the first.
UPSTREAM = {"rain_victoria_basin": (29.5, -3.0, 35.0, 1.0),
            "rain_uganda_lakes": (30.0, 1.0, 34.5, 3.5),
            "rain_ethiopian_highlands": (34.0, 6.0, 37.0, 9.5)}
# GloFAS river points (lat, lon): approximate town locations, each snapped to the
# largest-flow GloFAS cell within SNAP_DEG, so small coordinate errors do not matter.
GLOFAS_POINTS = {"q_nimule": (3.60, 32.06),        # White Nile entering from Uganda
                 "q_mongalla": (5.20, 31.77),      # White Nile entering the Sudd
                 "q_bor": (6.21, 31.56),
                 "q_nasir_sobat": (8.60, 33.07),   # Sobat, fed by the Ethiopian highlands
                 "q_malakal": (9.53, 31.66)}       # White Nile below the Sobat junction
SNAP_DEG = 0.15
NEAR_DEG = 0.45        # each square takes discharge from the largest river within ~50 km
KERNEL_KM = (10, 30, 100)   # distance scales for the weighted sum of all nearby river flow
RIVER_MIN = 10.0            # m3/s long-run mean; smaller GloFAS cells are not counted as rivers
LOW_M = 2.0                 # MERIT: land less than this many metres above the nearest drainage counts as low-lying


def target(res):
    """CYGNSS labels averaged onto the --res grid; NaN outside South Sudan."""
    z = np.load(CYG)
    w, lat, lon, mon = z["w"], z["lat"], z["lon"], z["month"]
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


def nearest(grid_lat, grid_lon, lat2, lon2):
    """Flat index of the nearest grid node for every square (regular grids)."""
    dl, dn = grid_lat[1] - grid_lat[0], grid_lon[1] - grid_lon[0]
    i = np.clip(np.rint((lat2 - grid_lat[0]) / dl), 0, grid_lat.size - 1).astype(int)
    j = np.clip(np.rint((lon2 - grid_lon[0]) / dn), 0, grid_lon.size - 1).astype(int)
    return (i[:, None] * grid_lon.size + j[None, :]).astype(np.int32)


def monthly_on_clock(da, how):
    """Monthly sum/mean of a (time, lat, lon) DataArray, placed on MONTHS as [month, cell]."""
    m = getattr(da.resample({da.dims[0]: "MS"}), how)()
    out = np.full((len(MONTHS), m.shape[1] * m.shape[2]), np.nan, np.float32)
    pos = MONTHS.get_indexer(pd.PeriodIndex(pd.to_datetime(m[m.dims[0]].values), freq="M"))
    keep = pos >= 0
    out[pos[keep]] = m.values.reshape(m.shape[0], -1)[keep]
    return out


def era5(lat2, lon2):
    """Local monthly rainfall/runoff per ERA5 cell, and rainfall over the upstream boxes."""
    tp, ro, up = [], [], {k: [] for k in UPSTREAM}
    for y in ERA5_YEARS:
        ds = xr.open_dataset(RAW / "rainfall and runoff" / f"ERA5_{y}.nc",
                             chunks={"valid_time": 744})[["tp", "ro"]] * 1000.0   # m -> mm
        loc = ds.sel(latitude=slice(lat2.max() + 0.25, lat2.min() - 0.25),       # latitude runs N -> S
                     longitude=slice(lon2.min() - 0.25, lon2.max() + 0.25))
        tp.append(monthly_on_clock(loc.tp, "sum"))
        ro.append(monthly_on_clock(loc.ro, "sum"))
        for k, (x0, y0, x1, y1) in UPSTREAM.items():
            box = ds.tp.sel(latitude=slice(y1, y0), longitude=slice(x0, x1)).mean(["latitude", "longitude"])
            s = box.resample(valid_time="MS").sum().to_series()
            up[k].append(s)
        print(f"  ERA5 {y}", flush=True)
    tp, ro = np.fmax.reduce(tp), np.fmax.reduce(ro)          # each year fills only its own months
    idx = nearest(loc.latitude.values, loc.longitude.values, lat2, lon2)
    shared = pd.DataFrame({k: pd.concat(v) for k, v in up.items()})
    shared.index = pd.PeriodIndex(shared.index, freq="M")
    return tp, ro, idx, shared


def agera5(lat2, lon2):
    """Monthly mean reference ET per AgERA5 cell. Returns None (loudly) if it cannot be read."""
    var, out, grid = "ReferenceET_PenmanMonteith_FAO56", [], None
    try:
        for y in ERA5_YEARS:
            files = sorted((RAW / "evapotranspiration" / f"ET_{y}").glob("*.nc"))
            ds = xr.open_mfdataset(files, combine="by_coords")[var]
            la = ds.lat.values
            lat_sl = slice(lat2.max() + 0.1, lat2.min() - 0.1) if la[0] > la[-1] \
                else slice(lat2.min() - 0.1, lat2.max() + 0.1)
            ds = ds.sel(lat=lat_sl, lon=slice(lon2.min() - 0.1, lon2.max() + 0.1))
            if ds.size == 0:
                raise ValueError("selection is empty: check the AgERA5 extent")
            out.append(monthly_on_clock(ds, "mean"))
            grid = (ds.lat.values, ds.lon.values)
            print(f"  AgERA5 {y}", flush=True)
    except Exception as e:
        print(f"\n!!! ET NOT INCLUDED: {e}\n")
        return None, None
    return np.fmax.reduce(out), nearest(grid[0], grid[1], lat2, lon2)


def glofas(lat2, lon2):
    """GloFAS monthly mean discharge, log(1 + m3/s).

    Local: each square takes the largest river within NEAR_DEG (its long-run mean
    flow picks the river; river locations do not change, so using all years for
    that choice leaks nothing about flooding). Shared: the GLOFAS_POINTS series.
    """
    files = sorted((RAW / "glofas").glob("glofas_*.nc"))
    if not files:
        print("\n!!! GLOFAS NOT INCLUDED: no raw_data/glofas/glofas_*.nc files\n")
        return {}, None
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        var = max(ds.data_vars, key=lambda v: ds[v].ndim)
        da = ds[var].squeeze(drop=True)
        tdim = next(d for d in da.dims if "time" in d)
        la = next(d for d in da.dims if d.startswith("lat"))
        lo = next(d for d in da.dims if d.startswith("lon"))
        da = da.transpose(tdim, la, lo)
        parts.append(monthly_on_clock(da, "mean"))
        glat, glon = da[la].values, da[lo].values
        print(f"  GloFAS {f.name} ({var})", flush=True)
    q = np.fmax.reduce(parts)
    qbar = np.nanmean(q, 0).reshape(glat.size, glon.size)

    H, W = qbar.shape
    r = max(1, int(round(NEAR_DEG / abs(glat[1] - glat[0]))))
    pad = np.pad(np.nan_to_num(qbar, nan=-1.0), r, constant_values=-1.0)
    best, bi, bj = np.full((H, W), -1.0), *np.indices((H, W))
    ii, jj = np.indices((H, W))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            v = pad[r + di:r + di + H, r + dj:r + dj + W]
            better = v > best
            best = np.where(better, v, best)
            bi, bj = np.where(better, ii + di, bi), np.where(better, jj + dj, bj)
    river = np.clip(bi, 0, H - 1) * W + np.clip(bj, 0, W - 1)
    q_idx = river.ravel()[nearest(glat, glon, lat2, lon2)]
    rlat, rlon = glat[q_idx // W], glon[q_idx % W]        # the chosen river cell per square
    q_dist = np.hypot((rlat - lat2[:, None]) * 111.0,
                      (rlon - lon2[None, :]) * 111.0 * np.cos(np.radians(lat2[:, None])))   # km
    q_size = np.log1p(qbar.ravel()[q_idx])                # how big that river is

    # Weighted sum of ALL river flow around each square: weight exp(-distance / L), for
    # several L. The model gets every scale and learns which matters.
    near = nearest(glat, glon, lat2, lon2)
    dy = abs(glat[1] - glat[0]) * 111.0
    dx = abs(glon[1] - glon[0]) * 111.0 * np.cos(np.radians(glat.mean()))
    rivers = qbar >= RIVER_MIN
    cubes = {}
    for L in KERNEL_KM:
        R = int(np.ceil(3 * L / min(dx, dy)))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        ker = np.exp(-np.hypot(yy * dy, xx * dx) / L)
        cube = np.full((len(MONTHS),) + near.shape, np.nan, np.float16)
        for t in range(len(MONTHS)):
            if np.isnan(q[t]).all():
                continue
            f = np.where(rivers, np.nan_to_num(q[t].reshape(H, W)), 0.0)
            cube[t] = np.log1p(np.clip(fftconvolve(f, ker, mode="same"), 0, None)).ravel()[near]
        cubes[f"qk{L}"] = cube
        print(f"  weighted river flow, scale {L} km", flush=True)

    pts = {}
    for name, (pla, plo) in GLOFAS_POINTS.items():
        m = ((np.abs(glat - pla) <= SNAP_DEG)[:, None] & (np.abs(glon - plo) <= SNAP_DEG)[None, :]).ravel()
        k = np.flatnonzero(m)[np.nanargmax(qbar.ravel()[m])]
        pts[name] = np.log1p(q[:, k])
        print(f"  {name}: mean flow {qbar.ravel()[k]:,.0f} m3/s")
    shared = pd.DataFrame(pts, index=MONTHS.astype(str))
    return dict(q=np.log1p(q).astype(np.float32), q_idx=q_idx.astype(np.int32),
                q_dist=q_dist.astype(np.float32), q_size=q_size.astype(np.float32), **cubes), shared


def merit(lat2, lon2, res):
    """MERIT Hydro terrain per square, fixed in time (Yamazaki et al. 2019, 90 m):
    hnd_mean / hnd_min  average and lowest height above the nearest drainage (m)
    low_2m              share of the square less than LOW_M above the nearest drainage
    upa_max             log of the largest upstream drainage area in the square (km2)
    Reads the *_hnd.tif / *_upa.tif tiles under raw_data/MERITHydro (any subfolder) that
    overlap the grid; tile names give their lower-left corner, e.g. n05e030 = 5-10N, 30-35E."""
    dst_tr = from_origin(lon2[0] - res / 2, lat2[0] + res / 2, res, res)
    shape = (lat2.size, lon2.size)
    jobs = {"hnd_mean": ("hnd", Resampling.average, False), "hnd_min": ("hnd", Resampling.min, False),
            "low_2m": ("hnd", Resampling.average, True), "upa_max": ("upa", Resampling.max, False)}
    out = {k: np.full(shape, np.nan, np.float32) for k in jobs}
    def overlaps(f):
        la0, lo0 = int(f.name[1:3]) * (1 if f.name[0] == "n" else -1), \
            int(f.name[4:7]) * (1 if f.name[3] == "e" else -1)
        return (la0 < lat2.max() + res and la0 + 5 > lat2.min() - res
                and lo0 < lon2.max() + res and lo0 + 5 > lon2.min() - res)

    for var in ("hnd", "upa"):
        tiles = [f for f in sorted((RAW / "MERITHydro").rglob(f"*_{var}.tif")) if overlaps(f)]
        if not tiles:
            print(f"\n!!! MERIT {var} NOT INCLUDED: no *_{var}.tif over South Sudan under raw_data/MERITHydro\n")
            continue
        for f in tiles:
            with rasterio.open(f) as src:
                a = src.read(1).astype(np.float32)
                tr, crs = src.transform, src.crs
            a[a <= -9999] = np.nan                       # ocean / undefined
            for k, (v, how, low) in jobs.items():
                if v != var:
                    continue
                src_a = np.where(np.isnan(a), np.nan, (a < LOW_M).astype(np.float32)) if low else a
                tmp = np.full(shape, np.nan, np.float32)
                reproject(src_a, tmp, src_transform=tr, src_crs=crs, dst_transform=dst_tr,
                          dst_crs="EPSG:4326", resampling=how, src_nodata=np.nan, dst_nodata=np.nan)
                out[k] = np.where(np.isnan(tmp), out[k], tmp)
            print(f"  MERIT {f.name}", flush=True)
    out["upa_max"] = np.log1p(out["upa_max"])
    return {k: v for k, v in out.items() if not np.isnan(v).all()}


def course_series():
    """Lake levels and Dartmouth discharge, monthly, via the course loaders."""
    sys.path.insert(0, str(ROOT / "processing_data"))
    os.chdir(ROOT)                                          # the loaders use relative paths
    from loading import load_dartmouth_data, load_lake_stations
    cols = {}
    lakes = load_lake_stations()
    for name, col in (("victoria", "height_wrt_ref"), ("Kyoga", "height_wrt_ref"),
                      ("Albert", "water_level")):
        s = lakes[name][col]
        cols[f"lake_{name.lower()}"] = s.groupby(s.index.to_period("M")).mean()
    for sid, df in load_dartmouth_data().items():
        s = df.select_dtypes("number").iloc[:, 0]
        cols[f"dfo_{sid}"] = s.groupby(s.index.to_period("M")).mean()
    # Altimetry revisits every ~10-35 days, so a month can be empty: carry the last
    # level forward for at most 2 months.
    return pd.DataFrame(cols).reindex(MONTHS).ffill(limit=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--add", nargs="+", choices=["glofas", "merit"],
                    help="add these to an existing panel instead of rebuilding everything")
    a = ap.parse_args()
    res = round(a.res, 2)
    out = ROOT / "processed_data" / f"panel_{res:.2f}"
    out.mkdir(parents=True, exist_ok=True)

    if a.add:
        arrays = dict(np.load(out / "inputs.npz"))
        shared = pd.read_csv(out / "shared.csv", index_col=0)
        if "glofas" in a.add:
            shared = shared[[c for c in shared.columns if not c.startswith("q_")]]
        lat2, lon2 = arrays["lat"], arrays["lon"]
        steps = a.add
    else:
        y, nbr, lat2, lon2 = target(res)
        tp, ro, era5_idx, upstream = era5(lat2, lon2)
        et, et_idx = agera5(lat2, lon2)
        shared = course_series().join(upstream.reindex(MONTHS))
        shared.index = shared.index.astype(str)
        arrays = dict(y=y.astype(np.float16), nbr=nbr.astype(np.float16), lat=lat2, lon=lon2,
                      months=np.array(MONTHS.astype(str), dtype="U7"), tp=tp, ro=ro,
                      era5_idx=era5_idx)
        if et is not None:
            arrays.update(et=et, et_idx=et_idx)
        steps = ["glofas", "merit"]

    if "glofas" in steps:
        g_arrays, q_pts = glofas(lat2, lon2)
        if g_arrays:
            arrays.update(g_arrays)
            shared = shared.join(q_pts)
    if "merit" in steps:
        arrays.update(merit(lat2, lon2, res))
    shared.to_csv(out / "shared.csv")
    np.savez_compressed(out / "inputs.npz", **arrays)
    print(f"\nwrote {out}: local streams "
          f"{[k for k in ('tp', 'ro', 'et', 'q', *[f'qk{L}' for L in KERNEL_KM]) if k in arrays]}, "
          f"shared series {list(shared.columns)}, fixed per square "
          f"{[k for k in ('q_dist', 'q_size', 'hnd_mean', 'hnd_min', 'low_2m', 'upa_max') if k in arrays]}")
    print("shared series, months with data:")
    print(shared.notna().sum().to_string())


if __name__ == "__main__":
    main()
