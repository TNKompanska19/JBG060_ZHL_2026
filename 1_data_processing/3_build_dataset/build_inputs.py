"""Build every model input on the target's grid and monthly clock (the target itself comes from build_target.py).

    python 1_data_processing/3_build_dataset/build_inputs.py --res 0.02            full rebuild
    python 1_data_processing/3_build_dataset/build_inputs.py --add rainx hetero    add or refresh steps only
    (run from the repo root; steps: see ADD_STEPS)

Needs processed_data/cygnss/ssd_watermask.npz (from build_target.py) and the raw data from 1_download/.
Output in processed_data/panel_<res>/:  inputs.npz (arrays below)  and  shared.csv (one row per month, one column
per series that is the same for every square). Every array uses the monthly clock `months` (Aug 2017 - Jul 2026);
months a source does not cover are NaN. [month, cell] arrays come with an *_idx [row, col] array giving the cell each
square takes its value from. Nothing is averaged across years here, so nothing leaks between test folds.

Functions are grouped by the input families of the model and the report:
  TARGET (build_target.target)  y [month, row, col] flood share of the square's land; nbr = its ~5 km neighbourhood
  WEATHER            tp, ro (ERA5 rain, runoff); et (AgERA5 evapotranspiration); r20, rx1 (heavy-rain days, wettest
                     day); sm1, sm3 (ERA5 soil moisture); t2m, d2m, si10, ssrd (ERA5 atmosphere); wetd, wspell,
                     conc5, dsd (rain shape within the month); uprain (rain over the square's upstream catchment);
                     shared: rain_* over three upstream areas
  RIVERS & STORAGE   q, qk10/30/100, q_dist, q_size (GloFAS); grace (GRACE storage); hw, hw_dist (Hydroweb levels,
                     off in the main model); shared: lake_* levels, dfo_* Dartmouth gauges, q_<place> GloFAS points
  LAND               hnd_mean, hnd_min, low_2m, upa_max (MERIT HAND, upstream area); slope, twi (terrain);
                     hnd_q1, hnd_sd, slope_sd, depress (within-square spread); clay, sand, vertisol (SoilGrids)
  CLIMATE INDICES    shared: nino34, dmi
  SCORING ONLY       crop_pct (ASAP cropland %), never a model input
"""
import argparse
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
from scipy.ndimage import uniform_filter  # noqa: F401  (kept for the neighbourhood helpers' history)
from scipy.signal import fftconvolve

from build_target import MONTHS, target  # noqa: E402  (same folder)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_data"
ERA5_YEARS = range(2017, 2026)
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
SLOPE_MIN_DEG = 0.01        # flat-land floor for the wetness index (our choice): tan(0) is undefined
HEAVY_MM = 20.0             # heavy-rain day: >= 20 mm (ETCCDI index R20mm)
CROP = RAW / "farmland" / "asap_mask_crop_v04.tif"
HW_MIN_OBS = 60             # Hydroweb: minimum measurements per station (our choice)
HW_START_BY = "2019-06-30"  # Hydroweb: record must start by this date, so training years give it a calendar-month mean
WET_MM = 1.0                # rain day: >= 1 mm (common convention, our choice)
DEPRESS_M = 1.0             # local depression: > 1 m below the mean elevation of the surrounding ~1 km (our choice)


# ==================== HELPERS =================================================================================


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


def _tiles(var, lat2, lon2, res):
    """MERIT tiles of one variable that overlap the grid (any subfolder of raw_data/MERITHydro)."""
    def overlaps(f):
        la0 = int(f.name[1:3]) * (1 if f.name[0] == "n" else -1)
        lo0 = int(f.name[4:7]) * (1 if f.name[3] == "e" else -1)
        return (la0 < lat2.max() + res and la0 + 5 > lat2.min() - res
                and lo0 < lon2.max() + res and lo0 + 5 > lon2.min() - res)
    return {f.name[:7]: f for f in sorted((RAW / "MERITHydro").rglob(f"*_{var}.tif")) if overlaps(f)}


def _to_grid(a, tr, crs, lat2, lon2, res, how=Resampling.average):
    tmp = np.full((lat2.size, lon2.size), np.nan, np.float32)
    reproject(a.astype(np.float32), tmp, src_transform=tr, src_crs=crs,
              dst_transform=from_origin(lon2[0] - res / 2, lat2[0] + res / 2, res, res),
              dst_crs="EPSG:4326", resampling=how, src_nodata=np.nan, dst_nodata=np.nan)
    return tmp


# ==================== WEATHER =================================================================================


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


def heavy_rain(lat2, lon2, n_cells):
    """Per ERA5 cell and month: r20 = days with >= HEAVY_MM rain, rx1 = wettest day (mm).
    Days are summed from the hourly files (UTC days). Same cells as tp, so era5_idx applies."""
    r20, rx1 = [], []
    for y in ERA5_YEARS:
        ds = xr.open_dataset(RAW / "rainfall and runoff" / f"ERA5_{y}.nc", chunks={"valid_time": 744})["tp"] * 1000.0
        loc = ds.sel(latitude=slice(lat2.max() + 0.25, lat2.min() - 0.25),
                     longitude=slice(lon2.min() - 0.25, lon2.max() + 0.25))
        day = loc.resample(valid_time="1D").sum()
        r20.append(monthly_on_clock((day >= HEAVY_MM).astype("float32"), "sum"))
        rx1.append(monthly_on_clock(day, "max"))
        print(f"  heavy rain {y}", flush=True)
    r20, rx1 = np.fmax.reduce(r20), np.fmax.reduce(rx1)
    if r20.shape[1] != n_cells:
        raise ValueError(f"heavy-rain cells {r20.shape[1]} != tp cells {n_cells}")
    return dict(r20=r20, rx1=rx1)


def soil_moisture(lat2, lon2):
    """ERA5 monthly volumetric soil water: sm1 = layer 1 (0-7 cm), sm3 = layer 3 (28-100 cm)."""
    files = sorted((RAW / "era5_soil_moisture").glob("*.nc"))
    if not files:
        print("\n!!! SOIL MOISTURE NOT INCLUDED: no raw_data/era5_soil_moisture/*.nc\n")
        return {}
    with xr.open_dataset(files[0]) as ds:
        out = {k: monthly_on_clock(ds[v].clip(min=0).transpose("valid_time", "latitude", "longitude"), "mean")
               for k, v in (("sm1", "swvl1"), ("sm3", "swvl3"))}
        out["sm_idx"] = nearest(ds.latitude.values, ds.longitude.values, lat2, lon2)
    print("  ERA5 soil moisture", flush=True)
    return out


def atmosphere(lat2, lon2):
    """ERA5 monthly means per 0.25 deg cell: t2m and d2m (deg C), si10 (m/s), ssrd (MJ/m2 per day)."""
    files = sorted((RAW / "era5_atmosphere").glob("*.nc"))
    if not files:
        print("\n!!! ATMOSPHERE NOT INCLUDED: no raw_data/era5_atmosphere/*.nc\n")
        return {}
    out = {}
    f = files[0]
    import zipfile
    if zipfile.is_zipfile(f):                   # CDS zips variables from different streams, even when named .nc
        d = f.parent / f.stem
        with zipfile.ZipFile(f) as z:
            z.extractall(d)
        ds = xr.merge([xr.open_dataset(p) for p in sorted(d.glob("*.nc"))], compat="override", join="outer")
    else:
        ds = xr.open_dataset(f)
    with ds:
        for k, f in (("t2m", lambda a: a - 273.15), ("d2m", lambda a: a - 273.15), ("si10", lambda a: a),
                     ("ssrd", lambda a: a / 1e6)):
            if k not in ds:
                print(f"  !! atmosphere: {k} not in {files[0].name} (variables: {list(ds.data_vars)})")
                continue
            out[k] = monthly_on_clock(f(ds[k]).transpose("valid_time", "latitude", "longitude"), "mean")
        out["atm_idx"] = nearest(ds.latitude.values, ds.longitude.values, lat2, lon2)
    print(f"  atmosphere: {[k for k in out if k != 'atm_idx']}", flush=True)
    return out


def rain_shape_month(d):
    """Shape of one month of daily rain d [days, cells]: rain days, longest wet spell, share of the month's
    rain in its 5 wettest days, day-to-day SD."""
    wet = d >= WET_MM
    run = np.zeros(d.shape[1], np.float32)
    best = np.zeros(d.shape[1], np.float32)
    for w in wet:
        run = (run + 1) * w
        best = np.maximum(best, run)
    tot = d.sum(0)
    top5 = np.sort(d, 0)[-5:].sum(0)
    return {"wetd": wet.sum(0).astype(np.float32), "wspell": best,
            "conc5": np.where(tot >= 1, top5 / np.maximum(tot, 1e-6), np.nan).astype(np.float32),
            "dsd": d.std(0).astype(np.float32)}


def upstream_matrix(ids, next_down, area):
    """Sparse [basin, basin] matrix: row b holds the area of every basin draining into b (b included)."""
    from scipy.sparse import csr_matrix
    pos = {int(i): k for k, i in enumerate(ids)}
    children = {}
    for k, nd in enumerate(next_down):
        if int(nd) in pos:
            children.setdefault(pos[int(nd)], []).append(k)
    rows, cols = [], []
    for b in range(len(ids)):
        stack, seen = [b], {b}
        while stack:
            u = stack.pop()
            for c in children.get(u, []):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        rows += [b] * len(seen)
        cols += list(seen)
    return csr_matrix((np.asarray(area, float)[cols], (rows, cols)), shape=(len(ids), len(ids)))


def rain_extra(lat2, lon2, n_cells, res):
    """From the hourly ERA5 files (one pass):
    shape of each month's daily rain per local cell (wetd, wspell, conc5, dsd; same cells as tp, so era5_idx
    applies), and uprain = mean rain over each square's whole upstream catchment, routed along HydroBASINS
    level-7 sub-basins (raw_data/hydrobasins/hybas_af_lev07*.shp; skipped if absent). The ERA5 files end at
    3 S, so the far south of the Lake Victoria catchment is not covered."""
    shape_out, full, glat, glon = None, None, None, None
    for y in ERA5_YEARS:
        ds = xr.open_dataset(RAW / "rainfall and runoff" / f"ERA5_{y}.nc", chunks={"valid_time": 744})["tp"]
        day = (ds.resample(valid_time="1D").sum() * 1000.0).load()
        glat, glon = day.latitude.values, day.longitude.values
        D = day.values.astype(np.float32)
        per = pd.PeriodIndex(pd.to_datetime(day.valid_time.values), freq="M")
        li = (glat <= lat2.max() + 0.25) & (glat >= lat2.min() - 0.25)
        lj = (glon >= lon2.min() - 0.25) & (glon <= lon2.max() + 0.25)
        L = D[:, li][:, :, lj].reshape(len(D), -1)
        if L.shape[1] != n_cells:
            raise ValueError(f"rain-shape cells {L.shape[1]} != tp cells {n_cells}")
        if shape_out is None:
            shape_out = {k: np.full((len(MONTHS), n_cells), np.nan, np.float32) for k in ("wetd", "wspell", "conc5", "dsd")}
            full = np.full((len(MONTHS), glat.size, glon.size), np.nan, np.float32)
        for p in per.unique():
            if p not in MONTHS:
                continue
            i, sel = MONTHS.get_loc(p), per == p
            for k, v in rain_shape_month(L[sel]).items():
                shape_out[k][i] = v
            full[i] = D[sel].sum(0)
        print(f"  rain shape {y}", flush=True)
    out = dict(shape_out)
    shp = sorted((RAW / "hydrobasins").rglob("hybas_af_lev07*.shp"))
    if not shp:
        print("\n!!! UPSTREAM-CATCHMENT RAIN NOT INCLUDED: no raw_data/hydrobasins/hybas_af_lev07*.shp\n")
        return out
    import geopandas as gpd
    from scipy.sparse import csr_matrix
    b = gpd.read_file(shp[0], bbox=(float(glon.min()) - 1, float(glat.min()) - 1, float(glon.max()) + 1, float(glat.max()) + 1))
    b.columns = [c if c == "geometry" else c.upper() for c in b.columns]     # field case differs between releases
    nb = len(b)
    # basins rasterised on a 0.05 deg grid aligned with the ERA5 cells (5 x 5 fine pixels per cell)
    f = 5
    H, W = glat.size * f, glon.size * f
    tr = from_origin(float(glon.min()) - 0.125, float(glat.max()) + 0.125, 0.25 / f, 0.25 / f)
    lab = rasterize(((g, k + 1) for k, g in enumerate(b.geometry)), out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    rr, cc = np.nonzero(lab)
    W_ = csr_matrix((np.ones(rr.size), (lab[rr, cc] - 1, (rr // f) * glon.size + cc // f)), shape=(nb, glat.size * glon.size))
    cnt = np.asarray(W_.sum(1)).ravel()
    flat = np.nan_to_num(full.reshape(len(MONTHS), -1)).T                        # [cells, months]
    R = (W_ @ flat) / np.maximum(cnt, 1)[:, None]                                  # [basin, month] mean rain
    area = b["SUB_AREA"].to_numpy(float) * (cnt > 0)                              # basins outside ERA5 get no weight
    A = upstream_matrix(b["HYBAS_ID"].to_numpy(), b["NEXT_DOWN"].to_numpy(), area)
    up = (A @ R) / np.maximum(np.asarray(A.sum(1)).ravel(), 1e-6)[:, None]       # A already holds basin areas
    up[:, np.isnan(full).all((1, 2))] = np.nan                                      # months without ERA5
    up = np.vstack([up, np.full((1, len(MONTHS)), np.nan)]).T.astype(np.float32)    # last column: outside all basins
    sq = rasterize(((g, k + 1) for k, g in enumerate(b.geometry)), out_shape=(lat2.size, lon2.size),
                   transform=from_origin(lon2[0] - res / 2, lat2[0] + res / 2, res, res), fill=0, dtype="int32")
    out.update(uprain=up, uprain_idx=np.where(sq > 0, sq - 1, nb).astype(np.int32))
    print(f"  upstream-catchment rain: {nb} sub-basins, {(sq > 0).mean():.0%} of squares inside one", flush=True)
    return out


# ==================== RIVERS & STORAGE ========================================================================


def course_series():
    """Lake levels and Dartmouth discharge, monthly, via the course loaders."""
    sys.path.insert(0, str(ROOT / "1_data_processing" / "2_loaders"))
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


def grace(lat2, lon2):
    """GRACE/GRACE-FO JPL mascon water storage anomaly (cm, scale factor applied), per 0.5 deg cell.
    Months without a solution (incl. the 2017-18 mission gap) stay NaN."""
    files = sorted((RAW / "grace").glob("*.nc"))
    if not files:
        print("\n!!! GRACE NOT INCLUDED: no raw_data/grace/*.nc\n")
        return {}
    with xr.open_dataset(files[0]) as ds:
        a = (ds["lwe_thickness"] * ds["scale_factor"]).sel(
            lat=slice(lat2.min() - 1, lat2.max() + 1), lon=slice(lon2.min() - 1, lon2.max() + 1))
        vals = a.transpose("time", "lat", "lon").values.astype(np.float32)
        glat, glon = a.lat.values, a.lon.values
        per = pd.PeriodIndex(pd.to_datetime(ds.time.values), freq="M")
    out = np.full((len(MONTHS), glat.size * glon.size), np.nan, np.float32)
    for p in per.unique():
        if p in MONTHS:
            out[MONTHS.get_loc(p)] = np.nanmean(vals[per == p], 0).ravel()
    print(f"  GRACE: {np.isfinite(out).any(1).sum()} of {len(MONTHS)} months", flush=True)
    return dict(grace=out, grace_idx=nearest(glat, glon, lat2, lon2))


def hydroweb(lat2, lon2):
    """Hydroweb river water levels (satellite altimetry): monthly mean orthometric height per station.
    Kept: Nile-basin stations with >= HW_MIN_OBS measurements whose record starts by HW_START_BY (so a calendar-month
    mean can be computed from training years). Months without a pass stay NaN, single-month gaps are interpolated.
    hw [month, station], hw_idx [row, col] = nearest kept station, hw_dist [row, col] = its distance (km)."""
    files = sorted((RAW / "hydroweb").rglob("hydroprd_R_*.txt"))
    if not files:
        print("\n!!! HYDROWEB NOT INCLUDED: no raw_data/hydroweb/**/hydroprd_R_*.txt\n")
        return {}
    series, lat_s, lon_s, names = [], [], [], []
    for f in files:
        head, dates, h = {}, [], []
        for line in f.read_text(encoding="utf8", errors="replace").splitlines():
            if line.startswith("#"):
                k, sep, v = line[1:].partition("::")
                if sep:
                    head[k.strip().upper()] = v.strip()
            else:
                p = line.split()
                if len(p) >= 3 and len(p[0]) == 10 and p[0][4] == "-":
                    dates.append(p[0])
                    h.append(float(p[2]))
        if head.get("BASIN") != "NILE" or len(dates) < HW_MIN_OBS or min(dates) > HW_START_BY:
            continue
        s = pd.Series(h, index=pd.PeriodIndex(pd.to_datetime(dates), freq="M"))
        s = s[s.between(s.median() - 20, s.median() + 20)]               # drop gross altimetry outliers
        m = s.groupby(level=0).mean().reindex(MONTHS).interpolate(limit=1, limit_area="inside")
        series.append(m.to_numpy(np.float32))
        lat_s.append(float(head["REFERENCE LATITUDE"]))
        lon_s.append(float(head["REFERENCE LONGITUDE"]))
        names.append(f.stem.replace("hydroprd_R_NILE_", "").replace("_exp", ""))
    if not series:
        print("\n!!! HYDROWEB NOT INCLUDED: no Nile station passed the record-length rules\n")
        return {}
    lat_s, lon_s = np.array(lat_s), np.array(lon_s)
    best = np.full((lat2.size, lon2.size), np.inf, np.float32)
    idx = np.zeros((lat2.size, lon2.size), np.int32)
    for k in range(lat_s.size):
        d = np.hypot((lat2[:, None] - lat_s[k]) * 111.0,
                     (lon2[None, :] - lon_s[k]) * 111.0 * np.cos(np.radians(lat2[:, None])))
        better = d < best
        best = np.where(better, d, best)
        idx = np.where(better, k, idx)
    print(f"  Hydroweb: {len(names)} Nile stations kept: {', '.join(names)}", flush=True)
    return dict(hw=np.stack(series, 1), hw_idx=idx, hw_dist=best.astype(np.float32))


# ==================== LAND ====================================================================================


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


def terrain(lat2, lon2, res):
    """From MERIT Hydro elevation (and upstream area), per 90 m pixel, then averaged per square:
    slope  mean slope (degrees)
    twi    mean topographic wetness index ln(a / tan(slope)), a = upstream area / pixel width
           (Beven & Kirkby 1979); slope floored at SLOPE_MIN_DEG."""
    elv, upa = _tiles("elv", lat2, lon2, res), _tiles("upa", lat2, lon2, res)
    if not elv:
        print("\n!!! TERRAIN NOT INCLUDED: no *_elv.tif over South Sudan under raw_data/MERITHydro\n")
        return {}
    out = {k: np.full((lat2.size, lon2.size), np.nan, np.float32) for k in ("slope", "twi")}
    for name, f in elv.items():
        with rasterio.open(f) as src:
            z = src.read(1).astype(np.float32)
            tr, crs, d = src.transform, src.crs, abs(src.res[0])
        z[z <= -9999] = np.nan
        rows_lat = tr.f + (np.arange(z.shape[0]) + 0.5) * tr.e
        dy = d * 111320.0
        dx = (d * 111320.0 * np.cos(np.radians(rows_lat)))[:, None]
        gy, gx = np.gradient(z)
        slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy))).astype(np.float32)
        del gy, gx
        g = _to_grid(slope, tr, crs, lat2, lon2, res)
        out["slope"] = np.where(np.isnan(g), out["slope"], g)
        if name in upa:
            with rasterio.open(upa[name]) as src:
                u = src.read(1).astype(np.float32)
            if u.shape == z.shape:
                u[u <= -9999] = np.nan
                a = u * 1e6 / dx                                   # km2 -> m2, per metre of pixel width
                tanb = np.tan(np.radians(np.maximum(slope, SLOPE_MIN_DEG)))
                g = _to_grid(np.log(a / tanb), tr, crs, lat2, lon2, res)
                out["twi"] = np.where(np.isnan(g), out["twi"], g)
            else:
                print(f"  !! {name}: upa and elv shapes differ, no wetness index for this tile")
        print(f"  terrain {name}", flush=True)
    return {k: v for k, v in out.items() if not np.isnan(v).all()}


def heterogeneity(lat2, lon2, res):
    """Within-square spread from the 90 m MERIT pixels: hnd_q1 (25th percentile of HAND, m), hnd_sd (SD of HAND),
    slope_sd (SD of slope, deg), depress (share of pixels > DEPRESS_M below the mean elevation of the surrounding
    ~1 km, i.e. local depressions)."""
    from scipy.ndimage import uniform_filter as uf
    out = {k: np.full((lat2.size, lon2.size), np.nan, np.float32) for k in ("hnd_q1", "hnd_sd", "slope_sd", "depress")}
    put = lambda k, g: out.__setitem__(k, np.where(np.isnan(g), out[k], g))   # noqa: E731
    sd = lambda m, m2: np.sqrt(np.clip(m2 - m * m, 0, None))                    # noqa: E731
    for name, f in _tiles("hnd", lat2, lon2, res).items():
        with rasterio.open(f) as src:
            a = src.read(1).astype(np.float32)
            tr, crs = src.transform, src.crs
        a[a <= -9999] = np.nan
        put("hnd_q1", _to_grid(a, tr, crs, lat2, lon2, res, Resampling.q1))
        put("hnd_sd", sd(_to_grid(a, tr, crs, lat2, lon2, res), _to_grid(a * a, tr, crs, lat2, lon2, res)))
        print(f"  heterogeneity hnd {name}", flush=True)
    for name, f in _tiles("elv", lat2, lon2, res).items():
        with rasterio.open(f) as src:
            z = src.read(1).astype(np.float32)
            tr, crs, d = src.transform, src.crs, abs(src.res[0])
        z[z <= -9999] = np.nan
        rows_lat = tr.f + (np.arange(z.shape[0]) + 0.5) * tr.e
        dx = (d * 111320.0 * np.cos(np.radians(rows_lat)))[:, None]
        gy, gx = np.gradient(z)
        s = np.degrees(np.arctan(np.hypot(gx / dx, gy / (d * 111320.0)))).astype(np.float32)
        del gy, gx
        put("slope_sd", sd(_to_grid(s, tr, crs, lat2, lon2, res), _to_grid(s * s, tr, crs, lat2, lon2, res)))
        zf = np.where(np.isnan(z), np.nanmean(z), z)
        dep = np.where(np.isnan(z), np.nan, (z < uf(zf, size=11) - DEPRESS_M).astype(np.float32))
        put("depress", _to_grid(dep, tr, crs, lat2, lon2, res))
        print(f"  heterogeneity elv {name}", flush=True)
    return {k: v for k, v in out.items() if not np.isnan(v).all()}


def soil(lat2, lon2, res):
    """SoilGrids 250 m: clay and sand (%), thickness-weighted over 0-30 cm; Vertisol probability (%)."""
    d = RAW / "soilgrids"
    if not d.exists():
        print("\n!!! SOIL NOT INCLUDED: no raw_data/soilgrids\n")
        return {}

    def read(f):
        with rasterio.open(f) as src:
            a = src.read(1).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            return _to_grid(a, src.transform, src.crs, lat2, lon2, res)

    out = {}
    for p in ("clay", "sand"):
        layers = [(read(d / f"{p}_{z}_mean.tif"), w) for z, w in (("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15))]
        out[p] = (sum(a * w for a, w in layers) / 30.0 / 10.0).astype(np.float32)   # g/kg -> %
        print(f"  soil {p}", flush=True)
    out["vertisol"] = read(d / "wrb_Vertisols.tif")
    print("  soil vertisol", flush=True)
    return out


# ==================== CLIMATE INDICES =========================================================================


def climate_indices():
    """Monthly Nino3.4 anomaly (CPC sstoi.indices, ERSST) and Dipole Mode Index (CPC)."""
    d = RAW / "climate_indices"
    n = pd.read_csv(d / "sstoi.indices", sep=r"\s+")
    if n.columns[8] != "NINO3.4":
        raise ValueError(f"unexpected sstoi.indices columns {list(n.columns)}")
    per = lambda yy, mm: pd.PeriodIndex(pd.to_datetime(pd.DataFrame({"year": yy, "month": mm, "day": 1})), freq="M")  # noqa: E731
    nino = pd.Series(n.iloc[:, 9].to_numpy(float), index=per(n.YR, n.MON))
    m = pd.read_csv(d / "dmi.csv")
    dmi = pd.Series(m.dmi.to_numpy(float), index=per(m.year, m.month))
    df = pd.DataFrame({"nino34": nino, "dmi": dmi}).reindex(MONTHS)
    df.index = df.index.astype(str)
    return df


# ==================== SCORING ONLY ============================================================================


def cropland(lat2, lon2, res):
    """Mean ASAP crop % per square (non-crop and no-data count as 0). For scoring and sampling only, never an input."""
    from rasterio.windows import from_bounds
    with rasterio.open(CROP) as src:
        win = from_bounds(lon2[0] - res, lat2[-1] - res, lon2[-1] + res, lat2[0] + res, src.transform)
        win = win.round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32)
        bad = (a < 0) | (a > 100) | ~np.isfinite(a)
        if src.nodata is not None:
            bad |= a == src.nodata
        a[bad] = 0
        out = np.zeros((lat2.size, lon2.size), np.float32)
        reproject(a, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=-1,
                  dst_transform=from_origin(lon2[0] - res / 2, lat2[0] + res / 2, res, res),
                  dst_crs="EPSG:4326", dst_nodata=-1, resampling=Resampling.average)
    print("  cropland mask", flush=True)
    return dict(crop_pct=np.clip(out, 0, 100))


# ==============================================================================================================

ADD_STEPS = ["target", "glofas", "merit", "terrain", "soil", "grace", "sm", "heavy", "crop", "indices", "hydroweb",
             "atmos", "rainx", "hetero"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--add", nargs="+", choices=ADD_STEPS,
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
        if "target" in steps:                 # rebuild only the target (same grid), keep every input
            y, nbr, tl, tn = target(res)
            if tl.size != lat2.size or tn.size != lon2.size:
                raise ValueError("target grid differs from the panel grid: rebuild the whole panel")
            arrays.update(y=y.astype(np.float16), nbr=nbr.astype(np.float16))
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
        steps = ADD_STEPS

    if "glofas" in steps:
        g_arrays, q_pts = glofas(lat2, lon2)
        if g_arrays:
            arrays.update(g_arrays)
            shared = shared.join(q_pts)
    if "merit" in steps:
        arrays.update(merit(lat2, lon2, res))
    if "terrain" in steps:
        arrays.update(terrain(lat2, lon2, res))
    if "soil" in steps:
        arrays.update(soil(lat2, lon2, res))
    if "grace" in steps:
        arrays.update(grace(lat2, lon2))
    if "sm" in steps:
        arrays.update(soil_moisture(lat2, lon2))
    if "heavy" in steps:
        arrays.update(heavy_rain(lat2, lon2, arrays["tp"].shape[1]))
    if "crop" in steps:
        arrays.update(cropland(lat2, lon2, res))
    if "hydroweb" in steps:
        arrays.update(hydroweb(lat2, lon2))
    if "atmos" in steps:
        arrays.update(atmosphere(lat2, lon2))
    if "rainx" in steps:
        arrays.update(rain_extra(lat2, lon2, arrays["tp"].shape[1], res))
    if "hetero" in steps:
        arrays.update(heterogeneity(lat2, lon2, res))
    if "indices" in steps:
        shared = shared[[c for c in shared.columns if c not in ("nino34", "dmi")]].join(climate_indices())
    shared.to_csv(out / "shared.csv")
    np.savez_compressed(out / "inputs.npz", **arrays)
    print(f"\nwrote {out}: {len(arrays)} arrays ({', '.join(sorted(arrays))})")
    print("shared series, months with data:")
    print(shared.notna().sum().to_string())


if __name__ == "__main__":
    main()
