"""GFM (Copernicus Global Flood Monitoring, Sentinel-1 radar) evidence for the report.

GFM was tested as the flood target and failed: it shows almost no water before
2022, including the 2019-2022 record flood years. This script regenerates every
GFM number we quote. Output of the last full run: output/gfm_verdict.txt.

    python flood_pred_modelling/gfm_check.py meta     # A. national series from GFM's own metadata, offline, seconds
    python flood_pred_modelling/gfm_check.py sites    # B. pixel truth at four Sudd sites

A needs raw_data/gfm/stac/ (from gfm_inventory.py). B reads rasters from EODC
over the network; every read is cached under processed_data/gfm/cache, so
reruns reuse it.

Three traps. Each produced an invalid result before it was caught:
- One Sentinel-1 acquisition becomes one STAC item per Equi7 tile, and each of
  those items repeats the same swath-level flooded/floodable pixel counts.
  Deduplicate to acquisitions before aggregating; never divide by a tile count.
- Rasters are on the Equi7 grid (projected metres). A lon/lat box must be
  reprojected and clipped to the raster, or the read silently returns nothing.
- GFM writes pixels it cannot judge (exclusion mask) as 0, i.e. as dry. B
  removes them from the denominator.
"""
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Without these, every remote raster read lists directories and refetches
# headers, which dominates the run time. Set before rasterio loads GDAL.
for _k, _v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                   VSI_CACHE="TRUE", VSI_CACHE_SIZE="10000000").items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pystac_client import Client  # noqa: E402
from rasterio.errors import WindowError  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import Window, from_bounds, intersection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAC = ROOT / "raw_data" / "gfm" / "stac"
CACHE = ROOT / "processed_data" / "gfm" / "cache"
OUT = Path(__file__).resolve().parent / "output"
API = "https://stac.eodc.eu/api/v1"
WORKERS = 12

_local = threading.local()


def catalog():
    """One STAC client per thread: pystac_client is not thread-safe."""
    if not hasattr(_local, "cat"):
        _local.cat = Client.open(API)
    return _local.cat


def retry(fn, *args):
    """Call fn, once more on an exception (EODC drops connections under load)."""
    for _ in range(2):
        try:
            return fn(*args)
        except Exception:
            time.sleep(1)
    return None


# ------------------------------------------------------- A. metadata series

ACQ = re.compile(r"^(.*?)_[EW]\d+[NS]\d+T\d+$")   # item id minus its Equi7 tile suffix


def meta():
    """A: national flooded share per year and per processor version, metadata only."""
    frames = [pd.read_parquet(f) for f in sorted(STAC.glob("*.parquet"))]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df["year"] = pd.to_datetime(df["datetime"], format="mixed", utc=True).dt.year
    df["acq"] = df["item_id"].str.replace(ACQ, r"\1", regex=True)
    df["nrt"] = ~df["processing:software"].astype(str).str.contains("archive")

    # The trap: counts must not vary within one acquisition.
    v = df.groupby("acq")[["flooded_pixels", "floodable_pixels"]].nunique().max()
    print(f"{len(df)} tile items -> {df.acq.nunique()} acquisitions. Distinct counts within "
          f"an acquisition (must be 1/1): {v.flooded_pixels}/{v.floodable_pixels}")

    d = df.drop_duplicates("acq")
    g = d.groupby("year")
    print("\nper year, one row per acquisition (record flood years 2019-2021 should NOT be lowest):")
    print(pd.DataFrame({
        "acquisitions": g.size(),
        "flooded_pct": (g.flooded_pixels.sum() / g.floodable_pixels.sum() * 100).round(4),
        "floodable_per_acq_M": (g.floodable_pixels.mean() / 1e6).round(1),
        "pct_nrt": (g.nrt.mean() * 100).round(0),
    }).to_string())

    g = d.groupby("processing:version")
    print("\nper processor version (the archive is one processing for 2015-2023):")
    print(pd.DataFrame({
        "acquisitions": g.size(),
        "years": g.year.agg(lambda s: f"{s.min()}-{s.max()}"),
        "flooded_pct": (g.flooded_pixels.sum() / g.floodable_pixels.sum() * 100).round(4),
    }).to_string())


# ------------------------------------------------------------- B. Sudd sites
#
# Fixed places, only time varies. In the last run only Bentiu and Bor had
# usable scenes; site-months without one are left out, never scored as dry.

SITES = {"Bentiu": (29.80, 9.26),       # Unity
         "OldFangak": (30.87, 9.07),    # Jonglei
         "Bor": (31.56, 6.21),          # Jonglei
         "Malakal": (31.66, 9.53)}      # Upper Nile
HALF = 0.15                             # degrees: a ~33 km box around each site
YEARS = range(2016, 2026)
# March: driest. August: rain peak. October: inside the Oct-Dec Sudd flood
# season (Hardy et al. 2023).
MONTHS = {"dry": 3, "wet": 8, "peak": 10}
SCENES_PER_MONTH = 3
MAX_TRIES = 12
MIN_VALID = 0.30                        # at least 30% of the box must be judgeable


def read_window(href, bounds, key):
    """Read a lon/lat box from a remote GFM raster, cached as npz. None if it misses the raster."""
    f = CACHE / f"{key}.npz"
    if f.exists():
        try:
            return np.load(f)["a"]
        except Exception:
            f.unlink(missing_ok=True)
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bounds)
        w = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        try:
            w = intersection(w, Window(0, 0, src.width, src.height))
        except WindowError:              # disjoint windows raise, they do not come back empty
            return None
        if w.width <= 0 or w.height <= 0:
            return None
        a = src.read(1, window=w)
    np.savez_compressed(f, a=a)
    return a


def scene(item, bounds):
    """Water % of the judgeable pixels in one scene, or None if the box is not judgeable.

    Values: 0 dry, 1 water, 255 no data. Excluded pixels are written as 0, so
    they are removed from the denominator here.
    """
    a = item.assets
    if "ensemble_water_extent" not in a:
        return None
    ew = read_window(a["ensemble_water_extent"].href, bounds, f"{item.id}__wat")
    if ew is None:
        return None
    covered = ew != 255
    if covered.mean() < 0.01:
        return None                      # outside the swath, not dry
    valid = covered
    if "exclusion_mask" in a:
        ex = read_window(a["exclusion_mask"].href, bounds, f"{item.id}__exc")
        if ex is not None and ex.shape == ew.shape:
            valid = covered & (ex == 0)
    if valid.mean() < MIN_VALID:
        return None
    soft = str(item.properties.get("processing:software", ""))
    return {"item": item.id, "proc": "archive" if "archive" in soft else "nrt",
            "water_pct": float((ew[valid] == 1).mean() * 100)}


def search(bbox, year, month):
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    return list(catalog().search(collections=["GFM"], bbox=bbox,
                                 datetime=f"{year}-{month:02d}-01/{end:%Y-%m-%d}").items())


def site_month(site, lon, lat, year, season, month):
    """Up to SCENES_PER_MONTH judgeable scenes for one site and month."""
    bounds = (lon - HALF, lat - HALF, lon + HALF, lat + HALF)
    try:
        items = search(bounds, year, month)
    except Exception as e:
        print(f"  search failed {site} {year}-{month:02d}: {str(e)[:60]}")
        return []
    items.sort(key=lambda it: it.id)    # fixed order: the sample must not depend on the server
    rows = []
    for it in items[:MAX_TRIES]:
        r = retry(scene, it, bounds)
        if r is not None:
            rows.append({**r, "site": site, "year": year, "season": season})
        if len(rows) == SCENES_PER_MONTH:
            break
    return rows


def sites():
    """B: water % at fixed Sudd sites in March, August and October, 2016-2025."""
    jobs = [(s, lon, lat, y, season, m) for s, (lon, lat) in SITES.items()
            for y in YEARS for season, m in MONTHS.items()]
    rows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(lambda j: site_month(*j), jobs), 1):
            rows.extend(res)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} site-months", flush=True)
    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("no usable scene anywhere")
    r.to_csv(OUT / "gfm_scenes.csv", index=False)
    filled = r.groupby(["site", "year", "season"]).ngroups
    print(f"{len(r)} usable scenes. {len(jobs) - filled} of {len(jobs)} site-months had none.")

    for season, m in MONTHS.items():
        print(f"\nwater % of judgeable pixels, {season} (month {m}):")
        print(r[r.season == season].pivot_table(index="year", columns="site",
                                                values="water_pct").round(2).to_string())

    # Test 1. The record flood years should be among the wettest: OCHA reports
    # record numbers affected in 2019-2021, Hardy et al. 2023 the largest Sudd
    # extents of 1984-2022 in 2019-2022.
    octo = (r[r.season == "peak"].pivot_table(index="year", columns="site", values="water_pct")
            .mean(axis=1).dropna())
    print("\nTEST 1. October water %, mean over sites, rank 1 = wettest. "
          "2019-2022 should rank at the top:")
    print(pd.DataFrame({"water_pct": octo.round(2),
                        "rank": octo.rank(ascending=False).astype(int)}).to_string())

    # Test 2. Seasonality per site-year, so a missing month cannot bias it.
    w = r.pivot_table(index=["site", "year"], columns="season", values="water_pct")
    for a, b in (("wet", "dry"), ("peak", "dry")):
        if {a, b} <= set(w.columns):
            ratio = (w[a] / w[b]).replace([np.inf, -np.inf], np.nan).dropna()
            print(f"TEST 2. {a}/{b}: median {ratio.median():.2f} over {len(ratio)} "
                  f"site-years. Should be clearly > 1.")


if __name__ == "__main__":
    modes = {"meta": meta, "sites": sites}
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in modes:
        sys.exit(f"usage: python flood_pred_modelling/gfm_check.py [{' | '.join(modes)}]")
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    modes[mode]()
