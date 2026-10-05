"""CYGNSS (L-band GNSS-R) Sudd inundation: does it pass the tests GFM failed?

Product: UC Berkeley CYGNSS L3 Monthly RWAWC Watermask v3.1 (NASA PO.DAAC),
0.01 deg, monthly, Aug 2018 -> present. watermask: 0 land, 1 water, -99 no data.

Two tests, read from the printed table:
  1. the record years 2019-2022 should rank at the top for Oct-Dec water
     (Hardy et al. 2023: largest Sudd extents of 1984-2022, 2022 the largest)
  2. Oct-Dec (flood peak) / Mar-Apr (dry) should be clearly > 1

Needs a free NASA Earthdata account (urs.earthdata.nasa.gov). The first run asks
for username and password and stores them in your home folder. Rerunning skips
files already downloaded.

    pip install earthaccess
    python flood_pred_modelling/get_cygnss.py
"""
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "cygnss"
OUT = Path(__file__).resolve().parent / "output"
SHORT = "CYGNSS_L3_UC_BERKELEY_WATERMASK_V3.1"
SUDD = (29.0, 6.0, 32.5, 10.0)       # lon_min, lat_min, lon_max, lat_max; same box as the GFM work


def granule_start(g):
    """Start date of a granule from its NASA metadata, so months are not guessed from file names."""
    te = g["umm"]["TemporalExtent"]
    return pd.Timestamp(te.get("RangeDateTime", {}).get("BeginningDateTime")
                        or te.get("SingleDateTime"))


def sudd_water_pct(path):
    """% of Sudd-box cells classified as water, and how many cells had data."""
    with xr.open_dataset(path, decode_times=False) as ds:   # time units are 'months since', which xarray cannot decode; month comes from metadata
        v = ds["watermask"]
        lat = next((d for d in v.dims if "lat" in d.lower()), None)
        lon = next((d for d in v.dims if "lon" in d.lower()), None)
        if lat is None or lon is None:
            raise SystemExit(f"unexpected dimensions {v.dims} in {path.name}: check the print above")
        la = ds[lat].to_numpy()
        lat_slice = slice(SUDD[1], SUDD[3]) if la[0] < la[-1] else slice(SUDD[3], SUDD[1])
        a = v.sel({lon: slice(SUDD[0], SUDD[2]), lat: lat_slice}).to_numpy()
    ok = a >= 0                          # -99 = no data
    return (float((a[ok] == 1).mean() * 100) if ok.any() else np.nan), int(ok.sum())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    earthaccess.login(persist=True)
    granules = earthaccess.search_data(short_name=SHORT, bounding_box=SUDD)
    print(f"{len(granules)} monthly files")
    earthaccess.download(granules, str(RAW))

    start = {Path(link).name: granule_start(g) for g in granules for link in g.data_links()}
    files = sorted(p for p in RAW.glob("*.nc") if p.name in start)
    if not files:
        raise SystemExit(f"no downloaded CYGNSS files in {RAW}")
    with xr.open_dataset(files[0], decode_times=False) as ds:
        print(ds)                        # the real variable and dimension names, once

    rows = []
    for f in files:
        pct, n = sudd_water_pct(f)
        t = start[f.name]
        rows.append({"file": f.name, "year": t.year, "month": t.month,
                     "water_pct": pct, "cells": n})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cygnss_sudd_monthly.csv", index=False)

    print("\n% of the Sudd box classified as water, year x month:")
    print(d.pivot_table(index="year", columns="month", values="water_pct").round(1).to_string())

    peak = d[d.month.isin([10, 11, 12])].groupby("year").water_pct.mean()
    dry = d[d.month.isin([3, 4])].groupby("year").water_pct.mean()
    print("\nTEST 1: 2019-2022 should rank top.   TEST 2: peak/dry clearly > 1.")
    print(pd.DataFrame({"peak_OctDec": peak, "dry_MarApr": dry, "peak/dry": peak / dry,
                        "rank_peak": peak.rank(ascending=False)}).round(2).to_string())


if __name__ == "__main__":
    main()
