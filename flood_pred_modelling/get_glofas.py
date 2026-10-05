"""Download GloFAS-ERA5 daily river discharge, South Sudan, as NetCDF (no ecCodes needed).

Needs ~/.cdsapirc:
    url: https://ewds.climate.copernicus.eu/api
    key: <EWDS token>
and the licence accepted on the cems-glofas-historical dataset page.

Resumable: rerun to retry failures.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi

DEST = Path(__file__).resolve().parent.parent / "raw_data" / "glofas"
DEST.mkdir(parents=True, exist_ok=True)

YEARS = [str(y) for y in range(2017, 2027)]     # the model's input clock starts Aug 2017
CHUNK = 1        # years per request; lower if rejected as too large
WORKERS = 10      # parallel requests

REQUEST = {
    "system_version": ["version_4_0"],
    "hydrological_model": ["lisflood"],
    "product_type": ["consolidated"],# "intermediate"],
    "variable": ["average_river_discharge_in_the_last_24_hours"],
    "timespan": ["time_mean"],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [13.0, 23.5, 3.0, 36.5],      # N, W, S, E
}


def fetch(years):
    out = DEST / f"glofas_{years[0]}_{years[-1]}.nc"
    if out.exists() and out.stat().st_size:
        return print(f"skip {out.name}")
    try:
        cdsapi.Client().retrieve(
            "cems-glofas-historical", {**REQUEST, "year": years}
        ).download(str(out))
        print(f"ok {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    except Exception as exc:
        print(f"FAIL {out.name}: {exc}")


chunks = [YEARS[i:i + CHUNK] for i in range(0, len(YEARS), CHUNK)]
with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(fetch, chunks))