"""Download every source except CYGNSS and GloFAS (they have their own scripts). Rerunning skips what is there.

    python flood_pred_modelling/1_download/get_covariates.py                 (all steps)
    python flood_pred_modelling/1_download/get_covariates.py --only soil grace

  indices   ENSO (Nino3.4 anomaly, CPC sstoi.indices) and IOD (DMI, CPC, ERSSTv6)   -> raw_data/climate_indices/
  grace     GRACE/GRACE-FO JPL mascon CRI RL06.3 v4 (NASA Earthdata login)          -> raw_data/grace/
  soil      SoilGrids 250 m clay and sand (0-30 cm), Vertisol probability           -> raw_data/soilgrids/
  era5sm    ERA5 monthly soil water, layers 1-4, 2017-2025 (CDS)                    -> raw_data/era5_soil_moisture/
  atmos     ERA5 monthly 2 m temperature, dewpoint, 10 m wind, solar radiation (CDS) -> raw_data/era5_atmosphere/
  jrc       JRC Global Surface Water occurrence tiles (permanent-water mask)         -> raw_data/jrc_gsw/
  hydroweb  Hydroweb river water levels (needs HYDROWEB_API_KEY)                     -> raw_data/hydroweb/
  merit     checks that the MERIT Hydro elevation packages are present

CDS steps: the ERA5 licence must be accepted once on the CDS website; the key in ~/.cdsapirc is used.
Manual downloads (registration or licence click), unpacked into raw_data/:
  MERIT Hydro hnd, upa, elv packages n00e000 and n00e030      -> raw_data/MERITHydro/<variable>_<tile>/
  HydroBASINS Standard, Africa (level 7 is used)              -> raw_data/hydrobasins/
Daily rain statistics (heavy-rain days, rain shape) are computed from the hourly ERA5 files already in
raw_data/rainfall and runoff/, so nothing is downloaded for them.
"""
import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_data"
BBOX = (23.0, 3.0, 36.5, 12.5)          # lon_min, lat_min, lon_max, lat_max; South Sudan plus margin
YEARS = range(2017, 2026)               # same years as the ERA5 rainfall used in build_inputs.py


def get(url, path):
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def indices():
    import pandas as pd
    out = RAW / "climate_indices"
    out.mkdir(parents=True, exist_ok=True)
    nino = out / "sstoi.indices"
    if not nino.exists():
        get("https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices", nino)
    cols = ["year", "month", "nino12", "nino12_anom", "nino3", "nino3_anom",
            "nino4", "nino4_anom", "nino34", "nino34_anom"]
    n = pd.read_csv(nino, sep=r"\s+", skiprows=1, names=cols)
    print(f"Nino3.4: {len(n)} months, last {int(n.year.iloc[-1])}-{int(n.month.iloc[-1]):02d}")

    dmi = out / "DMI_month.html"
    if not dmi.exists():
        get("https://www.cpc.ncep.noaa.gov/products/international/ocean_monitoring/IODMI/DMI_month.html", dmi)
    # the table's header row uses <td>, and some values sit inside <strong>: parse rows directly
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", dmi.read_text(encoding="utf8", errors="replace"), re.S):
        c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if c and re.fullmatch(r"\d{4}", c[0]):
            rows += [(int(c[0]), m, float(v)) for m, v in enumerate(c[1:13], 1)
                     if re.fullmatch(r"-?\d+(\.\d+)?", v)]
    t = pd.DataFrame(rows, columns=["year", "month", "dmi"])
    if t.empty:
        raise SystemExit(f"DMI: no values parsed from {dmi}; the page layout changed")
    t.to_csv(out / "dmi.csv", index=False)
    print(f"DMI: {len(t)} months, last {t.year.iloc[-1]}-{t.month.iloc[-1]:02d}")


def grace():
    import earthaccess
    out = RAW / "grace"
    out.mkdir(parents=True, exist_ok=True)
    earthaccess.login(persist=True)
    g = earthaccess.search_data(short_name="TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4")
    if not g:
        raise SystemExit("GRACE: no granules found; check the short name on podaac.jpl.nasa.gov")
    earthaccess.download(g, str(out))
    print(f"GRACE: {len(g)} file(s) in {out}")


def soil():
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    out = RAW / "soilgrids"
    out.mkdir(parents=True, exist_ok=True)
    base = "https://files.isric.org/soilgrids/latest/data"
    layers = {f"{p}_{d}_mean": f"{base}/{p}/{p}_{d}_mean.vrt"
              for p in ("clay", "sand") for d in ("0-5cm", "5-15cm", "15-30cm")}
    layers["wrb_Vertisols"] = f"{base}/wrb/Vertisols.vrt"
    for name, url in layers.items():
        dst = out / f"{name}.tif"
        if dst.exists():
            continue
        try:
            src = rasterio.open(f"/vsicurl/{url}")
        except rasterio.errors.RasterioIOError as e:
            raise SystemExit(f"{name}: cannot open {url}\n{e}\nCheck the path at {base}/") from e
        with src:
            b = transform_bounds("EPSG:4326", src.crs, *BBOX, densify_pts=21)
            w = from_bounds(*b, src.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, src.width, src.height))
            a = src.read(1, window=w)
            prof = {k: v for k, v in src.profile.items() if k not in ("blockxsize", "blockysize", "tiled")}
            prof |= {"driver": "GTiff", "width": w.width, "height": w.height,
                     "transform": src.window_transform(w), "compress": "deflate"}
            with rasterio.open(dst, "w", **prof) as d:
                d.write(a, 1)
        print(f"soil: {name} {a.shape}, nodata {prof.get('nodata')}")


def era5sm():
    import cdsapi
    out = RAW / "era5_soil_moisture"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"era5_swvl_monthly_{YEARS[0]}_{YEARS[-1]}.nc"
    if dst.exists():
        return
    rc = dict(line.split(":", 1) for line in (Path.home() / ".cdsapirc").read_text().splitlines() if ":" in line)
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=rc["key"].strip())
    c.retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [f"volumetric_soil_water_layer_{i}" for i in (1, 2, 3, 4)],
        "year": [str(y) for y in YEARS], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "data_format": "netcdf", "download_format": "unarchived",
        "area": [BBOX[3], BBOX[0], BBOX[1], BBOX[2]],          # N, W, S, E
    }, str(dst))
    print(f"ERA5 soil moisture: {dst}")


def era5atmos():
    """ERA5 monthly means: 2 m temperature and dewpoint, 10 m wind speed, surface solar radiation."""
    import cdsapi
    out = RAW / "era5_atmosphere"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"era5_atmos_monthly_{YEARS[0]}_{YEARS[-1]}.nc"
    if dst.exists():
        return
    rc = dict(line.split(":", 1) for line in (Path.home() / ".cdsapirc").read_text().splitlines() if ":" in line)
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=rc["key"].strip())
    c.retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature", "2m_dewpoint_temperature", "10m_wind_speed",
                     "surface_solar_radiation_downwards"],
        "year": [str(y) for y in YEARS], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "data_format": "netcdf", "download_format": "unarchived",
        "area": [BBOX[3], BBOX[0], BBOX[1], BBOX[2]],
    }, str(dst))
    print(f"ERA5 atmosphere: {dst}")


def jrc():
    """JRC Global Surface Water 1984-2021 occurrence tiles (10 x 10 deg, 30 m) covering South Sudan.
    Tiles are named by a corner; all candidates around the country are tried, missing ones are skipped."""
    import requests
    out = RAW / "jrc_gsw"
    out.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence"
    for lo in ("20E", "30E"):
        for la in ("0N", "10N", "20N"):
            name = f"occurrence_{lo}_{la}v1_4_2021.tif"
            dst = out / name
            if dst.exists():
                continue
            with requests.get(f"{base}/{name}", stream=True, timeout=120) as r:
                if r.status_code != 200:
                    print(f"JRC {name}: HTTP {r.status_code}, skipped")
                    continue
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            print(f"JRC {name}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
    import rasterio
    for f in sorted(out.glob("*.tif")):
        with rasterio.open(f) as src:
            print(f"  {f.name}: bounds {tuple(round(b, 2) for b in src.bounds)}, res {src.res[0]:.6f}")


def hydroweb():
    """Hydroweb river virtual stations (satellite-altimetry water levels) inside BBOX, operational collection.
    Needs a free hydroweb.next API key in the environment ($env:HYDROWEB_API_KEY = "..."; account settings ->
    API key) and `pip install py_hydroweb`. The zip is moved to and unpacked in raw_data/hydroweb/."""
    import shutil
    import zipfile
    out = RAW / "hydroweb"
    if any(out.rglob("hydroprd_*.txt")):
        print(f"Hydroweb: station files already in {out}")
        return
    key = os.environ.get("HYDROWEB_API_KEY")
    if not key:
        print("Hydroweb: skipped, set HYDROWEB_API_KEY first")
        return
    import py_hydroweb
    out.mkdir(parents=True, exist_ok=True)
    basket = py_hydroweb.DownloadBasket("south_sudan_rivers")
    basket.add_collection("HYDROWEB_RIVERS_OPE", bbox=list(BBOX))
    z = out / Path(py_hydroweb.Client(api_key=key).submit_and_download_zip(basket)).name
    shutil.move(Path.cwd() / z.name, z)
    with zipfile.ZipFile(z) as fh:
        fh.extractall(out)
    print(f"Hydroweb: {len(list(out.rglob('hydroprd_*.txt')))} station files in {out}")


def merit():
    d = RAW / "MERITHydro"
    need = ["elv_n00e000", "elv_n00e030"]
    missing = [n for n in need if not (d / n).is_dir() or not any((d / n).rglob("*.tif"))]
    if missing:
        print(f"MERIT: missing {missing}. Download {', '.join(n + '.tar' for n in missing)} from the MERIT Hydro "
              f"page (same Dropbox as hnd/upa) and extract each into raw_data/MERITHydro/<name>/")
    else:
        print("MERIT: elevation present")


STEPS = {"indices": indices, "grace": grace, "soil": soil, "era5sm": era5sm, "atmos": era5atmos, "jrc": jrc,
         "hydroweb": hydroweb, "merit": merit}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(STEPS), default=list(STEPS))
    for s in ap.parse_args().only:
        print(f"--- {s}")
        STEPS[s]()
