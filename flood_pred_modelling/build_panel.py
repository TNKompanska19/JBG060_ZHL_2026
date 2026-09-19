"""
Builds the modelling panel: one row per ERA5 0.25 degree cell per day.

    python flood_pred_modelling/build_panel.py

Targets come from the MODIS/VIIRS flood masks, features from ERA5 rainfall and
runoff, AgERA5 evapotranspiration and lake altimetry. See README.md for why the
panel is shaped this way and which loader bugs are worked around here.
"""

import io
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

# xarray and pandas are noisy about API changes that do not affect us. Keep the
# filter narrow: a blanket ignore is what hid the ET bug for a week.
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW = REPO / "raw_data"
OUT = HERE / "output"

STATES = ["Unity", "Jonglei"]
YEARS = np.arange(2015, 2026)

GRID = 0.25                                        # ERA5 cell size, degrees
PIXEL = 0.00208333                                 # MODIS/VIIRS pixel, degrees
PIXELS_PER_CELL = int(round(GRID / PIXEL)) ** 2    # 14400

WINDOWS = [3, 7, 14, 30, 60, 90]

# Mission codes as they actually appear in the Hydroweb files. The course loader
# filters on 'JASON-1' and friends, which never occur, and truncates at 2002.
MISSIONS = {"TOPEX", "JASN1", "JASN2", "JASN3", "SEN6A", "POSDN"}

LAKE_COLS = [
    "mission", "cycle", "date", "hour", "minute",
    "height_wrt_ref", "height_err", "backscatter_ku",
    "wet_tropo_corr", "iono_corr", "dry_tropo_corr",
    "mode1", "mode2", "ice_flag", "height_egm2008", "data_source_flag",
]


def snap(a):
    """Snap coordinates to the nearest ERA5 grid node."""
    return np.round(np.asarray(a, dtype="float64") / GRID) * GRID


def coord_slice(coord, lo, hi):
    """Build a slice for xarray .sel() that works whichever way the axis runs.

    ERA5 latitude runs 33 -> -3 and AgERA5 latitude does too. Passing
    slice(lo, hi) to a descending axis returns nothing at all, silently.
    """
    values = np.asarray(coord)
    if values[0] > values[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def load_aoi():
    """Dissolved AOI polygon plus its bounding box."""
    adm2 = gpd.read_file(RAW / "Administrative boundaries" / "ssd_admin2.geojson")
    sel = adm2[adm2["adm1_name"].isin(STATES)].copy()
    if sel.empty:
        raise ValueError(f"no counties matched {STATES}, check adm1_name spellings")

    minx, miny, maxx, maxy = sel.total_bounds
    bbox = {
        "lon_min": float(minx), "lon_max": float(maxx),
        "lat_min": float(miny), "lat_max": float(maxy),
    }
    print(f"AOI: {len(sel)} counties in {STATES}")
    print(f"  lon {bbox['lon_min']:.2f}..{bbox['lon_max']:.2f}  "
          f"lat {bbox['lat_min']:.2f}..{bbox['lat_max']:.2f}")
    if bbox["lat_max"] > 10.0:
        print("  note: AOI reaches above 10N but the flood-mask tiles stop there, "
              "so the northern strip has no inundation data")
    return sel.dissolve(), bbox


def cells_in_aoi(aoi, bbox):
    """ERA5 cells whose centre falls inside the AOI polygon."""
    lats = np.arange(snap(bbox["lat_min"]), snap(bbox["lat_max"]) + GRID, GRID)
    lons = np.arange(snap(bbox["lon_min"]), snap(bbox["lon_max"]) + GRID, GRID)
    grid = pd.DataFrame(
        [(la, lo) for la in lats for lo in lons], columns=["cell_lat", "cell_lon"]
    )
    pts = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid.cell_lon, grid.cell_lat),
        crs="EPSG:4326",
    )
    inside = gpd.sjoin(pts, aoi[["geometry"]], predicate="within", how="inner")
    out = inside[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    print(f"  {len(out)} ERA5 cells inside AOI (of {len(grid)} in the bbox)")
    return out


def build_targets(years, bbox):
    """Aggregate 250 m flood pixels up to (cell, day) counts."""
    base = RAW / "flood_masks"
    frames = []

    for year in years:
        for kind, code in (("compact_recurring", 0), ("compact_unusual", 1)):
            for tile in ("h20v08", "h21v08"):
                path = base / kind / f"flood_events_{tile}_{year}.parquet"
                if not path.exists():
                    print(f"  missing {path.name}, skipped")
                    continue

                df = pd.read_parquet(path, columns=["date", "lat", "lon"])
                df = df[
                    df.lat.between(bbox["lat_min"], bbox["lat_max"])
                    & df.lon.between(bbox["lon_min"], bbox["lon_max"])
                ]
                if df.empty:
                    continue

                df["date"] = pd.to_datetime(df["date"].astype(str))
                df["cell_lat"] = snap(df.lat)
                df["cell_lon"] = snap(df.lon)
                df["flood_type"] = code
                frames.append(
                    df.groupby(["date", "cell_lat", "cell_lon", "flood_type"])
                      .size().rename("n").reset_index()
                )
        print(f"  flood masks {year} done")

    if not frames:
        raise RuntimeError("no flood-mask data loaded, check RAW path and YEARS")

    long = pd.concat(frames, ignore_index=True)
    wide = (
        long.pivot_table(
            index=["date", "cell_lat", "cell_lon"],
            columns="flood_type", values="n", aggfunc="sum", fill_value=0,
        )
        .rename(columns={0: "n_recurring", 1: "n_unusual"})
        .reset_index()
    )
    # A year may contain only one of the two mask kinds, in which case the pivot
    # produces only one column. Add the missing one so the schema is stable.
    for c in ("n_recurring", "n_unusual"):
        if c not in wide:
            wide[c] = 0

    # Recurring and unusual are two views of the same water, so the target sums
    # them. flood_type itself is never a feature: it comes from a single static
    # mask built from the 2003-2024 archive and applied retroactively, so using
    # it would leak the future into the past.
    wide["n_flood"] = wide.n_recurring + wide.n_unusual
    wide["flood_frac"] = wide.n_flood / PIXELS_PER_CELL
    wide["unusual_frac"] = wide.n_unusual / PIXELS_PER_CELL
    return wide


def load_era5(years, bbox):
    """Daily tp and ro per cell, plus a domain mean as a crude upstream proxy."""
    paths = [RAW / "rainfall and runoff" / f"ERA5_{y}.nc" for y in years]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise RuntimeError("no ERA5 files found under 'rainfall and runoff'")

    ds = xr.open_mfdataset(paths, combine="by_coords", chunks={"valid_time": 500})
    ds = ds[["tp", "ro"]]
    # tp and ro are hourly accumulations in metres, so a daily sum is correct
    # and the x1000 below converts to millimetres.
    daily = ds.resample(valid_time="1D").sum()

    basin = (
        daily.mean(dim=["latitude", "longitude"])
        .to_dataframe().reset_index()
        .rename(columns={"valid_time": "date", "tp": "tp_basin", "ro": "ro_basin"})
    )
    basin[["tp_basin", "ro_basin"]] *= 1000.0       # m -> mm

    local = daily.sel(
        latitude=coord_slice(ds.latitude, bbox["lat_min"] - GRID, bbox["lat_max"] + GRID),
        longitude=coord_slice(ds.longitude, bbox["lon_min"] - GRID, bbox["lon_max"] + GRID),
    ).to_dataframe().reset_index()

    local = local.rename(
        columns={"valid_time": "date", "latitude": "cell_lat", "longitude": "cell_lon"}
    )
    local[["tp", "ro"]] *= 1000.0
    local["cell_lat"] = snap(local.cell_lat)
    local["cell_lon"] = snap(local.cell_lon)

    if local.empty:
        raise RuntimeError("ERA5 selection came back empty, check the bbox and axis order")

    return local[["date", "cell_lat", "cell_lon", "tp", "ro"]], basin[["date", "tp_basin", "ro_basin"]]


def load_et(years, bbox):
    """Daily reference ET, averaged from the 0.1 degree AgERA5 grid onto 0.25."""
    var = "ReferenceET_PenmanMonteith_FAO56"
    out = []

    for year in years:
        files = sorted((RAW / "evapotranspiration" / f"ET_{year}").glob("*.nc"))
        if not files:
            print(f"  no ET files for {year}, skipped")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 60})
        sub = ds[var].sel(
            lat=coord_slice(ds.lat, bbox["lat_min"] - GRID, bbox["lat_max"] + GRID),
            lon=coord_slice(ds.lon, bbox["lon_min"] - GRID, bbox["lon_max"] + GRID),
        )
        if sub.size == 0:
            raise RuntimeError(
                f"ET {year}: the lat/lon selection is empty. AgERA5 latitude is "
                f"descending, so slice(lo, hi) returns nothing."
            )

        df = sub.to_dataframe().reset_index()
        df["cell_lat"] = snap(df.lat)
        df["cell_lon"] = snap(df.lon)
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()

        out.append(
            df.groupby(["date", "cell_lat", "cell_lon"])[var]
              .mean().rename("et").reset_index()
        )
        ds.close()
        print(f"  ET {year} done ({len(files)} files)")

    if not out:
        raise RuntimeError("no evapotranspiration loaded at all")

    et = pd.concat(out, ignore_index=True)
    years_got = set(et.date.dt.year.unique())
    missing = set(int(y) for y in years) - years_got
    if missing:
        raise RuntimeError(f"ET is missing whole years: {sorted(missing)}")
    return et


def _read_hydroweb(path):
    """Parse one Hydroweb altimetry text file into date and height.

    The files carry one row per satellite pass with a mission code in the first
    column, interleaved with header and comment lines. Filtering on the real
    mission codes is both how we select data rows and how we skip the headers.
    """
    lines = [
        ln.strip() for ln in path.read_text(encoding="latin-1").splitlines()
        if ln.strip() and ln.strip().split()[0] in MISSIONS
    ]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)), sep=r"\s+", names=LAKE_COLS,
        na_values=["999.99", "99.999", "9999.99"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    # Drop only on the two columns we use. The course loader drops on every
    # column, which throws away rows whose correction terms happen to be blank.
    return df.dropna(subset=["date", "height_egm2008"])[["date", "height_egm2008"]]


def load_lakes():
    """Daily levels for Victoria, Kyoga and Albert.

    Victoria and Kyoga come from Hydroweb text files, Albert from a netCDF, so
    the three are read differently and then concatenated onto one daily index.
    """
    base = RAW / "Water levels lakes"
    series = {}

    for name, fname in (("victoria", "water_level_victoria.txt"),
                        ("kyoga", "water_level_Kyoga.txt")):
        d = _read_hydroweb(base / fname)
        if d.empty:
            raise RuntimeError(f"lake {name} came back empty, check the mission codes")
        series[name] = d.groupby("date").height_egm2008.mean()
        print(f"  lake {name}: {len(d)} obs, {d.date.min().date()} -> {d.date.max().date()}")

    ds = xr.open_dataset(base / "water_level_altimetry_Albert.nc")
    alb = ds.to_dataframe().reset_index(drop=True)
    alb["date"] = pd.to_datetime(alb["datetime"]).dt.normalize()
    alb = alb.dropna(subset=["water_level"])
    series["albert"] = alb.groupby("date").water_level.mean()
    print(f"  lake albert: {len(alb)} obs, {alb.date.min().date()} -> {alb.date.max().date()}")

    wide = pd.concat(series, axis=1)
    wide.columns = [f"lake_{c}" for c in wide.columns]

    # Altimetry repeats every 6 to 10 days, so put it on a daily index.
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full).interpolate(method="time", limit_direction="both")
    return wide.rename_axis("date").reset_index()


def add_lagged(panel):
    """Trailing accumulations, lake deltas and day-of-year terms.

    min_periods is the full window on purpose. With min_periods=1 the first
    89 days of every cell carry a short sum that reads as dry weather.
    """
    panel = panel.sort_values(["cell_lat", "cell_lon", "date"])
    g = panel.groupby(["cell_lat", "cell_lon"], sort=False)

    for w in WINDOWS:
        panel[f"tp_sum{w}"] = g.tp.transform(lambda s: s.rolling(w, min_periods=w).sum())
        panel[f"ro_sum{w}"] = g.ro.transform(lambda s: s.rolling(w, min_periods=w).sum())
        panel[f"et_mean{w}"] = g.et.transform(lambda s: s.rolling(w, min_periods=w).mean())

    # Basin and lake terms vary with date only, so roll them once on the date axis
    # rather than once per cell.
    daily = panel[["date", "tp_basin", "ro_basin",
                   "lake_victoria", "lake_kyoga", "lake_albert"]].drop_duplicates("date")
    daily = daily.sort_values("date").set_index("date")

    for w in WINDOWS:
        daily[f"tp_basin_sum{w}"] = daily.tp_basin.rolling(w, min_periods=w).sum()
        daily[f"ro_basin_sum{w}"] = daily.ro_basin.rolling(w, min_periods=w).sum()
    for lake in ("victoria", "kyoga", "albert"):
        for w in (30, 90):
            daily[f"lake_{lake}_d{w}"] = daily[f"lake_{lake}"].diff(w)

    new = [c for c in daily.columns if c not in panel.columns]
    panel = panel.merge(daily[new].reset_index(), on="date", how="left")

    doy = panel.date.dt.dayofyear
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return panel


def preflight():
    """Check every input exists before doing twenty minutes of work.

    Failing here with a readable list beats failing deep inside a loader after
    the flood masks have already been aggregated.
    """
    needed = [
        RAW / "Administrative boundaries" / "ssd_admin2.geojson",
        RAW / "flood_masks" / "compact_recurring",
        RAW / "flood_masks" / "compact_unusual",
        RAW / "rainfall and runoff",
        RAW / "evapotranspiration",
        RAW / "Water levels lakes" / "water_level_victoria.txt",
        RAW / "Water levels lakes" / "water_level_Kyoga.txt",
        RAW / "Water levels lakes" / "water_level_altimetry_Albert.nc",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print(f"raw_data resolved to {RAW}")
        for p in missing:
            print(f"  MISSING  {p.relative_to(REPO)}")
        raise SystemExit("\nExpected raw_data/ next to flood_pred_modelling/ in the repo root.")
    print(f"raw_data OK at {RAW}")


def main():
    """Build the panel end to end and write it to output/panel.parquet."""
    preflight()
    OUT.mkdir(parents=True, exist_ok=True)

    print("area of interest")
    aoi, bbox = load_aoi()
    cells = cells_in_aoi(aoi, bbox)

    print("flood masks")
    targets = build_targets(YEARS, bbox)

    print("ERA5 rainfall and runoff")
    era5, basin = load_era5(YEARS, bbox)

    print("evapotranspiration")
    et = load_et(YEARS, bbox)

    print("lake levels")
    lakes = load_lakes()

    print("assembling")
    # Cross join first: a cell-day with no flood record is a dry day, not a
    # missing one, and that distinction is the whole point of the skeleton.
    days = pd.DataFrame({"date": pd.date_range(f"{YEARS.min()}-01-01",
                                               f"{YEARS.max()}-12-31", freq="D")})
    panel = cells.merge(days, how="cross")

    panel = panel.merge(targets, on=["date", "cell_lat", "cell_lon"], how="left")
    for c in ("n_recurring", "n_unusual", "n_flood", "flood_frac", "unusual_frac"):
        panel[c] = panel[c].fillna(0)
    panel["flood_any"] = (panel.n_flood > 0).astype("int8")

    panel = panel.merge(era5, on=["date", "cell_lat", "cell_lon"], how="left")
    panel = panel.merge(et, on=["date", "cell_lat", "cell_lon"], how="left")
    panel = panel.merge(basin, on="date", how="left")
    panel = panel.merge(lakes, on="date", how="left")

    et_missing = panel.et.isna().mean()
    if et_missing > 0.05:
        raise RuntimeError(f"et is {et_missing:.1%} missing after the merge, "
                           f"the ET grid probably does not line up with the cells")

    panel = add_lagged(panel)

    path = OUT / "panel.parquet"
    panel.to_parquet(path, index=False)

    print(f"\nwrote {path}  shape={panel.shape}")
    print(f"dates      {panel.date.min().date()} -> {panel.date.max().date()}")
    print(f"cells      {len(cells)}")
    print(f"flood_any  {panel.flood_any.mean():.4f} positive rate")
    print(f"flood_frac mean {panel.flood_frac.mean():.5f}, max {panel.flood_frac.max():.4f}")
    print(f"et         mean {panel.et.mean():.3f} mm/day")

    miss = panel.isna().mean().sort_values(ascending=False)
    print("\ntop missingness:")
    print(miss[miss > 0].head(10).round(4).to_string())


if __name__ == "__main__":
    main()
