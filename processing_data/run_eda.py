

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust these two imports if your loader files aren't in the same folder as this script
from loading import (
    load_dartmouth_data,
    load_lake_stations,
    load_rainfall_runoff,
    load_processed_ET,
    load_flood_masks,
)
from loading_impact_data import (
    load_admin_boundaries,
    load_worldpop_area,
    load_health_facilities,
    load_cattle,
    load_farmland_mask,
    load_ipc_data,
    load_GDP,
)

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from the project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
    )

# Rough South Sudan bounding box - reused across several loaders
SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS_FULL = np.arange(2000, 2026)
YEARS_POP = np.arange(2015, 2026)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def eda_discharge():
    section("1. RIVER DISCHARGE (Dartmouth Flood Observatory)")
    data = load_dartmouth_data()

    info = pd.read_excel("./raw_data/Darthmouth Flood Observatory/information.xlsx")
    print(f"\nTotal stations loaded: {len(data)}")
    print(f"Columns in information.xlsx: {info.columns.tolist()}")

    col_map = {c.lower().strip(): c for c in info.columns}
    country_col = col_map.get("country")
    area_id_col = col_map.get("area id") or col_map.get("area_id") or col_map.get("areaid")

    if country_col is None or area_id_col is None:
        print("\nCould not auto-detect 'country' / 'area id' columns - "
              "inspect info.columns above and adjust eda_discharge() manually.")
        return data, info

    print("\nStations by country:")
    print(info.groupby(country_col).size())

    ssd_ids = info.loc[info[country_col] == "South Sudan", area_id_col].tolist()
    print(f"\nStation IDs actually in South Sudan: {ssd_ids}")

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 2.2 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (area_id, df) in zip(axes, data.items()):
        col = df.columns[0]
        ax.plot(df.index, df[col], linewidth=0.6)
        country = info.loc[info[area_id_col] == area_id, country_col].values
        country = country[0] if len(country) else "?"
        ax.set_title(f"Station {area_id} ({country}) - n={len(df)}, "
                      f"missing={df[col].isna().sum()}", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_discharge_all_stations.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '01_discharge_all_stations.png'}")

    # Coverage table: date range and completeness per station
    rows = []
    for area_id, df in data.items():
        col = df.columns[0]
        rows.append({
            "area_id": area_id,
            "start": df.index.min(),
            "end": df.index.max(),
            "n_obs": len(df),
            "n_missing": df[col].isna().sum(),
            "mean_discharge": df[col].mean(),
            "max_discharge": df[col].max(),
        })
    coverage = pd.DataFrame(rows).set_index("area_id")
    print("\nCoverage summary per station:")
    print(coverage)
    coverage.to_csv(OUT_DIR / "01_discharge_coverage.csv")

    return data, info

def eda_lakes():
    section("2. LAKE WATER LEVELS")
    data = load_lake_stations()

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 3 * len(data)), sharex=False)
    for ax, (name, df) in zip(axes, data.items()):
        value_col = "water_level" if "water_level" in df.columns else "height_egm2008"
        ax.plot(df.index, df[value_col], linewidth=0.7)
        ax.set_title(f"Lake {name} - n={len(df)}, "
                      f"{df.index.min().date()} to {df.index.max().date()}", fontsize=9)
        ax.set_ylabel(value_col)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_lake_levels.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '02_lake_levels.png'}")

    for name, df in data.items():
        print(f"\n{name}: {len(df)} obs, {df.index.min()} to {df.index.max()}")
        print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]]
              if "water_level" not in df.columns
              else df["water_level"].describe())

    return data

def eda_rainfall_runoff(years=None):
    section("3. RAINFALL & RUNOFF (ERA5)")
    years = years if years is not None else YEARS_FULL
    ds = load_rainfall_runoff(years)

    # Spatial mean time series (whole Nile Basin grid) - cheap first look
    tp_mean = ds["tp"].mean(dim=["latitude", "longitude"]).compute()
    ro_mean = ds["ro"].mean(dim=["latitude", "longitude"]).compute()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(ds.valid_time.values, tp_mean.values, linewidth=0.5, color="tab:blue")
    axes[0].set_title("Mean daily precipitation across Nile Basin grid (m/day)")
    axes[1].plot(ds.valid_time.values, ro_mean.values, linewidth=0.5, color="tab:orange")
    axes[1].set_title("Mean daily runoff across Nile Basin grid (m/day)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_rainfall_runoff_timeseries.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '03_rainfall_runoff_timeseries.png'}")

    # Spatial mean map (average over all time) - shows where it rains most
    tp_spatial = ds["tp"].mean(dim="valid_time").compute()
    fig, ax = plt.subplots(figsize=(8, 6))
    tp_spatial.plot(ax=ax, cmap="Blues")
    ax.set_title("Mean daily precipitation by location (m/day), full period")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_rainfall_spatial_mean.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '03_rainfall_spatial_mean.png'}")

    print(f"\nPrecipitation - overall mean: {float(tp_mean.mean()):.5f} m/day, "
          f"max single day: {float(tp_mean.max()):.5f} m/day")
    print(f"Runoff        - overall mean: {float(ro_mean.mean()):.5f} m/day, "
          f"max single day: {float(ro_mean.max()):.5f} m/day")

    return ds


def eda_et(years=None, target_lon=30.725, target_lat=9.475):
    section("4. EVAPOTRANSPIRATION (single grid cell)")
    years = years if years is not None else YEARS_FULL

    et_dir = Path("./processing_data/evapotranspiration")
    lat_str = f"{target_lat:.3f}N"
    lon_str = f"{target_lon:.3f}E"
    available_years = np.array([
        y for y in years
        if (et_dir / f"ET_{y}_{lat_str}_{lon_str}_processed.csv").exists()
    ])

    if len(available_years) == 0:
        print(f"\nNo processed ET files found in {et_dir.resolve()} "
              f"for target ({target_lat}N, {target_lon}E).")
        print("This means process_ET() hasn't been run yet for this location, "
              "or you're not running from the project root - check that "
              "'processing_data/evapotranspiration/' contains ET_<year>_..._processed.csv files.")
        return None

    if len(available_years) < len(years):
        missing = sorted(set(years) - set(available_years))
        print(f"\nNOTE: {len(missing)} year(s) not yet processed, skipping: {missing}")

    data = load_processed_ET(available_years, target_lon, target_lat)

    combined = pd.concat(
        [df.assign(year=y) for y, df in data.items()], ignore_index=True
    )
    combined["date"] = pd.to_datetime(combined["date"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(combined["date"], combined["gridcell"], linewidth=0.4)
    ax.set_title(f"Reference ET at ({target_lat}N, {target_lon}E), {years.min()}-{years.max()}")
    ax.set_ylabel("mm/day")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_evapotranspiration.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '04_evapotranspiration.png'}")

    print(combined["gridcell"].describe())
    print(f"\nNOTE: this is ET at a single fixed point ({target_lat}N, {target_lon}E), "
          f"not spatially distributed. If your project direction needs ET across an area, "
          f"process_ET() will need to be re-run for multiple grid cells.")

    return combined

def eda_flood_masks(years=None, bbox=None):
    section("5. FLOOD MASKS (MODIS/VIIRS)")
    years = years if years is not None else YEARS_FULL
    bbox = bbox if bbox is not None else SSD_BBOX
    df = load_flood_masks(years, bbox=bbox)

    print(f"Total flood pixel-records in bbox: {len(df):,}")
    print(f"Recurring: {(df['flood_type'] == 0).sum():,}  "
          f"Unusual: {(df['flood_type'] == 1).sum():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique days with any flood record: {df['date'].nunique():,}")
    print(f"Unique pixel locations flooded at least once: "
          f"{df[['lat', 'lon']].drop_duplicates().shape[0]:,}")

    # Flood events per year, split recurring vs unusual
    df["year"] = df["date"].dt.year
    yearly = df.groupby(["year", "flood_type"]).size().unstack(fill_value=0)
    yearly.columns = ["recurring", "unusual"]

    fig, ax = plt.subplots(figsize=(12, 4))
    yearly.plot(kind="bar", stacked=True, ax=ax, color=["tab:blue", "tab:red"])
    ax.set_title("Flood pixel-records per year (South Sudan bbox)")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_flood_events_per_year.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '05_flood_events_per_year.png'}")

    # Spatial scatter - where do floods happen
    sample = df.sample(min(200_000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    for ftype, color, label in [(0, "tab:blue", "recurring"), (1, "tab:red", "unusual")]:
        subset = sample[sample["flood_type"] == ftype]
        ax.scatter(subset["lon"], subset["lat"], s=0.5, alpha=0.3, color=color, label=label)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Flood pixel locations (sampled)")
    ax.legend(markerscale=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_flood_spatial_scatter.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '05_flood_spatial_scatter.png'}")

    yearly.to_csv(OUT_DIR / "05_flood_events_per_year.csv")
    return df

def eda_admin_population(bbox=None):
    section("6. ADMIN BOUNDARIES & POPULATION (WorldPop, area-based)")
    bbox = bbox if bbox is not None else SSD_BBOX

    admin1, admin2 = load_admin_boundaries()
    print(f"Admin1 (states): {len(admin1)} polygons")
    print(f"Admin2 (counties): {len(admin2)} polygons")

    fig, ax = plt.subplots(figsize=(8, 8))
    admin1.boundary.plot(ax=ax, color="black", linewidth=1)
    admin2.boundary.plot(ax=ax, color="gray", linewidth=0.3)
    ax.set_title("South Sudan admin1 (black) / admin2 (gray) boundaries")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_admin_boundaries.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '06_admin_boundaries.png'}")

    pop_area = load_worldpop_area(bbox)
    pop_series = pd.Series(pop_area).sort_index()
    print("\nPopulation in bbox, by year:")
    print(pop_series)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pop_series.index, pop_series.values, marker="o")
    ax.set_title(f"Total population in bbox {bbox}, 2015-2025")
    ax.set_ylabel("people")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_population_trend.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '06_population_trend.png'}")

    return admin1, admin2, pop_series

def eda_health_facilities():
    section("7. HEALTH FACILITIES")
    hf = load_health_facilities()
    print(f"Total health facilities in South Sudan: {len(hf)}")
    print("\nBy facility type:")
    print(hf["Facility_t"].value_counts())
    print("\nBy ownership:")
    print(hf["Ownership"].value_counts())
    print("\nBy admin1 (state):")
    print(hf["Admin1"].value_counts())

    fig, ax = plt.subplots(figsize=(8, 8))
    hf.plot(ax=ax, markersize=3, column="Facility_t", legend=True,
            legend_kwds={"fontsize": 7, "loc": "lower left"})
    ax.set_title("Health facilities in South Sudan, by type")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_health_facilities_map.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '07_health_facilities_map.png'}")

    return hf

def eda_cattle_farmland(bbox=None):
    section("8. CATTLE & FARMLAND (crop / rangeland cover)")
    bbox = bbox if bbox is not None else SSD_BBOX

    cattle_df, lons, lats = load_cattle()
    n_nonzero = (cattle_df.notna() & (cattle_df != 0)).sum().sum()
    print(f"Cattle raster: {cattle_df.shape[0]}x{cattle_df.shape[1]} pixels, "
          f"{n_nonzero:,} pixels with cattle present")
    print(f"Cattle count - min: {np.nanmin(cattle_df.values):.1f}, "
          f"max: {np.nanmax(cattle_df.values):.1f}, "
          f"mean (nonzero): {cattle_df.values[cattle_df.values > 0].mean():.1f}")

    crop_da = load_farmland_mask(bbox, "crop")
    range_da = load_farmland_mask(bbox, "rangeland")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    crop_da.plot(ax=axes[0], cmap="YlGn", vmin=0, vmax=100)
    axes[0].set_title("Crop cover (%)")
    range_da.plot(ax=axes[1], cmap="YlOrBr", vmin=0, vmax=100)
    axes[1].set_title("Rangeland cover (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_crop_rangeland_masks.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '08_crop_rangeland_masks.png'}")

    print(f"\nCrop cover - mean: {float(crop_da.mean()):.1f}%, "
          f"% of pixels with any crop: {float((crop_da > 0).mean()) * 100:.1f}%")
    print(f"Rangeland cover - mean: {float(range_da.mean()):.1f}%, "
          f"% of pixels with any rangeland: {float((range_da > 0).mean()) * 100:.1f}%")

    return cattle_df, crop_da, range_da

def eda_ipc():
    section("9. IPC FOOD INSECURITY (Phase 3+ population, by county)")
    ipc = load_ipc_data()
    print(f"Shape: {ipc.shape}")
    print(f"Time windows covered: {ipc['Start Date'].min()} to {ipc['End Date'].max()}")
    n_counties = ipc.shape[1] - 2  # minus Start Date, End Date
    print(f"Counties with data: {n_counties}")

    county_cols = [c for c in ipc.columns if c not in ("Start Date", "End Date")]
    totals = ipc[county_cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ipc["Start Date"], totals, marker="o")
    ax.set_title("Total Phase 3+ population (summed across counties) over time")
    ax.set_ylabel("people in Phase 3+")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_ipc_phase3plus_trend.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '09_ipc_phase3plus_trend.png'}")

    missing_pct = ipc[county_cols].isna().mean().sort_values(ascending=False)
    print("\nCounties with most missing data (top 10):")
    print(missing_pct.head(10))

    return ipc

if __name__ == "__main__":
    discharge_data, station_info = eda_discharge()
    lake_data = eda_lakes()

    et_data = eda_et()
    flood_df = eda_flood_masks()
    admin1, admin2, pop_series = eda_admin_population()
    hf = eda_health_facilities()
    cattle_df, crop_da, range_da = eda_cattle_farmland()
    ipc = eda_ipc()

    section("DONE")
    print(f"All plots and CSVs saved to: {OUT_DIR.resolve()}")