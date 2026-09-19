

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from the project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
    )

from loading import load_flood_masks, load_rainfall_runoff
from loading_impact_data import load_admin_boundaries

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

# Default: a mix of the highest-flood-volume counties (from exposure_analysis.py)
TARGET_COUNTIES = [
    "Aweil East", "Pibor", "Bor South", "Ayod", "Baliet", "Lafon",
    "Kapoeta East", "Uror", "Akobo", "Fangak", "Gogrial West",
]
LEAD_DAYS = 14
YEARS = np.arange(2015, 2026)
SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_county_bbox(admin2, county_col, county_name, pad=0.15):
    match = admin2[admin2[county_col].str.strip().str.lower() == county_name.lower()]
    if match.empty:
        return None, None
    geom = match.iloc[0].geometry
    b = geom.bounds
    bbox = {"lon_min": b[0] - pad, "lat_min": b[1] - pad,
            "lon_max": b[2] + pad, "lat_max": b[3] + pad}
    return geom, bbox


def get_flood_events(flood_df_all, county_geom, admin2_crs):
    """Filter pooled flood data to one county's polygon and collapse to event dates."""
    if len(flood_df_all) == 0:
        return pd.Series(dtype="datetime64[ns]")
    gdf = gpd.GeoDataFrame(
        flood_df_all,
        geometry=[Point(lon, lat) for lon, lat in zip(flood_df_all["lon"], flood_df_all["lat"])],
        crs=admin2_crs,
    )
    gdf = gdf[gdf.within(county_geom)]
    if len(gdf) == 0:
        return pd.Series(dtype="datetime64[ns]")
    dates = pd.Series(sorted(gdf["date"].unique()))
    gaps = dates.diff().dt.days.fillna(999)
    event_id = (gaps > 3).cumsum()
    return dates.groupby(event_id).first()


def get_county_rainfall(rain_ds, bbox):
    lat_descending = bool(rain_ds.latitude.values[0] > rain_ds.latitude.values[-1])
    lat_slice = slice(bbox["lat_max"], bbox["lat_min"]) if lat_descending \
        else slice(bbox["lat_min"], bbox["lat_max"])
    subset = rain_ds.sel(latitude=lat_slice, longitude=slice(bbox["lon_min"], bbox["lon_max"]))
    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        return None, None
    tp = subset["tp"].mean(dim=["latitude", "longitude"]).compute().to_series()
    ro = subset["ro"].mean(dim=["latitude", "longitude"]).compute().to_series()
    tp.index = pd.to_datetime(tp.index)
    ro.index = pd.to_datetime(ro.index)
    return tp, ro


def main():
    section("Load shared data once (flood masks + rainfall for whole country)")
    admin1, admin2 = load_admin_boundaries()
    name_candidates = [c for c in admin2.columns if "adm2" in c.lower() and "name" in c.lower()]
    county_col = name_candidates[0] if name_candidates else admin2.columns[0]

    print("Loading flood mask data for the whole country bbox (reused for every county)...")
    flood_df_all = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df_all["date"] = pd.to_datetime(flood_df_all["date"])
    print(f"Loaded {len(flood_df_all):,} flood pixel-day records nationally.")

    print("\nLoading rainfall/runoff for the whole country grid (reused for every county)...")
    rain_ds_full = load_rainfall_runoff(YEARS)
    print("Rainfall/runoff loaded.")

    section("Per-county flood events + lead-time rainfall/runoff")
    all_lead_records = []
    per_county_event_counts = {}

    for county_name in TARGET_COUNTIES:
        county_geom, bbox = get_county_bbox(admin2, county_col, county_name)
        if county_geom is None:
            print(f"  SKIP '{county_name}': not found in admin2 boundaries.")
            continue

        event_starts = get_flood_events(flood_df_all, county_geom, admin2.crs)
        if len(event_starts) == 0:
            print(f"  SKIP '{county_name}': no flood events found.")
            continue

        tp_series, ro_series = get_county_rainfall(rain_ds_full, bbox)
        if tp_series is None:
            print(f"  SKIP '{county_name}': rainfall grid subset came back empty.")
            continue

        per_county_event_counts[county_name] = len(event_starts)
        print(f"  {county_name}: {len(event_starts)} flood events")

        for event_date in event_starts:
            for lead in range(0, LEAD_DAYS + 1):
                check_date = event_date - pd.Timedelta(days=lead)
                if check_date in tp_series.index:
                    all_lead_records.append({
                        "county": county_name,
                        "event_date": event_date,
                        "lead_days": lead,
                        "rainfall": tp_series.loc[check_date],
                        "runoff": ro_series.loc[check_date],
                    })

    lead_df = pd.DataFrame(all_lead_records)
    total_events = sum(per_county_event_counts.values())
    print(f"\nPooled {total_events} flood events across {len(per_county_event_counts)} counties "
          f"({len(lead_df)} lead-time data points total).")

    section("Baseline across all counties (random non-flood days)")
    baseline_rain_vals, baseline_runoff_vals = [], []
    rng = np.random.default_rng(42)
    for county_name in per_county_event_counts:
        county_geom, bbox = get_county_bbox(admin2, county_col, county_name)
        tp_series, ro_series = get_county_rainfall(rain_ds_full, bbox)
        event_starts = get_flood_events(flood_df_all, county_geom, admin2.crs)
        non_flood_dates = tp_series.index.difference(pd.DatetimeIndex(event_starts))
        sample_n = min(100, len(non_flood_dates))
        if sample_n == 0:
            continue
        sample = rng.choice(non_flood_dates, size=sample_n, replace=False)
        baseline_rain_vals.extend(tp_series.loc[pd.DatetimeIndex(sample)].tolist())
        baseline_runoff_vals.extend(ro_series.loc[pd.DatetimeIndex(sample)].tolist())

    baseline_rain = np.mean(baseline_rain_vals)
    baseline_runoff = np.mean(baseline_runoff_vals)
    print(f"Pooled baseline rainfall (typical non-flood day): {baseline_rain:.5f} m/day")
    print(f"Pooled baseline runoff (typical non-flood day):   {baseline_runoff:.5f} m/day")

    section("Pooled average rainfall/runoff by lead time")
    profile = lead_df.groupby("lead_days")[["rainfall", "runoff"]].mean().sort_index()
    profile_std = lead_df.groupby("lead_days")[["rainfall", "runoff"]].std().sort_index()
    print("\nPooled average rainfall/runoff at each lead time (all counties combined):")
    print(profile.round(5))

    profile.to_csv(OUT_DIR / "14_rainfall_lead_time_allcounties.csv")
    print(f"\nSaved: {OUT_DIR / '14_rainfall_lead_time_allcounties.csv'}")

    section("Per-county event counts (check before trusting the pooled shape)")
    print(pd.Series(per_county_event_counts).sort_values(ascending=False).to_string())
    print("\nIf one or two counties dominate the event count, the pooled pattern mostly "
          "reflects those counties, not a true national average.")

    section("Plot - pooled rainfall/runoff lead-time profile, with spread")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(profile.index, profile["rainfall"], marker="o", color="tab:blue", linewidth=2)
    axes[0].fill_between(profile.index,
                          profile["rainfall"] - profile_std["rainfall"],
                          profile["rainfall"] + profile_std["rainfall"],
                          color="tab:blue", alpha=0.15)
    axes[0].axhline(baseline_rain, color="gray", linestyle="--", linewidth=1.2)
    axes[0].text(LEAD_DAYS * 0.7, baseline_rain, "typical (non-flood) day", fontsize=9,
                 va="bottom", color="gray")
    axes[0].set_ylabel("Rainfall (m/day)")
    axes[0].set_title(f"Rainfall before a flood event - {len(per_county_event_counts)} counties, "
                       f"{total_events} events pooled")
    axes[0].invert_xaxis()

    axes[1].plot(profile.index, profile["runoff"], marker="o", color="tab:orange", linewidth=2)
    axes[1].fill_between(profile.index,
                          profile["runoff"] - profile_std["runoff"],
                          profile["runoff"] + profile_std["runoff"],
                          color="tab:orange", alpha=0.15)
    axes[1].axhline(baseline_runoff, color="gray", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Runoff (m/day)")
    axes[1].set_xlabel("Days before flood event (0 = flood day)")
    axes[1].set_title(f"Runoff before a flood event - {len(per_county_event_counts)} counties, "
                       f"{total_events} events pooled")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "14_rainfall_lead_time_allcounties.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '14_rainfall_lead_time_allcounties.png'}")
    return profile, per_county_event_counts


if __name__ == "__main__":
    profile, per_county_event_counts = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")