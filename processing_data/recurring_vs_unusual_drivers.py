
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


def assign_counties_once(flood_df_all, admin2, county_col, target_counties):
    target_polys = admin2[admin2[county_col].isin(target_counties)][[county_col, "geometry"]]

    unique_locs = flood_df_all[["lat", "lon"]].drop_duplicates()
    loc_gdf = gpd.GeoDataFrame(
        unique_locs,
        geometry=[Point(lon, lat) for lon, lat in zip(unique_locs["lon"], unique_locs["lat"])],
        crs=admin2.crs,
    )
    joined = gpd.sjoin(loc_gdf, target_polys, how="inner", predicate="within")
    loc_to_county = joined.set_index(["lat", "lon"])[county_col]

    flood_df_with_county = flood_df_all.join(loc_to_county, on=["lat", "lon"])
    flood_df_with_county = flood_df_with_county.dropna(subset=[county_col])
    return flood_df_with_county


def get_flood_events(flood_df_with_county, county_col, county_name, flood_type):
    """Filter the pre-joined dataframe to one county + flood_type, collapse to events."""
    subset = flood_df_with_county[
        (flood_df_with_county[county_col] == county_name)
        & (flood_df_with_county["flood_type"] == flood_type)
    ]
    if len(subset) == 0:
        return pd.Series(dtype="datetime64[ns]")
    dates = pd.Series(sorted(subset["date"].unique()))
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


def build_lead_profile(flood_df_joined, county_col_name, admin2, rain_ds_full, flood_type, label):
    """Runs the full pooled lead-time pipeline for ONE flood_type across all counties."""
    print(f"\n--- Building profile for {label} floods (flood_type={flood_type}) ---")
    all_lead_records = []
    per_county_event_counts = {}

    for county_name in TARGET_COUNTIES:
        event_starts = get_flood_events(flood_df_joined, county_col_name, county_name, flood_type)
        if len(event_starts) == 0:
            print(f"  {county_name}: 0 {label} events - skipped")
            continue

        _, bbox = get_county_bbox(admin2, county_col_name, county_name)
        if bbox is None:
            continue
        tp_series, ro_series = get_county_rainfall(rain_ds_full, bbox)
        if tp_series is None:
            continue

        per_county_event_counts[county_name] = len(event_starts)
        print(f"  {county_name}: {len(event_starts)} {label} events")

        for event_date in event_starts:
            for lead in range(0, LEAD_DAYS + 1):
                check_date = event_date - pd.Timedelta(days=lead)
                if check_date in tp_series.index:
                    all_lead_records.append({
                        "county": county_name,
                        "lead_days": lead,
                        "rainfall": tp_series.loc[check_date],
                        "runoff": ro_series.loc[check_date],
                    })

    lead_df = pd.DataFrame(all_lead_records)
    total_events = sum(per_county_event_counts.values())
    print(f"  TOTAL: {total_events} {label} events pooled across "
          f"{len(per_county_event_counts)} counties ({len(lead_df)} lead-time points)")

    if len(lead_df) == 0:
        return None, per_county_event_counts, total_events

    profile = lead_df.groupby("lead_days")[["rainfall", "runoff"]].mean().sort_index()
    return profile, per_county_event_counts, total_events


def main():
    section("Load shared data once (flood masks + rainfall for whole country)")
    admin1, admin2 = load_admin_boundaries()
    name_candidates = [c for c in admin2.columns if "adm2" in c.lower() and "name" in c.lower()]
    county_col = name_candidates[0] if name_candidates else admin2.columns[0]

    print("Loading flood mask data (whole country bbox, reused for every county/type)...")
    flood_df_all = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df_all["date"] = pd.to_datetime(flood_df_all["date"])
    print(f"Loaded {len(flood_df_all):,} flood pixel-day records nationally "
          f"({(flood_df_all['flood_type']==0).sum():,} recurring, "
          f"{(flood_df_all['flood_type']==1).sum():,} unusual).")

    print("\nAssigning flood points to counties ONCE (single spatial join, "
          "not one join per county - this is the main speedup vs. re-checking "
          ".within() against each county's polygon separately)...")
    flood_df_joined = assign_counties_once(flood_df_all, admin2, county_col, TARGET_COUNTIES)
    print(f"{len(flood_df_joined):,} flood pixel-day records matched to a target county.")

    print("\nLoading rainfall/runoff for the whole country grid (reused for every county)...")
    rain_ds_full = load_rainfall_runoff(YEARS)
    print("Rainfall/runoff loaded.")

    section("Build lead-time profile for RECURRING floods")
    recurring_profile, recurring_counts, recurring_total = build_lead_profile(
        flood_df_joined, county_col, admin2, rain_ds_full, flood_type=0, label="recurring"
    )

    section("Build lead-time profile for UNUSUAL floods")
    unusual_profile, unusual_counts, unusual_total = build_lead_profile(
        flood_df_joined, county_col, admin2, rain_ds_full, flood_type=1, label="unusual"
    )

    if recurring_profile is None or unusual_profile is None:
        raise SystemExit("\nOne of the two flood types had no events in the selected "
                          "counties/years - widen TARGET_COUNTIES or YEARS and re-run.")

    section("Save data")
    recurring_profile.to_csv(OUT_DIR / "15_recurring_flood_lead_time.csv")
    unusual_profile.to_csv(OUT_DIR / "15_unusual_flood_lead_time.csv")
    print(f"Saved: {OUT_DIR / '15_recurring_flood_lead_time.csv'}")
    print(f"Saved: {OUT_DIR / '15_unusual_flood_lead_time.csv'}")

    section("Side-by-side comparison plot")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(recurring_profile.index, recurring_profile["rainfall"],
                 marker="o", color="tab:blue", linewidth=2, label=f"Recurring (n={recurring_total})")
    axes[0].plot(unusual_profile.index, unusual_profile["rainfall"],
                 marker="s", color="tab:red", linewidth=2, label=f"Unusual (n={unusual_total})")
    axes[0].set_ylabel("Rainfall (m/day)")
    axes[0].set_title("Rainfall before flood events: recurring vs. unusual")
    axes[0].invert_xaxis()
    axes[0].legend()

    axes[1].plot(recurring_profile.index, recurring_profile["runoff"],
                 marker="o", color="tab:blue", linewidth=2, label=f"Recurring (n={recurring_total})")
    axes[1].plot(unusual_profile.index, unusual_profile["runoff"],
                 marker="s", color="tab:red", linewidth=2, label=f"Unusual (n={unusual_total})")
    axes[1].set_ylabel("Runoff (m/day)")
    axes[1].set_xlabel("Days before flood event (0 = flood day)")
    axes[1].set_title("Runoff before flood events: recurring vs. unusual")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "15_recurring_vs_unusual_lead_time.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '15_recurring_vs_unusual_lead_time.png'}")

    section("Event count comparison by county")
    compare = pd.DataFrame({
        "recurring_events": pd.Series(recurring_counts),
        "unusual_events": pd.Series(unusual_counts),
    }).fillna(0).astype(int)
    compare["ratio_unusual_to_recurring"] = (
        compare["unusual_events"] / compare["recurring_events"].replace(0, np.nan)
    ).round(2)
    print(compare.sort_values("recurring_events", ascending=False).to_string())
    compare.to_csv(OUT_DIR / "15_event_counts_by_type.csv")
    print(f"\nSaved: {OUT_DIR / '15_event_counts_by_type.csv'}")

    return recurring_profile, unusual_profile


if __name__ == "__main__":
    recurring_profile, unusual_profile = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")