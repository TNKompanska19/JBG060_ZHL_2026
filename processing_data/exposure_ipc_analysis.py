
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
        f"\nERROR: run this script from your project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
        f"Example:\n"
        f"  cd D:\\TUE\\Y3\\JBG060_ZHL_2026\n"
        f"  python processing_data\\flood_vs_ipc_windows.py\n"
    )

from loading import load_flood_masks
from loading_impact_data import load_admin_boundaries, load_ipc_data

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def main():
    section("Load IPC data (defines the time windows we align flood data to)")
    ipc = load_ipc_data()
    ipc = ipc.sort_values("Start Date").reset_index(drop=True)

    ipc["Start Date"] = pd.to_datetime(ipc["Start Date"]).dt.tz_localize(None)
    ipc["End Date"] = pd.to_datetime(ipc["End Date"]).dt.tz_localize(None)

    county_cols = [c for c in ipc.columns if c not in ("Start Date", "End Date")]
    print(f"{len(ipc)} IPC time windows found:")
    for _, row in ipc.iterrows():
        print(f"  {row['Start Date'].date()} to {row['End Date'].date()}")

    if len(ipc) < 2:
        raise SystemExit("Need at least 2 IPC windows to compare window-over-window. Aborting.")

    section("Load flood mask data spanning all IPC windows")
    # Only need the years covered by the IPC windows, not the full 2000-2025 -
    # much faster than the full historical load.
    min_year = ipc["Start Date"].min().year
    max_year = ipc["End Date"].max().year
    years = np.arange(min_year, max_year + 1)
    print(f"Loading flood data for years {min_year}-{max_year} only "
          f"(matches IPC coverage, avoids loading the full 2000-2025 history).")
    flood_df = load_flood_masks(years, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    print(f"Loaded {len(flood_df):,} flood pixel-day records.")

    section("Assign flood points to admin2 counties")
    admin1, admin2 = load_admin_boundaries()
    name_candidates = [c for c in admin2.columns if "adm2" in c.lower() and "name" in c.lower()]
    county_col = name_candidates[0] if name_candidates else admin2.columns[0]
    print(f"Using '{county_col}' as the county name column.")

    unique_locs = flood_df[["lat", "lon"]].drop_duplicates()
    loc_gdf = gpd.GeoDataFrame(
        unique_locs,
        geometry=[Point(lon, lat) for lon, lat in zip(unique_locs["lon"], unique_locs["lat"])],
        crs=admin2.crs,
    )
    loc_gdf = gpd.sjoin(loc_gdf, admin2[[county_col, "geometry"]], how="left", predicate="within")
    loc_to_county = loc_gdf.set_index(["lat", "lon"])[county_col]

    flood_df = flood_df.join(loc_to_county, on=["lat", "lon"])
    n_unmatched = flood_df[county_col].isna().sum()
    if n_unmatched:
        print(f"NOTE: {n_unmatched:,} flood pixel-days ({n_unmatched/len(flood_df)*100:.1f}%) "
              f"had no matching county (bbox edge effects) - excluded below.")
    flood_df = flood_df.dropna(subset=[county_col])

    section("Count flood activity per county per IPC window")
    records = []
    for i, row in ipc.iterrows():
        window_mask = (flood_df["date"] >= row["Start Date"]) & (flood_df["date"] <= row["End Date"])
        window_floods = flood_df[window_mask]
        counts = window_floods.groupby(county_col).size()
        for county, n_floods in counts.items():
            records.append({
                "window_idx": i,
                "start_date": row["Start Date"],
                "county": county,
                "n_flood_pixeldays": n_floods,
            })
    flood_by_window = pd.DataFrame(records)
    print(f"Computed flood pixel-day counts for {flood_by_window['county'].nunique()} "
          f"counties across {ipc.shape[0]} IPC windows.")

    section("Match county names and build window-over-window comparison")
    ipc_indexed = ipc.reset_index(drop=True).copy()
    ipc_indexed["window_idx"] = ipc_indexed.index

    ipc_long = ipc_indexed.melt(
        id_vars=["window_idx", "Start Date", "End Date"], value_vars=county_cols,
        var_name="ipc_county", value_name="ipc_phase3plus"
    )
    ipc_long = ipc_long.dropna(subset=["ipc_phase3plus"])
    ipc_long["county_norm"] = ipc_long["ipc_county"].apply(normalize_name)

    flood_by_window["county_norm"] = flood_by_window["county"].apply(normalize_name)

    merged = pd.merge(
        flood_by_window, ipc_long,
        on=["window_idx", "county_norm"], how="inner"
    )
    print(f"Matched {merged['county_norm'].nunique()} counties with both flood and IPC data "
          f"across windows ({len(merged)} county-window rows total).")

    # Sort chronologically within each county, then compute window-over-window change
    merged = merged.sort_values(["county_norm", "window_idx"])
    merged["flood_change"] = merged.groupby("county_norm")["n_flood_pixeldays"].diff()
    merged["ipc_change"] = merged.groupby("county_norm")["ipc_phase3plus"].diff()

    change_data = merged.dropna(subset=["flood_change", "ipc_change"])
    print(f"\n{len(change_data)} county-window pairs have a valid window-over-window change "
          f"(i.e. not the first window for that county).")

    change_data.to_csv(OUT_DIR / "12_flood_ipc_window_changes.csv", index=False)
    print(f"Saved: {OUT_DIR / '12_flood_ipc_window_changes.csv'}")

    section("STEP 6: National totals per IPC window (for the timeline chart)")
    national = merged.groupby(["window_idx", "start_date"]).agg(
        total_ipc=("ipc_phase3plus", "sum"),
        total_flood_pixeldays=("n_flood_pixeldays", "sum"),
    ).reset_index().sort_values("start_date")
    print(national[["start_date", "total_ipc", "total_flood_pixeldays"]].to_string(index=False))

    section("STEP 7: Timeline chart - IPC over time, with flood activity marked")
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(national["start_date"], national["total_ipc"],
              marker="o", markersize=9, linewidth=2.5, color="#C62828", zorder=3)
    ax1.set_ylabel("People in IPC Phase 3+ (crisis or worse)", color="#C62828", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#C62828")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    bar_width = 40  # days - wide enough to see, narrow enough not to overlap
    ax2.bar(national["start_date"], national["total_flood_pixeldays"],
             width=bar_width, color="#90CAF9", alpha=0.6, zorder=1,
             label="Flood activity")
    ax2.set_ylabel("Flood activity (pixel-days, national total)", color="#1565C0", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#1565C0")

    for _, row in national.iterrows():
        ax1.annotate(f"{row['total_ipc']:,.0f}",
                      (row["start_date"], row["total_ipc"]),
                      textcoords="offset points", xytext=(0, 12),
                      ha="center", fontsize=9, color="#C62828", fontweight="bold")

    ax1.set_title("Food insecurity (red line) vs. flood activity (blue bars) over time",
                   fontsize=13)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_flood_ipc_timeline.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '12_flood_ipc_timeline.png'}")
    return national



if __name__ == "__main__":
    national = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")
