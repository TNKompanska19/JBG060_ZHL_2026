
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point
from scipy.spatial import cKDTree

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from the project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
    )

from loading import load_flood_masks
from loading_impact_data import load_admin_boundaries, load_farmland_mask

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS = np.arange(2000, 2026)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def nearest_neighbor_lookup(query_lats, query_lons, grid_lats, grid_lons, grid_values):
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)
    grid_points = np.column_stack([grid_lat_mesh.ravel(), grid_lon_mesh.ravel()])
    tree = cKDTree(grid_points)

    query_points = np.column_stack([query_lats, query_lons])
    _, nearest_idx = tree.query(query_points, k=1)

    return grid_values.ravel()[nearest_idx]


def main():
    section("Load flood mask points (unique locations ever flooded)")
    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)

    flood_points = (
        flood_df.groupby(["lat", "lon"])["flood_type"]
        .max()
        .reset_index()
    )
    print(f"Unique flooded locations: {len(flood_points):,}")
    print(f"  Recurring only: {(flood_points['flood_type'] == 0).sum():,}")
    print(f"  Unusual (at least once): {(flood_points['flood_type'] == 1).sum():,}")

    section("Load crop and rangeland masks")
    crop_da = load_farmland_mask(SSD_BBOX, "crop")
    range_da = load_farmland_mask(SSD_BBOX, "rangeland")
    print(f"Crop/rangeland grid shape: {crop_da.shape} "
          f"({crop_da.sizes['latitude']} lat x {crop_da.sizes['longitude']} lon)")

    grid_lats = crop_da["latitude"].values
    grid_lons = crop_da["longitude"].values
    crop_values = crop_da.values
    range_values = range_da.values

    section("Snap flood points to nearest crop/rangeland pixel")
    flood_points["crop_pct"] = nearest_neighbor_lookup(
        flood_points["lat"].values, flood_points["lon"].values,
        grid_lats, grid_lons, crop_values
    )
    flood_points["rangeland_pct"] = nearest_neighbor_lookup(
        flood_points["lat"].values, flood_points["lon"].values,
        grid_lats, grid_lons, range_values
    )

    section("Assign flood points to admin2 counties")
    admin1, admin2 = load_admin_boundaries()

    print(f"admin2 columns available: {admin2.columns.tolist()}")
    # Guess the county name column - adjust the candidates below if this
    # picks the wrong one; print statement above shows you all options.
    name_candidates = [c for c in admin2.columns if "adm2" in c.lower() and "name" in c.lower()]
    if not name_candidates:
        name_candidates = [c for c in admin2.columns if "name" in c.lower()]
    if not name_candidates:
        raise SystemExit(
            f"\nCould not auto-detect a county name column in admin2.columns "
            f"({admin2.columns.tolist()}). Set county_col manually below and re-run."
        )
    county_col = name_candidates[0]
    print(f"Using '{county_col}' as the county name column.")

    flood_gdf = gpd.GeoDataFrame(
        flood_points,
        geometry=[Point(lon, lat) for lon, lat in zip(flood_points["lon"], flood_points["lat"])],
        crs=admin2.crs,
    )
    # Spatial join: which county polygon contains each flood point
    flood_gdf = gpd.sjoin(flood_gdf, admin2[[county_col, "geometry"]],
                           how="left", predicate="within")

    n_unmatched = flood_gdf[county_col].isna().sum()
    if n_unmatched:
        print(f"NOTE: {n_unmatched:,} flood points ({n_unmatched/len(flood_gdf)*100:.1f}%) "
              f"fell outside all admin2 polygons (likely just outside the border, or bbox "
              f"edge effects) - excluded from per-county aggregation below.")
    flood_gdf = flood_gdf.dropna(subset=[county_col])

    section("Aggregate exposure by county")
    # "Exposed" = flooded location with >0% crop or rangeland cover at that spot
    flood_gdf["crop_exposed"] = flood_gdf["crop_pct"] > 0
    flood_gdf["rangeland_exposed"] = flood_gdf["rangeland_pct"] > 0

    county_summary = flood_gdf.groupby(county_col).agg(
        n_flooded_locations=("crop_pct", "size"),
        n_crop_exposed=("crop_exposed", "sum"),
        mean_crop_pct_at_flood=("crop_pct", "mean"),
        n_rangeland_exposed=("rangeland_exposed", "sum"),
        mean_rangeland_pct_at_flood=("rangeland_pct", "mean"),
        n_unusual_flood_points=("flood_type", lambda s: (s == 1).sum()),
        n_recurring_flood_points=("flood_type", lambda s: (s == 0).sum()),
    ).sort_values("n_flooded_locations", ascending=False)

    county_summary["pct_crop_exposed"] = (
        county_summary["n_crop_exposed"] / county_summary["n_flooded_locations"] * 100
    )
    county_summary["pct_rangeland_exposed"] = (
        county_summary["n_rangeland_exposed"] / county_summary["n_flooded_locations"] * 100
    )

    print("\nTop 15 counties by number of flooded locations:")
    print(county_summary.head(15)[
        ["n_flooded_locations", "pct_crop_exposed", "mean_crop_pct_at_flood",
         "pct_rangeland_exposed", "mean_rangeland_pct_at_flood"]
    ].round(1))

    print("\nTop 10 counties by SHARE of flood points that are cropland-exposed "
          "(min 100 flood points, to avoid tiny-sample noise):")
    reliable = county_summary[county_summary["n_flooded_locations"] >= 100]
    print(reliable.sort_values("pct_crop_exposed", ascending=False).head(10)[
        ["n_flooded_locations", "pct_crop_exposed", "pct_rangeland_exposed"]
    ].round(1))

    county_summary.to_csv(OUT_DIR / "10_flood_exposure_by_county.csv")
    print(f"\nSaved full table: {OUT_DIR / '10_flood_exposure_by_county.csv'}")

    section("STEP 5b: Flood volume vs. cropland share")
    # Plain scatter: does flooding more often relate to more cropland exposure?
    plot_data = county_summary[county_summary["n_flooded_locations"] >= 50].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        plot_data["n_flooded_locations"],
        plot_data["pct_crop_exposed"],
        s=80, color="tab:orange", alpha=0.7, edgecolors="black", linewidths=0.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of flooded locations (log scale) - how often this county floods")
    ax.set_ylabel("% of flood points with cropland present")
    ax.set_title("Flood volume vs. cropland exposure by county")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_flood_volume_vs_crop_share.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '10_flood_volume_vs_crop_share.png'}")

    section("STEP 5c: Flood volume vs. rangeland share")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        plot_data["n_flooded_locations"],
        plot_data["pct_rangeland_exposed"],
        s=80, color="tab:green", alpha=0.7, edgecolors="black", linewidths=0.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of flooded locations (log scale) - how often this county floods")
    ax.set_ylabel("% of flood points with rangeland present")
    ax.set_title("Flood volume vs. rangeland exposure by county")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_flood_volume_vs_rangeland_share.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '10_flood_volume_vs_rangeland_share.png'}")

    section("STEP 5d: Top 10 counties by cropland exposure and by rangeland exposure")
    reliable = county_summary[county_summary["n_flooded_locations"] >= 50].copy()

    top10_crop = reliable.sort_values("pct_crop_exposed", ascending=False).head(10)[
        ["n_flooded_locations", "pct_crop_exposed", "mean_crop_pct_at_flood"]
    ].round(1)
    print("\nTop 10 counties by % cropland-exposed (min 50 flood points):")
    print(top10_crop)

    top10_range = reliable.sort_values("pct_rangeland_exposed", ascending=False).head(10)[
        ["n_flooded_locations", "pct_rangeland_exposed", "mean_rangeland_pct_at_flood"]
    ].round(1)
    print("\nTop 10 counties by % rangeland-exposed (min 50 flood points):")
    print(top10_range)

    top10_crop.to_csv(OUT_DIR / "10_top10_cropland_exposed_counties.csv")
    top10_range.to_csv(OUT_DIR / "10_top10_rangeland_exposed_counties.csv")
    print(f"\nSaved: {OUT_DIR / '10_top10_cropland_exposed_counties.csv'}")
    print(f"Saved: {OUT_DIR / '10_top10_rangeland_exposed_counties.csv'}")

    section("STEP 6: Overall national picture")
    total = len(flood_points)
    crop_exposed_total = (flood_points["crop_pct"] > 0).sum()
    range_exposed_total = (flood_points["rangeland_pct"] > 0).sum()
    print(f"Of {total:,} unique flooded locations nationally:")
    print(f"  {crop_exposed_total:,} ({crop_exposed_total/total*100:.1f}%) "
          f"are at a location with SOME cropland cover")
    print(f"  {range_exposed_total:,} ({range_exposed_total/total*100:.1f}%) "
          f"are at a location with SOME rangeland cover")
    print(f"  Mean crop%      at flooded locations: {flood_points['crop_pct'].mean():.1f}%")
    print(f"  Mean rangeland% at flooded locations: {flood_points['rangeland_pct'].mean():.1f}%")

    return county_summary, flood_points


if __name__ == "__main__":
    county_summary, flood_points = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")