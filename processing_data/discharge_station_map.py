
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

from loading_impact_data import load_admin_boundaries

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

CORR_SUMMARY_CSV = OUT_DIR / "18_discharge_flood_correlation_summary.csv"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("Load station coordinates")
    info = pd.read_excel("./raw_data/Darthmouth Flood Observatory/information.xlsx")
    col_map = {c.lower().strip(): c for c in info.columns}
    area_id_col = col_map.get("area id")
    country_col = col_map.get("country")
    lat_col = col_map.get("latitude")
    lon_col = col_map.get("longitude")
    print(info[[area_id_col, country_col, lat_col, lon_col]].to_string(index=False))

    section("Load correlation results from the previous script")
    if not CORR_SUMMARY_CSV.exists():
        raise SystemExit(
            f"\n{CORR_SUMMARY_CSV} not found. Run discharge_flood_correlation.py first."
        )
    corr_summary = pd.read_csv(CORR_SUMMARY_CSV)
    print(corr_summary.to_string(index=False))

    # Merge coordinates onto the correlation results
    merged = corr_summary.merge(
        info[[area_id_col, lat_col, lon_col]],
        left_on="station_id", right_on=area_id_col, how="left"
    )

    ssd_station = info[info[country_col] == "South Sudan"]
    if not ssd_station.empty:
        ssd_row = ssd_station.iloc[0]
        print(f"\n(For reference: South Sudan's own station {ssd_row[area_id_col]} "
              f"at lat={ssd_row[lat_col]}, lon={ssd_row[lon_col]})")

    section("Load South Sudan boundary")
    admin1, admin2 = load_admin_boundaries()
    ssd_boundary = admin1.dissolve()

    section("Compute distance from each station to South Sudan's border")
    ssd_geom = ssd_boundary.geometry.iloc[0]
    merged["distance_to_ssd_deg"] = merged.apply(
        lambda row: Point(row[lon_col], row[lat_col]).distance(ssd_geom), axis=1
    )
    merged["distance_to_ssd_km_approx"] = (merged["distance_to_ssd_deg"] * 111).round(0)
    print(merged[["station_id", "country", "best_correlation", "distance_to_ssd_km_approx"]]
          .sort_values("best_correlation", ascending=False).to_string(index=False))
    merged.to_csv(OUT_DIR / "19_station_distance_vs_correlation.csv", index=False)
    print(f"\nSaved: {OUT_DIR / '19_station_distance_vs_correlation.csv'}")

    section("Map - stations colored by correlation strength")
    fig, ax = plt.subplots(figsize=(9, 10))
    ssd_boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
    admin1.boundary.plot(ax=ax, color="gray", linewidth=0.4)

    scatter = ax.scatter(
        merged[lon_col], merged[lat_col],
        c=merged["best_correlation"], cmap="RdBu_r", vmin=-0.7, vmax=0.7,
        s=200, edgecolors="black", linewidths=1, zorder=5,
    )
    for _, row in merged.iterrows():
        ax.annotate(f"{int(row['station_id'])}\n(r={row['best_correlation']:.2f})",
                     (row[lon_col], row[lat_col]), fontsize=8, ha="left",
                     xytext=(5, 5), textcoords="offset points")

    # Mark South Sudan's own station too
    if not ssd_station.empty:
        ax.scatter(ssd_row[lon_col], ssd_row[lat_col], c="lime", s=250,
                   edgecolors="black", linewidths=1.5, marker="*", zorder=6,
                   label="South Sudan's own station (100205)")
        ax.legend(loc="lower left", fontsize=8)

    plt.colorbar(scatter, label="Correlation with South Sudan flood extent")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Discharge stations vs. South Sudan boundary\n(color = correlation strength)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "19_station_map.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '19_station_map.png'}")

    return merged


if __name__ == "__main__":
    merged = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")