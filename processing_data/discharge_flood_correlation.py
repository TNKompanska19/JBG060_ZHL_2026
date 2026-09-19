import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from the project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
    )

from loading import load_dartmouth_data, load_flood_masks

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
MAX_LAG = 30
YEARS = np.arange(2015, 2026)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("Load all discharge stations + location info")
    all_stations = load_dartmouth_data()
    info = pd.read_excel("./raw_data/Darthmouth Flood Observatory/information.xlsx")
    col_map = {c.lower().strip(): c for c in info.columns}
    country_col = col_map.get("country")
    area_id_col = col_map.get("area id")
    lat_col = col_map.get("latitude")
    lon_col = col_map.get("longitude")

    upstream_ids = info.loc[info[country_col] != "South Sudan", area_id_col].tolist()
    print(info[[area_id_col, country_col, lat_col, lon_col]].to_string(index=False))

    section("Build DAILY South Sudan flood extent time series")
    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    print(f"{len(flood_df):,} flood pixel-day records loaded nationally.")

    daily_extent = flood_df.groupby("date").size()
    full_date_range = pd.date_range(daily_extent.index.min(), daily_extent.index.max(), freq="D")
    daily_extent = daily_extent.reindex(full_date_range, fill_value=0)
    print(f"Daily flood extent series built: {len(daily_extent)} days, "
          f"{(daily_extent > 0).sum()} days with any flooding, "
          f"max {daily_extent.max():,} pixels flooded in a single day.")

    section("Correlation vs. lag, per station")
    lags = np.arange(0, MAX_LAG + 1)
    results = {}

    for station_id in upstream_ids:
        if station_id not in all_stations:
            continue
        station_country = info.loc[info[area_id_col] == station_id, country_col].values[0]
        discharge = all_stations[station_id].iloc[:, 0]

        corrs = []
        for lag in lags:

            shifted_flood = daily_extent.shift(-lag)
            common_idx = discharge.index.intersection(shifted_flood.dropna().index)
            if len(common_idx) < 30:
                corrs.append(np.nan)
                continue
            corr = discharge.loc[common_idx].corr(shifted_flood.loc[common_idx])
            corrs.append(corr)

        results[station_id] = {"country": station_country, "corrs": pd.Series(corrs, index=lags)}
        best_lag = results[station_id]["corrs"].idxmax()
        best_corr = results[station_id]["corrs"].max()
        print(f"  Station {station_id} ({station_country}): "
              f"best correlation = {best_corr:.3f} at lag {best_lag} days")

    section("Plot - correlation vs. lag, all stations")
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for (station_id, data), color in zip(results.items(), colors):
        ax.plot(data["corrs"].index, data["corrs"].values, marker="o", markersize=3,
                linewidth=1.3, color=color, label=f"{station_id} ({data['country']})")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (days) - discharge on day D vs. flood extent on day D+lag")
    ax.set_ylabel("Correlation (Pearson)")
    ax.set_title("Correlation between upstream discharge and South Sudan flood extent, by lag")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "18_discharge_flood_correlation_by_lag.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '18_discharge_flood_correlation_by_lag.png'}")


    section("STEP 5: Ranked summary - which stations show the strongest relationship")
    summary = pd.DataFrame([
        {"station_id": sid, "country": d["country"],
         "best_lag_days": d["corrs"].idxmax(), "best_correlation": round(d["corrs"].max(), 3)}
        for sid, d in results.items()
    ]).sort_values("best_correlation", ascending=False)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "18_discharge_flood_correlation_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / '18_discharge_flood_correlation_summary.csv'}")

    return summary


if __name__ == "__main__":
    summary = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")