
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

STATION_ID = 100205
SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS = np.arange(2015, 2026)
ANOMALY_START = "2022-01-01"   # the discharge dip period you spotted earlier
ANOMALY_END = "2023-12-31"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("Load discharge and build daily flood extent series")
    all_stations = load_dartmouth_data()
    discharge = all_stations[STATION_ID].iloc[:, 0]

    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    daily_extent = flood_df.groupby("date").size()
    full_range = pd.date_range(daily_extent.index.min(), daily_extent.index.max(), freq="D")
    daily_extent = daily_extent.reindex(full_range, fill_value=0)

    common_idx = discharge.index.intersection(daily_extent.index)
    discharge = discharge.loc[common_idx]
    daily_extent = daily_extent.loc[common_idx]
    print(f"Aligned {len(common_idx)} days of discharge + flood extent data.")

    full_corr = discharge.corr(daily_extent)
    print(f"Full-period correlation: {full_corr:.3f}")

    section("Plot both series together over time")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(discharge.index, discharge.values, color="tab:purple", linewidth=0.8)
    axes[0].set_ylabel("Discharge (m3/s)\nstation 100205")
    axes[0].axvspan(pd.Timestamp(ANOMALY_START), pd.Timestamp(ANOMALY_END),
                     color="orange", alpha=0.15, label="Discharge dip period")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].fill_between(daily_extent.index, daily_extent.values, color="tab:blue", alpha=0.6)
    axes[1].set_ylabel("Daily flood extent\n(pixels flooded, national)")
    axes[1].set_xlabel("Date")
    axes[1].axvspan(pd.Timestamp(ANOMALY_START), pd.Timestamp(ANOMALY_END),
                     color="orange", alpha=0.15)

    fig.suptitle("Station 100205 discharge vs. South Sudan daily flood extent, over time")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "20_discharge_vs_flood_timeseries.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '20_discharge_vs_flood_timeseries.png'}")

    section("Correlation WITHOUT the 2022-2023 anomaly period")
    outside_anomaly = (discharge.index < ANOMALY_START) | (discharge.index > ANOMALY_END)
    corr_excl = discharge.loc[outside_anomaly].corr(daily_extent.loc[outside_anomaly])
    print(f"Correlation EXCLUDING {ANOMALY_START} to {ANOMALY_END}: {corr_excl:.3f}")
    print(f"Correlation INCLUDING that period (full period):         {full_corr:.3f}")
    section("Seasonal pattern - average by month")
    monthly_discharge = discharge.groupby(discharge.index.month).mean()
    monthly_flood = daily_extent.groupby(daily_extent.index.month).mean()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    months = range(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax1.plot(months, monthly_discharge.reindex(months).values, marker="o",
             color="tab:purple", linewidth=2, label="Discharge")
    ax1.set_ylabel("Average discharge (m3/s)", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_labels)
    ax1.set_xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(months, monthly_flood.reindex(months).values, marker="s",
             color="tab:blue", linewidth=2, label="Flood extent")
    ax2.set_ylabel("Average daily flood extent (pixels)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Seasonal pattern: discharge vs. flood extent, averaged by month across all years")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "20_seasonal_discharge_vs_flood.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '20_seasonal_discharge_vs_flood.png'}")

    section("Save diagnostic data")
    diag = pd.DataFrame({"discharge": discharge, "flood_extent": daily_extent})
    diag.to_csv(OUT_DIR / "20_discharge_flood_daily_aligned.csv")
    print(f"Saved: {OUT_DIR / '20_discharge_flood_daily_aligned.csv'}")

    return full_corr, corr_excl


if __name__ == "__main__":
    full_corr, corr_excl = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")
    print(f"\nSummary: full-period correlation = {full_corr:.3f}, "
          f"excluding 2022-2023 anomaly = {corr_excl:.3f}")