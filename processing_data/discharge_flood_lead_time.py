
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

STATION_ID = 100205       # South Sudan's only in-country discharge station
STATION_LAT = 9.6
STATION_LON = 31.6
BOX_RADIUS_DEG = 1.0       # bounding box half-width around the station
LEAD_DAYS = 14
YEARS = np.arange(2015, 2026)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section(f"Load discharge for station {STATION_ID}")
    all_stations = load_dartmouth_data()
    if STATION_ID not in all_stations:
        raise SystemExit(f"\nStation {STATION_ID} not found. Available: {list(all_stations.keys())}")
    discharge = all_stations[STATION_ID]
    discharge_col = discharge.columns[0]
    discharge_series = discharge[discharge_col]
    print(f"Loaded {len(discharge_series)} daily discharge readings, "
          f"{discharge_series.index.min().date()} to {discharge_series.index.max().date()}")

    section("Find flood events near the station (bounding box around it)")
    bbox = {
        "lon_min": STATION_LON - BOX_RADIUS_DEG, "lat_min": STATION_LAT - BOX_RADIUS_DEG,
        "lon_max": STATION_LON + BOX_RADIUS_DEG, "lat_max": STATION_LAT + BOX_RADIUS_DEG,
    }
    print(f"Bounding box around station: {bbox}")
    flood_df = load_flood_masks(YEARS, bbox=bbox)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    print(f"{len(flood_df):,} flood pixel-day records found in this box.")

    flood_dates = pd.Series(sorted(flood_df["date"].unique()))
    gaps = flood_dates.diff().dt.days.fillna(999)
    event_id = (gaps > 3).cumsum()
    event_starts = flood_dates.groupby(event_id).first()
    print(f"{len(flood_dates)} distinct flood dates collapse into {len(event_starts)} events.")

    section("Discharge at each lead time before flood events")
    lead_records = []
    for event_date in event_starts:
        for lead in range(0, LEAD_DAYS + 1):
            check_date = event_date - pd.Timedelta(days=lead)
            if check_date in discharge_series.index:
                lead_records.append({
                    "event_date": event_date,
                    "lead_days": lead,
                    "discharge": discharge_series.loc[check_date],
                })
    lead_df = pd.DataFrame(lead_records)
    print(f"{len(lead_df)} lead-time discharge readings collected.")

    section("Baseline discharge (random non-flood days)")
    non_flood_dates = discharge_series.index.difference(pd.DatetimeIndex(event_starts))
    rng = np.random.default_rng(42)
    sample_n = min(300, len(non_flood_dates))
    sample = rng.choice(non_flood_dates, size=sample_n, replace=False)
    baseline_discharge = discharge_series.loc[pd.DatetimeIndex(sample)].mean()
    print(f"Baseline (typical non-flood day) discharge: {baseline_discharge:.1f} m3/s")

    section("Average discharge by lead time")
    profile = lead_df.groupby("lead_days")["discharge"].mean().sort_index()
    profile_std = lead_df.groupby("lead_days")["discharge"].std().sort_index()
    print("\nAverage discharge at each lead time before a flood event "
          "(lead_days=0 is the flood day itself):")
    print(profile.round(1))

    profile.to_csv(OUT_DIR / "16_discharge_lead_time.csv")
    print(f"\nSaved: {OUT_DIR / '16_discharge_lead_time.csv'}")

    section("STEP 6: Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profile.index, profile.values, marker="o", color="tab:purple", linewidth=2)
    ax.fill_between(profile.index, profile - profile_std, profile + profile_std,
                     color="tab:purple", alpha=0.15)
    ax.axhline(baseline_discharge, color="gray", linestyle="--", linewidth=1.2)
    ax.text(LEAD_DAYS * 0.7, baseline_discharge, "typical (non-flood) day", fontsize=9,
            va="bottom", color="gray")
    ax.set_xlabel("Days before flood event (0 = flood day)")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"Discharge before flood events near station {STATION_ID} "
                 f"({len(event_starts)} events)")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "16_discharge_lead_time.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '16_discharge_lead_time.png'}")

    return profile, event_starts


if __name__ == "__main__":
    profile, event_starts = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")
