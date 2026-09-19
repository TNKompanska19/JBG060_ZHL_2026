"""
Robustness Check: Does the Rainfall Lead-Time Pattern Survive Detection Bias?
(JBG060 - South Sudan)

Follow-up to data_quality_2021_shift.py, which found a strong correlation
(-0.773) between monthly flood-detection volume and mean rainfall at flood
points - a warning sign that the earlier "sustained wetness, not a spike"
rainfall lead-time finding might partly be a detection artifact (only the
most severe/rainy floods get detected during low-visibility months) rather
than a true hydrological signal.

APPROACH
--------
1. Classify each calendar month as HIGH-detection or LOW-detection, based
   on the monthly detection volume already measured (wet season months
   Jun-Sep were found to have ~4.5x LOWER detection than other months).
2. Split flood EVENTS into two groups by which month they occurred in.
3. Rebuild the same 14-day rainfall/runoff lead-time profile SEPARATELY
   for each group.
4. Compare the two profiles directly:
   - If both groups show the SAME shape (sustained wetness, no clean
     spike), the pattern is likely real and not a detection artifact -
     it holds regardless of how reliable detection was.
   - If the LOW-detection group shows a much higher/different rainfall
     profile than the HIGH-detection group, that's evidence the earlier
     finding was distorted by which floods happened to be detectable.

CAVEATS
-------
This doesn't fully resolve the issue (detection reliability likely varies
within a month too, not just between months), but it's a strong first test:
if the pattern doesn't even survive a coarse high/low split, it needs
serious re-examination before going in the final report.

Run from your project root.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from your project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
        f"Example:\n"
        f"  cd D:\\TUE\\Y3\\JBG060_ZHL_2026\n"
        f"  python processing_data\\rainfall_detection_bias_check.py\n"
    )

from loading import load_flood_masks, load_rainfall_runoff

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS = np.arange(2015, 2026)
LEAD_DAYS = 14

# Directly from your measured monthly detection volumes (data_quality_2021_shift.py
# output) - low-detection months are the confirmed wet-season/cloud-obscured ones.
LOW_DETECTION_MONTHS = [4, 5, 6, 7, 8, 9]     # under ~2.3M detections/month
HIGH_DETECTION_MONTHS = [1, 2, 3, 10, 11, 12]  # over ~3.2M detections/month


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_lead_profile(event_dates, rain_series, runoff_series, label):
    """Builds the pooled 14-day lead-time profile for a given set of event dates."""
    records = []
    for event_date in event_dates:
        for lead in range(0, LEAD_DAYS + 1):
            check_date = event_date - pd.Timedelta(days=lead)
            if check_date in rain_series.index:
                records.append({
                    "lead_days": lead,
                    "rainfall": rain_series.loc[check_date],
                    "runoff": runoff_series.loc[check_date],
                })
    df = pd.DataFrame(records)
    print(f"  {label}: {len(event_dates)} events, {len(df)} lead-time data points")
    if len(df) == 0:
        return None
    return df.groupby("lead_days")[["rainfall", "runoff"]].mean().sort_index()


def main():
    section("STEP 1: Load national flood events and rainfall/runoff")
    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    print(f"{len(flood_df):,} flood pixel-day records loaded.")

    rain_ds = load_rainfall_runoff(YEARS)
    tp_national = rain_ds["tp"].mean(dim=["latitude", "longitude"]).compute().to_series()
    ro_national = rain_ds["ro"].mean(dim=["latitude", "longitude"]).compute().to_series()
    tp_national.index = pd.to_datetime(tp_national.index)
    ro_national.index = pd.to_datetime(ro_national.index)

    section("STEP 2: Build national flood ONSET events (extent rising), split by month group")
    # The earlier "any day with flooding nationally" event definition breaks
    # at national scale: with flooding present almost every day somewhere in
    # the country (especially post-2021), gap>3-day event grouping barely
    # ever splits the series - it collapsed into just 1-3 giant "events"
    # covering years each, which is why the previous run showed n=1-2.
    #
    # Instead, define an "onset day" as a day where national flood extent
    # rises SHARPLY above its recent trend - a genuine escalation, not just
    # "flooding present". This gives many distinct, meaningful reference
    # points instead of a handful of multi-year blobs.
    daily_extent = flood_df.groupby("date").size()
    full_range = pd.date_range(daily_extent.index.min(), daily_extent.index.max(), freq="D")
    daily_extent = daily_extent.reindex(full_range, fill_value=0)

    # 7-day rolling average as the "recent trend"; an onset day is one where
    # today's extent exceeds that trailing average by a wide margin (50%)
    # AND is a local increase (today > yesterday), so it captures genuine
    # escalation points rather than every day of an already-large flood.
    rolling_avg = daily_extent.rolling(7, min_periods=3).mean()
    is_rising = daily_extent.diff() > 0
    is_spike = daily_extent > rolling_avg * 1.5
    onset_days = daily_extent.index[is_rising & is_spike]

    # Still collapse consecutive onset days within 3 days of each other into
    # one event, so we don't double-count the same escalation multiple times.
    onset_series = pd.Series(sorted(onset_days))
    if len(onset_series) == 0:
        raise SystemExit("\nNo onset days found - check YEARS/bbox settings.")
    gaps = onset_series.diff().dt.days.fillna(999)
    event_id = (gaps > 3).cumsum()
    event_starts = onset_series.groupby(event_id).first()
    print(f"{len(event_starts)} national flood ONSET events found (2015-2025).")

    event_months = event_starts.dt.month
    low_events = event_starts[event_months.isin(LOW_DETECTION_MONTHS)]
    high_events = event_starts[event_months.isin(HIGH_DETECTION_MONTHS)]
    print(f"  Low-detection-month events (Apr-Sep):  {len(low_events)}")
    print(f"  High-detection-month events (Oct-Mar): {len(high_events)}")

    if len(low_events) < 10 or len(high_events) < 10:
        print(f"\nWARNING: one group has very few events (<10) - results below will be noisy. "
              f"Consider loosening the onset threshold (currently extent > 1.5x its 7-day "
              f"rolling average) if this keeps happening.")

    section("STEP 3: Build lead-time profiles for each group separately")
    low_profile = build_lead_profile(low_events, tp_national, ro_national, "Low-detection events")
    high_profile = build_lead_profile(high_events, tp_national, ro_national, "High-detection events")

    if low_profile is None or high_profile is None:
        raise SystemExit("\nOne of the two groups had no valid events - cannot compare.")

    low_profile.to_csv(OUT_DIR / "22_lowdetect_lead_time.csv")
    high_profile.to_csv(OUT_DIR / "22_highdetect_lead_time.csv")

    section("STEP 4: Compare the two profiles directly")
    print("\nLow-detection-month events:")
    print(low_profile.round(5))
    print("\nHigh-detection-month events:")
    print(high_profile.round(5))

    # Quantify how different the two shapes are: correlate the two profiles'
    # values across lead_days (high correlation = similar SHAPE even if
    # different absolute level; also compare absolute levels directly).
    shape_corr_rain = low_profile["rainfall"].corr(high_profile["rainfall"])
    shape_corr_runoff = low_profile["runoff"].corr(high_profile["runoff"])
    level_ratio_rain = low_profile["rainfall"].mean() / high_profile["rainfall"].mean()
    level_ratio_runoff = low_profile["runoff"].mean() / high_profile["runoff"].mean()

    print(f"\nShape correlation (rainfall, across lead_days): {shape_corr_rain:.3f}")
    print(f"Shape correlation (runoff, across lead_days):   {shape_corr_runoff:.3f}")
    print(f"Level ratio (low-detection mean / high-detection mean), rainfall: {level_ratio_rain:.2f}x")
    print(f"Level ratio (low-detection mean / high-detection mean), runoff:   {level_ratio_runoff:.2f}x")

    if shape_corr_rain > 0.7 and 0.5 < level_ratio_rain < 2.0:
        print("\n>> REASSURING: shapes are similar and absolute levels are within 2x of each "
              "other - the 'sustained wetness' pattern appears to hold regardless of detection "
              "reliability, NOT purely a detection artifact.")
    else:
        print("\n>> WARNING: shapes and/or levels differ substantially between the two groups - "
              "the earlier rainfall lead-time finding may be distorted by detection bias, and "
              "should be caveated or re-examined before being presented as a solid finding.")

    section("STEP 5: Plot - side by side comparison")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(low_profile.index, low_profile["rainfall"], marker="o", color="tab:red",
                 linewidth=2, label=f"Low-detection months (n={len(low_events)} events)")
    axes[0].plot(high_profile.index, high_profile["rainfall"], marker="s", color="tab:blue",
                 linewidth=2, label=f"High-detection months (n={len(high_events)} events)")
    axes[0].set_ylabel("Rainfall (m/day)")
    axes[0].set_title("Rainfall before flood events: low- vs. high-detection months")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=9)

    axes[1].plot(low_profile.index, low_profile["runoff"], marker="o", color="tab:red",
                 linewidth=2, label=f"Low-detection months (n={len(low_events)} events)")
    axes[1].plot(high_profile.index, high_profile["runoff"], marker="s", color="tab:blue",
                 linewidth=2, label=f"High-detection months (n={len(high_events)} events)")
    axes[1].set_ylabel("Runoff (m/day)")
    axes[1].set_xlabel("Days before flood event (0 = flood day)")
    axes[1].set_title("Runoff before flood events: low- vs. high-detection months")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "22_detection_bias_robustness_check.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '22_detection_bias_robustness_check.png'}")
    print("\nRead this as: if red (low-detection) and blue (high-detection) lines have a "
          "SIMILAR SHAPE (both flat/noisy, both rising, etc.) even if at different absolute "
          "levels, the pattern is likely real. If one group shows a completely different "
          "shape than the other, the original pooled finding was likely an artifact of mixing "
          "reliable and unreliable detection periods together.")

    return low_profile, high_profile


if __name__ == "__main__":
    low_profile, high_profile = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")