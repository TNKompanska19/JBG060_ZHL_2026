import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

_required = ["raw_data", "processing_data"]
_missing = [d for d in _required if not Path(d).is_dir()]
if _missing:
    raise SystemExit(
        f"\nERROR: run this script from the project root, not from '{Path.cwd()}'.\n"
        f"Missing expected folder(s): {_missing}\n"
    )

from loading import load_dartmouth_data, load_flood_masks, load_rainfall_runoff

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS = np.arange(2015, 2026)
SHIFT_DATE = "2021-01-01"
WET_SEASON_MONTHS = [6, 7, 8, 9]  # Jun-Sep, per the group's own claim


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compare_before_after(series, label, shift_date=SHIFT_DATE):
    """Runs a Mann-Whitney U test comparing a series before vs. after shift_date."""
    before = series[series.index < shift_date].dropna()
    after = series[series.index >= shift_date].dropna()
    if len(before) < 10 or len(after) < 10:
        print(f"  {label}: not enough data on one side to test (before={len(before)}, after={len(after)})")
        return None
    stat, pval = stats.mannwhitneyu(before, after, alternative="two-sided")
    pct_change = (after.mean() - before.mean()) / before.mean() * 100 if before.mean() != 0 else np.nan
    print(f"  {label}: before mean={before.mean():.4f}, after mean={after.mean():.4f} "
          f"({pct_change:+.1f}%), Mann-Whitney p={pval:.2e} "
          f"{'*** SIGNIFICANT SHIFT' if pval < 0.01 else '(not significant at p<0.01)'}")
    return {"label": label, "before_mean": before.mean(), "after_mean": after.mean(),
            "pct_change": pct_change, "p_value": pval, "before": before, "after": after}


def main():
    section("Load all key variables")
    print("Loading national daily flood extent...")
    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    daily_extent = flood_df.groupby("date").size()
    full_range = pd.date_range(daily_extent.index.min(), daily_extent.index.max(), freq="D")
    daily_extent = daily_extent.reindex(full_range, fill_value=0)

    print("Loading rainfall/runoff...")
    rain_ds = load_rainfall_runoff(YEARS)
    tp_national = rain_ds["tp"].mean(dim=["latitude", "longitude"]).compute().to_series()
    ro_national = rain_ds["ro"].mean(dim=["latitude", "longitude"]).compute().to_series()
    tp_national.index = pd.to_datetime(tp_national.index)
    ro_national.index = pd.to_datetime(ro_national.index)

    print("Loading discharge (station 100205)...")
    all_stations = load_dartmouth_data()
    discharge = all_stations[100205].iloc[:, 0]

    section("Before vs. after 2021")
    results = {}
    results["flood_extent"] = compare_before_after(daily_extent, "National flood extent (pixels/day)")
    results["rainfall"] = compare_before_after(tp_national, "National mean rainfall (m/day)")
    results["runoff"] = compare_before_after(ro_national, "National mean runoff (m/day)")
    results["discharge"] = compare_before_after(discharge, "Discharge at 100205 (m3/s)")

    section("Boxplot comparison, before vs. after 2021")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, r) in zip(axes.flat, results.items()):
        if r is None:
            ax.set_visible(False)
            continue
        ax.boxplot([r["before"].values, r["after"].values], labels=["Before 2021", "After 2021"],
                   showfliers=False)
        ax.set_title(f"{r['label']}\np={r['p_value']:.2e}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "21_before_after_2021_boxplots.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '21_before_after_2021_boxplots.png'}")

    section("Flood detection volume by YEAR (reproduce the '50x variation' claim)")
    yearly_detection = flood_df.groupby(flood_df["date"].dt.year).size()
    print(yearly_detection.to_string())
    max_year, min_year = yearly_detection.idxmax(), yearly_detection.idxmin()
    ratio = yearly_detection.max() / yearly_detection.min()
    print(f"\nMax year: {max_year} ({yearly_detection.max():,} pixel-days)")
    print(f"Min year: {min_year} ({yearly_detection.min():,} pixel-days)")
    print(f"Ratio (max/min): {ratio:.1f}x")

    section("Flood detection volume by MONTH (reproduce 'wet season bias' claim)")
    monthly_detection = flood_df.groupby(flood_df["date"].dt.month).size()
    print(monthly_detection.to_string())
    wet_season_total = monthly_detection.loc[WET_SEASON_MONTHS].sum()
    other_months_total = monthly_detection.drop(WET_SEASON_MONTHS).sum()
    wet_season_avg = wet_season_total / len(WET_SEASON_MONTHS)
    other_avg = other_months_total / (12 - len(WET_SEASON_MONTHS))
    print(f"\nAverage detections/month in wet season (Jun-Sep): {wet_season_avg:,.0f}")
    print(f"Average detections/month in other months:          {other_avg:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(yearly_detection.index.astype(str), yearly_detection.values, color="steelblue")
    axes[0].set_title("Flood detections by year")
    axes[0].set_ylabel("Pixel-day detections")
    axes[0].tick_params(axis="x", rotation=45)

    colors = ["orange" if m in WET_SEASON_MONTHS else "steelblue" for m in monthly_detection.index]
    axes[1].bar([str(m) for m in monthly_detection.index], monthly_detection.values, color=colors)
    axes[1].set_title("Flood detections by month\n(orange = claimed wet season Jun-Sep)")
    axes[1].set_ylabel("Pixel-day detections")
    axes[1].set_xlabel("Month")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "21_detection_volume_by_year_month.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '21_detection_volume_by_year_month.png'}")

    section("Does detection volume correlate with rainfall lead-time pattern strength?")
    flood_df["month"] = flood_df["date"].dt.month
    monthly_rain_at_floods = flood_df.merge(
        tp_national.rename("rainfall"), left_on="date", right_index=True, how="left"
    ).groupby("month")["rainfall"].mean()

    comparison = pd.DataFrame({
        "detection_volume": monthly_detection,
        "mean_rainfall_at_flood_points": monthly_rain_at_floods,
    })
    print(comparison.round(6).to_string())
    corr = comparison["detection_volume"].corr(comparison["mean_rainfall_at_flood_points"])
    print(f"\nCorrelation between monthly detection volume and mean rainfall at flood points: {corr:.3f}")

    comparison.to_csv(OUT_DIR / "21_detection_vs_rainfall_by_month.csv")
    print(f"\nSaved: {OUT_DIR / '21_detection_vs_rainfall_by_month.csv'}")

    section("SUMMARY - bring this to your meeting")
    print("2021 SHIFT:")
    for key, r in results.items():
        if r is not None:
            flag = "SIGNIFICANT" if r["p_value"] < 0.01 else "not significant"
            print(f"  {r['label']}: {r['pct_change']:+.1f}% change, {flag}")
    print(f"\nDETECTION BIAS:")
    print(f"  Year-to-year ratio: {ratio:.1f}x (group's claim: ~50x)")
    print(f"  Wet season vs. other months: {'LOWER' if wet_season_avg < other_avg else 'NOT lower'} "
          f"detection in wet season")
    print(f"  Detection volume vs. rainfall-at-flood-points correlation: {corr:.3f}")

    return results, yearly_detection, monthly_detection


if __name__ == "__main__":
    results, yearly_detection, monthly_detection = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")