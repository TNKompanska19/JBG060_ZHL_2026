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

from loading import load_flood_masks

OUT_DIR = Path("./eda_output")
OUT_DIR.mkdir(exist_ok=True)

SSD_BBOX = {"lon_min": 23.5, "lat_min": 3.5, "lon_max": 36.0, "lat_max": 12.5}
YEARS = np.arange(2015, 2026)
SHIFT_DATE = "2021-01-01"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("Load flood mask data")
    flood_df = load_flood_masks(YEARS, bbox=SSD_BBOX)
    flood_df["date"] = pd.to_datetime(flood_df["date"])
    flood_df["period"] = np.where(flood_df["date"] < SHIFT_DATE, "before_2021", "after_2021")
    print(f"{len(flood_df):,} flood pixel-day records loaded.")

    n_years_before = (pd.Timestamp(SHIFT_DATE) - flood_df["date"].min()).days / 365.25
    n_years_after = (flood_df["date"].max() - pd.Timestamp(SHIFT_DATE)).days / 365.25
    print(f"Before 2021: ~{n_years_before:.1f} years of data")
    print(f"After 2021:  ~{n_years_after:.1f} years of data")

    section("Pixel-days (the original +764% finding)")
    pixeldays_by_period = flood_df.groupby("period").size()
    pixeldays_per_year = pixeldays_by_period / pd.Series(
        {"before_2021": n_years_before, "after_2021": n_years_after}
    )
    print("Total pixel-days:")
    print(pixeldays_by_period.to_string())
    print("\nPixel-days PER YEAR (normalizes for the two periods being different lengths):")
    print(pixeldays_per_year.round(0).to_string())
    pixday_ratio = pixeldays_per_year["after_2021"] / pixeldays_per_year["before_2021"]
    print(f"Ratio (after/before), per-year basis: {pixday_ratio:.2f}x")

    section("UNIQUE LOCATIONS ever flooded, before vs. after 2021")
    unique_locs_by_period = flood_df.groupby("period").apply(
        lambda g: g[["lat", "lon"]].drop_duplicates().shape[0], include_groups=False
    )
    print(unique_locs_by_period.to_string())
    loc_ratio = unique_locs_by_period["after_2021"] / unique_locs_by_period["before_2021"]
    print(f"Ratio (after/before) in NUMBER OF DISTINCT LOCATIONS ever flooded: {loc_ratio:.2f}x")

    section(" PERSISTENCE - average flood-days per location, before vs. after 2021")
    # For each period, average number of days each individual location was
    # recorded as flooded (a location flooded 20 separate days counts as 20).
    days_per_location = flood_df.groupby(["period", "lat", "lon"]).size().reset_index(name="n_days")
    persistence_by_period = days_per_location.groupby("period")["n_days"].mean()
    print(persistence_by_period.round(2).to_string())
    persist_ratio = persistence_by_period["after_2021"] / persistence_by_period["before_2021"]
    print(f"Ratio (after/before) in average days-per-location: {persist_ratio:.2f}x")

    section("Does the shift affect recurring vs. unusual floods differently?")
    type_period_counts = flood_df.groupby(["period", "flood_type"]).size().unstack(fill_value=0)
    type_period_counts.columns = ["recurring", "unusual"]
    print(type_period_counts.to_string())

    type_period_peryear = type_period_counts.div(
        pd.Series({"before_2021": n_years_before, "after_2021": n_years_after}), axis=0
    )
    recurring_ratio = type_period_peryear.loc["after_2021", "recurring"] / type_period_peryear.loc["before_2021", "recurring"]
    unusual_ratio = type_period_peryear.loc["after_2021", "unusual"] / type_period_peryear.loc["before_2021", "unusual"]
    print(f"\nRecurring floods, per-year ratio (after/before): {recurring_ratio:.2f}x")
    print(f"Unusual floods, per-year ratio (after/before):   {unusual_ratio:.2f}x")

    section("Plot - decomposition summary")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    period_order = ["before_2021", "after_2021"]
    period_labels = ["Before 2021", "After 2021"]

    axes[0].bar(period_labels, pixeldays_per_year.loc[period_order].values, color=["steelblue", "coral"])
    axes[0].set_title(f"Pixel-days per year\n({pixday_ratio:.1f}x increase)")
    axes[0].set_ylabel("Pixel-days / year")

    axes[1].bar(period_labels, unique_locs_by_period.loc[period_order].values, color=["steelblue", "coral"])
    axes[1].set_title(f"Distinct locations ever flooded\n({loc_ratio:.1f}x increase)")
    axes[1].set_ylabel("Unique (lat, lon) locations")

    axes[2].bar(period_labels, persistence_by_period.loc[period_order].values, color=["steelblue", "coral"])
    axes[2].set_title(f"Avg. days flooded per location\n({persist_ratio:.1f}x increase)")
    axes[2].set_ylabel("Mean days flooded")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "23_2021_shift_decomposition.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUT_DIR / '23_2021_shift_decomposition.png'}")

    section("SUMMARY - which explanation does the evidence favor?")
    print(f"Pixel-days/year jump:        {pixday_ratio:.2f}x")
    print(f"Distinct locations jump:     {loc_ratio:.2f}x")
    print(f"Persistence (days/location): {persist_ratio:.2f}x")
    print(f"Recurring floods/year:       {recurring_ratio:.2f}x")
    print(f"Unusual floods/year:         {unusual_ratio:.2f}x")
    print()

    return pixday_ratio, loc_ratio, persist_ratio


if __name__ == "__main__":
    pixday_ratio, loc_ratio, persist_ratio = main()
    section("DONE")
    print(f"Output saved to: {OUT_DIR.resolve()}")