"""MODIS/VIIRS flood masks: when and how much they detect across South Sudan.

    python flood_pred_modelling/modis_audit.py

Reads the raw parquets directly (an audit must not depend on the loaders it
audits). One detection = one 250 m pixel flagged as water on one day.
Writes to output/:
    modis_audit_month.png                share of detections per calendar month
    modis_audit_year.png                 detections per year
    modis_detections_by_year_month.csv   the counts behind both plots

What the plots show, and what they do not:
- Month: detections collapse in Jun-Sep, the cloudy months. That is measured.
  Cloud as the cause is documented by NASA (user guide section 7.1), not shown
  here. Sudd water itself peaks in Oct-Dec (Hardy et al. 2023), so never
  compare detections with same-month rainfall.
- Year: detections step up after 2020. The cause is NOT established; the
  2019-2022 floods were real and record-sized (Hardy et al. 2023), so never
  call the step an instrument change.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw_data" / "flood_masks"
OUT = HERE / "output"

BOX = dict(lat_min=3.48, lat_max=12.24, lon_min=23.44, lon_max=35.95)  # South Sudan admin-0 envelope
WET = [6, 7, 8, 9]
NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def count():
    """Detections per year (rows) and calendar month (columns 1-12), both tiles, both mask types."""
    files = sorted(RAW.glob("compact_*/flood_events_*.parquet"))
    if not files:
        raise SystemExit(f"no flood-mask parquets under {RAW}")
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["date", "lat", "lon"])
        d = d[d.lat.between(BOX["lat_min"], BOX["lat_max"])
              & d.lon.between(BOX["lon_min"], BOX["lon_max"])]
        t = pd.to_datetime(d["date"].astype(str))
        parts.append(pd.crosstab(t.dt.year.rename("year"), t.dt.month.rename("month")))
        print(f"  {f.parent.name}/{f.name}: {len(d):,}")
    tab = pd.concat(parts).groupby(level=0).sum()
    return tab.reindex(columns=range(1, 13)).fillna(0).astype("int64")


def main():
    OUT.mkdir(exist_ok=True)
    tab = count()
    tab.to_csv(OUT / "modis_detections_by_year_month.csv")

    share = tab.sum() / tab.to_numpy().sum() * 100            # % of all detections per month
    wet_all = share.loc[WET].sum()
    wet_year = (tab[WET].sum(axis=1) / tab.sum(axis=1) * 100).dropna()
    print("\n% of all detections per calendar month, all years pooled:")
    print(share.set_axis(NAMES).round(1).to_string())
    print(f"\nJun-Sep share, all years pooled: {wet_all:.1f}% (a flat year gives 33.3%)")
    print(f"Jun-Sep share per year: {wet_year.min():.1f}-{wet_year.max():.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(NAMES, share.to_numpy(), color=["tab:red" if m in WET else "grey" for m in range(1, 13)])
    ax.axhline(100 / 12, color="black", ls="--", lw=1, label="unfiorm distribution baseline (8.3%)")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Share of all flood detections (%)")
    ax.set_title(f"MODIS/VIIRS flood detections by month, South Sudan, "
                 f"{tab.index.min()}-{tab.index.max()}\n"
                 f"red = Jun-Sep, the cloudy months: {wet_all:.1f}% of all detections")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "modis_audit_month.png", dpi=120)

    per_year = tab.sum(axis=1)
    per_year = per_year[per_year > 0]
    print(f"\ndetections per year (max/min = {per_year.max() / per_year.min():.0f}x):")
    print(per_year.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(per_year.index.astype(str), per_year.to_numpy() / 1e6, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Flood detections (millions, log scale)")
    ax.set_title("MODIS/VIIRS flood detections per year, South Sudan\n"
                 "one detection = one 250 m pixel flagged as water on one day")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "modis_audit_year.png", dpi=120)
    print(f"\nwrote modis_audit_month.png, modis_audit_year.png and the CSV to {OUT}")


if __name__ == "__main__":
    main()
