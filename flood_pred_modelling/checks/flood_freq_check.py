"""Check the flood-frequency thresholds against the data, at the 1 km CYGNSS cells, inside South Sudan.

    python flood_pred_modelling/checks/flood_freq_check.py > flood_pred_modelling/checks/output/flood_freq_check.txt

Prints:
1. distribution of flood frequency (share of valid months a cell is water), as share of land per band
2. permanent water under three definitions: >= 90 %, >= 95 % of months, water in every month of every full year
3. scope sensitivity: cells (and cropland cells) kept for lower bounds 2/5/10 % and the three permanent definitions
4. the same distribution for cropland cells only (crop >= 10 %)
5. mixed 2 km squares under the current rule: squares holding both permanent and non-permanent cells
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_build_dataset"))
import build_inputs as bi  # noqa: E402
from build_target import counties, crop_all  # noqa: E402

CROP_MIN = 10.0


def main():
    w, lat, lon, months = crop_all()
    cid, _, _ = counties(lat, lon)
    inside = cid > 0
    valid = w >= 0
    wet = w == 1
    nval = valid.sum(0)
    ok = inside & (nval >= 24)
    freq = np.where(nval > 0, wet.sum(0) / np.maximum(nval, 1), np.nan)
    f = freq[ok]
    print(f"cells inside South Sudan with >= 24 valid months: {ok.sum():,} of {inside.sum():,}; "
          f"months {months[0]} to {months[-1]} ({len(months)})")

    bands = [0, 1e-9, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1 - 1e-9, 1.0 + 1e-9]
    labels = ["0 (never)", "0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-70%",
              "70-80%", "80-90%", "90-95%", "95-<100%", "100% (always)"]
    h = pd.cut(pd.Series(f), bands, labels=labels, right=False).value_counts(sort=False)
    print("\n1. flood frequency, share of land (all cells)")
    print((h / h.sum() * 100).round(2).to_string())

    yrs = np.array([m[:4] for m in months])
    full_years = [y for y in np.unique(yrs) if (yrs == y).sum() == 12]
    every = np.ones(w.shape[1:], bool)
    for y in full_years:
        sel = yrs == y
        every &= (wet[sel] | ~valid[sel]).all(0) & (valid[sel].sum(0) >= 10)
    perm = {">=90%": ok & (freq >= 0.90), ">=95%": ok & (freq >= 0.95), "every month, every full year": ok & every}
    print(f"\n2. permanent water (full years used: {', '.join(full_years)})")
    for k, v in perm.items():
        print(f"  {k:30s} {v.sum():>8,} cells  ({v.sum() / ok.sum() * 100:.2f}% of land)")

    crop = bi.cropland(lat, lon, abs(float(lat[1] - lat[0])))["crop_pct"] >= CROP_MIN
    print(f"\n3. scope sensitivity (cropland cell = crop >= {CROP_MIN:.0f}%; {(ok & crop).sum():,} cropland cells in total)")
    rows = []
    for lo in (0.02, 0.05, 0.10):
        for k, p in perm.items():
            s = ok & (freq >= lo) & ~p
            rows.append({"lower": f"{lo:.0%}", "permanent": k, "cells": int(s.sum()),
                         "% of land": round(s.sum() / ok.sum() * 100, 1),
                         "cropland cells": int((s & crop).sum()),
                         "% of cropland": round((s & crop).sum() / max((ok & crop).sum(), 1) * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    fc = freq[ok & crop]
    hc = pd.cut(pd.Series(fc), bands, labels=labels, right=False).value_counts(sort=False)
    print("\n4. flood frequency, share of cropland cells")
    print((hc / max(hc.sum(), 1) * 100).round(2).to_string())

    pm = (ok & (freq >= 0.90)).astype(int)
    H, W = (pm.shape[0] // 2) * 2, (pm.shape[1] // 2) * 2
    nperm = pm[:H, :W].reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    nok = ok[:H, :W].astype(int).reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    mixed = (nperm > 0) & (nperm < nok)
    print(f"\n5. 2 km squares (2x2 cells): {int((nok > 0).sum()):,} with data; holding permanent water: "
          f"{int((nperm > 0).sum()):,}; mixed (permanent + land cells): {int(mixed.sum()):,}")


if __name__ == "__main__":
    main()
