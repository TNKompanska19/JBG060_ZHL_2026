"""GFM STAC metadata inventory, whole South Sudan, 2015-2026.

Metadata only: no rasters are downloaded. One parquet per year-month, so a
crash or a 502 loses at most one month and re-running skips what exists.

    python gfm_inventory.py fetch      # pull (resumable)
    python gfm_inventory.py summary    # what came back
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://stac.eodc.eu/api/v1/search"
COLLECTION = "GFM"
BBOX = [23.44, 3.48, 35.95, 12.24]  # South Sudan adm0 envelope, incl. Renk
YEARS = range(2015, 2027)
OUT = Path(__file__).resolve().parents[1] / "raw_data" / "gfm" / "stac"


def request(method, url, **kw):
    """Retry 429/5xx with exponential backoff; EODC 502s are transient."""
    for i in range(6):
        r = requests.request(method, url, timeout=120, **kw)
        if r.status_code == 200:
            return r.json()
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
        time.sleep(5 * 2**i)
    raise RuntimeError(f"{url} -> {r.status_code} after 6 tries")


def flatten(feature):
    """One row per STAC item. Keep every property; stringify nested values."""
    row = dict(feature.get("properties", {}))
    row["item_id"] = feature.get("id")
    row["collection"] = feature.get("collection")
    row["bbox"] = json.dumps(feature.get("bbox"))
    row["assets"] = "|".join(sorted(feature.get("assets", {}).keys()))
    return {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in row.items()}


def fetch_month(year, month):
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    payload = {
        "collections": [COLLECTION],
        "bbox": BBOX,
        "datetime": f"{year}-{month:02d}-01T00:00:00Z/{end}T00:00:00Z",
        "limit": 500,
    }
    features, page = [], request("POST", API, json=payload)
    while True:
        features += page.get("features", [])
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            return features
        if nxt.get("body"):  # POST-style paging (token in body)
            page = request("POST", nxt.get("href", API), json={**payload, **nxt["body"]})
        else:  # GET-style paging (token in href)
            page = request("GET", nxt["href"])


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        for month in range(1, 13):
            path = OUT / f"{year}-{month:02d}.parquet"
            if path.exists():
                continue
            feats = fetch_month(year, month)
            df = pd.DataFrame([flatten(f) for f in feats])
            df.to_parquet(path)  # empty months written too, so they aren't retried
            print(f"{year}-{month:02d}  {len(df):>6} items", flush=True)


def summary():
    files = sorted(OUT.glob("*.parquet"))
    if not files:
        sys.exit("nothing fetched yet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["year"] = pd.to_datetime(df["datetime"], format="mixed", utc=True).dt.year

    print(f"\n{len(df)} items, {df['year'].nunique()} years\n")
    print("items per year:")
    print(df.groupby("year").size().to_string())

    # Does the asset list change over time? A thinner pre-2022 asset set would
    # mean those items were never fully processed.
    print("\ndistinct asset sets per year:")
    for year, grp in df.groupby("year"):
        for assets, n in grp["assets"].value_counts().items():
            print(f"  {year}  n={n:>6}  {assets}")

    # Any property that is constant within a year but differs between years is
    # a processing-version discontinuity. This is the actual layer check.
    print("\nproperties that differ across years:")
    for col in df.columns:
        if col in ("item_id", "bbox", "datetime", "year", "assets"):
            continue
        per_year = df.groupby("year")[col].agg(lambda s: tuple(sorted(s.dropna().astype(str).unique()))[:4])
        if per_year.nunique() > 1:
            print(f"\n  {col}")
            print(per_year.to_string())


if __name__ == "__main__":
    {"fetch": fetch, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "fetch"]()
