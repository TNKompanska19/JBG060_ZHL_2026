"""Check that three course loaders run: the OpenStreetMap road network, the flood masks of 2000 and 2021, and the
cattle raster. Prints what each returns.

    python 2_eda/1_data_overview/geotiff_parquet.py
"""
import sys

import numpy as np

sys.path.insert(0, "1_data_processing/2_loaders")   # course loaders; run from the repo root
from loading import load_flood_masks
from loading_impact_data import download_OSM_network, load_cattle

print("download OSM")
print(download_OSM_network(name="OSM"))

print("loading mask data")
print(load_flood_masks(years=np.array((2000, 2021))).head())

print("loading cattle data")
print(load_cattle())
