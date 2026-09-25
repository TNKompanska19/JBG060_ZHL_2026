from loading import load_flood_masks
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from loading_impact_data import download_OSM_network

os.chdir('C:/Users/0233608/PycharmProjects/JBG060_ZHL_2026')
print(os.getcwd())
#OpenStreetMap
from pathlib import Path
from tqdm import tqdm

raster_path = '../raw_data/farmland/asap_mask_rangeland_v04.tif'
#loading the mask data
masks = load_flood_masks(years=np.array((2000, 2021)))
print(masks.head())
#Loading TIF files
