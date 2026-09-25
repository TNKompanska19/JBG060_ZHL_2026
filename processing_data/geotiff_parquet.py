raster_path = '../raw_data/farmland/asap_mask_rangeland_v04.tif'
from loading import load_flood_masks
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

masks = load_flood_masks(years=np.array((2000, 2021)))