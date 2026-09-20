import rasterio
import numpy as np
raster_path = 'raw_data/farmland/asap_mask_rangeland_v04.tif'
with rasterio.open(raster_path) as src:
    print(src.meta)  # Metadata (CRS, bounds, etc.)
    data = src.read()  # Pixel data as numpy array
print("Min:", data.min(), "Max:", data.max(), "Mean:", data.mean())
print("NaN count:", np.isnan(data).sum())  # If float type