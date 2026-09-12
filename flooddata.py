from importlib.metadata import pass_none

import pandas as pd
import xarray as xr
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

#Loading the data first.
import os
os.chdir("C:/Users/20233608/PyCharmProjects/JBG060_ZHL_2026")
def load_data():
    base = "./raw_data/Darthmouth Flood Observatory/"
    data_per_item: dict={}
    for i in os.listdir(base):
        if i.endswith(".csv"):
            data_per_item[i.split('_')[0]] = pd.read_csv(base + i)
    return data_per_item

data_per_item = load_data()
#Having loaded the files, I now set X and Y values:
Y = data_per_item["100205"]['Discharge (m3/s)']
data_per_item.pop("100205")
X_pre: dict={}
for i in data_per_item.keys():
    X_pre[i] = data_per_item[i]["Discharge (m3/s)"]

X = pd.DataFrame(X_pre)
print(X)