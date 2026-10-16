import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

root_folder = '../processed_data'
##This code is used to generate the visualizations in this project. It uses npy, csv and npz files to create the visualizations.

##Loading the county data
county_month = pd.read_csv(root_folder + '/cygnss/county_month.csv')
print(county_month.head())

##Loading the npy data
jrc_perm = np.load(root_folder + '/cygnss/jrc_perm_1km.npy')
ssd_water = np.load(root_folder + '/cygnss/ssd_watermask.npz')
print(jrc_perm.shape)
print(ssd_water.shape)

##panel_0.02
inputs = np.load(root_folder + '/panel_0.02/inputs.npz')
shared = pd.read_csv(root_folder + '/panel_0.02/shared.csv')

##Evaporation