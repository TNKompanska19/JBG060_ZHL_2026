import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#For directory selection
import os
root_folder = '../processed_data'
##This code is used to generate the visualizations in this project. It uses npy, csv and npz files to create the visualizations.

##Loading the county data
print("Input County_Month")
county_month = pd.read_csv(root_folder + '/cygnss/county_month.csv')

##Loading the npy data
print("Input Watermask")
jrc_perm = np.load(root_folder + '/cygnss/jrc_perm_1km.npy')
ssd_water = np.load(root_folder + '/cygnss/ssd_watermask.npz')


##panel_0.02
print("Input Panel 0.02")
inputs = np.load(root_folder + '/panel_0.02/inputs.npz')
shared = pd.read_csv(root_folder + '/panel_0.02/shared.csv')

##Evaporation
os_list = os.listdir(root_folder + '/evapotranspiration')
base_list = []
for i in os_list:
    data = pd.read_csv(root_folder + '/evapotranspiration/' + i)
    base_list.append(data)
    print("Data:" + i)
assert len(base_list) == len(os_list)
print("All data loaded")

