from processing_data import loading as ld
from processing_data import loading_impact_data as ld_imp
import xarray as xr
import matplotlib.pyplot as plt
import cfgrib as grib
import pandas as pd
import datetime as dt

def main():
    
    years_list = range(2000,2026)
    longitude: float = 30.725
    latitude: float = 9.475
    for year in years_list:
        ld.process_ET(year=year, target_latitude=latitude,target_longitude=longitude)

    evo = ld.load_processed_ET(years=years_list, target_latitude=latitude,target_longitude=longitude)
    list_et = []
    df_evo= pd.DataFrame()
    for year in years_list:
       list_et.append(evo[year])
    df_evo: pd.DataFrame = pd.concat(list_et, ignore_index=True)
    df_evo = pd.DataFrame(df_evo)
    print(df_evo.describe())
    plot = df_evo.plot(x='date', y='gridcell')
    #plot.set_xlim(right = '2000-12-31')
    plot2000 = evo[2000].plot(x='date', y='gridcell')
    plt.show()

    #rainRun: xr.Dataset = ld.load_rainfall_runoff(years)
    #print(type(rainRun))
    #pp = rainRun['tp'].drop_vars(['latitude', 'longitude'])
    #rainRun.info()    

    #plt.plot(rainRun['tp'])

if __name__ == "__main__":
    main()
