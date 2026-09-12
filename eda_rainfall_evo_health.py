from processing_data import loading as ld
from processing_data import loading_impact_data as ld_imp
import xarray as xr
import matplotlib.pyplot as plt
import cfgrib as grib
import pandas as pd
import datetime as dt

def ET_eda(years_list,latitude,longitude):
    
    
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


def health_eda():
    health = ld_imp.load_health_facilities()
    print(health.value_counts('Facility_t'))
    
    h_type_plot = health.plot('Facility_t', legend=True)
    h_type_plot.set_title('Healthcare locations by type')
    health_no_PHCU = health[~health['Facility_t'].isin(['Primary Health Care Unit', 'Primary Health Care Centre'])]
    print(health_no_PHCU.value_counts('Facility_t'))
    h_no_phcu_plot = health_no_PHCU.plot('Facility_t', legend=True)

    plt.show()

def rainrun(years):
    rainRun: xr.Dataset = ld.load_rainfall_runoff(years)
    #print(type(rainRun))
    #pp = rainRun['tp'].drop_vars(['latitude', 'longitude'])
    #rainRun.info()    

    #plt.plot(rainRun['tp'])

def main():
    years_list = range(2000,2026)
    longitude: float = 30.725
    latitude: float = 9.475
    ET_eda(years_list=years_list,longitude=longitude,latitude=latitude)
    health_eda()
    #rainrun(years= years_list)


if __name__ == "__main__":
    main()
