"""Daily reference evapotranspiration at one grid cell (9.475N, 30.725E), 2000-2025, and the health facilities by type
(all, and without primary health care units and centres). Shows the plots.

    python 2_eda/1_data_overview/eda_et_health.py
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "1_data_processing/2_loaders")   # course loaders; run from the repo root
import loading as ld
import loading_impact_data as ld_imp


def ET_eda(years_list, latitude, longitude):
    for year in years_list:
        ld.process_ET(year=year, target_latitude=latitude, target_longitude=longitude)
    evo = ld.load_processed_ET(years=years_list, target_latitude=latitude, target_longitude=longitude)
    df_evo = pd.concat([evo[year] for year in years_list], ignore_index=True)
    print(df_evo.describe())
    df_evo.plot(x="date", y="gridcell")
    evo[2000].plot(x="date", y="gridcell")
    plt.show()


def health_eda():
    health = ld_imp.load_health_facilities()
    print(health.value_counts("Facility_t"))
    health.plot("Facility_t", legend=True).set_title("Healthcare locations by type")
    health_no_PHCU = health[~health["Facility_t"].isin(["Primary Health Care Unit", "Primary Health Care Centre"])]
    print(health_no_PHCU.value_counts("Facility_t"))
    health_no_PHCU.plot("Facility_t", legend=True)
    plt.show()


def main():
    ET_eda(years_list=range(2000, 2026), latitude=9.475, longitude=30.725)
    health_eda()


if __name__ == "__main__":
    main()
