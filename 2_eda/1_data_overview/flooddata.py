"""Linear regression of daily discharge at Dartmouth station 100205 (the only station in South Sudan) on the other
stations' discharge. Prints the test R^2 and the weight of each station.

    python 2_eda/1_data_overview/flooddata.py
"""
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_data():
    """One DataFrame per station file, keyed by station ID."""
    base = "./raw_data/Darthmouth Flood Observatory/"
    return {f.split("_")[0]: pd.read_csv(base + f) for f in os.listdir(base) if f.endswith(".csv")}


data_per_item = load_data()
data = pd.DataFrame({k: v["Discharge (m3/s)"] for k, v in data_per_item.items()}).dropna()
Y = data["100205"]
X = data.drop("100205", axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
linear_regression = LinearRegression().fit(X_train, Y_train)
print("score: " + str(linear_regression.score(X_test, Y_test)))

weight_param = {p: float(w) for p, w in zip(X.columns, linear_regression.coef_)}
print(sorted(weight_param.items(), key=lambda x: x[1], reverse=True))
