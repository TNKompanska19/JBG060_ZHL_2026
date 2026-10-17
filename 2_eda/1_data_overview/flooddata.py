from email.mime import base

import pandas as pd
import os
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


#Loading the data first.
def load_data():
    base = "./raw_data/Darthmouth Flood Observatory/"
    data_per_item: dict={}
    for i in os.listdir(base):
        if i.endswith(".csv"):
            data_per_item[i.split('_')[0]] = pd.read_csv(base + i)
    return data_per_item

data_per_item = load_data()
date = data_per_item['100205']["Date"]
#Having loaded the files, I now set X and Y values:
data_first = {}
for i in data_per_item.keys():
    data_first[i] = data_per_item[i]['Discharge (m3/s)']
data = pd.DataFrame(data_first)
data_clean = data.dropna()
Y = data_clean['100205']
X = data_clean.drop('100205', axis=1)
assert len(X) == len(Y)
#Linear regression and weights comparison of all Sudanese/Ethiopian stations to the one in South Sudan
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
assert len(X_train) == len(Y_train)
assert len(X_test) == len(Y_test)
linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)
print("score: " + str(linear_regression.score(X_test, Y_test)))
weights = linear_regression.coef_
params = list(X.keys())
weight_param = {}
for i in range(len(weights)):
    weight_param[params[i]] = float(weights[i])

sorted_weights = sorted(weight_param.items(), key=lambda x: x[1], reverse=True)
print(sorted_weights)