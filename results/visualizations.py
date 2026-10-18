import matplotlib as plb
import numpy as np
import pandas as pd
#For directory selection
import os
base = ('run_all')
data = pd.read_csv(base + '/ablation.csv')
data.dropna(inplace=True)
notstring = ['cv', 'model', 'informative', 'direction']
for i in notstring:
    del data[i]
delta = 'years_positive'
cue = 'target'
sort = 'subset'
target = 'h'
data.groupby(by=sort)
#create sort dictionary
unique=data.subset.unique()
#separate on occurence/persistence
dataDict = {elem: data.loc[data[sort] == elem] for elem in unique}
means = {}
for i in dataDict.keys():
    dataset = dataDict[i]
    u = dataset.target.unique()
    for j in u:
        d = list(dataset.loc[dataset['target'] == j][delta])
        d_clean = []
        for l in d:
            d_clean.append(int(l[0])/int(l[2]))
        means[(i,j)] = np.mean(d_clean)

print(means)
n = len(means.keys()) / 2