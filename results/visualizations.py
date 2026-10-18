import seaborn as sns
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
dataDict = {elem: data.loc[data[sort] == elem] for elem in unique}
print(dataDict)
