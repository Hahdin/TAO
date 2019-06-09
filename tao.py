import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

column_names = ['obs', 'year', 'month', 'day', 'date','latitude','longitude','zon.winds','mer.winds','humidity','air temp.','s.s.temp.']
# data set at https://archive.ics.uci.edu/ml/datasets/El+Nino
df = pd.read_csv('data/tao-all2.dat/tao-all2.dat', sep=' ', names=column_names, na_values='.', dtype={'humidity': np.float64, 'air temp.': np.float64,})

#pvAirTemp = df.dropna().pivot_table(values=['air temp.', 's.s.temp.'],index='month',columns='year')
pvAirTempM = df.dropna().pivot_table(values=['air temp.'],index=['month','day'],columns='year')
# this would require standardizing the data to work
# pvAirTempH = df.dropna().pivot_table(values=['air temp.', 'humidity'],index=['month','day'],columns='year')
sns.heatmap(pvAirTempM, cmap='inferno')
plt.show()