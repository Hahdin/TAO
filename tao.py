import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set_style('whitegrid')

column_names = ['obs', 'year', 'month', 'day', 'date','latitude','longitude','zon.winds','mer.winds','humidity','air temp.','s.s.temp.']
# data set at https://archive.ics.uci.edu/ml/datasets/El+Nino
df = pd.read_csv('tao-all2.dat', sep=' ', names=column_names, na_values='.', dtype={'humidity': np.float64, 'air temp.': np.float64,})

# pivot table of air temp by day/month/year
pvAirTempM = df.dropna().pivot_table(values=['air temp.'],index=['month','day'],columns='year')
sns.heatmap(pvAirTempM, cmap='inferno')

# this requires standardizing the data to work
scaler = StandardScaler()
df_part = df.drop(['zon.winds','mer.winds','humidity','air temp.','s.s.temp.'], axis=1)# keep the date and lat/lon
df_tmp = df.drop(['obs', 'year', 'month', 'day', 'date','latitude','longitude'], axis=1)# keep the fields we want to normalize
scaler.fit(df_tmp)
scaled_features = scaler.transform(df_tmp)
df_feat = pd.DataFrame(scaled_features,columns=df.columns[7:])# create the features from the scaled data
df_comb = df_feat.join(df_part)# combine them with the date and lat/lon data

#pivot table of air temp and humidity by day/month/year
pvAirTempStd = df_comb.dropna().pivot_table(values=['air temp.', 'humidity'],index=['month','day'],columns='year')
#print(df_comb[df_comb['humidity'] > 0].head())

sns.heatmap(pvAirTempStd, cmap='inferno')
plt.show()