import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from fiona.crs import from_epsg
from shapely.geometry import Point
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# # 8/3 - clipped
# df_457049_clipped = pd.read_csv('../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv')
# df_457012_clipped = pd.read_csv('../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv')

# df_457049_clipped['datetime'] = pd.to_datetime(df_457049_clipped['datetime'])
# df_457012_clipped['datetime'] = pd.to_datetime(df_457012_clipped['datetime'])

# # find earliest time
# if min(df_457049_clipped['datetime']) < min(df_457012_clipped['datetime']): 
#     df_earliest_time = min(df_457049_clipped['datetime'])
# else: 
#     df_earliest_time = min(df_457012_clipped['datetime'])

# print(type(df_earliest_time))

# # remove timesteps after 4500 seconds
# timestep_4000 = df_earliest_time + timedelta(seconds = 4500)
# # timestep_500 = df_earliest_time + timedelta(seconds = 500)

# for index, row in df_457049_clipped.iterrows(): 
#     if row['datetime'] > timestep_4000: 
#         df_457049_clipped.drop(index, inplace=True)
#     # elif row['datetime'] < timestep_500:
#     #     df_457049_clipped.drop(index, inplace=True)

# for index, row in df_457012_clipped.iterrows(): 
#     if row['datetime'] > timestep_4000: 
#         df_457012_clipped.drop(index, inplace=True)
#     # elif row['datetime'] < timestep_500:
#     #     df_457012_clipped.drop(index, inplace=True)

# # print(timestep_500)
# print(timestep_4000)

# print(df_457049_clipped)
# print(df_457012_clipped)

# df_457049_clipped.to_csv('../data/06-08-2022/clip_tag78_50m_increment_long_beach_test_457049_0.csv')
# df_457012_clipped.to_csv('../data/06-08-2022/clip_tag78_50m_increment_long_beach_test_457012_0.csv')

# plotting the total distance 
# df_457049_clipped = pd.read_csv('../data/06-08-2022/clip_tag78_50m_increment_long_beach_test_457049_0.csv')

# plot the total distance and gps distance versus time


# get the error at the beginning and at the end (the drift)
# subtract and plot again

# Just look at the error at the last groundtruth, and at the first groundtruth, find the line that passes through those two and subtract it off the time of flight distance measurements
# The offset should be a linear function of time

# df = pd.read_csv('df_range.csv')

# groundtruth = []
# groundtruth = []

# estimated_range_x = []
# estimated_range_y = []

# for i in range(df.shape[0]): 
#     print("here")
#     groundtruth_x.append(df['Groundtruth range x'])
#     groundtruth_y.append(df['Groundtruth range y'])

#     estimated_range_x.append(df['Estimated range x'])
#     estimated_range_y.append(df['Estimated range y']) 
# print("HERE", np.sqrt(df.iloc[100]['Groundtruth range x']**2 + df.iloc[100]['Groundtruth range y']**2))

# datapath1 = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_VR100_0.csv'
#     datapath2 = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_457049_0.csv'
#     data1 = pd.read_csv(datapath1)
#     data2 = pd.read_csv(datapath2)
#     data1 = data1[data1['tag_id'] == 65478].reset_index(drop=True)
#     data2 = data2[data2['tag_id'] == 65478].reset_index(drop=True)
#     data1['datetime'] = pd.to_datetime(data1['datetime'])
#     data2['datetime'] = pd.to_datetime(data2['datetime'])

# Code for plotting range
df = pd.read_csv('df_range.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df.index = df['datetime']
print(df.dtypes)
print(df.index)

df = df[df['Groundtruth range'].notnull()]

print(df['Groundtruth range'].min)

print(df['Groundtruth range'].isnull().sum())

plt.scatter(df.index, df['Groundtruth range'], label = 'True Measured Range')
plt.scatter(df.index, df['Estimated range'], label = 'Estimated Range')
plt.scatter(df.index, df['ToF Distance'], label = 'ToF Distance')
plt.ylabel("Range")
plt.legend()


# plt.scatter(df['datetime'], df['Groundtruth range'])
# plt.scatter(df['datetime'], df['Estimated range'])
# plt.scatter(df['datetime'], df['ToF Distance'])
plt.show()

# df = pd.read_csv('dataframe_error.csv')

# error = np.sqrt(df['Best Particle Path X']**2 + df['Best Particle Path Y']**2)
# df['index column'] = df.index * 10

# # relabel x axes to be accurate to the number of time steps

# plt.plot(df['index column'], error)
# plt.title('Error plot as a function of time of estimated range')
# plt.xlabel('Frames')
# plt.ylabel('Error (m)')

# plt.show()

