import pandas as pd
from datetime import datetime
import utils
import os
import numpy as np

datapath = '../data/07-19-2022/VR100_10913_D2022.07.20T09.29.53.csv'
filepath, fileext = os.path.splitext(datapath)
savepath = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_VR100_0.csv'
# savepath = '{}_converted.{}'.format(filepath, fileext)

# Read vr100 csv
data = pd.read_csv(datapath)

# Create datetime objects
datetimes = [datetime.fromisoformat('{} {}'.format(row.Date, row.Time)) for _, row in data[['Date', 'Time']].iterrows()]
data['datetime'] = datetimes

# Sort data by datetime
data.sort_values('datetime', inplace=True)
data.reset_index(drop=True, inplace=True)
datetimes = data['datetime']

columns = [
    'serial_no',
    'datetime',
    'code_space',
    'tag_id',
    'signal_level',
    'latitude',
    'longitude',
    'total_dt',
    'dt',
    'delta_tof',
    'delta_distance', 
    'total_distance'
]

# Extract lines for each tag
tags = {}
for i, row in data.iterrows():
    if row.ID not in tags:
        tags[row.ID] = []
    tags[row.ID].append((i, datetimes[i]))

total_dt, dt, delta_tof, delta_distance, total_distance = np.zeros((5, len(data)))
for tag_id, tag_data in tags.items():
    start = tag_data[0][1]

    # Create total_dt column
    for i, timestamp in tag_data:
        total_dt[i] = (timestamp - start).total_seconds()
    
    # Create dt column
    for (prev_i, prev_timestamp), (i, timestamp) in utils.pairwise(tag_data):
        dt[i] = (timestamp - prev_timestamp).total_seconds()
    
    # Create delta_tof, delta_distance, and total_distance columns columns
    partial_delta_tof = utils.get_delta_tof(data['datetime'][data['ID'] == tag_id].reset_index(drop=True), utils.avg_dt_dict.get(tag_id, 8.179))
    partial_delta_distance = partial_delta_tof * utils.SPEED_OF_SOUND
    partial_total_distance = np.cumsum(partial_delta_distance)
    for j, (a_delta_tof, a_delta_distance, a_total_distance) in enumerate(zip(partial_delta_tof, partial_delta_distance, partial_total_distance)):
        pd_index = tag_data[j][0]
        delta_tof[pd_index] = a_delta_tof
        delta_distance[pd_index] = a_delta_distance
        total_distance[pd_index] = a_total_distance

converted_data = pd.DataFrame(np.array([
            ['VR100'] * len(data),
            datetimes,
            data['Code Space'],
            data['ID'],
            data['Signal (dB)'],
            data['Latitude'],
            data['Longitude'],
            total_dt,
            dt,
            delta_tof,
            delta_distance,
            total_distance
        ], dtype=object).T, columns=columns)

converted_data.to_csv(savepath, index=None)