import yaml
import pandas as pd
import os
import utils
import numpy as np
from datetime import datetime, timedelta

# Where to find the data
dataset_name = 'tag78_swimming_test_1'
datasets_path = '../data/datasets.yaml'
datasets = yaml.safe_load(open(datasets_path, 'r'))
dataset_obj = datasets[dataset_name]

# Add the datasets.yaml file's path to a path contained in it
def get_path(relpath):
    return os.path.join(os.path.split(datasets_path)[0], relpath)

# Retrieve the CSVs making up the data and put them into a single data frame
hydrophone_pds = []
for _, hydrophone_dict in dataset_obj['hydrophones'].items():
    print(hydrophone_dict['path'])
    if type(hydrophone_dict['path']) is list:
        hydrophone_pds = hydrophone_pds + [pd.read_csv(get_path(path)) for path in hydrophone_dict['path']]
    else:
        hydrophone_pds.append(pd.read_csv(get_path(hydrophone_dict['path'])))
hydrophone_data = pd.concat(hydrophone_pds)

# Sort the data by datetime
hydrophone_data['datetime'] = pd.to_datetime(hydrophone_data['datetime'])
hydrophone_data.sort_values('datetime', inplace=True)

# Filter the data for the detections matching the tag id
tag_id = dataset_obj['tag']
hydrophone_data = hydrophone_data[hydrophone_data['tag_id'] == tag_id]
hydrophone_data.reset_index(drop=True, inplace=True)
 
# Add tag coordinates to each detection
tag_coords_obj = dataset_obj['tag_coords']
if 'path' in tag_coords_obj:
    tag_coords_pd = pd.read_csv(get_path(tag_coords_obj['path']))
    tag_coords_pd['datetime'] = pd.to_datetime(tag_coords_pd['datetime'])

    tag_latitude = np.full(len(hydrophone_data), np.nan)
    tag_longitude = np.full(len(hydrophone_data), np.nan)

    key = lambda x: x[1]['datetime']
    for (i, hydrophone_row), tag_tuple in utils.project_time_series(hydrophone_data.iterrows(), tag_coords_pd.iterrows(), key=key, reversed=True):
        if tag_tuple is not None:
            _, tag_row = tag_tuple
            # if hydrophone_row['datetime'] - tag_row['datetime'] < timedelta(minutes=1):
            tag_latitude[i] = tag_row['latitude']
            tag_longitude[i] = tag_row['longitude']
    
    hydrophone_data['tag_latitude'] = tag_latitude
    hydrophone_data['tag_longitude'] = tag_longitude
else:
    tag_latitude, tag_longitude = tag_coords_obj['latitude'], tag_coords_obj['longitude']
    hydrophone_data['tag_latitude'] = [tag_latitude] * len(hydrophone_data)
    hydrophone_data['tag_longitude'] = [tag_longitude] * len(hydrophone_data)
