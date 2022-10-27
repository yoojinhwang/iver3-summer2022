import yaml
import pandas as pd
import os
import utils
import numpy as np
# from datetime import datetime, timedelta
# from functools import reduce, partial
import geopy.distance
import utils
from collections import namedtuple
import matplotlib.pyplot as plt

Dataset = namedtuple('Dataset', ['df', 'attrs'])

def merge_dataset(dataset_name, datasets_path='../data/datasets.yaml'):
    # Where to find the data
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
            hydrophone_pds += [pd.read_csv(get_path(path)) for path in hydrophone_dict['path']]
        else:
            hydrophone_pds.append(pd.read_csv(get_path(hydrophone_dict['path'])))
    data = pd.concat(hydrophone_pds)

    # Sort the data by datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.sort_values('datetime', inplace=True)

    # Filter the data for the detections matching the tag id
    tag_id = dataset_obj['tag_id']
    data = data[data['tag_id'] == tag_id]
    data.reset_index(drop=True, inplace=True)

    # Retrieve origin of cartesian coordinate frame
    origin_obj = dataset_obj.get('origin', {'latitude': None, 'longitude': None})
    origin = (origin_obj.get('latitude', None), origin_obj.get('longitude', None))
    if origin == (None, None):
        # TODO implement something to default to
        raise ValueError('Origin was not given')

    # Add tag coordinates to each detection
    tag_coords_obj = dataset_obj['tag_coords']
    if 'path' in tag_coords_obj:
        tag_coords_pd = pd.read_csv(get_path(tag_coords_obj['path']))
        tag_coords_pd['datetime'] = pd.to_datetime(tag_coords_pd['datetime'])

        tag_latitude = np.full(len(data), np.nan)
        tag_longitude = np.full(len(data), np.nan)

        key = lambda x: x[1]['datetime']
        for (i, hydrophone_row), tag_tuple in utils.project_time_series(data.iterrows(), tag_coords_pd.iterrows(), key=key, reversed=True):
            if tag_tuple is not None:
                _, tag_row = tag_tuple
                # if hydrophone_row['datetime'] - tag_row['datetime'] < timedelta(minutes=1):
                tag_latitude[i] = tag_row['latitude']
                tag_longitude[i] = tag_row['longitude']
        
        data['tag_latitude'] = tag_latitude
        data['tag_longitude'] = tag_longitude
    else:
        tag_latitude, tag_longitude = tag_coords_obj['latitude'], tag_coords_obj['longitude']
        data['tag_latitude'] = [tag_latitude] * len(data)
        data['tag_longitude'] = [tag_longitude] * len(data)
    
    # Add gps distance using latitude and longitude columns as well as the tag latitude and longitude columns
    if utils.columns_exist(['latitude', 'longitude', 'tag_latitude', 'tag_longitude'], data):
        # Add gps distance from hydrophone to tag
        gps_distances = utils.get_column('gps_distance', data)
        for i, row in data.iterrows():
            sensor_coords = (row['latitude'], row['longitude'])
            tag_coords = (row['tag_latitude'], row['tag_longitude'])
            try:
                gps_distances[i] = geopy.distance.geodesic(tag_coords, sensor_coords).m
            except Exception as err:
                print(err)
        data['gps_distance'] = gps_distances

        # Add bearing relative to the tag using latitude and longitude columns
        latitudes, longitudes = utils.get_column('latitude', data), utils.get_column('longitude', data)
        sensor_coords = utils.to_cartesian((latitudes, longitudes), origin)
        data['x'] = sensor_coords[:, 0]
        data['y'] = sensor_coords[:, 1]
        tag_latitudes, tag_longitudes = utils.get_column('tag_latitude', data), utils.get_column('tag_longitude', data)
        tag_coords = utils.to_cartesian((tag_latitudes, tag_longitudes), origin)
        data['tag_x'] = tag_coords[:, 0]
        data['tag_y'] = tag_coords[:, 1]

        heading = utils.get_column('logged_heading', data)
        compass_vecs = utils.unit_2d(utils.convert_heading(heading))
        data['relative_tag_bearing'] = utils.angle_between(compass_vecs, sensor_coords)
        data['tag_bearing'] = utils.angle_between((1, 0), -sensor_coords)

        # Add speed and delta time of flight calculated from the GPS coordinates
        for serial_no in dataset_obj['hydrophones'].keys():
            # Extract the data from a single hydrophone
            bool_indices = data['serial_no'] == serial_no
            hydrophone_data = data[bool_indices]

            # Create columns for a single hydrophone
            gps_speeds = utils.get_column('gps_speed', hydrophone_data)
            gps_delta_tof = utils.get_column('gps_delta_tof', hydrophone_data)
            gps_vels = utils.get_column('gps_vel', hydrophone_data)
            gps_thetas = utils.get_column('gps_theta', hydrophone_data)
            dts = np.array(hydrophone_data.get('dt', [0] + [(t_next - t).total_seconds() for t, t_next in utils.pairwise(hydrophone_data['datetime'])]))
            for i, ((_, row), (_, next_row)) in enumerate(utils.pairwise(hydrophone_data.iterrows())):
                dt = dts[i + 1]
                if dt != 0:
                    delta_distance = (next_row['gps_distance'] - row['gps_distance'])
                    gps_speeds[i + 1] = delta_distance / dt
                    gps_delta_tof[i + 1] = delta_distance / utils.SPEED_OF_SOUND
                    gps_vels[i + 1] = np.sqrt((next_row['y'] - row['y']) ** 2 + (next_row['x'] - row['x']) ** 2)
                gps_thetas[i + 1] = np.arctan2(next_row['y'] - row['y'], next_row['x'] - row['x'])
            data.loc[bool_indices, 'gps_speed'] = gps_speeds
            data.loc[bool_indices, 'gps_delta_tof'] = gps_delta_tof
            data.loc[bool_indices, 'gps_theta'] = gps_thetas
            data.loc[bool_indices, 'gps_vel'] = gps_vels

            # Add heading calculated from cartesian coordinates
            sensor_x = utils.get_column('x', hydrophone_data)
            sensor_y = utils.get_column('y', hydrophone_data)
            gps_heading = utils.get_column('gps_heading', hydrophone_data)
            coords_iter = utils.pairwise(zip(sensor_x, sensor_y))
            for i, ((x, y), (next_x, next_y)) in enumerate(coords_iter):
                dx = next_x - x
                dy = next_y - y
                gps_heading[i] = utils.angle_between((1, 0), (dx, dy))
            data.loc[bool_indices, 'gps_heading'] = gps_heading
    return Dataset(data, dataset_obj)

'''
Explanation of columns in a merged dataset:
    serial_no:
    line_counter:
    datetime:
    code_space:
    tag_id:
    sensor_adc:
    signal_level:
    noise_level:
    channel:
    latitude:
    longitude:
    total_dt:
    dt:
    delta_tof:
    delta_distance:
    total_distance:
    tag_latitude:
    tag_longitude:
    gps_distance:
    x:
    y:
    tag_x:
    tag_y:
    relative_tag_bearing:
    tag_bearing:
    gps_speed:
    gps_delta_tof:
    gps_theta:
    gps_vel:
'''

# def visualize_dataset(dataset):
#     '''
#     Plot the times at which:
#     - we receive a detection for each hydrophone
#     - we have gps coordinates for each hydrophone
#     - we have gps coordinates for the tag
#     '''
#     fig, ax = plt.subplot()
#     for serial_no in dataset.df['serial_no'].unique():


if __name__ == '__main__':
    dataset_name = 'tag78_swimming_test_1'
    data = merge_dataset(dataset_name)