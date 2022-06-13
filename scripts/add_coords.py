import pandas as pd
import geopy.distance
import numpy as np
from common import pairwise

knots_per_meter = 1.944
meters_per_knot = 1 / knots_per_meter

tag_coords = (33.7421588, -118.12206)
targetpath = '../data/06-08-2022/tag78_cowling_none_long_beach_test_457049_0.csv'
sourcepath = None
# sourcepath = r'../data\06-08-2022\20220608-171235-CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013\Logs\20220608-171318--CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013.log'
# sourcepath = r'../data\06-08-2022\20220608-153927-CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013\Logs\20220608-153932--CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013.log'
# sourcepath = r'../data\06-08-2022\20220608-154731-CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013\Logs\20220608-154736--CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013.log'
# sourcepath = r'../data\06-08-2022\20220608-183307-CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013\Logs\20220608-183312--CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013.log'
# sourcepath = '../data/06-01-2022/20220601-164716-CH_Long_Beach_Mission_05_31_2022-IVER3-3013/Logs/20220601-164722--CH_Long_Beach_Mission_05_31_2022-IVER3-3013.log'

hydrophone_data = pd.read_csv(targetpath)
hydrophone_data['datetime'] = pd.to_datetime(hydrophone_data['datetime'])

# Add latitude and longitude columns and a speed column using the mission logs
if sourcepath is not None:
    mission_data = pd.read_csv(sourcepath, sep=';')
    mission_data['datetime'] = pd.to_datetime(mission_data['Date'] + ' ' + mission_data['Time'])
    latitudes = np.array(hydrophone_data.get('latitude', [np.nan] * len(hydrophone_data)))
    longitudes = np.array(hydrophone_data.get('longitude', [np.nan] * len(hydrophone_data)))
    logged_speeds = np.array(hydrophone_data.get('logged_speed', [np.nan] * len(hydrophone_data)))
    mission_iter = pairwise(mission_data.iterrows())
    for i, hydrophone_row in hydrophone_data.iterrows():
        if hydrophone_row['datetime'] >= mission_data['datetime'][0]:
            hydrophone_time = hydrophone_row['datetime']
            try:
                closest_row = None
                while closest_row is None:
                    (_, row), (_, next_row) = next(mission_iter)
                    if next_row['datetime'] > hydrophone_time >= row['datetime']:
                        if next_row['datetime'] - hydrophone_time > hydrophone_time - row['datetime']:
                            closest_row = row
                        else:
                            closest_row = next_row
                iver_coords = (closest_row['Latitude'], closest_row['Longitude'])
                latitudes[i] = iver_coords[0]
                longitudes[i] = iver_coords[1]
                logged_speeds[i] = closest_row['Vehicle Speed (Kn)'] * meters_per_knot
            except StopIteration:
                continue
    hydrophone_data['latitude'] = latitudes
    hydrophone_data['longitude'] = longitudes
    hydrophone_data['logged_speed'] = logged_speeds

# Add gps distance using latitude and longitude columns
gps_distances = np.array(hydrophone_data.get('gps_distance', [np.nan] * len(hydrophone_data)))
for i, hydrophone_row in hydrophone_data.iterrows():
    sensor_coords = (hydrophone_row['latitude'], hydrophone_row['longitude'])
    try:
        gps_distances[i] = geopy.distance.geodesic(tag_coords, sensor_coords).m
    except:
        gps_distances[i] = np.nan
hydrophone_data['gps_distance'] = gps_distances

# Add absolute distances using the first non-nan gps distance
distances_iter = zip(hydrophone_data['total_distance'], gps_distances)
try:
    while True:
        total_distance, gps_distance = next(distances_iter)
        if not np.isnan(gps_distance):
            offset = gps_distance - total_distance
            hydrophone_data['absolute_distance'] = hydrophone_data['total_distance'] + offset
            break
except StopIteration:
    pass

# Add speed calculated by the GPS
gps_distances_iter = pairwise(hydrophone_data['gps_distance'])
gps_speeds = np.array(hydrophone_data.get('gps_speed', [np.nan] * len(hydrophone_data)))
for i, (current_gps_distance, next_gps_distance) in enumerate(gps_distances_iter):
    dt = float(hydrophone_data['dt'][i])
    if np.isnan(current_gps_distance) or np.isnan(next_gps_distance) or dt == 0:
        gps_speeds[i] = np.nan
    else:
        gps_speeds[i] = (next_gps_distance - current_gps_distance) / dt
hydrophone_data['gps_speed'] = gps_speeds

hydrophone_data.to_csv(targetpath, index=None)