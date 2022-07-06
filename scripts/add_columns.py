import pandas as pd
import geopy.distance
import numpy as np
import utils

knots_per_meter = 1.944
meters_per_knot = 1 / knots_per_meter
speed_of_sound = 1460

# First and second Long Beach deployment buoy location
# tag_coords = (33.7421588, -118.12206)

# Third (failed) Long Beach deployment buoy location
# tag_coords = (34.100821, -117.706509)

# Inside Parsons
# tag_coords = (34.106539,-117.7123235)

# 06-28-2022 BFS buoy location
# tag_coords = (34.109179, -117.712774)

# 06-29-2022 BFS buoy location
tag_coords = (34.1090865, -117.712575)

targetpath = '../data/06-29-2022/tag78_cowling_small_snail_BFS_test_457012_0.csv'
sourcepath = '../data/06-29-2022/tag78_cowling_small_snail_BFS_test_uvc_log_0.csv'

# 06-01-2022/50m_increment_2
# sourcepath = r'../data\06-01-2022\20220601-155249-CH_Long_Beach_Mission_05_31_2022-IVER3-3013\Logs\20220601-155325--CH_Long_Beach_Mission_05_31_2022-IVER3-3013.log'

# 06-01-2022/50m_increment_3
# sourcepath = r'../data/06-01-2022/20220601-164716-CH_Long_Beach_Mission_05_31_2022-IVER3-3013/Logs/20220601-164722--CH_Long_Beach_Mission_05_31_2022-IVER3-3013.log'

# 06-08-2022/50m_increment
# sourcepath = r'../data\06-08-2022\20220608-153927-CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013\Logs\20220608-153932--CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013.log'
# sourcepath = r'../data\06-08-2022\20220608-154731-CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013\Logs\20220608-154736--CH_Long_Beach_Mission_0_06_07_2022-IVER3-3013.log'

# 06-08-2022/cowling_none
# sourcepath = r'../data\06-08-2022\20220608-171235-CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013\Logs\20220608-171318--CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013.log'

# 06-08-2022/cowling_front_1
# sourcepath = r'../data\06-08-2022\20220608-180213-CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013\Logs\20220608-180220--CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013.log'

# 06-08-2022/cowling_back_1
# sourcepath = r'../data\06-08-2022\20220608-183307-CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013\Logs\20220608-183312--CH_Long_Beach_Mission_1_06_07_2022-IVER3-3013.log'

# sourcepath = r'../data\06-27-2022\20220627-202050-CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-202135--CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'
# sourcepath = r'../data\06-27-2022\20220627-203120-CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-203125--CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'
# sourcepath = r'../data\06-27-2022\20220627-203256-SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-203342--SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'
# sourcepath = r'../data\06-27-2022\20220627-203523-SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-203610--SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'
# sourcepath = r'../data\06-27-2022\20220627-203730-SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-203817--SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'
# sourcepath = r'../data\06-27-2022\20220627-203936-SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013\Logs\20220627-204023--SRP_CH_Pool_Mission_6_05_31_2022-IVER3-3013.log'

hydrophone_data = pd.read_csv(targetpath)
hydrophone_data['datetime'] = pd.to_datetime(hydrophone_data['datetime'])

def get_hydrophone_column(name):
    return np.array(hydrophone_data.get(name, [np.nan] * len(hydrophone_data)))

# Add latitude, longitude, and logged speed column using the mission logs
if sourcepath is not None:
    mission_data = pd.read_csv(sourcepath, sep=';')
    if 'datetime' not in mission_data.columns:
        mission_data['datetime'] = pd.to_datetime(mission_data['Date'] + ' ' + mission_data['Time'])
    else:
        mission_data['datetime'] = pd.to_datetime(mission_data['datetime'])
    latitudes = get_hydrophone_column('latitude')
    longitudes = get_hydrophone_column('longitude')
    logged_speeds = get_hydrophone_column('logged_speed')
    logged_heading = get_hydrophone_column('logged_heading')
    mission_iter = utils.pairwise(mission_data.iterrows())
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
                logged_heading[i] = closest_row['C True Heading']
            except StopIteration:
                continue
    hydrophone_data['latitude'] = latitudes
    hydrophone_data['longitude'] = longitudes
    hydrophone_data['logged_speed'] = logged_speeds
    hydrophone_data['logged_heading'] = logged_heading

# Add gps distance using latitude and longitude columns relative to the tag
if 'latitude' in hydrophone_data.columns and 'longitude' in hydrophone_data.columns:
    gps_distances = np.array(hydrophone_data.get('gps_distance', [np.nan] * len(hydrophone_data)))
    for i, hydrophone_row in hydrophone_data.iterrows():
        sensor_coords = (hydrophone_row['latitude'], hydrophone_row['longitude'])
        try:
            gps_distances[i] = geopy.distance.geodesic(tag_coords, sensor_coords).m
        except:
            pass
    hydrophone_data['gps_distance'] = gps_distances

    # Add bearing relative to the tag using latitude and longitude columns
    latitudes, longitudes = get_hydrophone_column('latitude'), get_hydrophone_column('longitude')
    sensor_coords = utils.to_cartesian((latitudes, longitudes), tag_coords)
    hydrophone_data['x'] = sensor_coords[:, 0]
    hydrophone_data['y'] = sensor_coords[:, 1]
    # hydrophone_data['gps_distance'] = np.sqrt(np.square(sensor_x) + np.square(sensor_y))

    heading = get_hydrophone_column('logged_heading')
    compass_vecs = utils.unit_2d(utils.convert_heading(heading))
    hydrophone_data['relative_tag_bearing'] = utils.angle_between(compass_vecs, sensor_coords)
    hydrophone_data['tag_bearing'] = utils.angle_between((1, 0), -sensor_coords)

# Add absolute distances using the first non-nan gps distance
gps_distances = get_hydrophone_column('gps_distance')

# distances_iter = zip(hydrophone_data['total_distance'], gps_distances)
# try:
#     while True:
#         total_distance, gps_distance = next(distances_iter)
#         if not np.isnan(gps_distance):
#             offset = gps_distance - total_distance
#             hydrophone_data['absolute_distance'] = hydrophone_data['total_distance'] + offset
#             break
# except StopIteration:
#     pass

# Add speed and delta time of flight calculated from the GPS coordinates
gps_distances_iter = utils.pairwise(gps_distances)
gps_speeds = get_hydrophone_column('gps_speed')
gps_delta_tof = get_hydrophone_column('gps_delta_tof')
dts = np.array(hydrophone_data.get('dt', [0] + np.diff(hydrophone_data['total_dt']).tolist()))
for i, (current_gps_distance, next_gps_distance) in enumerate(gps_distances_iter):
    dt = dts[i]
    if not np.isnan(current_gps_distance) and not np.isnan(next_gps_distance) and dt != 0:
        delta_distance = (next_gps_distance - current_gps_distance)
        gps_speeds[i] = delta_distance / dt
        gps_delta_tof[i] = delta_distance / speed_of_sound
hydrophone_data['gps_speed'] = gps_speeds
hydrophone_data['gps_delta_tof'] = gps_delta_tof

# Add heading calculated from cartesian coordinates
sensor_x = get_hydrophone_column('x')
sensor_y = get_hydrophone_column('y')
gps_heading = get_hydrophone_column('gps_heading')
coords_iter = utils.pairwise(zip(sensor_x, sensor_y))
for i, ((x, y), (next_x, next_y)) in enumerate(coords_iter):
    dx = next_x - x
    dy = next_y - y
    gps_heading[i] = utils.angle_between((1, 0), (dx, dy))
hydrophone_data['gps_heading'] = gps_heading

# Save dataframe to csv with all of the new columns
hydrophone_data.to_csv(targetpath, index=None)