import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import utils
import os
import contextily as ctx
import mercantile as mt

save = True
replace = True
map_dir = '../maps/OpenStreetMap/Mapnik'
# origin = (10.92641059, -85.7963883)
origin = None

def gaussian(x, mu=0, std=1):
    '''PDF for a 1D gaussian'''
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/std)**2)

def reject_outliers(data, m=2):
    '''Remove datapoints more than m standard deviations away from the mean'''
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# datapath = '../data/06-08-2022/tag78_cowling_none_long_beach_test_457012_0.csv'
# datapath = '../data/06-27-2022/tag78_overnight_test_457049_0.csv'
# datapath = '../data/07-13-2022/santa_elena_bay_coords.csv'
# datapath = '../data/07-15-2022/snorkeling_and_return_coords.csv'
# datapath = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_457049_0.csv'
datapath = '../data/07-21-2022/tag78_swimming_test_tag_coords_1.csv'
name = os.path.splitext(os.path.split(datapath)[1])[0]
data = pd.read_csv(datapath)
distances = np.array(data.get('total_distance', [np.nan] * len(data)))
times = np.array(data.get('total_dt', [np.nan] * len(data)))
signal = np.array(data.get('signal_level', [np.nan] * len(data)))

dt = np.diff(times[~np.isnan(times)])
if len(dt) == 0:
    print('All times are nan')
    has_times = False
else:
    has_times = True

if has_times:
    n = np.round(dt / np.min(dt))

# Plot a histogram of times between tag detections
if has_times:
    normed_dt = reject_outliers(dt / n)
    if len(normed_dt) != 0:
        dt_mu, dt_std = np.mean(normed_dt), np.std(normed_dt)
        x = np.linspace(np.min(normed_dt), np.max(normed_dt), 1001)
        plt.hist(normed_dt, density=True)
        plt.plot(x, gaussian(x, dt_mu, dt_std), label='N(mu={:.8f}, std={:.8f})'.format(dt_mu, dt_std))
        plt.legend()
        fig = plt.gcf()
        plt.show()

        if save:
            savepath = utils.get_savepath(datapath, '_histogram', replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)

# Plot trajectory
if 'x' in data.columns and 'y' in data.columns:
    if origin is not None:
        plt.plot([0], [0], marker='o', color='#ff7f0e', label='Tag')
    elif 'latitude' in data.columns and 'longitude' in data.columns:
        origin = data['latitude'][0], data['longitude'][0]
    x, y = np.array(data['x']), np.array(data['y'])
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    dx = np.concatenate([np.diff(x), [0]])
    dy = np.concatenate([np.diff(y), [0]])
    
    plt.quiver(x, y, dx, dy, label='Trajectory', facecolor='#1f77b4', units='xy', angles='xy', scale_units='xy', scale=1)
    # plt.plot(data['x'], data['y'], marker='.', label='Hydrophone trajectory')
    plt.plot(x[0], y[0], marker='o', color='blue', label='Start')
    plt.plot(x[-1], y[-1], marker='o', color='red', label='End')

    cartesian_bounds = np.array([plt.gca().get_xlim(), plt.gca().get_ylim()])
    cartesian_bounds = utils.pad_bounds(cartesian_bounds.T, f=2).T
    if origin is not None:
        coord_bounds = utils.to_coords(cartesian_bounds, origin)
        (south, west), (north, east) = coord_bounds
        img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir=map_dir)
        true_ext = utils.to_cartesian(np.flip(np.array(ext).reshape(2, 2), axis=0), origin).T.flatten()

    plt.imshow(img, extent=true_ext)
    plt.xlim(cartesian_bounds[0])
    plt.ylim(cartesian_bounds[1])
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()
    plt.suptitle('{} trajectory'.format(name))
    fig = plt.gcf()
    plt.show()

    if save:
        savepath = utils.get_savepath(datapath, '_trajectory', replace=replace)
        print('Saving to {}'.format(savepath))
        utils.savefig(fig, savepath)

# Plot distances
plt.plot(times, distances, label='TOF distance', marker='.')
if 'absolute_distance' in data.columns:
    plt.plot(times, data['absolute_distance'], label='Absolute TOF distance', marker='.')
if 'gps_distance' in data.columns:
    plt.plot(times, data['gps_distance'], label='GPS distance')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.legend()
plt.suptitle('{} distance'.format(name))
fig = plt.gcf()
plt.show()

if save:
    savepath = utils.get_savepath(datapath, '_distance', replace=replace)
    print('Saving to {}'.format(savepath))
    utils.savefig(fig, savepath)

# Plot speeds
if 'gps_speed' in data.columns:
    plt.plot(times, data['gps_speed'], label='GPS speed', marker='.')
    if 'logged_speed' in data.columns:
        plt.plot(times, data['logged_speed'], label='Logged speed', marker='.')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.suptitle('{} speed'.format(name))
    fig = plt.gcf()
    plt.show()

    if save:
        savepath = utils.get_savepath(datapath, '_speed', replace=replace)
        print('Saving to {}'.format(savepath))
        utils.savefig(fig, savepath)

# Plot robot heading and bearing to tag
if 'gps_heading' in data.columns:
    plt.plot(times, utils.wrap_to_180(np.degrees(data['gps_heading'])), label='Hydrophone heading (relative to east)', marker='.')
if 'logged_heading' in data.columns:
    plt.plot(times, utils.wrap_to_180(np.degrees(utils.convert_heading(data['logged_heading']))), label='Logged robot heading (relative to east)', marker='.')
if 'tag_bearing' in data.columns:
    plt.plot(times, utils.wrap_to_180(np.degrees(data['tag_bearing'])), label='Tag bearing (relative to east)', marker='.')
if 'relative_tag_bearing' in data.columns:
    plt.plot(times, utils.wrap_to_180(np.degrees(data['relative_tag_bearing'])), label='Tag bearing (relative to robot heading)', marker='.')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.suptitle('{} direction'.format(name))
fig = plt.gcf()
plt.show()

if save:
    savepath = utils.get_savepath(datapath, '_direction', replace=replace)
    print('Saving to {}'.format(savepath))
    utils.savefig(fig, savepath)

# start_time = datetime.fromisoformat(data['datetime'][0])
# move_0 = datetime.fromisoformat('2022-05-31 16:15:02')
# move_1 = datetime.fromisoformat('2022-05-31 16:19:10')
# move_2 = datetime.fromisoformat('2022-05-31 16:23:00')
# move_3 = datetime.fromisoformat('2022-05-31 16:26:12')
# move_4 = datetime.fromisoformat('2022-05-31 16:29:36')
# move_5 = datetime.fromisoformat('2022-05-31 16:33:30')
# move_6 = datetime.fromisoformat('2022-05-31 16:44:25')
# end_time = datetime.fromisoformat(data['datetime'][len(data)-1])

# groundtruth_times = [start_time, move_0, move_1, move_2, move_3, move_4, move_5, move_6, end_time]
# groundtruth_distances = [0, 20, np.sqrt(20**2 + 5**2), np.sqrt(20**2 + 10**2), np.sqrt(20**2 + 15**2), np.sqrt(20**2 + 20**2), np.sqrt(20**2 + 25**2), 20]
# # groundtruth_distances = [20, 0, 5, 10, 15, 20, 25, 0]

# for i, groundtruth_distance in enumerate(groundtruth_distances):
#     start = groundtruth_times[i]
#     end = groundtruth_times[i + 1]
#     line, = plt.plot(
#         np.linspace((start - start_time).total_seconds(), (end - start_time).total_seconds(), 100),
#         groundtruth_distance * np.ones(100))
    
#     # Set labels
#     if i == 0:
#         line.set(label='estimated groundtruth')
#     if i == 3:
#         line.set(label='groundtruth')
    
#     # Set colors
#     if i >= 3:
#         line.set(color='red')
#     elif i >= 0:
#         line.set(color='orange')
      
# plt.legend()
# plt.show()