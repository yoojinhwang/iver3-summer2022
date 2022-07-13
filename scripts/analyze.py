import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import utils
import os

replace = True

def gaussian(x, mu=0, std=1):
    '''PDF for a 1D gaussian'''
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/std)**2)

def reject_outliers(data, m=2):
    '''Remove datapoints more than m standard deviations away from the mean'''
    return data[abs(data - np.mean(data)) < m * np.std(data)]

datapath = '../data/06-29-2022/tag78_cowling_small_snail_BFS_test_457012_0.csv'
# datapath = '../data/06-30-2022/test_circle_around.csv'
# datapath = '/Users/Declan/Desktop/HMC/AUV/iver3-summer2022/data/06-30-2022/test_circle_around.csv'
name = os.path.splitext(os.path.split(datapath)[1])[0]
data = pd.read_csv(datapath)
latitude = np.array(data['Latitude'])
longitude = np.array(data['Longitude'])
# distances = np.array(data['total_distance'])
# times = np.array(data['total_dt'])
# signal = np.array(data['signal_level'])
# dt = np.diff(times[~np.isnan(times)])
# n = np.round(dt / np.min(dt))

x, y = np.array(data['Latitude']), np.array(data['Longitude'])
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]

#print(x)
#print(y)

cartesian_coords = []
#origin = np.array([34.1064, -117.7125])
origin = np.array([34.106129, -117.713168])

coords = np.array([x[1], y[1]])
ref = origin


R = 6371009
coords = np.radians(coords)
ref = np.radians(ref)
delta_x = R * (coords[1] - ref[1]) * np.cos(ref[0])

print(coords[1] - ref[1])
print(ref[1])


for i in range(0, len(x)):
    lat = x[i]
    lon = y[i]
    cartesian_coords.append(utils.to_cartesian(np.array([lat, lon]), origin))

#y = cartesian_coords[:,0]
#x = cartesian_coords[:,1]
x = [coord[0] for coord in cartesian_coords]
y = [coord[1] for coord in cartesian_coords]

plt.plot(x[0], y[0], marker='o', color='blue', label='Start')
plt.plot(x[-1], y[-1], marker='o', color='red', label='End')

plt.scatter(x=x, y = y)
plt.title("GPS Coordinate Plots")
#plt.xlabel("Latitude")
#plt.ylabel("Longitude")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
#fig = plt.gcf()
#savepath = utils.get_savepath(datapath, '_histogram', replace=replace)
#print('Saving to {}'.format(savepath))
#utils.savefig(fig, savepath)

plt.show()

# plt.plot(x[0], y[0], marker='o', color='blue', label='Start')
# plt.plot(x[-1], y[-1], marker='o', color='red', label='End')
# plt.plot([0], [0], marker='o', color='#ff7f0e', label='Tag')
# plt.show()

# # Plot a histogram of times between tag detections
# normed_dt = reject_outliers(dt / n)
# dt_mu, dt_std = np.mean(normed_dt), np.std(normed_dt)
# x = np.linspace(np.min(normed_dt), np.max(normed_dt), 1001)
# plt.hist(normed_dt, density=True)
# plt.plot(x, gaussian(x, dt_mu, dt_std), label='N(mu={:.8f}, std={:.8f})'.format(dt_mu, dt_std))
# plt.legend()
# fig = plt.gcf()
# plt.show()

# savepath = utils.get_savepath(datapath, '_histogram', replace=replace)
# print('Saving to {}'.format(savepath))
# utils.savefig(fig, savepath)

# Plot trajectory
# if 'Latitude' in data.columns and 'Longitude' in data.columns:
#     x, y = np.array(data['Latitude']), np.array(data['Longitude'])
#     x = x[~np.isnan(x)]
#     y = y[~np.isnan(y)]
#     dx = np.concatenate([np.diff(x), [0]])
#     dy = np.concatenate([np.diff(y), [0]])
    
#     plt.quiver(x, y, dx, dy, label='Hydrophone trajectory', facecolor='#1f77b4', units='xy', angles='xy', scale_units='xy', scale=1)
#     # plt.plot(data['x'], data['y'], marker='.', label='Hydrophone trajectory')
#     plt.plot(x[0], y[0], marker='o', color='blue', label='Start')
#     plt.plot(x[-1], y[-1], marker='o', color='red', label='End')
#     plt.plot([0], [0], marker='o', color='#ff7f0e', label='Tag')
#     plt.xlabel('East (m)')
#     plt.ylabel('North (m)')
#     plt.legend()
#     plt.suptitle('{} trajectory'.format(name))
#     fig = plt.gcf()
#     plt.show()

#     savepath = utils.get_savepath(datapath, '_trajectory', replace=replace)
#     print('Saving to {}'.format(savepath))
#     utils.savefig(fig, savepath)

# # Plot distances
# plt.plot(times, distances, label='TOF distance', marker='.')
# if 'absolute_distance' in data.columns:
#     plt.plot(times, data['absolute_distance'], label='Absolute TOF distance', marker='.')
# if 'gps_distance' in data.columns:
#     plt.plot(times, data['gps_distance'], label='GPS distance')
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.legend()
# plt.suptitle('{} distance'.format(name))
# fig = plt.gcf()
# plt.show()

# savepath = utils.get_savepath(datapath, '_distance', replace=replace)
# print('Saving to {}'.format(savepath))
# utils.savefig(fig, savepath)

# # Plot speeds
# if 'gps_speed' in data.columns:
#     plt.plot(times, data['gps_speed'], label='GPS speed', marker='.')
#     if 'logged_speed' in data.columns:
#         plt.plot(times, data['logged_speed'], label='Logged speed', marker='.')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Speed (m/s)')
#     plt.legend()
#     plt.suptitle('{} speed'.format(name))
#     fig = plt.gcf()
#     plt.show()

#     savepath = utils.get_savepath(datapath, '_speed', replace=replace)
#     print('Saving to {}'.format(savepath))
#     utils.savefig(fig, savepath)

# # Plot robot heading and bearing to tag
# if 'gps_heading' in data.columns:
#     plt.plot(times, utils.wrap_to_180(np.degrees(data['gps_heading'])), label='Hydrophone heading (relative to east)', marker='.')
# if 'logged_heading' in data.columns:
#     plt.plot(times, utils.wrap_to_180(np.degrees(utils.convert_heading(data['logged_heading']))), label='Logged robot heading (relative to east)', marker='.')
# if 'tag_bearing' in data.columns:
#     plt.plot(times, utils.wrap_to_180(np.degrees(data['tag_bearing'])), label='Tag bearing (relative to east)', marker='.')
# if 'relative_tag_bearing' in data.columns:
#     plt.plot(times, utils.wrap_to_180(np.degrees(data['relative_tag_bearing'])), label='Tag bearing (relative to robot heading)', marker='.')
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (deg)')
# plt.legend()
# plt.suptitle('{} direction'.format(name))
# fig = plt.gcf()
# plt.show()

# savepath = utils.get_savepath(datapath, '_direction', replace=replace)
# print('Saving to {}'.format(savepath))
# utils.savefig(fig, savepath)

# Plot signal level vs angle
plt.scatter(data['relative_tag_bearing'], data['signal_level'], label='Relative tag bearing')
plt.scatter(utils.convert_heading(data['logged_heading']), data['signal_level'], label='Compass heading (from east)')
plt.xlabel('Angle (rad)')
plt.ylabel('Signal level (dB)')
plt.legend()
plt.suptitle('{} signal level vs angle'.format(name))
fig = plt.gcf()
plt.show()
savepath = utils.get_savepath(datapath, '_signal_vs_angle', replace=replace)
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