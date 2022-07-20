#from appscript import k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import utils
import os
import matplotlib.dates as mdates
import matplotlib.animation as animation

replace = True

def gaussian(x, mu=0, std=1):
    '''PDF for a 1D gaussian'''
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/std)**2)

def reject_outliers(data, m=2):
    '''Remove datapoints more than m standard deviations away from the mean'''
    return data[abs(data - np.mean(data)) < m * np.std(data)]

datapath = '../data/07-07-2022/7_7_22_A_75_D_5_R_3_attempt9'
# datapath = '/Users/Declan/Desktop/HMC/AUV/iver3-summer2022/data/07-06-2022/BFS_PCONTROL_LINE_A_75_R_6_D_5_attempt9'
name = os.path.splitext(os.path.split(datapath)[1])[0]
data = pd.read_csv(datapath)
latitude = np.array(data['Latitude'])
longitude = np.array(data['Longitude'])
# distances = np.array(data['total_distance'])
times = np.array(data['datetime'])
# signal = np.array(data['signal_level'])

x, y = np.array(data['Latitude']), np.array(data['Longitude'])
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]

cartesian_coords = []
origin = np.array([34.109191, -117.712723])
waypoints = np.array([[34.109191, -117.712723],
                      [34.109096, -117.712535],
                      [34.109191, -117.712723]])
origin_cart = utils.to_cartesian(origin, origin)
waypoints_cart = [utils.to_cartesian(waypoint, origin) for waypoint in waypoints]
print("origin_cart", origin_cart)
print("waypoints_cart", waypoints_cart)

for i in range(0, len(x)):
    lat = x[i]
    lon = y[i]
    cartesian_coords.append(utils.to_cartesian(np.array([lat, lon]), origin))

x = [coord[0] for coord in cartesian_coords]
y = [coord[1] for coord in cartesian_coords]

x_way = [coord[0] for coord in waypoints_cart]
y_way = [coord[1] for coord in waypoints_cart]

x = x[2:]
y = y[2:]

## TIME STUFF
# convert to seconds
time = np.array(data['datetime'])
time_object = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in time]
timedelta = time_object[0] - datetime(1900, 1, 1)
seconds = timedelta.total_seconds() #time since epoch

time_object = [((x-datetime(1900, 1, 1))- timedelta) for x in time_object]
time_total_seconds = [y.total_seconds() for y in time_object]

x = x[2:]
y = y[2:]

# Plot gif for the robot moving at each time step overlay the robot vector on top


# Scatterplot for X,Y 
plt.plot(x[0], y[0], marker='o', color='blue', label='Start')
plt.plot(x[-1], y[-1], marker='o', color='red', label='End')
plt.plot(origin_cart[0], origin_cart[1], marker='o', color = 'green', label = 'Origin')
plt.plot(x_way, y_way ,marker='o', color = 'black',label = 'Waypoints')

for i in range(len(x_way)):
    circle = plt.Circle((x_way[i], y_way[i]), 5, color='red', fill=False)
    plt.gca().add_patch(circle)
# plt.axis('scaled')

for i in range(len(x)):
    plt.scatter(x[i], y[i], color=plt.cm.RdYlBu(i))

plt.title("Cartesian Coordinate Plot")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()

# fig = plt.gcf()
# savepath = utils.get_savepath(datapath, '_histogram', replace=replace)
# print('Saving to {}'.format(savepath))
# utils.savefig(fig, savepath)

plt.show()


# Plot controls
"""
yaw_control = 75*(des-true)+128
yaw_control -128 = 75*(des-true)
(yaw_control - 128)/75 = des - true
des = (yaw_control - 128)/75 + true
"""

desired_yaw = data['C True Heading'] - (data['Yaw Control'].apply(int, base=16) - 128)/75
# desired_yaw = (data['Yaw Control'].apply(int, base=16) - 128)/75 + data['C True Heading']
# desired_yaw = -1*(data['Yaw Control'].apply(int, base=16) * (1/75) - data['C True Heading'] -(255/2))
yaw = data['C True Heading']
thrust_control = np.array(data['Thrust Control'])

plt.plot(time_total_seconds, desired_yaw, label = "Desired Yaw")
plt.plot(time_total_seconds, yaw, label = "True Yaw")
plt.xlabel("Seconds")
plt.ylabel("Yaw Control Angle")
plt.legend()

plt.show()

# Plot error between vectors
yaw_error = desired_yaw - yaw
plt.plot(time_total_seconds, yaw_error)
plt.xlabel("Seconds")
plt.ylabel("Yaw Error Angle")
plt.title("Yaw Error Angle")
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
