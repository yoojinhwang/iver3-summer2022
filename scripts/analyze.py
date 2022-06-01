import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def regression(x, y):
    return np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)[0]

def gaussian(x, mu=0, std=1):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/std)**2)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

data = pd.read_csv('../data/tag78_60m_long_beach_test_457049_0.csv')
distances = np.array(data['total_distance'])
times = np.array(data['total_dt'])
dt = np.diff(times)
n = np.round(dt / np.min(dt))

# data = reject_outliers(dt / n)
# dt_mu, dt_std = np.mean(data), np.std(data)
# x = np.linspace(np.min(data), np.max(data), 1001)
# plt.hist(data, density=True)
# plt.plot(x, gaussian(x, dt_mu, dt_std), label='N(mu={:.8f}, std={:.8f})'.format(dt_mu, dt_std))
# plt.legend()
# plt.show()

plt.plot(times, distances, label='measured')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.show()

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