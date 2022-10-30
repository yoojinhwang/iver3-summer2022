import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils
from filter import Filter
from datetime import timedelta
from dataset import Dataset

class KalmanFilter(Filter):
    def from_csv(data):
        start_time = data['datetime'][0]
        end_time = data['datetime'][len(data)-1]
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))

        # Get relevant values from the dataframe
        times = np.array(data['total_dt'])
        delta_tof = np.array(data['delta_tof'])
        signal_level = utils.iir_filter(np.array(data['signal_level']), ff=1)
        global serial_no 
        serial_no = data['serial_no'][0]
        print(serial_no)

        kf = KalmanFilter(save_history=True)

        # Queue predictions
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            kf.queue_prediction(timestamp, np.array([]))
        
        # Queue measurements
        for total_dt, delta_tof, signal_level in zip(times, delta_tof, signal_level):
            timestamp = start_time + timedelta(seconds=total_dt)
            kf.queue_correction(timestamp, np.array([delta_tof, signal_level]))

        return kf

    def from_dataset(dataset, serial_no, save_history=False, ff=1, **kwargs):
        start_time = dataset.start_time
        end_time = dataset.end_time
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))

        # Get relevant values from the dataset
        data = dataset.hydrophones[serial_no].detections
        times = data.index
        delta_tof = np.array(data['delta_tof'])
        signal_level = utils.iir_filter(np.array(data['signal_level']), ff=ff)

        kf = KalmanFilter(save_history=save_history, **kwargs)

        # Queue predictions
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            kf.queue_prediction(timestamp, np.array([]))
        
        # Queue measurements
        for timestamp, delta_tof, signal_level in zip(times, delta_tof, signal_level):
            kf.queue_correction(timestamp, np.array([delta_tof, signal_level]))
        
        return kf

    def __init__(self, save_history=False, **kwargs):
        self._save_history = save_history
        self._kwargs = kwargs
        super().__init__()

        self._speed_of_sound = utils.SPEED_OF_SOUND  # speed of sound in water (m/s)

        self._m = kwargs.get('m', -0.064755099)  # conversion between distance (m) and signal intensity (dB)
        self._l = kwargs.get('l', -1.36582584)  # conversion between speed (m/s) and signal intensity (dB)
        self._b = kwargs.get('b', 77.7946280)  # intercept for conversion between distance and signal intensity

        self._delta_tof_var = kwargs.get('delta_tof_var', 0.0005444242032405411**2)  # variance in the tag's time of flight when stationary (s)

        # self._delta_tof_var = 1e-6
        # self._signal_var = 5.513502243014629**2  # variance in the signal intensity not explained by distance
        # self._signal_var = kwargs.get('signal_var', 15.297)
        self._signal_var = kwargs.get('signal_var', 25)
        # self._signal_var = kwargs.get('signal_var', 100)

        self._distance_var = kwargs.get('distance_var', 1e-3)
        # self._velocity_var = 0.0604
        self._velocity_var = kwargs.get('velocity_var', 1)

    def reset(self):
        super().reset()
        self._state_mean = np.array([self._kwargs.get('initial_r', 0), self._kwargs.get('initial_r_dot', 0)])  # r (m), dr/dt (m/s)
        self._state_cov = np.array([[1e4, 0], [0, 1e4]])  # covariance matrix for the state

        self._history = np.zeros((0,) + self._state_mean.shape)
        self._cov_history = np.zeros((0,) + self._state_cov.shape)
        self._step_history = np.zeros((0,), dtype=np.int32)
        self._time_history = np.zeros((0,), dtype=object)
        self._measurement_history = np.zeros((0, 2))
        # self._update_history(step_type=Filter.PREDICTION)
    
    def _update_history(self, timestamp, step_type, measurement=None):
        if self._save_history:
            self._history = np.concatenate([self._history, [self._state_mean]])
            self._cov_history = np.concatenate([self._cov_history, [self._state_cov]])
            self._step_history = np.concatenate([self._step_history, [step_type]])
            self._time_history = np.concatenate([self._time_history, [timestamp]])
            if step_type == Filter.CORRECTION:
                self._measurement_history = np.concatenate([self._measurement_history, [measurement]])

    def _prediction_step(self, timestamp, data, dt):
        super()._prediction_step(timestamp, data, dt)

        # Perform motion model equations
        F_k = np.array([[1, dt], [0, 1]])  # describes how x_k depends on x_{k-1} free from noise or control
        self._Q_k = np.array([[self._distance_var, 0], [0, self._velocity_var]])  # covariance of noise in state from external forces
        new_state_mean = F_k @ self._state_mean
        new_state_cov = F_k @ self._state_cov @ F_k.T + self._Q_k
        
        # Update object state
        self._state_mean = new_state_mean
        self._state_cov = new_state_cov
        self._update_history(timestamp, Filter.PREDICTION)
    
    def _correction_step(self, timestamp, data, dt):
        super()._correction_step(timestamp, data, dt)

        # Remove extraneous data for the purposes of the particle filter
        measurement = np.array(data[:2])

        # Perform kalman filter update equations
        H_k = np.array([[0, dt/self._speed_of_sound], [self._m, self._l]])
        self._R_k = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])
        K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + self._R_k)
        new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([0, self._b])))
        new_state_cov = self._state_cov - K @ H_k @ self._state_cov

        # Update object state
        self._state_mean = new_state_mean
        self._state_cov = new_state_cov
        self._update_history(timestamp, Filter.CORRECTION, measurement)

    def get_state(self):
        return self._state_mean
    
    def get_state_cov(self):
        return self._state_cov

    def plot(self, dataset=None, serial_no=None, show=True, save=False, replace=True):
        if dataset is None:
            fig, ax1 = plt.subplots()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Retrieve dataset for this hydrophone
        if dataset is not None:
            data = dataset.hydrophones[serial_no]

        # x values are timestamps
        times = self._time_history
        is_correction = self._step_history == Filter.CORRECTION
        correction_times = self._time_history[is_correction]

        # Draw lines to show when detections ocurred
        for time in correction_times:
            ax1.axvline(time, ymin=0, ymax=1, color='gray', linewidth=0.5)

        # Plot kf distances and error bars for it
        kf_distances = self._history[:, 0]
        ax1.plot(times, kf_distances, label='kf distance')
        ax1.scatter(times, kf_distances, marker='.')
        stdev = np.sqrt(self._cov_history[:, 0, 0])
        ax1.fill_between(self._time_history, kf_distances-stdev, kf_distances+stdev, facecolor='#1f77b4', alpha=0.5)
        ax1.fill_between(self._time_history, kf_distances-2*stdev, kf_distances+2*stdev, facecolor='#1f77b4', alpha=0.3)

        if dataset is None:
            tof_distances = np.cumsum(self._measurement_history[:, 0]) * self._speed_of_sound
        else:
            tof_distances = data.detections['total_distance']
        ax1.plot(correction_times, tof_distances, label='tof distance')
        ax1.scatter(correction_times, tof_distances, marker='.')

        # Plot signal distances
        signal_levels = np.array(Dataset.get(pd.Series(self._measurement_history[:, 1], correction_times), times, mode='interpolate')[1])
        if dataset is None:
            speeds = self._history[:, 1]
        else:
            speeds = dataset.get_gps_speed(serial_no, times)
        signal_distances = np.array((signal_levels - self._b - self._l * speeds) / self._m)
        ax1.plot(times, signal_distances, label='signal distance')
        ax1.scatter(correction_times, signal_distances[is_correction], marker='.')

        # Plot gps distances
        if dataset is not None:
            gps_distances = dataset.get_gps_distance(serial_no, times)
            ax1.plot(times, gps_distances, label='gps distance')
            ax1.scatter(data.coords.index, dataset.get_gps_distance(serial_no, data.coords.index), marker='.')

        # Print some parameters onto the plot
        ax1.text(0.99, 0.01, 'Q={}\nR={}\nm={}\nb={}\nl={}'.format(self._Q_k, self._R_k, self._m, self._b, self._l), horizontalalignment='right', transform = ax1.transAxes)

        # Set bounds on the y axis
        if dataset is not None:
            distances = np.concatenate([kf_distances, tof_distances, signal_distances, gps_distances])
        else:
            distances = np.concatenate([kf_distances, tof_distances, signal_distances])
        distances = distances[~np.isnan(distances)]
        min_y = distances.min()
        max_y = distances.max()
        mid_y = (max_y + min_y) / 2
        range_y = 1.2 * (max_y - min_y)
        min_y = mid_y - range_y / 2
        max_y = mid_y + range_y / 2

        # Label axes and set the title
        ax1.set_ybound(min_y, max_y)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance')
        ax1.legend(loc='upper left')

        # Plot errors on axis 2
        if dataset is not None:
            # Draw lines to show when detections ocurred
            for time in correction_times:
                ax2.axvline(time, ymin=0, ymax=1, color='gray', linewidth=0.5)

            # Calculate error only where we have groundtruth
            filtered_kf_distances = kf_distances[is_correction]
            filtered_signal_distances = signal_distances[is_correction]
            filtered_gps_distances = gps_distances[is_correction]
            n = len(correction_times)

            # Calculate RMS for each distance measure
            kf_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances - filtered_kf_distances)) / n)
            tof_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances - tof_distances)) / n)
            signal_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances[1:] - filtered_signal_distances[1:])) / (n-1))

            # Plot error and RMS
            ax2.plot(times, gps_distances - kf_distances, label='kf error, RMS={:.2f}'.format(kf_distance_rms))
            ax2.scatter(correction_times, filtered_gps_distances - filtered_kf_distances, marker='.')
            ax2.plot(correction_times, filtered_gps_distances - tof_distances, label='tof error, RMS={:.2f}'.format(tof_distance_rms))
            ax2.scatter(correction_times, filtered_gps_distances - tof_distances, marker='.')
            ax2.plot(times, gps_distances - signal_distances, label='signal error, RMS={:.2f}'.format(signal_distance_rms))
            ax2.scatter(correction_times, filtered_gps_distances - filtered_signal_distances, marker='.')
            ax2.plot(times, [0]*len(times))
            ax2.scatter(correction_times, [0]*len(correction_times), marker='.')

            # Label axes and set the title
            ax2.set_xlabel('Distance error (m)')
            ax2.set_ylabel('Time (s)')
            ax2.set_title('Error')
            ax2.legend(loc='upper left')

        if show:
            plt.show()
        if save and dataset is not None:
            savepath = utils.add_version('../datasets/{n}/kf_plots/{s}/{s}_kf_plot.png'.format(n=dataset.name, s=serial_no), replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        plt.close()

if __name__ == '__main__':
    # dataset = Dataset('tag78_50m_increment_long_beach_test_457012_2')
    # kf = KalmanFilter.from_dataset(
    #     dataset, 457012,
    #     save_history=True,
    #     m=-0.070397,
    #     l=-0.06,
    #     b=7.74520092e1,
    #     signal_var=500,
    #     initial_r=0,
    #     ff=1)
    # kf.run()
    # kf.plot(dataset, 457012, save=True, replace=False)

    dataset = Dataset('tag78_50m_increment_long_beach_test_0')
    kf = KalmanFilter.from_dataset(
        dataset, 457012,
        save_history=True,
        m=-0.070397,
        l=-0.06,
        b=7.74520092e1,
        signal_var=500,
        initial_r=0,
        ff=0.5)
    kf.run()
    kf.plot(dataset, 457012, save=True, replace=True)

    # Idea: make signal variance increase based on the average amount of time between the last n detections

    # dataset = Dataset('tag78_swimming_test_1_1')
    # kf = KalmanFilter.from_dataset(
    #     dataset, 457049,
    #     save_history=True,
    #     m=-0.20985953,
    #     l=5.5568182,
    #     b=76.90064068,
    #     signal_var=500,
    #     initial_r=120,
    #     ff=0.3)
    # kf.run()
    # kf.plot(dataset, 457049, save=True, replace=True)

    # dataset = Dataset('tag78_swimming_test_1_1')
    # kf = KalmanFilter.from_dataset(
    #     dataset, 'VR100',
    #     save_history=True,
    #     m=-0.10527966,
    #     l=-0.55164737,
    #     b=68.59493072,
    #     signal_var=500,
    #     initial_r=90,
    #     ff=0.3)
    # kf.run()
    # kf.plot(dataset, 'VR100', save=True, replace=True)

    # def plot(self, groundtruth_state=None, datapath=None, show=True, save=True, replace=False):
    #     if groundtruth_state is None:
    #         groundtruth_distance = None
    #         groundtruth_speed = 0
    #         fig, ax1 = plt.subplots()
    #     else:
    #         groundtruth_distance = groundtruth_state[:, 0]
    #         groundtruth_speed = groundtruth_state[:, 1]
    #         fig, (ax1, ax2) = plt.subplots(1, 2)
        
    #     is_prediction = self._step_history == Filter.PREDICTION
    #     all_times = np.array([(timestamp - self._time_history[0]).total_seconds() for timestamp in self._time_history])
    #     correction_times = all_times[~is_prediction]
    #     delta_tof = self._measurement_history[:, 0]
    #     signal_level = self._measurement_history[:, 1]
    #     filtered_distance = self._history[:, 0]

    #     # tof distance subtract the previously fitted line to it to calibrate
    #     # m = 0.007004
    #     # b = -42.240582 

    #     tof_distance = np.cumsum(delta_tof * self._speed_of_sound)
    #     print("before calibration")

    #     if datapath == '457012':
    #         print("calibration for 457012")
    #         tof_distance = tof_distance + 300.008905278
    #         # tof_distance = tof_distance - (0.007004*correction_times -42.240582)

    #     elif datapath == '457049':
    #         tof_distance = tof_distance - 852.19219604
    #     print("after calibration", tof_distance)
        
    #     signal_distance = np.array((signal_level - self._b - self._l * groundtruth_speed) / self._m)
    #     smoothed_signal_distance = utils.iir_filter(signal_distance, ff=0.3)

    #     # Colors: #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2
    #     # Plot filtered distance
    #     ax1.plot(all_times, filtered_distance, label='Filtered distance', marker='.')
    #     stdev = np.sqrt(self._cov_history[:, 0, 0])
    #     ax1.fill_between(all_times, self._history[:, 0]-stdev, self._history[:, 0]+stdev, facecolor='#1f77b4', alpha=0.5)
    #     ax1.fill_between(all_times, self._history[:, 0]-2*stdev, self._history[:, 0]+2*stdev, facecolor='#1f77b4', alpha=0.3)
    #     ax1.plot(correction_times, tof_distance, label='TOF distance', marker='.')
    #     ax1.plot(correction_times, signal_distance, label='Signal distance', marker='.')
    #     if groundtruth_distance is not None:
    #         ax1.plot(correction_times, groundtruth_distance, label='Groundtruth distance', marker='.')

    #     # Compute y bounds for the plot by scaling the minimum and maximum distances
    #     if groundtruth_distance is not None:
    #         distances = np.concatenate([filtered_distance, tof_distance, signal_distance, groundtruth_distance])
    #     else:
    #         distances = np.concatenate([filtered_distance, tof_distance, signal_distance])
    #     distances = distances[~np.isnan(distances)]
    #     min_y = distances.min()
    #     max_y = distances.max()
    #     mid_y = (max_y + min_y) / 2
    #     range_y = 1.2 * (max_y - min_y)
    #     min_y = mid_y - range_y / 2
    #     max_y = mid_y + range_y / 2
    #     ax1.text(0.99, 0.01, 'Q={}\nR={}\nm={}\nb={}\nl={}'.format(self._Q_k, self._R_k, self._m, self._b, self._l), horizontalalignment='right', transform = ax1.transAxes)
    #     ax1.set_ybound(min_y, max_y)
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Distance (m)')
    #     ax1.set_title('Distance')
    #     ax1.legend()

    #     if groundtruth_distance is not None:
    #         filtered_error = filtered_distance[~is_prediction] - groundtruth_distance
    #         total_filtered_error = np.sqrt(np.sum(np.square(filtered_error[~np.isnan(filtered_error)])))
    #         tof_error = tof_distance - groundtruth_distance
    #         print("toferror", tof_error)

    #         # print("start", tof_error)
    #         # print("first non nan value", tof_error[np.isfinite(tof_error)][0])
    #         # print("last non nan value", tof_error[np.isfinite(tof_error)][-1])

    #         # absolute tof error 
    #         # tof_error = tof_error + np.abs(tof_error[np.isfinite(tof_error)][0] - tof_error[np.isfinite(tof_error)][-1])

    #         total_tof_error = np.sqrt(np.sum(np.square(tof_error[~np.isnan(tof_error)])))
    #         signal_error = signal_distance - groundtruth_distance
    #         total_signal_error = np.sqrt(np.sum(np.square(signal_error[~np.isnan(signal_error)])))
    #         smoothed_signal_error = smoothed_signal_distance - groundtruth_distance
    #         total_smoothed_signal_error = np.sqrt(np.sum(np.square(smoothed_signal_error[~np.isnan(smoothed_signal_error)])))
    #         ax2.plot(correction_times, filtered_error, marker='.', label='Filtered error: {:.6f}'.format(total_filtered_error))
    #         ax2.plot(correction_times, tof_error, marker='.', label='TOF error: {:.6f}'.format(total_tof_error))
    #         ax2.plot(correction_times, signal_error, marker='.', label='Signal error: {:.6f}'.format(total_signal_error))
    #         # ax2.plot(self._times, smoothed_signal_error, marker='.', label='Smoothed signal error: {:.6f}'.format(total_smoothed_signal_error))

    #         isnan = np.logical_or(np.isnan(tof_error), np.isnan(correction_times))
    #         tof_error_subset = np.array(tof_error)[~isnan]
    #         correction_times_subset = np.array(correction_times)[~isnan]

    #         m, b = utils.fit_line(correction_times_subset, tof_error_subset)
    #         r_sqr = np.corrcoef(correction_times_subset, tof_error_subset)[0][1] ** 2
    #         ax2.plot(correction_times_subset, m*correction_times_subset + b, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr))

    #         ax2.set_xlabel('Time (s)')
    #         ax2.set_ylabel('Distance error (m)')
    #         ax2.set_title('Error')
    #         ax2.legend()

    #     # Add a title
    #     if datapath is not None:
    #         fig.suptitle(os.path.split(datapath)[1])

    #     # Show plot
    #     if show:
    #         plt.show()

    #     # Save figure
    #     if save:
    #         savepath = utils.get_savepath(datapath, '_kalman_filter', replace=replace)
    #         print('Saving to {}'.format(savepath))
    #         utils.savefig(fig, savepath)
        
    #     plt.close()

    #     self._filtered_distance = filtered_distance
    #     self._filtered_error = filtered_error
    #     self._total_filtered_error = total_filtered_error
    #     self._tof_distance = tof_distance
    #     self._tof_error = tof_error
    #     self._total_tof_error = total_tof_error
    #     self._signal_distance = signal_distance
    #     self._signal_error = signal_error
    #     self._total_signal_error = total_signal_error

# if __name__ == '__main__':
#     # datapath = '../data/07-18-2022/tag78_drift_test_VR100_0.csv'

#     # datapath = '../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv'
#     datapath = '../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv'
#     data = pd.read_csv(datapath)
#     data = data[data['tag_id'] == 65478].reset_index(drop=True)
#     data['datetime'] = pd.to_datetime(data['datetime'])
#     groundtruth_state = np.column_stack([data['gps_distance'], data['gps_speed']])
#     kf = KalmanFilter.from_csv(data)


#     # # VR100
#     # kf._m = -0.10527966
#     # kf._l = -0.55164737
#     # kf._b = 68.59493072
#     # kf._signal_var = 16.250003

#     # 457049
#     # # Found this from tag78_drift_test_457049_0_model_distance_gps_speed.png
#     # kf._m = -0.20985953
#     # kf._l = 5.5568182
#     # kf._b = 76.90064068
#     # kf._signal_var = 9.400336

#     # 457012
#     # Found this from clip_tag78_50m_increment_long_beach_test_457012
#     kf._m = -0.0580963639
#     kf._l = -0.791950975
#     kf._b = 76.890277
#     kf._signal_var = 13.522776

#     # # TODO: 4570
#     # # Found this from clip_tag78_50m_increment_long_beach_test_4570
#     # kf._m = -0.0580963639
#     # kf._l = -0.791950975
#     # kf._b = 76.890277
#     # kf._signal_var = 13.522776
    
#     kf.run()
#     kf.plot(groundtruth_state=groundtruth_state, datapath=datapath, save=False)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import utils

# class KalmanFilter():
#     def __init__(self, data_path, prediction_rate=1):
#         # Retrieve dataframe
#         self._data_path = data_path
#         self._data = pd.read_csv(data_path)

#         # Amount of time to wait between each prediction step
#         self._prediction_period = 1 / prediction_rate

#         # Get relevant values from dataframe
#         self._times = np.array(self._data['total_dt'])
#         dt = np.diff(self._times)
#         avg_dt = utils.avg_dt_dict[65478]
#         # avg_dt = 8.18
#         self._delta_tof = np.array(self._data['delta_tof'])
#         # self._delta_tof = np.where(np.isnan(self._data['gps_delta_tof']), 0, self._data['gps_delta_tof'])
#         # self._delta_tof = np.array([0] + ((dt - avg_dt / 2) % avg_dt - avg_dt / 2).tolist())
#         self._signal_level = utils.iir_filter(np.array(self._data['signal_level']), ff=1)
#         self._start_time = 0
#         self._end_time = self._times[-1]

#         self._speed_of_sound = 1460  # speed of sound in water (m/s)
#         # self._speed_of_sound = 343  # speed of sound in air (m/s)
#         self._l = -1.36582584
#         # self._l = 0
#         self._m = -0.064755099  # conversion between distance (m) and signal intensity (dB)
#         self._b = 77.7946280  # intercept for conversion between distance and signal intensity
#         # self._m = -0.059907  # conversion between distance (m) and signal intensity (dB)
#         # self._b = 77.074412  # intercept for conversion between distance and signal intensity
#         # self._m = -0.068464  # conversion between distance (m) and signal intensity (dB)
#         # self._b = 79.036371  # intercept for conversion between distance and signal intensity
#         self._delta_tof_var = 0.0005444242032405411**2  # variance in the tag's time of flight when stationary (s)
#         # self._delta_tof_var = 1e-6
#         # self._signal_var = 5.513502243014629**2  # variance in the signal intensity not explained by distance
#         self._signal_var = 15.297

#         self._distance_var = 1e-3
#         self._velocity_var = 0.0604

#         # Setup the rest of the variables that need to change when run is called
#         self.reset()

#     def reset(self):
#         self._current_time = self._start_time
#         self._last_prediction_time = self._start_time
#         self._last_measurement_time = self._times[0]
#         # self._state_mean = np.array([self._m * self._signal_level[0] + self._b, 0])  # r (m), dr/dt (m/s)
#         self._state_mean = np.array([0, 0])  # r (m), dr/dt (m/s)
#         self._state_cov = np.array([[1e4, 0], [0, 1e4]])  # covariance matrix for the state
#         self._next_measurement = 0
#         self._history = [self._state_mean]
#         self._cov_history = [self._state_cov]
    
#     def prediction_step(self, control, dt):
#         F_k = np.array([[1, dt], [0, 1]])  # describes how x_k depends on x_{k-1} free from noise or control
#         self._Q_k = np.array([[self._distance_var, 0], [0, self._velocity_var]])  # covariance of noise in state from external forces
#         new_state_mean = F_k @ self._state_mean
#         new_state_cov = F_k @ self._state_cov @ F_k.T + self._Q_k
#         return new_state_mean, new_state_cov
    
#     def correction_step(self, measurement, dt):
#         H_k = np.array([[0, dt/self._speed_of_sound], [self._m, self._l]])
#         self._R_k = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])
#         K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + self._R_k)
#         new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([0, self._b])))
#         new_state_cov = self._state_cov - K @ H_k @ self._state_cov
#         return new_state_mean, new_state_cov
    
#     def run(self):
#         while self._next_measurement < len(self._times):
#             measurement_time = self._times[self._next_measurement]
#             if self._current_time <= measurement_time:
#                 # print('Prediction step {}, {}'.format(self._current_time, measurement_time))
#                 # Update state
#                 self._state_mean, self._state_cov = self.prediction_step(None, self._current_time - self._last_prediction_time)
#                 self._history.append(self._state_mean)
#                 self._cov_history.append(self._state_cov)

#                 # Update last prediction time and current time
#                 self._last_prediction_time = self._current_time
#                 self._current_time += self._prediction_period
#             else:
#                 # print('Update step {}, {}'.format(self._current_time, measurement_time))
#                 # Update state
#                 measurement = np.array([self._delta_tof[self._next_measurement], self._signal_level[self._next_measurement]])
#                 self._state_mean, self._state_cov = self.correction_step(measurement, measurement_time - self._last_measurement_time)
#                 self._history[-1] = self._state_mean
#                 self._cov_history[-1] = self._state_cov

#                 # Update last measurement time and what the next measurement is, but not the current time since we still want to run a prediction step
#                 self._last_measurement_time = measurement_time
#                 self._next_measurement += 1
#         self._history = np.array(self._history)
#         self._cov_history = np.array(self._cov_history)

#     def plot(self, show=True, save=True, replace=False):
#         prediction_times = np.arange(len(self._history)) * self._prediction_period
#         filtered_distance = self._history[:, 0]
#         tof_distance = np.cumsum(self._delta_tof * self._speed_of_sound)
#         signal_distance = np.array((self._signal_level - self._b - self._l * self._data['gps_speed']) / self._m)
#         smoothed_signal_distance = utils.iir_filter(signal_distance, ff=0.3)
#         if 'gps_distance' in self._data.columns:
#             groundtruth_distance = np.array(self._data['gps_distance'])
#             fig, (ax1, ax2) = plt.subplots(1, 2)
#         else:
#             groundtruth_distance = None
#             fig, ax1 = plt.subplots()

#         # Colors: #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2
#         # Plot filtered distance
#         ax1.plot(prediction_times, filtered_distance, label='Filtered distance', marker='.')
#         stdev = np.sqrt(self._cov_history[:, 0, 0])
#         ax1.fill_between(prediction_times, self._history[:, 0]-stdev, self._history[:, 0]+stdev, facecolor='#1f77b4', alpha=0.5)
#         ax1.fill_between(prediction_times, self._history[:, 0]-2*stdev, self._history[:, 0]+2*stdev, facecolor='#1f77b4', alpha=0.3)
#         ax1.plot(self._times, tof_distance, label='TOF distance', marker='.')
#         ax1.plot(self._times, signal_distance, label='Signal distance', marker='.')
#         if groundtruth_distance is not None:
#             ax1.plot(self._times, groundtruth_distance, label='Groundtruth distance', marker='.')

#         # Compute y bounds for the plot by scaling the minimum and maximum distances
#         if groundtruth_distance is not None:
#             distances = np.concatenate([filtered_distance, tof_distance, signal_distance, groundtruth_distance])
#         else:
#             distances = np.concatenate([filtered_distance, tof_distance, signal_distance])
#         distances = distances[~np.isnan(distances)]
#         min_y = distances.min()
#         max_y = distances.max()
#         mid_y = (max_y + min_y) / 2
#         range_y = 1.2 * (max_y - min_y)
#         min_y = mid_y - range_y / 2
#         max_y = mid_y + range_y / 2
#         ax1.text(0.99, 0.01, 'Q={}\nR={}\nm={}\nb={}\nl={}'.format(self._Q_k, self._R_k, self._m, self._b, self._l), horizontalalignment='right', transform = ax1.transAxes)
#         ax1.set_ybound(min_y, max_y)
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Distance (m)')
#         ax1.set_title('Distance')
#         ax1.legend()

#         if groundtruth_distance is not None:
#             filtered_error = filtered_distance[self._times.astype(int)] - groundtruth_distance
#             total_filtered_error = np.sqrt(np.sum(np.square(filtered_error[~np.isnan(filtered_error)])))
#             tof_error = tof_distance - groundtruth_distance
#             total_tof_error = np.sqrt(np.sum(np.square(tof_error[~np.isnan(tof_error)])))
#             signal_error = signal_distance - groundtruth_distance
#             total_signal_error = np.sqrt(np.sum(np.square(signal_error[~np.isnan(signal_error)])))
#             smoothed_signal_error = smoothed_signal_distance - groundtruth_distance
#             total_smoothed_signal_error = np.sqrt(np.sum(np.square(smoothed_signal_error[~np.isnan(smoothed_signal_error)])))
#             ax2.plot(self._times, filtered_error, marker='.', label='Filtered error: {:.6f}'.format(total_filtered_error))
#             ax2.plot(self._times, tof_error, marker='.', label='TOF error: {:.6f}'.format(total_tof_error))
#             ax2.plot(self._times, signal_error, marker='.', label='Signal error: {:.6f}'.format(total_signal_error))
#             # ax2.plot(self._times, smoothed_signal_error, marker='.', label='Smoothed signal error: {:.6f}'.format(total_smoothed_signal_error))
#             ax2.set_xlabel('Time (s)')
#             ax2.set_ylabel('Distance error (m)')
#             ax2.set_title('Error')
#             ax2.legend()

#         # Add a title
#         fig.suptitle(os.path.split(self._data_path)[1])

#         # Show plot
#         if show:
#             plt.show()

#         # Save figure
#         if save:
#             savepath = utils.get_savepath(self._data_path, '_kalman_filter', replace=replace)
#             print('Saving to {}'.format(savepath))
#             utils.savefig(fig, savepath)
        
#         plt.close()

#         self._filtered_distance = filtered_distance
#         self._filtered_error = filtered_error
#         self._total_filtered_error = total_filtered_error
#         self._tof_distance = tof_distance
#         self._tof_error = tof_error
#         self._total_tof_error = total_tof_error
#         self._signal_distance = signal_distance
#         self._signal_error = signal_error
#         self._total_signal_error = total_signal_error


# if __name__ == '__main__':
#     save_to = '../plots/06-08-2022'
#     replace = False

#     # kf = KalmanFilter('../data/05-26-2022/tag78-0m-air-test-0.csv')
#     # kf = KalmanFilter('../data/06-01-2022/tag78_50m_increment_long_beach_test_457012_2.csv', prediction_rate=1)
#     kf = KalmanFilter('../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv', prediction_rate=1)
#     # kf = KalmanFilter('../data/06-08-2022/tag78_cowling_none_long_beach_test_457012_0.csv', prediction_rate=1)
#     # kf = KalmanFilter('../data/06-08-2022/tag78_cowling_none_long_beach_test_457049_0.csv', prediction_rate=1)
#     # kf = KalmanFilter('../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv', prediction_rate=1)
    
#     kf.run()
#     kf.plot(save=True, replace=replace)

#     # paths = [
#     #     '../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv',
#     #     '../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv',
#     #     '../data/06-08-2022/tag78_cowling_none_long_beach_test_457012_0.csv',
#     #     '../data/06-08-2022/tag78_cowling_none_long_beach_test_457049_0.csv'
#     # ]
#     # errors = []
#     # data_idx = 0  # 0 means total error. 3 means average error. 4 means error standard deviation.
#     # # xrange = range(-8, 3)
#     # # yrange = range(-8, 3)
#     # # xrange = np.linspace(1e-3, 1e-1, 11)
#     # # yrange = np.linspace(1e-3, 1e-1, 11)
#     # xrange = np.linspace(0.0001, 0.001, 11)
#     # yrange = np.linspace(0.0001, 0.001, 11)
#     # for path in paths:
#     #     kf = KalmanFilter(path, prediction_rate=1)
#     #     errors.append([])
#     #     for i in yrange:
#     #         errors[-1].append([])
#     #         for j in xrange:
#     #             kf._distance_var = i
#     #             kf._velocity_var = j
#     #             kf.reset()
#     #             kf.run()
#     #             kf.plot(show=False, save=False)
#     #             isnan = np.isnan(kf._filtered_error)
#     #             errors[-1][-1].append([
#     #                 kf._total_filtered_error,
#     #                 np.max(kf._filtered_error[~isnan]),
#     #                 np.min(kf._filtered_error[~isnan]),
#     #                 np.mean(kf._filtered_error[~isnan]),
#     #                 np.std(kf._filtered_error[~isnan])
#     #             ])
#     #             print('Distance var={}, velocity var={}, total error={}, error mean={}, error stdev={}'.format(
#     #                 kf._distance_var, kf._velocity_var, errors[-1][-1][-1][0], errors[-1][-1][-1][3], errors[-1][-1][-1][4]
#     #             ))
#     #     errors[-1] = np.array(errors[-1])
#     #     im = np.flip(np.abs(errors[-1][:, :, data_idx]), axis=0)
#     #     ax = sns.heatmap(np.log(im), annot=im, linewidth=0.5, xticklabels=xrange, yticklabels=list(reversed(yrange)))
#     #     ax.set_xlabel('Velocity variance')
#     #     ax.set_ylabel('Distance variance')
#     #     ax.set_title('{} KF total error '.format(os.path.split(os.path.splitext(path)[0])[1]))
#     #     fig = plt.gcf()
#     #     plt.show()

#     #     savepath = utils.get_savepath(path, '_kf_param_heatmap', replace=replace)
#     #     print('Saving to {}'.format(savepath))
#     #     utils.savefig(fig, savepath)
#     # errors = np.array(errors)
#     # # sums = np.apply_along_axis(np.sum, 1, np.abs(errors[:, :, :, data_idx]).reshape(len(paths), -1)).reshape(-1, 1, 1)
#     # sums = utils.apply_along_axes(np.sum, (1, 3), np.abs(errors[:, :, :, data_idx])).reshape(-1, 1, 1)
#     # im = np.flip(np.abs(np.sum(errors[:, :, :, data_idx] / (4 * sums), axis=0)), axis=0)
#     # ax = sns.heatmap(np.log(im), annot=im, linewidth=0.5, xticklabels=xrange, yticklabels=list(reversed(yrange)))
#     # ax.set_xlabel('Velocity variance')
#     # ax.set_ylabel('Distance variance')
#     # ax.set_title('KF total error (all)')
#     # fig = plt.gcf()
#     # plt.show()

#     # savepath = utils.add_version(os.path.join(save_to, 'kf_param_heatmap_all.png'), replace=replace)
#     # print('Saving to {}'.format(savepath))
#     # utils.savefig(fig, savepath)