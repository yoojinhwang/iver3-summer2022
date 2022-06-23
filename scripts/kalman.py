import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils

class KalmanFilter():
    def __init__(self, data_path, prediction_rate=1):
        # Retrieve dataframe
        self._data_path = data_path
        self._data = pd.read_csv(data_path)

        # Amount of time to wait between each prediction step
        self._prediction_period = 1 / prediction_rate

        # Get relevant values from dataframe
        self._times = np.array(self._data['total_dt'])
        dt = np.diff(self._times)
        avg_dt = utils.avg_dt_dict[65478]
        # avg_dt = 8.18
        self._delta_tof = np.array(self._data['delta_tof'])
        # self._delta_tof = np.where(np.isnan(self._data['gps_delta_tof']), 0, self._data['gps_delta_tof'])
        # self._delta_tof = np.array([0] + ((dt - avg_dt / 2) % avg_dt - avg_dt / 2).tolist())
        self._signal_level = utils.iir_filter(np.array(self._data['signal_level']), ff=1)
        self._start_time = 0
        self._end_time = self._times[-1]

        # self._speed_of_sound = 1460  # speed of sound in water (m)
        self._speed_of_sound = 1460  # speed of sound in air (m)
        self._l = -1.36582584
        # self._l = 0
        self._m = -0.064755099  # conversion between distance (m) and signal intensity (dB)
        self._b = 77.7946280  # intercept for conversion between distance and signal intensity
        # self._m = -0.059907  # conversion between distance (m) and signal intensity (dB)
        # self._b = 77.074412  # intercept for conversion between distance and signal intensity
        # self._m = -0.068464  # conversion between distance (m) and signal intensity (dB)
        # self._b = 79.036371  # intercept for conversion between distance and signal intensity
        self._delta_tof_var = 0.0005444242032405411**2  # variance in the tag's time of flight when stationary (s)
        # self._delta_tof_var = 1e-6
        # self._signal_var = 5.513502243014629**2  # variance in the signal intensity not explained by distance
        self._signal_var = 15.297

        self._distance_var = 1e-3
        self._velocity_var = 0.0604

        # Setup the rest of the variables that need to change when run is called
        self.reset()

    def reset(self):
        self._current_time = self._start_time
        self._last_prediction_time = self._start_time
        self._last_measurement_time = self._times[0]
        # self._state_mean = np.array([self._m * self._signal_level[0] + self._b, 0])  # r (m), dr/dt (m/s)
        self._state_mean = np.array([0, 0])  # r (m), dr/dt (m/s)
        self._state_cov = np.array([[1e4, 0], [0, 1e4]])  # covariance matrix for the state
        self._next_measurement = 0
        self._history = [self._state_mean]
        self._cov_history = [self._state_cov]
    
    def prediction_step(self, control, dt):
        F_k = np.array([[1, dt], [0, 1]])  # describes how x_k depends on x_{k-1} free from noise or control
        self._Q_k = np.array([[self._distance_var, 0], [0, self._velocity_var]])  # covariance of noise in state from external forces
        new_state_mean = F_k @ self._state_mean
        new_state_cov = F_k @ self._state_cov @ F_k.T + self._Q_k
        return new_state_mean, new_state_cov
    
    def correction_step(self, measurement, dt):
        H_k = np.array([[0, dt/self._speed_of_sound], [self._m, self._l]])
        self._R_k = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])
        K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + self._R_k)
        new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([0, self._b])))
        new_state_cov = self._state_cov - K @ H_k @ self._state_cov
        return new_state_mean, new_state_cov
    
    def run(self):
        while self._next_measurement < len(self._times):
            measurement_time = self._times[self._next_measurement]
            if self._current_time <= measurement_time:
                # print('Prediction step {}, {}'.format(self._current_time, measurement_time))
                # Update state
                self._state_mean, self._state_cov = self.prediction_step(None, self._current_time - self._last_prediction_time)
                self._history.append(self._state_mean)
                self._cov_history.append(self._state_cov)

                # Update last prediction time and current time
                self._last_prediction_time = self._current_time
                self._current_time += self._prediction_period
            else:
                # print('Update step {}, {}'.format(self._current_time, measurement_time))
                # Update state
                measurement = np.array([self._delta_tof[self._next_measurement], self._signal_level[self._next_measurement]])
                self._state_mean, self._state_cov = self.correction_step(measurement, measurement_time - self._last_measurement_time)
                self._history[-1] = self._state_mean
                self._cov_history[-1] = self._state_cov

                # Update last measurement time and what the next measurement is, but not the current time since we still want to run a prediction step
                self._last_measurement_time = measurement_time
                self._next_measurement += 1
        self._history = np.array(self._history)
        self._cov_history = np.array(self._cov_history)

    def plot(self, show=True, save=True, replace=False):
        prediction_times = np.arange(len(self._history)) * self._prediction_period
        filtered_distance = self._history[:, 0]
        tof_distance = np.cumsum(self._delta_tof * self._speed_of_sound)
        signal_distance = np.array((self._signal_level - self._b - self._l * self._data['gps_speed']) / self._m)
        smoothed_signal_distance = utils.iir_filter(signal_distance, ff=0.3)
        if 'gps_distance' in self._data.columns:
            groundtruth_distance = np.array(self._data['gps_distance'])
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            groundtruth_distance = None
            fig, ax1 = plt.subplots()

        # Colors: #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2
        # Plot filtered distance
        ax1.plot(prediction_times, filtered_distance, label='Filtered distance', marker='.')
        stdev = np.sqrt(self._cov_history[:, 0, 0])
        ax1.fill_between(prediction_times, self._history[:, 0]-stdev, self._history[:, 0]+stdev, facecolor='#1f77b4', alpha=0.5)
        ax1.fill_between(prediction_times, self._history[:, 0]-2*stdev, self._history[:, 0]+2*stdev, facecolor='#1f77b4', alpha=0.3)
        ax1.plot(self._times, tof_distance, label='TOF distance', marker='.')
        ax1.plot(self._times, signal_distance, label='Signal distance', marker='.')
        if groundtruth_distance is not None:
            ax1.plot(self._times, groundtruth_distance, label='Groundtruth distance', marker='.')

        # Compute y bounds for the plot by scaling the minimum and maximum distances
        if groundtruth_distance is not None:
            distances = np.concatenate([filtered_distance, tof_distance, signal_distance, groundtruth_distance])
        else:
            distances = np.concatenate([filtered_distance, tof_distance, signal_distance])
        distances = distances[~np.isnan(distances)]
        min_y = distances.min()
        max_y = distances.max()
        mid_y = (max_y + min_y) / 2
        range_y = 1.2 * (max_y - min_y)
        min_y = mid_y - range_y / 2
        max_y = mid_y + range_y / 2
        ax1.text(0.99, 0.01, 'Q={}\nR={}\nm={}\nb={}\nl={}'.format(self._Q_k, self._R_k, self._m, self._b, self._l), horizontalalignment='right', transform = ax1.transAxes)
        ax1.set_ybound(min_y, max_y)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance')
        ax1.legend()

        if groundtruth_distance is not None:
            filtered_error = filtered_distance[self._times.astype(int)] - groundtruth_distance
            total_filtered_error = np.sqrt(np.sum(np.square(filtered_error[~np.isnan(filtered_error)])))
            tof_error = tof_distance - groundtruth_distance
            total_tof_error = np.sqrt(np.sum(np.square(tof_error[~np.isnan(tof_error)])))
            signal_error = signal_distance - groundtruth_distance
            total_signal_error = np.sqrt(np.sum(np.square(signal_error[~np.isnan(signal_error)])))
            smoothed_signal_error = smoothed_signal_distance - groundtruth_distance
            total_smoothed_signal_error = np.sqrt(np.sum(np.square(smoothed_signal_error[~np.isnan(smoothed_signal_error)])))
            ax2.plot(self._times, filtered_error, marker='.', label='Filtered error: {:.6f}'.format(total_filtered_error))
            ax2.plot(self._times, tof_error, marker='.', label='TOF error: {:.6f}'.format(total_tof_error))
            ax2.plot(self._times, signal_error, marker='.', label='Signal error: {:.6f}'.format(total_signal_error))
            # ax2.plot(self._times, smoothed_signal_error, marker='.', label='Smoothed signal error: {:.6f}'.format(total_smoothed_signal_error))
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Distance error (m)')
            ax2.set_title('Error')
            ax2.legend()

        # Add a title
        fig.suptitle(os.path.split(self._data_path)[1])

        # Show plot
        if show:
            plt.show()

        # Save figure
        if save:
            savepath = utils.get_savepath(self._data_path, '_kalman_filter', replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        
        plt.close()

        self._filtered_distance = filtered_distance
        self._filtered_error = filtered_error
        self._total_filtered_error = total_filtered_error
        self._tof_distance = tof_distance
        self._tof_error = tof_error
        self._total_tof_error = total_tof_error
        self._signal_distance = signal_distance
        self._signal_error = signal_error
        self._total_signal_error = total_signal_error


if __name__ == '__main__':
    save_to = '../plots/06-08-2022'
    replace = False

    # kf = KalmanFilter('../data/05-26-2022/tag78-0m-air-test-0.csv')
    # kf = KalmanFilter('../data/06-01-2022/tag78_50m_increment_long_beach_test_457012_2.csv', prediction_rate=1)
    kf = KalmanFilter('../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv', prediction_rate=1)
    # kf = KalmanFilter('../data/06-08-2022/tag78_cowling_none_long_beach_test_457012_0.csv', prediction_rate=1)
    # kf = KalmanFilter('../data/06-08-2022/tag78_cowling_none_long_beach_test_457049_0.csv', prediction_rate=1)
    # kf = KalmanFilter('../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv', prediction_rate=1)
    
    kf.run()
    kf.plot(save=True, replace=replace)

    # paths = [
    #     '../data/06-08-2022/tag78_50m_increment_long_beach_test_457012_0.csv',
    #     '../data/06-08-2022/tag78_50m_increment_long_beach_test_457049_0.csv',
    #     '../data/06-08-2022/tag78_cowling_none_long_beach_test_457012_0.csv',
    #     '../data/06-08-2022/tag78_cowling_none_long_beach_test_457049_0.csv'
    # ]
    # errors = []
    # data_idx = 0  # 0 means total error. 3 means average error. 4 means error standard deviation.
    # # xrange = range(-8, 3)
    # # yrange = range(-8, 3)
    # # xrange = np.linspace(1e-3, 1e-1, 11)
    # # yrange = np.linspace(1e-3, 1e-1, 11)
    # xrange = np.linspace(0.0001, 0.001, 11)
    # yrange = np.linspace(0.0001, 0.001, 11)
    # for path in paths:
    #     kf = KalmanFilter(path, prediction_rate=1)
    #     errors.append([])
    #     for i in yrange:
    #         errors[-1].append([])
    #         for j in xrange:
    #             kf._distance_var = i
    #             kf._velocity_var = j
    #             kf.reset()
    #             kf.run()
    #             kf.plot(show=False, save=False)
    #             isnan = np.isnan(kf._filtered_error)
    #             errors[-1][-1].append([
    #                 kf._total_filtered_error,
    #                 np.max(kf._filtered_error[~isnan]),
    #                 np.min(kf._filtered_error[~isnan]),
    #                 np.mean(kf._filtered_error[~isnan]),
    #                 np.std(kf._filtered_error[~isnan])
    #             ])
    #             print('Distance var={}, velocity var={}, total error={}, error mean={}, error stdev={}'.format(
    #                 kf._distance_var, kf._velocity_var, errors[-1][-1][-1][0], errors[-1][-1][-1][3], errors[-1][-1][-1][4]
    #             ))
    #     errors[-1] = np.array(errors[-1])
    #     im = np.flip(np.abs(errors[-1][:, :, data_idx]), axis=0)
    #     ax = sns.heatmap(np.log(im), annot=im, linewidth=0.5, xticklabels=xrange, yticklabels=list(reversed(yrange)))
    #     ax.set_xlabel('Velocity variance')
    #     ax.set_ylabel('Distance variance')
    #     ax.set_title('{} KF total error '.format(os.path.split(os.path.splitext(path)[0])[1]))
    #     fig = plt.gcf()
    #     plt.show()

    #     savepath = utils.get_savepath(path, '_kf_param_heatmap', replace=replace)
    #     print('Saving to {}'.format(savepath))
    #     utils.savefig(fig, savepath)
    # errors = np.array(errors)
    # # sums = np.apply_along_axis(np.sum, 1, np.abs(errors[:, :, :, data_idx]).reshape(len(paths), -1)).reshape(-1, 1, 1)
    # sums = utils.apply_along_axes(np.sum, (1, 3), np.abs(errors[:, :, :, data_idx])).reshape(-1, 1, 1)
    # im = np.flip(np.abs(np.sum(errors[:, :, :, data_idx] / (4 * sums), axis=0)), axis=0)
    # ax = sns.heatmap(np.log(im), annot=im, linewidth=0.5, xticklabels=xrange, yticklabels=list(reversed(yrange)))
    # ax.set_xlabel('Velocity variance')
    # ax.set_ylabel('Distance variance')
    # ax.set_title('KF total error (all)')
    # fig = plt.gcf()
    # plt.show()

    # savepath = utils.add_version(os.path.join(save_to, 'kf_param_heatmap_all.png'), replace=replace)
    # print('Saving to {}'.format(savepath))
    # utils.savefig(fig, savepath)