import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from abc import ABC, abstractmethod
from common import iir_filter, avg_dt_dict

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
        avg_dt = avg_dt_dict[65478]
        # avg_dt = 8.18
        self._delta_tof = np.array([0] + ((dt - avg_dt / 2) % avg_dt - avg_dt / 2).tolist())
        # self._delta_tof = np.array(self._data['delta_tof'])
        self._signal_level = iir_filter(np.array(self._data['signal_level']), ff=1)
        self._start_time = 0
        self._end_time = self._times[-1]

        self._speed_of_sound = 1460  # speed of sound in water (m)
        # self._speed_of_sound = 343  # speed of sound in air (m)
        self._m = -0.086828  # conversion between distance (m) and signal intensity (dB)
        self._b = 77.708010  # intercept for conversion between distance and signal intensity
        self._delta_tof_var = 0.0005444242032405411**2  # variance in the tag's time of flight when stationary (s)
        # self._delta_tof_var = 1e-17
        # self._signal_var = 5.513502243014629**2  # variance in the signal intensity not explained by distance
        self._signal_var = 1e3

        # Setup the rest of the variables that need to change when run is called
        self.reset()

    def reset(self):
        self._last_time = self._start_time
        self._state_mean = np.array([50])  # r (m)
        self._state_cov = np.array([[1]])  # covariance matrix for the state
        self._next_measurement = 0
        self._history = [self._state_mean]
    
    def prediction_step(self, control, dt):
        F_k = np.array([[1]])  # describes how x_k depends on x_{k-1} free from noise or control
        B_k = np.array([[self._speed_of_sound]])  # describes how x_k depends on controls
        Q_k = np.array([[1]])  # covariance of noise in state from external forces
        new_state_mean = F_k @ self._state_mean + B_k @ control
        new_state_cov = F_k @ self._state_cov @ F_k.T + Q_k
        return new_state_mean, new_state_cov
    
    def correction_step(self, measurement, dt):
        H_k = np.array([[self._m]])
        R_k = np.array([[self._signal_var]])
        K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + R_k)
        new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([self._b])))
        new_state_cov = self._state_cov - K @ H_k @ self._state_cov
        return new_state_mean, new_state_cov

    def run(self):
        for current_time, delta_tof, signal_level in zip(self._times, self._delta_tof, self._signal_level):
            dt = current_time - self._last_time
            self._state_mean, self._state_cov = self.prediction_step(np.array([delta_tof]), dt)
            self._state_mean, self._state_cov = self.correction_step(np.array([signal_level]), dt)
            self._history.append(self._state_mean)
            self._last_time = current_time
        self._history = np.array(self._history)

    # def reset(self):
    #     self._current_time = self._start_time
    #     self._last_prediction_time = self._start_time
    #     self._last_measurement_time = self._times[0]
    #     self._state_mean = np.array([0, 0])  # r (m), previous_r (m)
    #     self._state_cov = np.array([[1, 0], [0, 1]])  # covariance matrix for the state
    #     self._next_measurement = 0
    #     self._history = [self._state_mean]
    
    # def prediction_step(self, control, dt):
    #     F_k = np.array([[1, 0], [0, 1]])  # describes how x_k depends on x_{k-1} free from noise or control
    #     Q_k = np.array([[1, 0], [0, 1]])  # covariance of noise in state from external forces
    #     new_state_mean = F_k @ self._state_mean
    #     new_state_cov = F_k @ self._state_cov @ F_k.T + Q_k
    #     return new_state_mean, new_state_cov
    
    # def correction_step(self, measurement, dt):
    #     H_k = np.array([[1/self._speed_of_sound, -1/self._speed_of_sound], [self._m, 0]])
    #     R_k = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])
    #     K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + R_k)
    #     new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([0, self._b])))
    #     new_state_cov = self._state_cov - K @ H_k @ self._state_cov
    #     return new_state_mean, new_state_cov

    # def reset(self):
    #     self._current_time = self._start_time
    #     self._last_prediction_time = self._start_time
    #     self._last_measurement_time = self._times[0]
    #     # self._state_mean = np.array([self._m * self._signal_level[0] + self._b, 0])  # r (m), dr/dt (m/s)
    #     self._state_mean = np.array([0, 0])  # r (m), dr/dt (m/s)
    #     self._state_cov = np.array([[1, 0], [0, 1]])  # covariance matrix for the state
    #     self._next_measurement = 0
    #     self._history = [self._state_mean]
    
    # def prediction_step(self, control, dt):
    #     F_k = np.array([[1, dt], [0, 1]])  # describes how x_k depends on x_{k-1} free from noise or control
    #     Q_k = np.array([[1, 0], [0, 10]])  # covariance of noise in state from external forces
    #     new_state_mean = F_k @ self._state_mean
    #     new_state_cov = F_k @ self._state_cov @ F_k.T + Q_k
    #     return new_state_mean, new_state_cov
    
    # def correction_step(self, measurement, dt):
    #     H_k = np.array([[0, dt/self._speed_of_sound], [self._m, 0]])
    #     R_k = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])
    #     K = self._state_cov @ H_k.T @ np.linalg.inv(H_k @ self._state_cov @ H_k.T + R_k)
    #     new_state_mean = self._state_mean + K @ (measurement - (H_k @ self._state_mean + np.array([0, self._b])))
    #     new_state_cov = self._state_cov - K @ H_k @ self._state_cov
    #     return new_state_mean, new_state_cov
    
    # def run(self):
    #     while self._next_measurement < len(self._times):
    #         measurement_time = self._times[self._next_measurement]
    #         if self._current_time <= measurement_time:
    #             # Update state
    #             self._state_mean, self._state_cov = self.prediction_step(None, self._current_time - self._last_prediction_time)
    #             self._history.append(self._state_mean)

    #             # Update last prediction time and current time
    #             self._last_prediction_time = self._current_time
    #             self._current_time += self._prediction_period
    #         else:
    #             # Update state
    #             measurement = np.array([self._delta_tof[self._next_measurement], self._signal_level[self._next_measurement]])
    #             self._state_mean, self._state_cov = self.correction_step(measurement, measurement_time - self._last_measurement_time)
    #             self._history.append(self._state_mean)

    #             # Update last measurement time and what the next measurement is, but not the current time since we still want to run a prediction step
    #             self._last_measurement_time = measurement_time
    #             self._next_measurement += 1
    #     self._history = np.array(self._history)

    def plot(self):
        # fig, axs = plt.subplots(2, 2)
        fig, ax = plt.subplots()

        prediction_times = np.arange(len(self._history)) * self._prediction_period
        filtered_distance = (self._times, self._history[1:, 0])
        # filtered_velocity = (prediction_times, self._history[:, 1])
        # integrated_velocity = (prediction_times, self._prediction_period * np.cumsum(self._history[:, 1]))
        tof_distance = (self._times, self._data['absolute_distance'])
        # tof_distance = (self._times, np.cumsum((self._delta_tof * self._speed_of_sound)))
        signal_distance = (self._times, (self._signal_level - self._b) / self._m)
        groundtruth_distance = (self._times, self._data['gps_distance'])
        # signal_level = (self._times, self._signal_level)
        # delta_tof = (self._times, self._data['delta_tof'] * self._speed_of_sound / 8.179)

        # axs[0][0].plot(*filtered_distance, color='#1f77b4')
        # axs[0][0].plot(*tof_distance, color='#d62728')
        # axs[0][0].plot(*groundtruth_distance, color='#9467bd')
        # # axs[0][0].plot(*integrated_velocity, color='#2ca02c')

        # # axs[0][1].plot(*filtered_velocity, color='#ff7f0e')
        # # axs[0][1].plot(*delta_tof, color='#e377c2')

        # axs[1][0].plot(*signal_level, color='#8c564b')

        # axs[1][1].plot(*filtered_distance, color='#1f77b4', label='Filtered distance')
        # # axs[1][1].plot(*filtered_velocity, color='#ff7f0e', label='Filtered velocity')
        # # axs[1][1].plot(*integrated_velocity, color='#2ca02c', label='Integrated velocity')
        # axs[1][1].plot(*tof_distance, color='#d62728', label='TOF distance')
        # axs[1][1].plot(*groundtruth_distance, color='#9467bd', label='Groundtruth distance')
        # axs[1][1].plot(*signal_level, color='#8c564b', label='Signal level')
        # # axs[1][1].plot(*delta_tof, color='#e377c2', label='Delta TOF')
        # axs[1][1].legend()

        ax.plot(*filtered_distance, label='Filtered distance')
        ax.plot(*tof_distance, label='TOF distance')
        ax.plot(*signal_distance, label='Signal distance')
        ax.plot(*groundtruth_distance, label='Groundtruth distance')
        ax.legend()

        plt.show()

if __name__ == '__main__':
    # kf = KalmanFilter('../data/05-26-2022/tag78-0m-air-test-0.csv')
    kf = KalmanFilter('../data/06-01-2022/tag78_50m_increment_manual_long_beach_test_457049_0.csv', prediction_rate=1)
    kf.run()
    kf.plot()