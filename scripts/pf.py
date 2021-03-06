from abc import ABC, abstractmethod
from filter import Filter
from kalman import KalmanFilter
import numpy as np
import pandas as pd
from datetime import timedelta
import utils
import scipy
from scipy.stats import multivariate_normal
from plotting import plot_df
import matplotlib.pyplot as plt

class MotionModelBase(ABC):
    '''
    Provides motion model functions
    '''
    def __init__(self, size, name, **kwargs):
        self._size = size
        self.name = name
        self.info = {'kwargs': kwargs}
    
    def particle_size(self):
        return self._size
    
    @abstractmethod
    def initialize_particles(self, num_particles, groundtruth=None):
        particles = np.zeros((num_particles, self._size))
        if groundtruth is not None:
            particles[:, 0] = groundtruth[0]
            particles[:, 1] = groundtruth[1]
            particles[:, 2] = groundtruth[2]
        else:
            particles[:, 0] = np.random.uniform(-150, 150, num_particles)
            particles[:, 1] = np.random.uniform(-150, 150, num_particles)
        return particles
    
    @abstractmethod
    def prediction_step(self, particles, dt):
        '''
        Given the list of particles from the previous time step, update them to the next time step
        '''
        pass

class RandomMotionModel(MotionModelBase):
    def __init__(self, **kwargs):
        # Particle: x, y, theta, linear velocity, angular velocity, weight
        kwargs['name'] = 'random'
        super().__init__(6, **kwargs)
        self._x_mean = kwargs.get('x_mean', 0)
        self._x_stdev = kwargs.get('x_stdev', 1e-1)
        self._y_mean = kwargs.get('y_mean', 0)
        self._y_stdev = kwargs.get('y_stdev', 1e-1)
    
    def initialize_particles(self, num_particles, groundtruth=None):
        return super().initialize_particles(num_particles, groundtruth=groundtruth)
    
    def prediction_step(self, particles, dt):
        x, y, theta, v, omega, weight = particles.T

        x_f = x + np.random.normal(self._x_mean, self._x_stdev, len(particles))
        y_f = y + np.random.normal(self._y_mean, self._y_stdev, len(particles))
        x_diff, y_diff = x_f - x, y_f - y
        theta_f = utils.angle_between([0, 1], np.column_stack([x_diff, y_diff]))
        if dt == 0:
            v_f = np.zeros(len(particles))
            omega_f = np.zeros(len(particles))
        else:
            v_f = np.sqrt(np.square(x_diff) + np.square(y_diff)) / dt
            omega_f = (theta_f - theta) / dt

        # Pack up the values and return them
        return np.column_stack([x_f, y_f, theta_f, v_f, omega_f, weight])

class VelocityMotionModel(MotionModelBase):
    def __init__(self, **kwargs):
        # Particle: x, y, theta, linear velocity, angular velocity, weight
        kwargs['name'] = 'velocity'
        super().__init__(6, **kwargs)
        self._velocity_mean = kwargs.get('velocity_mean', 0)
        self._velocity_stdev = kwargs.get('velocity_stdev', 1)
        self._omega_mean = kwargs.get('omega_mean', 0)
        self._omega_stdev = kwargs.get('omega_stdev', 1)
    
    def initialize_particles(self, num_particles, groundtruth=None):
        return super().initialize_particles(num_particles, groundtruth=groundtruth)
    
    def prediction_step(self, particles, dt):
        x, y, theta, v, omega, weight = particles.T

        # Sample velocity and omega
        v = np.abs(np.random.normal(self._velocity_mean, self._velocity_stdev, len(particles)))
        # omega = np.random.normal(self._omega_mean, self._omega_stdev, len(particles))
        omega = np.random.uniform(-np.pi, np.pi, len(particles))

        # Bound omega
        omega = np.where(np.abs(omega) < 1e-10, np.sign(omega) * 1e-10, omega)

        # Compute the new position and heading
        x_c = x - v/omega * np.sin(theta)
        y_c = y + v/omega * np.cos(theta)
        x_f = x_c + v/omega * np.sin(theta + omega * dt)
        y_f = y_c - v/omega * np.cos(theta + omega * dt)
        theta_f = utils.wrap_to_pi(theta + omega * dt)

        # Pack up the values and return them
        return np.column_stack([x_f, y_f, theta_f, v, omega, weight])

class ParticleFilter(Filter):
    def from_csvs(data1, data2, num_particles, motion_model_type, motion_model_params={}, save_history=False):
        start_time = min(data1['datetime'][0], data2['datetime'][0])
        end_time = max(data1['datetime'][len(data1)-1], data2['datetime'][len(data2)-1])
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))

        pf = ParticleFilter(num_particles, motion_model_type, motion_model_params, save_history)

        # Queue predictions
        print(num_predictions)
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            pf.queue_prediction(timestamp, [])
        
        # Queue measurements
        it = utils.imerge(
            zip(data1['total_dt'], data1['delta_tof'], data1['signal_level'], data1['x'], data1['y'], data1['gps_heading'], data1['gps_speed']),
            zip(data2['total_dt'], data2['delta_tof'], data2['signal_level'], data2['x'], data2['y'], data2['gps_heading'], data2['gps_speed']),
            enum=True)
        for hydrophone, (total_dt, *measurement) in it:
            timestamp = start_time + timedelta(seconds=total_dt)
            pf.queue_correction(timestamp, np.array([hydrophone+1] + measurement))

        return pf

    def __init__(self, num_particles, motion_model_type, motion_model_params={}, save_history=False):
        self._save_history = save_history
        self._num_particles = num_particles
        self._motion_model = motion_model_type(**motion_model_params)
        self._kf1 = KalmanFilter(save_history=save_history)
        self._kf2 = KalmanFilter(save_history=save_history)
        super().__init__()
    
    def reset(self):
        super().reset()
        self._kf1.reset()
        self._kf2.reset()
        self._particles = self._motion_model.initialize_particles(self._num_particles)
        self._history = np.zeros((0,) + self._particles.shape)
        self._step_history = np.zeros((0,), dtype=np.int32)
        self._time_history = np.zeros((0,), dtype=object)
        self._measurement_history = np.zeros((0, 4))
        self._hydrophone_state_history = np.zeros((0, 8))
        self._hydrophone1_state = np.full(4, np.nan)
        self._hydrophone2_state = np.full(4, np.nan)
        self._kf1_history = np.zeros((0, 2))
        self._kf2_history = np.zeros((0, 2))
        self._kf1_cov_history = np.zeros((0, 2, 2))
        self._kf2_cov_history = np.zeros((0, 2, 2))

    def _update_history(self, timestamp, step_type, measurement=None):
        if self._save_history:
            self._history = np.concatenate([self._history, [self._particles]])
            self._step_history = np.concatenate([self._step_history, [step_type]])
            self._time_history = np.concatenate([self._time_history, [timestamp]])
            hydrophone_state = np.concatenate([self._hydrophone1_state, self._hydrophone2_state])
            self._hydrophone_state_history = np.concatenate([self._hydrophone_state_history, [hydrophone_state]])
            self._kf1_history = np.concatenate([self._kf1_history, [self._kf1.get_state()]])
            self._kf2_history = np.concatenate([self._kf2_history, [self._kf2.get_state()]])
            self._kf1_cov_history = np.concatenate([self._kf1_cov_history, [self._kf1.get_state_cov()]])
            self._kf2_cov_history = np.concatenate([self._kf2_cov_history, [self._kf2.get_state_cov()]])
            if step_type == Filter.CORRECTION:
                self._measurement_history = np.concatenate([self._measurement_history, [measurement]])

    def _prediction_step(self, timestamp, data, dt):
        super()._prediction_step(timestamp, data, dt)

        # Update Kalman filters
        self._kf1.queue_prediction(timestamp, data)
        self._kf2.queue_prediction(timestamp, data)
        self._kf1.iterate()
        self._kf2.iterate()

        # Update particles
        self._particles = self._motion_model.prediction_step(self._particles, dt)
        self._update_history(timestamp, Filter.PREDICTION)

    def _correction_step(self, timestamp, data, dt):
        super()._correction_step(timestamp, data, dt)

        # Update the appropriate kalman filter
        hydrophone, delta_tof, signal_level, *hydrophone_state = data
        if hydrophone == 1:
            self._kf1.queue_correction(timestamp, np.array([delta_tof, signal_level]))
            self._kf1.iterate()
            self._hydrophone1_state = hydrophone_state
        else:
            self._kf2.queue_correction(timestamp, np.array([delta_tof, signal_level]))
            self._kf2.iterate()
            self._hydrophone2_state = hydrophone_state

        # Unpack the measurement data from the kalman filters and the cached hydrophone state data
        (r1, r_dot1), (r2, r_dot2) = self._kf1.get_state(), self._kf2.get_state()
        measurement = np.array([r1, r_dot1, r2, r_dot2])
        cov1, cov2 = self._kf1.get_state_cov(), self._kf2.get_state_cov()
        (x1, y1, theta1, v1), (x2, y2, theta2, v2) = self._hydrophone1_state, self._hydrophone2_state
        vx1 = v1 * np.cos(theta1)
        vy1 = v1 * np.sin(theta1)
        vx2 = v2 * np.cos(theta2)
        vy2 = v2 * np.sin(theta2)

        # Unpack the particles
        tag_x, tag_y, tag_theta, tag_v, tag_omega, old_weight = self._particles.T
        tag_vx = tag_v * np.cos(tag_theta)
        tag_vy = tag_v * np.sin(tag_theta)

        # Compute predicted measurement given the state
        x1_diff = x1 - tag_x
        y1_diff = y1 - tag_y
        r1_pred = np.sqrt(np.square(x1_diff) + np.square(y1_diff))
        r_dot1_pred = (x1_diff * (vx1 - tag_vx) + y1_diff * (vy1 - tag_vy)) / r1_pred

        x2_diff = x2 - tag_x
        y2_diff = y2 - tag_y
        r2_pred = np.sqrt(np.square(x2_diff) + np.square(y2_diff))
        r_dot2_pred = (x2_diff * (vx2 - tag_vx) + y2_diff * (vy2 - tag_vy)) / r2_pred

        # Compute weights
        measurement_cov = np.block([
            [cov1, np.zeros([2, 2])],
            [np.zeros([2, 2]), cov2]
        ])
        x = np.column_stack([r1_pred, r_dot1_pred, r2_pred, r_dot2_pred])
        try:
            dist = multivariate_normal(measurement, measurement_cov)
        except scipy.linalg.LinAlgError as err:
            print(err)
            print(measurement_cov)
            dist = multivariate_normal(measurement, measurement_cov, allow_singular=True)
        weight = dist.pdf(x)
        
        if np.sum(np.isnan(weight)) == 0:
            self._particles[:, -1] = weight
            self._resample_step()
        else:
            print('Skipped measurement due to nan weights')
        self._update_history(timestamp, Filter.CORRECTION, measurement)

    def _resample_step(self):
        min_weight = 1e-60
        self._particles[:, -1] = np.maximum(self._particles[:, -1], min_weight)
        w_tot = np.sum(self._particles[:, -1])  # sum up all the particles
        probs = self._particles[:, -1] = self._particles[:, -1] / w_tot
        indices = np.random.choice(len(self._particles), len(self._particles), p=probs)
        self._particles = self._particles[indices]

if __name__ == '__main__':
    replace = True

    datapath1 = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_VR100_0.csv'
    datapath2 = '../data/07-19-2022/tag78_shore_2_boat_all_static_test_457049_0.csv'
    data1 = pd.read_csv(datapath1)
    data2 = pd.read_csv(datapath2)
    data1 = data1[data1['tag_id'] == 65478].reset_index(drop=True)
    data2 = data2[data2['tag_id'] == 65478].reset_index(drop=True)
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['datetime'] = pd.to_datetime(data2['datetime'])
    pf = ParticleFilter.from_csvs(data1, data2, 1000, RandomMotionModel, save_history=True)

    # Set signal strength model parameters
    pf._kf1._m = -0.10527966
    pf._kf1._l = -0.55164737
    pf._kf1._b = 68.59493072
    pf._kf1._signal_var = 16.250003

    pf._kf2._m = -0.20985953
    pf._kf2._l = 5.5568182
    pf._kf2._b = 76.90064068
    pf._kf2._signal_var = 9.400336

    pf.run()

    # Extract all x and y values from the particle filter's history to create a bounding box for the plot
    x = np.concatenate([pf._history[:, :, 0].flatten(), pf._hydrophone_state_history[:, [0, 4]].flatten()])
    x = x[~np.isnan(x)]
    y = np.concatenate([pf._history[:, :, 1].flatten(), pf._hydrophone_state_history[:, [1, 5]].flatten()])
    y = y[~np.isnan(y)]
    bbox = (np.min(x), np.min(y), np.max(x), np.max(y))

    # Get groundtruth data
    groundtruth_path = np.zeros((len(pf._time_history), 6))
    groundtruth_state1 = np.column_stack([data1['gps_distance'], data1['gps_speed']])
    groundtruth_state2 = np.column_stack([data2['gps_distance'], data2['gps_speed']])
    pf._kf1.plot(groundtruth_state1, datapath1, save=True, replace=replace)
    pf._kf2.plot(groundtruth_state2, datapath2, save=True, replace=replace)

    # Create the plot
    savepath = utils.get_savepath(datapath1, '_particle_filter', extension='gif', replace=replace)
    plot_df(pf, groundtruth_path, bbox=bbox, square=True, save_to=savepath)