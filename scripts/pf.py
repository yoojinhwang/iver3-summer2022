from abc import ABC, abstractmethod
from filter import Filter
from kalman import KalmanFilter
import numpy as np
# import pandas as pd
from datetime import timedelta
import utils
import scipy
from scipy.stats import multivariate_normal
# from plotting import plot_df
# import matplotlib.pyplot as plt

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
    def initialize_particles(self, num_particles, **kwargs):
        uniform_random = kwargs.get('uniform_random', None)
        particles = np.zeros((num_particles, self._size))

        # Initialize weights to 1
        particles[:, -1] = 1

        # Initialize particle states according to a uniform random distribution
        if uniform_random is not None:
            for i, (low, high) in enumerate(uniform_random):
                particles[:, i] = np.random.uniform(low, high, num_particles)
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
    
    def initialize_particles(self, num_particles, **kwargs):
        return super().initialize_particles(num_particles, **kwargs)
    
    def prediction_step(self, particles, dt):
        x, y, theta, v, omega, weight = particles.T

        x_f = x + np.random.normal(self._x_mean, self._x_stdev, len(particles))
        y_f = x + np.random.normal(self._y_mean, self._y_stdev, len(particles))
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

class ParticleFilter(Filter):
    def from_dataset(df, tag_id, num_particles, motion_model_type, motion_model_params={}, save_history=False, hydrophone_params={}):
        # Remove rows that are not about the tag we are localizing
        df = df[df['tag_id'] == tag_id]

        # Assumes the data is sorted by datetime, which should be a valid assumption if it comes from merge_dataset
        start_time = df.iloc[0]['datetime']
        end_time = df.iloc[-1]['datetime']
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))

        pf = ParticleFilter(num_particles, motion_model_type, motion_model_params, save_history, hydrophone_params)

        # Queue predictions
        print(num_predictions)
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            pf._queue_prediction(timestamp, np.array([]))
        
        # Queue measurements
        for _, row in df.iterrows():
            timestamp = row['datetime']
            data = (
                row['serial_no'],
                np.array([
                    row['delta_tof'],
                    row['signal_level'],
                    row['x'],
                    row['y'],
                    row['gps_theta'],
                    row['gps_vel']]))

    def __init__(self, num_particles, motion_model_type, motion_model_params={}, save_history=False, hydrophone_params={}):
        self._save_history = save_history
        self._num_particles = num_particles
        self._motion_model_params = motion_model_params
        self._motion_model = motion_model_type(**motion_model_params)
        self._hydrophone_params = hydrophone_params
        super().__init__()
    
    def reset(self):
        super().reset()
        self._filters = {}
        self._particles = self._motion_model.initialize_particles(self._num_particles, **self._motion_model_params)
        self._history = np.zeros((0,) + self._particles.shape)
        self._step_history = np.zeros((0,), dtype=np.int32)
        self._time_history = np.zeros((0,), dtype=object)
        self._measurement_history = np.zeros((0, 2))
        self._hydrophone_state_history = np.zeros((0, 5), dtype=object)

    def _update_history(self, timestamp, step_type, measurement=None, hydrophone_state=None):
        if self._save_history:
            self._history = np.concatenate([self._history, [self._particles]])
            self._step_history = np.concatenate([self._step_history, [step_type]])
            self._time_history = np.concatenate([self._time_history, [timestamp]])
            if step_type == Filter.CORRECTION:
                self._measurement_history = np.concatenate([self._measurement_history, [measurement]])
                self._hydrophone_state_history = np.concatenate([self._hydrophone_state_history, [hydrophone_state]])

    def _prediction_step(self, timestamp, data, dt):
        super()._prediction_step(timestamp, data, dt)

        # Update Kalman filters
        for _, kf in self._filters.items():
            kf.queue_prediction(timestamp, data)
            kf.iterate()
        
        # Update particles
        self._particles = self._motion_model.prediction_step(self._particles, dt)
        self._update_history(timestamp, Filter.PREDICTION)
    
    def _correction_step(self, timestamp, data, dt):
        super()._correction_step(timestamp, data, dt)

        # Delta tof and signal level comprise the measurement for the kalman filter
        # The hydrophone state combined with the output of the kalman filter comprise the measurement for the particle filter
        serial_no, (delta_tof, signal_level, *hydrophone_state) = data

        # Update the appropriate kalman filter
        if serial_no in self._filters:
            kf = self._filters[serial_no]
        else:
            kf = self._filters[serial_no] = KalmanFilter(
                save_history=self._save_history,
                **self._hydrophone_params.get(serial_no, {}))
        kf.queue_correction(timestamp, np.array([delta_tof, signal_level]))
        kf.iterate()

        # Unpack the measurement data from the kalman filter and the cached hydrophone state data
        (r, r_dot) = kf.get_state()
        measurement = np.array([r, r_dot])
        measurement_cov = kf.get_state_cov()
        x, y, theta, v = hydrophone_state
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        # Unpack the particles
        tag_x, tag_y, tag_theta, tag_v, tag_omega, old_weight = self._particles.T
        tag_vx = tag_v * np.cos(tag_theta)
        tag_vy = tag_v * np.sin(tag_theta)

        # Compute the predicted measurement given the state
        x_diff = x - tag_x
        y_diff = y - tag_y
        r_pred = np.sqrt(np.square(x_diff) + np.square(y_diff))
        r_dot_pred = (x_diff * (vx - tag_vx) + y_diff * (vy - tag_vy)) / r_pred

        # Compute weights
        x = np.column_stack([r_pred, r_dot_pred])
        try:
            dist = multivariate_normal(measurement, measurement_cov)
        except scipy.linalg.LinAlgError as err:
            dist = multivariate_normal(measurement, measurement_cov, allow_singular=True)
        weight = dist.pdf(x)
        self._particles[:, -1] *= weight
        self._resample_step()
        self._update_history(timestamp, Filter.CORRECTION, measurement, np.array([serial_no, *hydrophone_state], dtype=object))
    
    def _resample_step(self):
        # Resample if possible
        weight = self._particles[:, -1]
        self._particles[:, -1] = 1
        if np.sum(np.isnan(weight)) == 0:
            min_weight = 1e-60
            weight = np.maximum(weight, min_weight)
            w_tot = np.sum(weight)  # sum up the weights of all particles
            probs = weight / w_tot  # normalize weights
            indices = np.random.choice(len(self._particles), len(self._particles), p=probs)
            self._particles = self._particles[indices]
            print('Resampled particles')
        else:
            print('Skipped measurement due to nan weights')

if __name__ == '__main__':
    pass