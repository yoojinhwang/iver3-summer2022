from abc import ABC, abstractmethod
import numpy as np
import utils
from filter import Filter
from scipy.stats import multivariate_normal
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from dataset import Dataset
from datetime import timedelta
from functools import partial
from pf import MotionModelBase

class MotionModel(MotionModelBase):
    def __init__(self, **kwargs):
        # Particle: r, r_dot, weight
        kwargs['name'] = 'velocity'
        super().__init__(3, **kwargs)
    
    def initialize_particles(self, num_particles, **kwargs):
        return super().initialize_particles(num_particles, **kwargs)
    
    def prediction_step(self, particles, dt):
        r, r_dot, weight = particles.T
        
        # Add some gaussian noise to the velocity
        r_dot_f = r_dot + np.random.normal(0, 1, len(particles))
        r_f = r + r_dot_f * dt

        # Pack up the values and return them
        return np.column_stack([r_f, r_dot_f, weight])

class ParticleFilter(Filter):
    def from_dataset(dataset, serial_no, num_particles, motion_model_type, motion_model_params={}, save_history=False, ff=1, **kwargs):
        save_history = True
        start_time = dataset.start_time
        end_time = dataset.end_time
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))
        
        # Get relevant values from the dataframe
        data = dataset.hydrophones[serial_no].detections
        times = data.index
        delta_tof = np.array(data['delta_tof'])
        signal_level = utils.iir_filter(np.array(data['signal_level']), ff=ff)

        # Create particle filter
        pf = ParticleFilter(num_particles, motion_model_type, motion_model_params, save_history, **kwargs)

        # Queue predictions
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            pf.queue_prediction(timestamp, np.array([]))
        
        # Queue measurements
        for timestamp, delta_tof, signal_level in zip(times, delta_tof, signal_level):
            pf.queue_correction(timestamp, np.array([delta_tof, signal_level]))
        return pf

    def __init__(self, num_particles, motion_model_type, motion_model_params={}, save_history=False, **kwargs):
        self._save_history = save_history
        self._num_particles = num_particles
        self._motion_model_params = motion_model_params
        self._motion_model = motion_model_type(**motion_model_params)
        self._last_step_type = None
        super().__init__()
    
        self._speed_of_sound = utils.SPEED_OF_SOUND

        self._m = kwargs.get('m', -0.064755099)  # conversion between distance (m) and signal intensity (dB)
        self._l = kwargs.get('l', -1.36582584)  # conversion between speed (m/s) and signal intensity (dB)
        self._b = kwargs.get('b', 77.7946280)  # intercept for conversion between distance and signal intensity

        self._delta_tof_var = kwargs.get('delta_tof_var', 0.0005444242032405411**2)  # variance in the tag's time of flight when stationary (s)
        self._signal_var = kwargs.get('signal_var', 25)  # variance in the signal intensity not explained by distance

        def on_prediction(*_):
            if self._last_step_type == Filter.CORRECTION:
                self._resample_step()
            self._last_step_type = Filter.PREDICTION
        
        def on_correction(*_):
            self._last_step_type = Filter.CORRECTION
        
        self.on_prediction(on_prediction)
        self.on_correction(on_correction)
    
    def reset(self):
        super().reset()
        self._filters = {}
        self._particles = self._motion_model.initialize_particles(self._num_particles, **self._motion_model_params)
        self._history = np.zeros((0,) + self._particles.shape)
        self._step_history = np.zeros((0,), dtype=np.int32)
        self._time_history = np.zeros((0,), dtype=object)
        self._measurement_history = np.zeros((0, 2))
    
    def _update_history(self, timestamp, step_type, measurement=None):
        if self._save_history:
            self._history = np.concatenate([self._history, [self._particles]])
            self._step_history = np.concatenate([self._step_history, [step_type]])
            self._time_history = np.concatenate([self._time_history, [timestamp]])
            if step_type == Filter.CORRECTION:
                self._measurement_history = np.concatenate([self._measurement_history, [measurement]])
    
    def _prediction_step(self, timestamp, data, dt):
        super()._prediction_step(timestamp, data, dt)

        # Update particles
        self._particles = self._motion_model.prediction_step(self._particles, dt)
        self._update_history(timestamp, Filter.PREDICTION)
    
    def _correction_step(self, timestamp, data, dt):
        super()._correction_step(timestamp, data, dt)

        # Unpack the measurement data
        delta_tof, signal_level = data

        # Unpack the particles
        r, r_dot, _ = self._particles.T

        # Compute the predicted measurement given the state
        delta_tof_pred = (1 / self._speed_of_sound) * dt * r_dot
        signal_level_pred = self._m * r + self._l * r_dot + self._b

        # Compute weights
        x = np.column_stack([delta_tof_pred, signal_level_pred])
        cov = np.array([[self._delta_tof_var, 0], [0, self._signal_var]])  # Covariance of noise in the measurement
        try:
            dist = multivariate_normal(data, cov)
        except scipy.linalg.LinAlgError as err:
            dist = multivariate_normal(data, cov, allow_singular=True)
        weight = dist.pdf(x)
        self._particles[:, -1] *= weight

        print('Correction step', timestamp)
        self._update_history(timestamp, Filter.CORRECTION, data)
    
    def _resample_step(self):
        # Resample if possible
        weight = self._particles[:, -1]
        if np.sum(np.isnan(weight)) == 0:
            min_weight = 1e-60
            weight = np.maximum(weight, min_weight)
            w_tot = np.sum(weight)  # sum up the weights of all particles
            probs = weight / w_tot  # normalize weights
            indices = np.random.choice(len(self._particles), len(self._particles), p=probs)
            self._particles = self._particles[indices]
        else:
            print('Skipped measurement(s) due to nan weights')
        self._particles[:, -1] = 1

    def plot(self, dataset=None, serial_no=None, show=True, save=False, replace=True):
        if dataset is None:
            fig, ax1 = plt.subplots()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        
        _history = np.average(self._history, axis=1)
        _cov_history = utils.apply_along_axes(partial(np.cov, rowvar=False), (1, 3), self._history[:, :, :-1])

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

        # Plot pf distances and error bars for it
        pf_distances = _history[:, 0]
        ax1.plot(times, pf_distances, label='pf distance')
        ax1.scatter(times, pf_distances, marker='.')
        stdev = np.sqrt(_cov_history[:, 0, 0])
        ax1.fill_between(self._time_history, pf_distances-stdev, pf_distances+stdev, facecolor='#1f77b4', alpha=0.5)
        ax1.fill_between(self._time_history, pf_distances-2*stdev, pf_distances+2*stdev, facecolor='#1f77b4', alpha=0.3)

        if dataset is None:
            tof_distances = np.cumsum(self._measurement_history[:, 0]) * self._speed_of_sound
        else:
            tof_distances = data.detections['total_distance']
        ax1.plot(correction_times, tof_distances, label='tof distance')
        ax1.scatter(correction_times, tof_distances, marker='.')

        # Plot signal distances
        signal_levels = np.array(Dataset.get(pd.Series(self._measurement_history[:, 1], correction_times), times, mode='interpolate')[1])
        if dataset is None:
            speeds = _history[:, 1]
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
        ax1.text(0.99, 0.01, 'm={}\nb={}\nl={}'.format(self._m, self._b, self._l), horizontalalignment='right', transform = ax1.transAxes)

        # Set bounds on the y axis
        if dataset is not None:
            distances = np.concatenate([pf_distances, tof_distances, signal_distances, gps_distances])
        else:
            distances = np.concatenate([pf_distances, tof_distances, signal_distances])
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
            filtered_pf_distances = pf_distances[is_correction]
            filtered_signal_distances = signal_distances[is_correction]
            filtered_gps_distances = gps_distances[is_correction]
            n = len(correction_times)

            # Calculate RMS for each distance measure
            pf_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances - filtered_pf_distances)) / n)
            tof_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances - tof_distances)) / n)
            signal_distance_rms = np.sqrt(np.sum(np.square(filtered_gps_distances[1:] - filtered_signal_distances[1:])) / (n-1))

            # Plot error and RMS
            ax2.plot(times, gps_distances - pf_distances, label='pf error, RMS={:.2f}'.format(pf_distance_rms))
            ax2.scatter(correction_times, filtered_gps_distances - filtered_pf_distances, marker='.')
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
            savepath = utils.add_version('../datasets/{n}/pf_plots/{s}/{s}_pf_plot.png'.format(n=dataset.name, s=serial_no), replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        plt.close()

if __name__ == '__main__':
    dataset = Dataset('tag78_shore_2_boat_all_static_test_0', shift_tof=True)
    # dataset = Dataset('tag78_shore_2_boat_all_static_test_1', shift_tof=True)
    # dataset = Dataset('tag78_50m_increment_long_beach_test_0', reset_tof=True, shift_tof=True)
    pf = ParticleFilter.from_dataset(dataset, 457049, 1000, MotionModel, motion_model_params={
        'init_uniform_random': [(0, 600), (-5, 5)]
    }, save_history=True, **{
        'm': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'delta_tof_var': 0.0005, 'signal_var': 100, 'ff': 1, 'initial_r': 0
    })
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(0, 200), (-150, 50)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))