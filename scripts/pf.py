from abc import ABC, abstractmethod
import numpy as np
import utils
from filter import Filter
from scipy.stats import multivariate_normal
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
from dataset import Dataset
from kalman import KalmanFilter
from datetime import timedelta
import bisect

class MotionModelBase(ABC):
    '''
    Provides motion model functions
    '''
    def __init__(self, size, name, **kwargs):
        self.size = size
        self.name = name
        self.info = {'kwargs': kwargs}

    @abstractmethod
    def initialize_particles(self, num_particles, **kwargs):
        init_uniform_random = kwargs.get('init_uniform_random', None)
        particles = np.zeros((num_particles, self.size))

        # Initialize weights to 1
        particles[:, -1] = 1

        # Initialize particle states according to a uniform random distribution
        if init_uniform_random is not None:
            for i, (low, high) in enumerate(init_uniform_random):
                particles[:, i] = np.random.uniform(low, high, num_particles)
        return particles
    
    @abstractmethod
    def prediction_step(self, particles, dt):
        '''
        Given the list of particles from the previous time step, update them to the next time step
        '''

class RandomMotionModel(MotionModelBase):
    def __init__(self, **kwargs):
        # Particle: x, y, theta, linear velocity, angular velocity, weight
        kwargs['name'] = 'random'
        super().__init__(6, **kwargs)
        self._x_mean = kwargs.get('x_mean', 0)
        self._x_stdev = kwargs.get('x_stdev', 1)
        self._y_mean = kwargs.get('y_mean', 0)
        self._y_stdev = kwargs.get('y_stdev', 1)
    
    def initialize_particles(self, num_particles, **kwargs):
        return super().initialize_particles(num_particles, **kwargs)
    
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
        self._gamma_mean = kwargs.get('gamma_mean', self._omega_mean)
        self._gamma_stdev = kwargs.get('gamma_stdev', self._omega_stdev)
    
    def initialize_particles(self, num_particles, **kwargs):
        return super().initialize_particles(num_particles, **kwargs)
    
    def prediction_step(self, particles, dt):
        # Don't do anything if no time has passed
        if dt == 0:
            return particles

        # Unpack particles
        x, y, theta, v, omega, weight = particles.T

        # Add some gaussian noise to the current velocity
        v += np.random.normal(self._velocity_mean, self._velocity_stdev, len(particles))

        # Add some gaussian noise to the current angular velocity
        omega += np.random.normal(self._omega_mean, self._omega_stdev, len(particles))
        omega = np.where(np.abs(omega) < 1e-10, np.sign(omega) * 1e-10, omega)  # Bound abs(omega) above 0

        # Add a random angle after the update
        gamma = np.random.normal(self._gamma_mean, self._gamma_stdev, len(particles))

        # Compute the new position and heading
        x_c = x - v/omega * np.sin(theta)
        y_c = y + v/omega * np.cos(theta)
        x_f = x_c + v/omega * np.sin(theta + omega * dt)
        y_f = y_c - v/omega * np.cos(theta + omega * dt)
        theta_f = utils.wrap_to_pi(theta + omega * dt + gamma * dt)

        # Compute the new velocity and angular velocity
        v_f = np.sqrt(np.square(x_f - x) + np.square(y_f - y)) / dt
        omega_f = (theta_f - theta) / dt

        # Pack up the values and return them
        return np.column_stack([x_f, y_f, theta_f, v_f, omega_f, weight])

class ParticleFilter(Filter):
    def from_dataset(dataset, num_particles, motion_model_type, motion_model_params={}, save_history=False, **kwargs):
        save_history = True
        start_time = dataset.start_time
        end_time = dataset.end_time
        num_predictions = int(np.floor((end_time - start_time).total_seconds()))
        use_tof = kwargs.get('use_tof', False)
        use_groundtruth = kwargs.get('use_groundtruth', False)

        if not use_tof and not use_groundtruth:
            # Create a kalman filter for each hydrophone
            hydrophone_params = kwargs.get('hydrophone_params', {})
            filters =   {serial_no :
                            KalmanFilter.from_dataset(
                                dataset, serial_no,
                                save_history=save_history,
                                **hydrophone_params.get(serial_no, {}))
                        for serial_no in dataset.hydrophones}
        
        # Create particle filter
        pf = ParticleFilter(num_particles, motion_model_type, motion_model_params, save_history)

        # Queue predictions
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            pf.queue_prediction(timestamp, np.array([]))
        
        # Queue measurements
        for serial_no in dataset.hydrophones:
            data = dataset.hydrophones[serial_no].detections
            times = data.index

            # Retrieve measurement covariances
            measurement_covs = np.array([kwargs.get('measurement_cov', [[10, 0], [0, 10]])] * len(times))
            if use_tof:
                # Use just time of flight ranges
                r = dataset.hydrophones[serial_no].detections['total_distance']
                # r_dot = np.zeros(len(r))
                r_dot = np.array(dataset.hydrophones[serial_no].detections['delta_distance'] / dataset.hydrophones[serial_no].detections['dt'])
                r_dot[0] = 0
                measurements = np.array([r, r_dot]).T
            elif use_groundtruth:
                # Use groundtruth ranges
                measurements = np.array([dataset.get_gps_distance(serial_no, times), dataset.get_gps_speed(serial_no, times)]).T
            else:
                # Use kalman filter to estimate ranges
                kf = filters[serial_no]
                kf.run()
                kf.plot(dataset, serial_no)
                is_correction = kf._step_history == Filter.CORRECTION
                measurements = kf._history[is_correction, :]
                if 'measurement_cov' not in kwargs:
                    measurement_covs = kf._cov_history[is_correction, :]

            # Build other parts of the measurements
            coords = utils.to_cartesian(dataset.get_hydrophone_coords(serial_no, times), dataset.origin)
            thetas = dataset.get_gps_theta(serial_no, times)
            speeds = dataset.get_gps_vel(serial_no, times)

            data = zip(times, measurements, measurement_covs, coords, thetas, speeds)
            for timestamp, (r, r_dot), measurement_cov, (x, y), theta, v in data:
                pf.queue_correction(timestamp, [serial_no, np.array([r, r_dot, x, y, theta, v]), measurement_cov])
        return pf

    def __init__(self, num_particles, motion_model_type, motion_model_params={}, save_history=False):
        self._save_history = save_history
        self._num_particles = num_particles
        self._motion_model_params = motion_model_params
        self._motion_model = motion_model_type(**motion_model_params)
        self._last_step_type = None
        super().__init__()
    
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
        self._hydrophone_history = np.zeros((0,), dtype=object)
        self._measurement_history = np.zeros((0, 6))
        self._measurement_cov_history = np.zeros((0, 2, 2))
    
    def _update_history(self, timestamp, step_type, serial_no=None, measurement=None, measurement_cov=None):
        if self._save_history:
            self._history = np.concatenate([self._history, [self._particles]])
            self._step_history = np.concatenate([self._step_history, [step_type]])
            self._time_history = np.concatenate([self._time_history, [timestamp]])
            if step_type == Filter.CORRECTION:
                self._hydrophone_history = np.concatenate([self._hydrophone_history, [serial_no]])
                self._measurement_history = np.concatenate([self._measurement_history, [measurement]])
                self._measurement_cov_history = np.concatenate([self._measurement_cov_history, [measurement_cov]])
    
    def _prediction_step(self, timestamp, data, dt):
        super()._prediction_step(timestamp, data, dt)

        # Update particles
        self._particles = self._motion_model.prediction_step(self._particles, dt)
        self._update_history(timestamp, Filter.PREDICTION)
    
    def _correction_step(self, timestamp, data, dt):
        super()._correction_step(timestamp, data, dt)

        # Unpack the measurement data
        serial_no, (r, r_dot, *hydrophone_state), measurement_cov = data
        measurement = np.array([r, r_dot])
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

        print('Correction step', timestamp)
        self._update_history(timestamp, Filter.CORRECTION, serial_no, np.array([r, r_dot, *hydrophone_state]), measurement_cov)
    
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

    def plot(self, dataset, show=True, save=False, replace=False, **kwargs):
        def ints():
            i = 0
            while True:
                yield i
                i += 1
        
        i = ints()
        n = 1 + 2 * len(dataset.hydrophones)

        # Get parameters
        padding = kwargs.get('padding', 1.1)
        no_titles = kwargs.get('exclude_titles', False)
        add_labels = kwargs.get('labels', False)
        width = kwargs.get('width', 3.5)
        ratio = kwargs.get('ratio', 5)
        seconds = kwargs.get('plot_total_seconds', False)

        # Set up figure and axes
        arr = mpl.figure.figaspect((1+ratio)/ratio)
        w, h = width * arr / arr[0]
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [ratio, 1]}, figsize=(w, h))

        avg_particle = np.average(self._history, axis=1)  # Find the average particle at each time step

        # Find a bounding box that always shows the average particle, the hydrophones, and the tag
        all_x = np.concatenate([
            avg_particle[:, 0].flatten(),
            dataset.tag.coords['x'].to_numpy(),
            self._measurement_history[:, 2].astype(np.float64)
        ])  # Concatenate all relevant x coordinates
        all_y = np.concatenate([
            avg_particle[:, 1].flatten(),
            dataset.tag.coords['y'].to_numpy(),
            self._measurement_history[:, 3].astype(np.float64)
        ])  # Concatenate all relevant y coordinates
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]
        (minx, miny), (maxx, maxy) = utils.pad_bounds(((np.min(all_x), np.min(all_y)), (np.max(all_x), np.max(all_y))), f=padding, square=True)
        bbox = (minx, miny, maxx, maxy)
        ax0.set_xlim((minx, maxx))
        ax0.set_ylim((miny, maxy))

        # Create background map
        origin = dataset.origin
        cartesian_bounds = np.array(bbox).reshape(2, 2).T
        cartesian_bounds = utils.pad_bounds(cartesian_bounds.T).T
        if origin is not None:
            coord_bounds = utils.to_coords(cartesian_bounds, origin)
            (south, west), (north, east) = coord_bounds
            img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir='../maps/OpenStreetMap/Mapnik')
            true_ext = utils.to_cartesian(np.flip(np.array(ext).reshape(2, 2), axis=0), origin).T.flatten()
        ax0.imshow(img, extent=true_ext)

        # Plot hydrophone paths
        for name, data in dataset.hydrophones.items():
            x, y = data.coords['x'], data.coords['y']
            dx, dy = np.concatenate([np.diff(x), [0]]), np.concatenate([np.diff(y), [0]])
            detection_x, detection_y = dataset.get_hydrophone_xy(name, data.detections.index).T
            l, = ax0.plot(x, y, marker='o', label='{} coords'.format(name))
            # ax0.plot(detection_x, detection_y, marker='.', linestyle='None', label='{} detections'.format(name))
            ax0.plot([], [], marker='o', linestyle='None', label='{} detections'.format(name))
            ax0.quiver(x, y, dx, dy, units='xy', angles='xy', scale_units='xy', scale=1, color=l.get_color())

            # Add times
            if seconds:
                coords_times = utils.total_seconds(data.coords.index, dataset.start_time)
                detections_times = utils.total_seconds(data.detections.index, dataset.start_time)
            else:
                coords_times = data.coords.index
                detections_times = data.detections.index
            ax1.scatter(coords_times, [n - next(i)]*len(data.coords))
            ax1.scatter(detections_times, [n - next(i)]*len(data.detections))

        # Plot groundtruth path
        x, y = np.array(dataset.tag.coords[['x', 'y']]).T
        dx, dy = np.concatenate([np.diff(x), [0]]), np.concatenate([np.diff(y), [0]])
        l, = ax0.plot(x, y, marker='o', label='Tag coords')
        ax0.quiver(x, y, dx, dy, units='xy', angles='xy', scale_units='xy', scale=1, color=l.get_color())
        if seconds:
            tag_times = utils.total_seconds(dataset.tag.coords.index, dataset.start_time)
        else:
            tag_times = dataset.tag.coords.index
        if type(dataset.tag.raw) is np.ndarray:
            ax1.plot(tag_times, [n - next(i)]*2, '-o', color=l.get_color())
        else:
            ax1.scatter(tag_times, [n - next(i)]*len(dataset.tag.coords))

        # Plot particle path
        x, y = avg_particle[:, [0, 1]].T
        dx, dy = np.concatenate([np.diff(x), [0]]), np.concatenate([np.diff(y), [0]])
        l, = ax0.plot(x, y, marker='.', label='Tag estimated coords', color='teal')
        ax0.quiver(x, y, dx, dy, units='xy', angles='xy', scale_units='xy', scale=1, color=l.get_color())

        # Final prep
        num_entries = 2*len(dataset.hydrophones)+2
        handles, labels = ax0.get_legend_handles_labels()
        actual_entries = len(handles)
        handles = np.array(handles + [None]*(num_entries-actual_entries)).reshape(-1, 2).T.flatten()[:num_entries]
        labels = np.array(labels + [None]*(num_entries-actual_entries)).reshape(-1, 2).T.flatten()[:num_entries]
        ax0.legend(handles, labels, loc='upper right', ncol=2)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        if not no_titles:
            ax0.set_title('Trajectories')

        if add_labels:
            ax0.set_title('(a)', loc='left', fontsize='medium')
            ax1.set_title('(b)', loc='left', fontsize='medium')

        if seconds:
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xlabel('Time')
        ax1.set_ylabel('Series')
        ax1.set_yticks([])
        ax1.set_yticklabels([])

        # Set axes and figure sizes
        ax1.set_box_aspect(1/ratio)
        fig.set_tight_layout(True)

        # Show and save the plot
        if show:
            plt.show()
        if save and dataset is not None:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/pf_plots/{n}_pf_trajectories.png'.format(n=dataset.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath, bbox_inches='tight')
        plt.close()

    def plot_heading(self, dataset):
        best_particle_idxs = np.argmax(self._history[:, :, -1], axis=1)
        best_particle = np.array([self._history[i, idx] for i, idx in enumerate(best_particle_idxs)])
        avg_particle = np.average(self._history, axis=1)
        times = self._time_history
        gps_heading = Dataset.get(dataset.tag.raw['gps_heading'], times)[1]

        # Note about averaging angles: convert to unit vectors, average x and y, then convert back to angles
        # Plot angles only after measurements. Use actual position to calculate angles

        start_time = times[0]
        times = utils.total_seconds(times, start_time)

        plt.plot(times, gps_heading, label='gps heading')
        plt.plot(times, best_particle[:, 2], label='best particle heading')
        plt.plot(times, avg_particle[:, 2], label='avg particle heading')
        plt.legend()
        plt.show()

    def plot_error(self, dataset, show=True, save=False, replace=False, **kwargs):
        zoom_times = kwargs.get('zoom_times', [])
        n = len(zoom_times)
        ratio = kwargs.get('ratio', 2)
        width = kwargs.get('width', 3.5)
        seconds = kwargs.get('plot_total_seconds', False)
        padding = kwargs.get('padding', 1.1)
        no_titles = kwargs.get('exclude_titles', False)
        add_labels = kwargs.get('labels', False)
        timeout = kwargs.get('timeout', 32)  # Number of seconds before the range stops being plotted

        # Set up figure and axes
        if n > 0:
            arr = mpl.figure.figaspect((1+ratio)/ratio)
            w, h = width * arr / arr[0]
            fig = plt.figure(constrained_layout=True, figsize=(width, width))
            gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[ratio, 1])
            gs1 = gs0[1].subgridspec(1, n)

            ax0 = fig.add_subplot(gs0[0])
            axs = [fig.add_subplot(gs1[0])]
            axs += [fig.add_subplot(gs1[i], sharey=axs[0]) for i in range(1, n)]

            for ax in axs:
                ax.set_aspect(1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        else:
            fig, ax0 = plt.figure(tight_layout=True)
            axs = []

        # Plot errors
        avg_particle = np.average(self._history, axis=1)  # Find the average particle at each time step
        times = self._time_history
        is_correction = self._step_history == Filter.CORRECTION
        correction_times = times[is_correction]
        true_x, true_y = dataset.get_tag_xy(times).T
        x_error = true_x - avg_particle[:, 0]
        x_error_corr = true_x[is_correction] - avg_particle[is_correction, 0]
        y_error = true_y - avg_particle[:, 1]
        y_error_corr = true_y[is_correction] - avg_particle[is_correction, 1]
        dist = np.sqrt(np.square(x_error) + np.square(y_error))
        dist_corr = np.sqrt(np.square(x_error_corr) + np.square(y_error_corr))

        # Convert to total seconds
        if seconds:
            start_time = times[0]
            times = utils.total_seconds(times, start_time)
            correction_times = utils.total_seconds(correction_times, start_time)
        ax0.plot(times, [0]*len(times))
        if kwargs.get('error_values', False):
            hline_kwargs_0 = {'xmin': 0, 'xmax': 1, 'linestyle': 'None', 'linewidth': 0.8}
            hline_kwargs_1 = {'xmin': 0, 'xmax': 1, 'linestyle': 'None', 'linewidth': 0.8}
            l0, = ax0.plot(times, x_error, label='x error'.format(np.max(np.abs(x_error)), np.average(x_error)))
            l1, = ax0.plot(times, y_error, label='y error'.format(np.max(np.abs(y_error)), np.average(y_error)))
            l2, = ax0.plot(times, dist, label='distance'.format(np.max(dist), np.average(dist)))
            ax0.scatter(correction_times, [0]*len(correction_times), marker='.', label='detections')

            max_abs = [max(np.max(x_error), np.min(x_error), key=np.abs), max(np.max(y_error), np.min(y_error), key=np.abs), np.max(dist)]
            ax0.axhline(max_abs[0], color=l0.get_color(), label='max abs = {:.2f}'.format(max_abs[0]), **hline_kwargs_0)
            ax0.axhline(max_abs[1], color=l1.get_color(), label='max abs = {:.2f}'.format(max_abs[1]), **hline_kwargs_0)
            ax0.axhline(max_abs[2], color=l2.get_color(), label='max = {:.2f}'.format(max_abs[2]), **hline_kwargs_0)

            avgs = [np.average(x_error), np.average(y_error), np.average(dist)]
            ax0.axhline(avgs[0], color=l0.get_color(), label='avg = {:.2f}'.format(avgs[0]), **hline_kwargs_1)
            ax0.axhline(avgs[1], color=l1.get_color(), label='avg = {:.2f}'.format(avgs[1]), **hline_kwargs_1)
            ax0.axhline(avgs[2], color=l2.get_color(), label='avg = {:.2f}'.format(avgs[2]), **hline_kwargs_1)
        else:
            # l, = ax0.plot(times, x_error, label='x error, max abs={:.2f}, avg={:.2f}'.format(np.max(np.abs(x_error)), np.average(x_error)))
            # l, = ax0.plot(times, y_error, label='y error, max abs={:.2f}, avg={:.2f}'.format(np.max(np.abs(y_error)), np.average(y_error)))
            # l, ax0.plot(times, dist, label='distance, max={:.2f}, avg={:.2f}'.format(np.max(dist), np.average(dist)))
            ax0.plot(times, x_error, label='x error')
            ax0.plot(times, y_error, label='y error')
            ax0.plot(times, dist, label='distance')
            ax0.scatter(correction_times, [0]*len(correction_times), marker='.', label='detections')

        ax0.scatter(correction_times, x_error_corr, marker='.')
        ax0.scatter(correction_times, y_error_corr, marker='.')
        ax0.scatter(correction_times, dist_corr, marker='.')
        if kwargs.get('error_values', False):
            ncol = 3
        else:
            ncol = 1
        if kwargs.get('legend_outside', False):
            ax0.legend(loc='center left', ncol=ncol, bbox_to_anchor=(1, 0.5))
        else:
            ax0.legend(loc='upper right', ncol=ncol)
        if add_labels:
            ax0.set_title('(a)', loc='left', fontsize='medium')

        # Plot zooms
        if n > 0:
            # Find a bounding box that always shows the average particle, the hydrophones, and the tag
            all_x = np.concatenate([
                self._history[:, :, 0].flatten(),
                dataset.tag.coords['x'].to_numpy(),
                self._measurement_history[:, 2].astype(np.float64)
            ])  # Concatenate all relevant x coordinates
            all_y = np.concatenate([
                self._history[:, :, 1].flatten(),
                dataset.tag.coords['y'].to_numpy(),
                self._measurement_history[:, 3].astype(np.float64)
            ])  # Concatenate all relevant y coordinates
            all_x = all_x[~np.isnan(all_x)]
            all_y = all_y[~np.isnan(all_y)]
            (minx, miny), (maxx, maxy) = utils.pad_bounds(((np.min(all_x), np.min(all_y)), (np.max(all_x), np.max(all_y))), f=padding, square=True)
            bbox = (minx, miny, maxx, maxy)
            for ax in axs:
                ax.set_xlim((minx, maxx))
                ax.set_ylim((miny, maxy))

            # Create background map
            origin = dataset.origin
            cartesian_bounds = np.array(bbox).reshape(2, 2).T
            cartesian_bounds = utils.pad_bounds(cartesian_bounds.T).T
            if origin is not None:
                coord_bounds = utils.to_coords(cartesian_bounds, origin)
                (south, west), (north, east) = coord_bounds
                img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir='../maps/OpenStreetMap/Mapnik')
                true_ext = utils.to_cartesian(np.flip(np.array(ext).reshape(2, 2), axis=0), origin).T.flatten()
            for ax in axs:
                ax.imshow(img, extent=true_ext)

            # Fill zoom axes
            for i, (time, ax, label) in enumerate(zip(zoom_times, axs, 'bcdefghijklmnopqrstuvwxyz')):
                if type(time) == int:
                    idx = time
                    timestamp = self._time_history[idx]
                    time = times[idx]
                else:
                    idx = max(bisect.bisect_left(self._time_history, time), 0)
                    timestamp = self._time_history[idx]
                    time = times[idx]
                
                # Plot particles
                ax.scatter(self._history[idx, :, 0], self._history[idx, :, 1], linestyle='None', marker='.', color='gold', label='Particles')

                # Plot hydrophone paths
                for name, data in dataset.hydrophones.items():
                    # Plot hydrophone coords
                    x, y = data.coords['x'][:timestamp], data.coords['y'][:timestamp]
                    if len(x) > 0 and len(y) > 0:
                        x, y = x[-1], y[-1]
                    ax.plot(x, y, marker='o', linestyle='None', label='{} position'.format(name))
                    ax.plot([], [])

                    # Draw circles showing the last measurement
                    pos = dataset.get_hydrophone_xy(name, timestamp, mode='last')
                    r_series = pd.Series(
                        self._measurement_history[self._hydrophone_history == name, 0],
                        index=self._time_history[self._step_history == Filter.CORRECTION][self._hydrophone_history == name])
                    r_std_series = pd.Series(
                        np.sqrt(self._measurement_cov_history[self._hydrophone_history == name, 0, 0]),
                        index=self._time_history[self._step_history == Filter.CORRECTION][self._hydrophone_history == name])
                    r_idx, r = Dataset.get(r_series, timestamp, mode='last')
                    r_std = Dataset.get(r_std_series, timestamp, mode='last')[1]
                    timed_out = (timestamp - r_series.index[r_idx]).total_seconds() > timeout
                    if r is None or pos is None or timed_out:
                        pos = (0, 0)
                        r = 0
                        r_std = 0
                    ax.add_patch(mpl.patches.Circle(pos, r, fill=False, linewidth=1, zorder=0))
                    ax.add_patch(mpl.patches.Circle(pos, r-r_std, fill=False, linewidth=1, color='gray', zorder=0))
                    ax.add_patch(mpl.patches.Circle(pos, r+r_std, fill=False, linewidth=1, color='gray', zorder=0))

                # Plot groundtruth path
                x, y = np.array(dataset.tag.coords[['x', 'y']][:timestamp]).T
                ax.plot(x, y, marker='.', label='Tag coords')

                # Plot particle path
                x, y = avg_particle[:idx, [0, 1]].T
                ax.plot(x, y, marker='.', label='Tag estimated coords', color='teal')

                # Add lines indicating where the zooms came from
                ax0.axvline(time, ymin=0, ymax=1, color='gray', linewidth=0.5)
                miny = ax0.get_ylim()[0]
                con_left = mpl.patches.ConnectionPatch(xyA=(time, miny), coordsA=ax0.transData, xyB=(0, 1), coordsB=ax.transAxes, color='gray', linewidth=0.5)
                con_right = mpl.patches.ConnectionPatch(xyA=(time, miny), coordsA=ax0.transData, xyB=(1, 1), coordsB=ax.transAxes, color='gray', linewidth=0.5)
                fig.add_artist(con_left)
                fig.add_artist(con_right)

                # Set title to the timestamp
                ax.set_title(str(time), fontsize='medium')
                if add_labels:
                    ax.set_title('({})'.format(label), loc='left', fontsize='medium')

                # Add legend to one zoom axis
                if i == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper left', ncol=5, fontsize='small', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)

        # Set and figure titles
        if seconds:
            ax0.set_xlabel('Time (s)')
        else:
            ax0.set_xlabel('Time')
        ax0.set_ylabel('Meters')
        if not no_titles:
            ax0.set_title('Errors')

        if show:
            plt.show()
        if save and dataset is not None:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/pf_plots/{n}_pf_error.png'.format(n=dataset.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath, bbox_inches='tight')
        plt.close()

    def tabulate_error(self, dataset, save=False, replace=False, **kwargs):
        times = kwargs.get('times', [0])

        # Calculate errors
        avg_particle = np.average(self._history, axis=1)  # Find the average particle at each time step
        true_x, true_y = dataset.get_tag_xy(self._time_history).T
        x_error = true_x - avg_particle[:, 0]
        y_error = true_y - avg_particle[:, 1]
        dist = np.sqrt(np.square(x_error) + np.square(y_error))

        # Tabular environment
        table_format =\
        '\\begin{{tabular}}{{rcccc}}\\toprule\n' +\
        '    Time span & \multicolumn{{3}}{{c}}{{Distance (m)}} & RMSE (m) \\\\\n' +\
        '    & Max & Average & Stdev & \\\\ \\midrule\n' +\
        '{}' +\
        '    \\bottomrule\n' +\
        '\\end{{tabular}}'

        # One row for each time
        row_format = '    {seconds:.0f}s to end & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'

        # Calculate for each time
        rows = []
        for time in times:
            if type(time) == int:
                idx = time
            else:
                idx = max(bisect.bisect_left(self._time_history, time), 0)
            x_error_t = x_error[idx:]
            y_error_t = y_error[idx:]
            dist_t = dist[idx:]
            rmse_t = np.sqrt(np.average(np.square(x_error_t) + np.square(y_error_t)))
            row_data = [np.max(dist_t), np.average(dist_t), np.std(dist_t), rmse_t]
            seconds = (self._time_history[idx] - self._time_history[0]).total_seconds()
            rows.append(row_format.format(*row_data, seconds=seconds))
        rows = ''.join(rows)
        table = table_format.format(rows)

        # Display table
        print(table)

        # Save table
        if save and dataset is not None:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/pf_plots/{n}_pf_error_table.txt'.format(n=dataset.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            with open(savepath, 'w') as f:
                print(table, end='', file=f)

    def animate(self, dataset, padding=1.1, show=True, save=False, replace=False):
        # Make sure we have the data to plot
        if not self._save_history:
            print('No history to plot')
            return

        fig, ax = plt.subplots()
        avg_particle = np.average(self._history, axis=1)  # Find the average particle at each time step
        num_steps = len(self._history)  # Total number of steps, including prediction and correction

        # Find a bounding box that always shows the average particle, the hydrophones, and the tag
        all_x = np.concatenate([
            avg_particle[:, 0].flatten(),
            dataset.tag.coords['x'].to_numpy(),
            self._measurement_history[:, 2].astype(np.float64)
        ])  # Concatenate all relevant x coordinates
        all_y = np.concatenate([
            avg_particle[:, 1].flatten(),
            dataset.tag.coords['y'].to_numpy(),
            self._measurement_history[:, 3].astype(np.float64)
        ])  # Concatenate all relevant y coordinates
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]
        (minx, miny), (maxx, maxy) = utils.pad_bounds(((np.min(all_x), np.min(all_y)), (np.max(all_x), np.max(all_y))), f=padding, square=True)
        bbox = (minx, miny, maxx, maxy)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])

        # Create artists
        # Create background map
        origin = dataset.origin
        cartesian_bounds = np.array(bbox).reshape(2, 2).T
        cartesian_bounds = utils.pad_bounds(cartesian_bounds.T, f=2).T
        if origin is not None:
            coord_bounds = utils.to_coords(cartesian_bounds, origin)
            (south, west), (north, east) = coord_bounds
            img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir='../maps/OpenStreetMap/Mapnik')
            true_ext = utils.to_cartesian(np.flip(np.array(ext).reshape(2, 2), axis=0), origin).T.flatten()
        background = ax.imshow(img, extent=true_ext)

        # Plot the hydrophones' locations and detections
        hydrophones = {}
        hydrophone_circles = {}
        hydrophone_outer_std_1 = {}
        hydrophone_inner_std_1 = {}
        hydrophone_r_series = {}
        hydrophone_r_std_series = {}
        for serial_no in dataset.hydrophones:
            hydrophones[serial_no], = ax.plot([], [], linestyle='None', marker='o', label=str(serial_no))
            hydrophone_circles[serial_no] = mpl.patches.Circle((0, 0), 0, fill=False, linewidth=1)
            hydrophone_outer_std_1[serial_no] = mpl.patches.Circle((0, 0), 0, fill=False, linewidth=1, color='gray')
            hydrophone_inner_std_1[serial_no] = mpl.patches.Circle((0, 0), 0, fill=False, linewidth=1, color='gray')
            hydrophone_r_series[serial_no] = pd.Series(
                self._measurement_history[self._hydrophone_history == serial_no, 0],
                index=self._time_history[self._step_history == Filter.CORRECTION][self._hydrophone_history == serial_no])
            hydrophone_r_std_series[serial_no] = pd.Series(
                np.sqrt(self._measurement_cov_history[self._hydrophone_history == serial_no, 0, 0]),
                index=self._time_history[self._step_history == Filter.CORRECTION][self._hydrophone_history == serial_no])
            ax.add_patch(hydrophone_circles[serial_no])
            ax.add_patch(hydrophone_outer_std_1[serial_no])
            ax.add_patch(hydrophone_inner_std_1[serial_no])
        
        # Plot the best particle's path
        best_particle_path_x = []
        best_particle_path_y = []
        best_particle_path, = ax.plot([], [], color='red', marker='.', label='est path')
        # best_particle_path_2, = ax.plot([], [], linestyle='None', marker='.', color='red')

        # Plot the tag's groundtruth path
        groundtruth_path_x = []
        groundtruth_path_y = []
        groundtruth_path, = ax.plot([], [], 'b-', label='true path')
        last_coords_idx = None
        groundtruth_coords_x = []
        groundtruth_coords_y = []
        groundtruth_coords, = ax.plot([], [], color='blue', marker='.', linestyle='None')

        # Plot particle positions
        particles, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='particles')

        # Plot the number of elapsed steps
        steps = ax.text(3, 6, 'Step = 0 / {}'.format(num_steps), horizontalalignment='center', verticalalignment='top')
        ax.legend()

        # Artists indexed later are drawn over ones indexed earlier
        artists = [
            background,
            particles,
            *hydrophones.values(),
            *hydrophone_inner_std_1.values(),
            *hydrophone_circles.values(),
            *hydrophone_outer_std_1.values(),
            # best_particle_path_2,
            best_particle_path,
            groundtruth_path,
            groundtruth_coords,
            steps
        ]

        def init():
            ax.set_title('Particle filter')
            return artists
        
        def update(frame):
            global last_coords_idx
            # print('Frame:', frame)
            curr_time = self._time_history[frame]

            # Reset paths on the first frame
            if frame == 0:
                best_particle_path_x.clear()
                best_particle_path_y.clear()
                groundtruth_path_x.clear()
                groundtruth_path_y.clear()
                groundtruth_coords_x.clear()
                groundtruth_coords_y.clear()
                last_coords_idx = None
            
            # Plot best particle path
            best_particle_path_x.append(avg_particle[frame, 0])
            best_particle_path_y.append(avg_particle[frame, 1])
            best_particle_path.set_data(best_particle_path_x, best_particle_path_y)
            # best_particle_path_2.set_data(best_particle_path_x, best_particle_path_y)

            # Plot groundtruth path
            pos = dataset.get_tag_xy(curr_time)
            idx, last_pos = Dataset.get(dataset.tag.coords, curr_time)
            groundtruth_path_x.append(pos[0])
            groundtruth_path_y.append(pos[1])
            groundtruth_path.set_data(groundtruth_path_x, groundtruth_path_y)
            if last_pos is not None and idx != last_coords_idx:
                groundtruth_coords_x.append(last_pos.x)
                groundtruth_coords_y.append(last_pos.y)
                groundtruth_coords.set_data(groundtruth_coords_x, groundtruth_coords_y)
                last_coords_idx = idx

            # Plot other particles' poses
            particles.set_data(pf._history[frame, :, 0], pf._history[frame, :, 1])

            # Plot hydrophones
            for serial_no, hydrophone in hydrophones.items():
                smooth_pos = dataset.get_hydrophone_xy(serial_no, curr_time, mode='interpolate')
                pos = dataset.get_hydrophone_xy(serial_no, curr_time, mode='last')
                if smooth_pos is not None:
                    hydrophone.set_data([smooth_pos[0]], [smooth_pos[1]])

                # r = dataset.get_gps_distance(serial_no, curr_time)
                r_series = hydrophone_r_series[serial_no]
                r_std_series = hydrophone_r_std_series[serial_no]
                r = Dataset.get(r_series, curr_time, mode='last')[1]
                r_std = Dataset.get(r_std_series, curr_time, mode='last')[1]
                if r is not None and pos is not None:
                    hydrophone_circles[serial_no].set(center=pos, radius=r)
                    hydrophone_outer_std_1[serial_no].set(center=pos, radius=r + r_std)
                    hydrophone_inner_std_1[serial_no].set(center=pos, radius=r - r_std)
            
            # Update steps
            steps.set_text('Step = {} / {}'.format(frame, num_steps))

            return artists
        anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, 1), init_func=init, blit=True, interval=2, repeat=True)
        if show:
            plt.show()
        if save and dataset is not None:
            savepath = utils.add_version('../datasets/{n}/pf_plots/{n}_pf_anim.gif'.format(n=dataset.name), replace=replace)
            print('Saving animation to {}'.format(savepath))
            utils.mkdir(savepath)
            writergif = animation.PillowWriter(fps=30)
            anim.save(savepath, writer=writergif)
            print('Animation Saved!')
        plt.close()

if __name__ == '__main__':
    save = False
    replace = False

    # dataset = Dataset('tag78_static_two_clusters', shift_tof=True)
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(-200, 100), (-150, 150)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    # pf.run()
    # pf.animate(dataset, save=False)
    # pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5)
    # pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, dataset.start_time + timedelta(seconds=50), dataset.start_time + timedelta(seconds=100), -1], ratio=1, width=6.5)

    dataset = Dataset('tag78_swimming_test_1_2', shift_tof=True)
    pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
                'init_uniform_random': [(0, 200), (-150, 50), (-np.pi, np.pi)]
            }, save_history=True, hydrophone_params={
                'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
                457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
            }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    pf.run()
    pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5, savepath='../paper/fig0.png')
    pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, 20, dataset.start_time + timedelta(seconds=41), 200, -6], ratio=1, width=6.5, savepath='../paper/fig1.png')
    pf.tabulate_error(dataset, save=save, replace=replace, times=[0, dataset.start_time + timedelta(seconds=30)], savepath='../paper/fig1_errors.txt')
    pf.plot_heading(dataset)

    # dataset = Dataset('tag78_shore_2_boat_all_static_test_0', shift_tof=True)
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(0, 200), (-150, 50), (-np.pi, np.pi)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    # pf.run()
    # pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5, savepath='../paper/fig2.png')
    # pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, dataset.start_time + timedelta(seconds=44), dataset.start_time + timedelta(seconds=60.5), dataset.start_time + timedelta(seconds=400), -2], ratio=1, width=6.5, savepath='../paper/fig3.png')
    # pf.tabulate_error(dataset, save=save, replace=replace, times=[0, dataset.start_time + timedelta(seconds=61)], savepath='../paper/fig3_errors.txt')

    # dataset = Dataset('tag78_swimming_test_0', shift_tof=True)
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(-50, 150), (-50, 150)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    # pf.run()
    # pf.animate(dataset)
    # pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5)
    # pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, -1], ratio=1, width=6.5)

    # dataset = Dataset('tag78_swimming_test_1_3', shift_tof=True)
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(-150, 100), (-200, 50)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    # pf.run()
    # pf.animate(dataset)
    # pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5)
    # pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, -1], ratio=1, width=6.5)

    # dataset = Dataset('tag78_shore_2_boat_all_static_test_1', shift_tof=True)
    # pf = ParticleFilter.from_dataset(dataset, 1000, RandomMotionModel, motion_model_params={
    #             'init_uniform_random': [(-200, 100), (-150, 150)]
    #         }, save_history=True, hydrophone_params={
    #             'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0},
    #             457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 1000, 'ff': 0.1, 'initial_r': 0}
    #         }, use_tof=True, measurement_cov=np.array([[10, 0], [0, 10]]))
    # pf.run()
    # pf.animate(dataset)
    # pf.plot(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, width=6.5)
    # pf.plot_error(dataset, show=True, save=save, replace=replace, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, -1], ratio=1, width=6.5)

    # dataset = Dataset('tag78_shore_2_boat_all_static_test_1', shift_tof=True)
    # dataset = Dataset('tag78_shore_2_boat_all_static_test_0', shift_tof=True)
    # dataset = Dataset('tag78_50m_increment_long_beach_test_0', reset_tof=True, shift_tof=True)
    # pf.plot_error(dataset, show=True, save=False, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, dataset.start_time + timedelta(seconds=44), dataset.start_time + timedelta(seconds=60.5), dataset.start_time + timedelta(seconds=400), -1], ratio=1, error_values=True, width=7, savepath='../paper/fig5.png')
    # pf.plot_error(dataset, show=True, save=True, exclude_titles=True, plot_total_seconds=True, zoom_times=[0, dataset.start_time + timedelta(seconds=500), dataset.start_time + timedelta(seconds=1500), dataset.start_time + timedelta(seconds=3500), -1], ratio=1, error_values=True, width=7, savepath='../paper/fig8.png')

    # Old hydrophone params
    # hydrophone_params = {
    #     'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 16.250003, 'ff': 0.3},
    #     457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 9.400336, 'ff': 0.3}
    # }

    # pf.animate(dataset)