from abc import ABC, abstractmethod
from filter import Filter
from kalman import KalmanFilter
import numpy as np
# import pandas as pd
from datetime import timedelta
import utils
import scipy
from scipy.stats import multivariate_normal
from merge_dataset import merge_dataset
# from plotting import plot_df
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import movingpandas as mpd
import bisect

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

class ParticleFilter(Filter):
    def from_dataset(dataset, num_particles, motion_model_type, motion_model_params={}, save_history=False, hydrophone_params={}):
        # Remove rows that are not about the tag we are localizing
        df = dataset.df[dataset.df['tag_id'] == dataset.attrs['tag_id']]

        # Assumes the data is sorted by datetime, which should be a valid assumption if it comes from merge_dataset
        start_time = df.iloc[0]['datetime']
        end_time = df.iloc[-1]['datetime']
        num_predictions = int((end_time - start_time).total_seconds()) + 2

        pf = ParticleFilter(num_particles, motion_model_type, motion_model_params, save_history, hydrophone_params)

        # Queue predictions
        print(num_predictions)
        for i in range(num_predictions):
            timestamp = start_time + timedelta(seconds=i)
            pf.queue_prediction(timestamp, np.array([]))
        
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
            pf.queue_correction(timestamp, data)
        return pf

    def __init__(self, num_particles, motion_model_type, motion_model_params={}, save_history=False, hydrophone_params={}):
        self._save_history = save_history
        self._num_particles = num_particles
        self._motion_model_params = motion_model_params
        self._motion_model = motion_model_type(**motion_model_params)
        self._hydrophone_params = hydrophone_params
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

        print('Correction step', timestamp, serial_no)
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
            # print('Resampled particles')
        else:
            print('Skipped measurement due to nan weights')
    
    def plot(self, dataset, padding=1.1, square=True):
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
            dataset.df['tag_x'].to_numpy(),
            self._hydrophone_state_history[:, 1].astype(np.float64)
        ])  # Concatenate all relevant x coordinates
        all_y = np.concatenate([
            avg_particle[:, 1].flatten(),
            dataset.df['tag_y'].to_numpy(),
            self._hydrophone_state_history[:, 2].astype(np.float64)
        ])  # Concatenate all relevant y coordinates
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]
        minx, miny, maxx, maxy = (np.min(all_x), np.min(all_y), np.max(all_x), np.max(all_y))

        # Expand the bounding box by (padding-1)%
        midx = (maxx + minx) / 2
        midy = (maxy + miny) / 2
        deltax = (maxx - minx) * padding
        deltay = (maxy - miny) * padding
        if square:
            deltax = deltay = max(deltax, deltay)
        minx = midx - deltax / 2
        maxx = midx + deltax / 2
        miny = midy - deltay / 2
        maxy = midy + deltay / 2
        bbox = (minx, miny, maxx, maxy)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])

        # Create artists
        # Create background map
        origin = (dataset.attrs['origin']['latitude'], dataset.attrs['origin']['longitude'])
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
        hydrophone_trajs = {}
        for serial_no in dataset.df['serial_no'].unique():
            hydrophone_trajs[serial_no] = mpd.Trajectory(
                dataset.df[
                    dataset.df['serial_no'] == serial_no
                ].set_index('datetime'),
                '{}_traj'.format(serial_no),
                x='longitude',
                y='latitude'
            )
            hydrophones[serial_no], = ax.plot([], [], linestyle='None', marker='o', label=str(serial_no))
            hydrophone_circles[serial_no] = mpl.patches.Circle((0, 0), 0, fill=False, linewidth=1)
            ax.add_patch(hydrophone_circles[serial_no])
        
        # Plot the best particle's path
        best_particle_path_x = []
        best_particle_path_y = []
        best_particle_path, = ax.plot([], [], 'r-', label='est path')
        best_particle_path_2, = ax.plot([], [], linestyle='None', marker='.', color='red')

        # Plot the tag's groundtruth path
        groundtruth_path_x = []
        groundtruth_path_y = []
        groundtruth_traj = mpd.Trajectory(dataset.df.set_index('datetime'), 'tag_traj', x='tag_longitude', y='tag_latitude')
        groundtruth_path, = ax.plot([], [], 'bo', label='true path')

        # Plot particle positions
        particles, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='particles')

        # Plot the number of elapsed steps
        steps = ax.text(3, 6, 'Step = 0 / {}'.format(num_steps), horizontalalignment='center', verticalalignment='top')

        # Artists indexed later are drawn over ones indexed earlier
        artists = [
            background,
            *hydrophones.values(),
            *hydrophone_circles.values(),
            best_particle_path_2,
            best_particle_path,
            groundtruth_path,
            particles,
            steps
        ]

        def init():
            ax.set_title('Particle filter')
            return artists
        
        def update(frame):
            print('Frame:', frame)
            curr_time = self._time_history[frame]

            # Reset paths on the first frame
            if frame == 0:
                groundtruth_path_x.clear()
                groundtruth_path_y.clear()
                best_particle_path_x.clear()
                best_particle_path_y.clear()
            
            # Plot best particle path
            best_particle_path_x.append(avg_particle[frame, 0])
            best_particle_path_y.append(avg_particle[frame, 1])
            best_particle_path.set_data(best_particle_path_x, best_particle_path_y)
            best_particle_path_2.set_data(best_particle_path_x, best_particle_path_y)

            # Plot groundtruth path
            pos = groundtruth_traj.get_position_at(curr_time)
            pos = utils.to_cartesian((pos.y, pos.x), origin)
            groundtruth_path_x.append(pos[0])
            groundtruth_path_y.append(pos[1])
            groundtruth_path.set_data(groundtruth_path_x, groundtruth_path_y)

            # Plot other particles' poses
            particles.set_data(pf._history[frame, :, 0], pf._history[frame, :, 1])

            # Plot hydrophones
            for serial_no, hydrophone in hydrophones.items():
                hydrophone_traj = hydrophone_trajs[serial_no]
                pos = hydrophone_traj.get_position_at(curr_time)
                pos = utils.to_cartesian((pos.y, pos.x), origin)
                hydrophone.set_data([pos[0]], [pos[1]])
                idx = bisect.bisect(hydrophone_traj.df.index, curr_time) - 1
                r = hydrophone_traj.df.iloc[idx]['gps_distance']
                hydrophone_circles[serial_no].set(center=pos, radius=r)
            
            # Update steps
            steps.set_text('Step = {} / {}'.format(frame, num_steps))

            return artists
        anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, 10), init_func=init, blit=False, interval=33, repeat=True)
        plt.show()

if __name__ == '__main__':
    dataset = merge_dataset('tag78_swimming_test_1')
    pf = ParticleFilter.from_dataset(dataset, 100, RandomMotionModel, motion_model_params={
        'uniform_random': [(-100, 100), (-100, 100)]
    }, save_history=True, hydrophone_params={
        'VR100': {'m': -0.10527966, 'l': -0.55164737, 'b': 68.59493072, 'signal_var': 16.250003},
        457049: {'m': -0.20985953, 'l': 5.5568182, 'b': 76.90064068, 'signal_var': 9.400336}
    })

    pf.run()
    for serial_no, kf in pf._filters.items():
        df = dataset.df[dataset.df['serial_no'] == serial_no]
        groundtruth = np.column_stack([df['gps_distance'], df['gps_speed']])
        kf.plot(groundtruth, save=False)
    # pf.plot(dataset)

