import yaml
import pandas as pd
import os
import utils
import numpy as np
import utils
from collections import namedtuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import bisect
from datetime import datetime

# Relevant distinct time series:
# - Detections from each hydrophone
# - Coordinates of each hydrophone
# - Coordinates of the tag

HydrophoneData = namedtuple('HydrophoneData', ['raw', 'detections', 'coords'])
TagData = namedtuple('TagData', ['raw', 'id', 'avg_dt', 'coords'])
Coordinate = namedtuple('Coordinate', ['latitude', 'longitude'])

class Dataset():
    datasets_path='../data/datasets.yaml'

    def get_path(relpath):
        return os.path.join(os.path.split(Dataset.datasets_path)[0], relpath)

    def make_path_array(path):
        if type(path) is list:
            return [Dataset.get_path(p) for p in path]
        else:
            return [Dataset.get_path(path)]

    def __init__(self, name, start_time=None, end_time=None, **kwargs):
        datasets = yaml.safe_load(open(Dataset.datasets_path, 'r'))
        self.name = name
        self.attrs = kwargs
        kwargs['from'] = name
        encountered_names = set()

        while 'from' in self.attrs:
            from_name = self.attrs['from']
            if from_name in encountered_names:
                raise ValueError('Recursive definition: {} depends on itself'.format(name))
            encountered_names.add(name)
            old_attrs = self.attrs
            self.attrs = datasets[from_name]

            # Overwrite attributes in the from dataset with the ones that are present in the top level dataset
            for key, value in old_attrs.items():
                if not key == 'from':
                    self.attrs[key] = value
        tag_id = self.attrs['tag_id']

        # Get start and end times if possible
        if start_time is None and 'start_time' in self.attrs:
            start_time = datetime.fromisoformat(self.attrs['start_time'])
        if end_time is None and 'end_time' in self.attrs:
            end_time = datetime.fromisoformat(self.attrs['end_time'])
        
        # Whether or not to reset time of flight and related columns
        reset_tof = self.attrs.get('reset_tof', False)

        # Whether or not to shift tof distance to match the first gps distance
        shift_tof = self.attrs.get('shift_tof', False)

        # Retrieve origin
        # self.origin = Coordinate(self.attrs['origin']['latitude'], self.attrs['origin']['longitude'])
        self.origin = np.array([self.attrs['origin']['latitude'], self.attrs['origin']['longitude']])

        # Read in detections and coordinates for each hydrophone
        self.hydrophones = {}
        for name, hydrophone_attrs in self.attrs['hydrophones'].items():
            # Concatenate all csvs given into one dataframe
            paths = Dataset.make_path_array(hydrophone_attrs['path'])
            df = pd.concat([pd.read_csv(Dataset.get_path(path), index_col='datetime', parse_dates=True) for path in paths])
            df.sort_index(inplace=True)  # Sort by datetime

            # Get just the detections
            columns = [
                'serial_no',            # The hydrophone's serial number
                'code_space',           #
                'tag_id',               # The id of the tag detected
                'signal_level',         # The signal level of the detection
                # 'noise_level',          # The noise in the signal level
                'total_dt',             # The total time since the first detection in seconds
                'dt',                   # The time since the last detection in seconds
                'delta_tof',            # The change in time of flight since the last detection
                'delta_distance',       # The change in distance since the last detection
                'total_distance'        # The total change in distance since the first detection
            ]
            detections = df[df['tag_id'] == tag_id][[column for column in columns if column in df.columns]]

            # Filter out rows that don't have valid gps coordinates
            idxs = []
            for i, ((prev_lat, prev_lon), (lat, lon)) in enumerate(utils.pairwise(zip(df['latitude'], df['longitude']))):
                if np.isnan(prev_lat) or np.isnan(prev_lon):
                    idxs.append(i)
                elif lat == prev_lat and lon == prev_lon:
                    idxs.append(i + 1)
            coords = df[~df.index.isin(df.index[idxs])][['latitude', 'longitude']]
            coords['x'], coords['y'] = utils.to_cartesian(np.array(coords).T, self.origin).T

            # Filter out rows by start and end time
            if start_time is not None:
                df = df[start_time:]
                detections = detections[start_time:]
                coords = coords[start_time:]
            if end_time is not None:
                df = df[:end_time]
                detections = detections[:end_time]
                coords = coords[:end_time]

            # Pack everything into a HydrophoneData object
            self.hydrophones[name] = HydrophoneData(df, detections, coords)
        
        # Read in tag coordinates
        tag_coords_dict = self.attrs['tag_coords']
        if 'path' in tag_coords_dict:
            # Concatenate all csvs given into one dataframe
            paths = Dataset.make_path_array(tag_coords_dict['path'])
            raw = pd.concat([pd.read_csv(Dataset.get_path(path), index_col='datetime', parse_dates=True) for path in paths])
            raw.sort_index(inplace=True)  # Sort by datetime

            # Filter out rows that don't have valid gps coordinates
            idxs = []
            for i, ((prev_lat, prev_lon), (lat, lon)) in enumerate(utils.pairwise(zip(raw['latitude'], raw['longitude']))):
                if np.isnan(prev_lat) or np.isnan(prev_lon):
                    idxs.append(i)
                elif lat == prev_lat and lon == prev_lon:
                    idxs.append(i + 1)
            coords = raw[~raw.index.isin(raw.index[idxs])][['latitude', 'longitude']]

            # Filter out rows by start and end time
            if start_time is not None:
                coords = coords[start_time:]
            if end_time is not None:
                coords = coords[:end_time]
        else:
            # Find the start time and end time for the whole data frame
            if start_time is None:
                for data in self.hydrophones.values():
                    if start_time is None or start_time > data.raw.index[0]:
                        start_time = data.raw.index[0]
            if end_time is None:
                for data in self.hydrophones.values():
                    if end_time is None or end_time < data.raw.index[-1]:
                        end_time = data.raw.index[-1]

            raw = np.array([tag_coords_dict['latitude'], tag_coords_dict['longitude']])
            coords = pd.DataFrame(
                [[raw[0], raw[1]]]*2,
                columns=['latitude', 'longitude'],
                index=[start_time, end_time])
        coords['x'], coords['y'] = utils.to_cartesian(np.array(coords).T, self.origin).T

        # Find the start time and end time for the whole dataset
        if start_time is None:
            for data in self.hydrophones.values():
                if start_time is None or start_time > data.raw.index[0]:
                    start_time = data.raw.index[0]
            if start_time is None or start_time > coords.index[0]:
                start_time = coords.index[0]
        if end_time is None:
            for data in self.hydrophones.values():
                if end_time is None or end_time < data.raw.index[-1]:
                    end_time = data.raw.index[-1]
            if end_time is None or start_time < coords.index[-1]:
                end_time = coords.index[-1]

        # Fill in tag data
        self.tag = TagData(raw, tag_id, utils.avg_dt_dict.get(tag_id, np.nan), coords)
        if np.isnan(self.tag.avg_dt):
            print('Warning: Tag {} has an unknown avg dt. Values relying on it will be NaN.'.format(self.tag.id))
        
        # Compute time of flight columns of the detections dataframe
        if reset_tof:
            self._update_detections()

        # Shift time of flight columns of the detections dataframe
        if shift_tof:
            for name, data in self.hydrophones.items():
                # Make sure we have detections
                if len(data.detections) == 0:
                    break
                    
                timestamp = data.detections.index[0]
                gps_distance = self.get_gps_distance(name, timestamp)
                data.detections['total_distance'] += gps_distance - data.detections['total_distance'][0]

        self.start_time = start_time
        self.end_time = end_time
    
    def _update_detections(self):
        for data in self.hydrophones.values():
            # Make sure we have detections
            if len(data.detections) == 0:
                break

            # Compute total_dt, dt, delta_tof, delta_distance, and total_distance
            detections = data.detections
            detections['total_dt'] = [delta.total_seconds() for delta in detections.index - detections.index[0]]
            detections['dt'] = [0] + [(timestamp1 - timestamp0).total_seconds() for timestamp0, timestamp1 in utils.pairwise(detections.index)]
            detections['delta_tof'] = np.concatenate([[0], utils.get_delta_tof(detections.index, self.tag.avg_dt)])
            detections['delta_distance'] = detections['delta_tof'] * utils.SPEED_OF_SOUND
            detections['total_distance'] = np.cumsum(detections['delta_distance'])

    def get(series, timestamp, mode='last', ordered=True, lo=0):
        '''
        Given a dataframe or series, retrieve a row or value respectively given a timestamp.
        mode determines what value is returned
        mode = last:        Returns the last row or value before timestamp
        mode = next:        Returns the next row or value after timestamp
        mode = nearest:     Returns the row or value closest in time to timestamp
        mode = interpolate: Linear interpolation between the last and next value. Only works with a series
        '''
        if not (type(timestamp) is pd.Timestamp or type(timestamp) is datetime or type(timestamp) is np.datetime64):
            idxs = []
            values = []
            for t in timestamp:
                # Use the index of the last timestamp, or lo as a bound
                if ordered and len(idxs) > 0 and idxs[-1] > lo:
                    idx, value = Dataset.get(series, t, mode=mode, ordered=ordered, lo=idxs[-1])
                else:
                    idx, value = Dataset.get(series, t, mode=mode, ordered=ordered, lo=lo)
                idxs.append(idx)
                values.append(value)
            return (idxs, values)
        else:
            # Retrieve the next and last times and values
            next_idx = bisect.bisect(series.index, timestamp, lo=lo)
            last_idx = next_idx - 1
            last_time = next_time = None
            last_value = next_value = None
            if last_idx >= 0:
                last_time = series.index[last_idx]
                last_value = series.iloc[last_idx]
            if next_idx < len(series):
                next_time = series.index[next_idx]
                next_value = series.iloc[next_idx]
            
            # Return the appropriate value depending on the mode
            if mode == 'last':
                return (last_idx, last_value)
            elif mode == 'next':
                return (last_idx, next_value)
            elif mode == 'nearest':
                if last_time is None:
                    return (last_idx, next_value)
                elif next_time is None:
                    return (last_idx, last_value)
                else:
                    if next_time - timestamp < timestamp - last_time:
                        return (last_idx, next_value)
                    else:
                        return (last_idx, last_value)
            elif mode == 'interpolate':
                if last_time is None:
                    return (last_idx, next_value)
                elif next_time is None:
                    return (last_idx, last_value)
                else:
                    interp_value = (next_value - last_value) * ((timestamp - last_time) / (next_time - last_time)) + last_value
                    return (last_idx, interp_value)
    
    def get_tag_coords(self, timestamp, mode='interpolate', ordered=True):
        '''
        Get the tag's coordinates at timestamp(s)
        '''
        rows = Dataset.get(self.tag.coords, timestamp, mode=mode, ordered=ordered)[1]
        if type(rows) == list:
            lats, lons = [], []
            for row in rows:
                lats.append(row.latitude)
                lons.append(row.longitude)
            return np.array([lats, lons])
        elif rows is None:
            return None
        else:
            return np.array([rows.latitude, rows.longitude])

    def get_tag_xy(self, timestamp, mode='interpolate', ordered=True):
        '''
        Get the tag's coordinates at timestamp(s)
        '''
        rows = Dataset.get(self.tag.coords, timestamp, mode=mode, ordered=ordered)[1]
        if type(rows) == list:
            x, y = [], []
            for row in rows:
                x.append(row.x)
                y.append(row.y)
            return np.array([x, y]).T
        elif rows is None:
            return None
        else:
            return np.array([rows.x, rows.y])
    
    def get_hydrophone_coords(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the given hydrophone's coordinates at timestamp(s)
        '''
        rows = Dataset.get(self.hydrophones[name].coords, timestamp, mode=mode, ordered=ordered)[1]
        if type(rows) == list:
            lats, lons = [], []
            for row in rows:
                lats.append(row.latitude)
                lons.append(row.longitude)
            return np.array([lats, lons])
        elif rows is None:
            return None
        else:
            return np.array([rows.latitude, rows.longitude])
    
    def get_hydrophone_xy(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the given hydrophone's coordinates at timestamp(s)
        '''
        rows = Dataset.get(self.hydrophones[name].coords, timestamp, mode=mode, ordered=ordered)[1]
        if type(rows) == list:
            x, y = [], []
            for row in rows:
                x.append(row.x)
                y.append(row.y)
            return np.array([x, y]).T
        elif rows is None:
            return None
        else:
            return np.array([rows.x, rows.y])
    
    def get_gps_distance(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the hydrophone's distance from the tag as measured by the GPS at timestamp(s)
        '''
        hydrophone_coords = self.get_hydrophone_coords(name, timestamp, mode=mode, ordered=ordered)
        tag_coords = self.get_tag_coords(timestamp, mode=mode, ordered=ordered)
        return utils.get_geodesic_distance(hydrophone_coords, tag_coords)
    
    def get_gps_speed(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the speed at which the hydrophone's distance from the tag is changing as measured by the GPS at timestamp(s)
        '''
        # times = np.concatenate([np.array(self.tag.coords.index), np.array(self.hydrophones[name].coords.index)])
        # all_distances = pd.Series(self.get_gps_distance(name, times, mode=mode, ordered=True), index=times)
        # all_distances.sort_index(inplace=True)
        # distances = Dataset.get(all_distances, timestamp, mode=mode, ordered=ordered)[1]
        distances = self.get_gps_distance(name, timestamp, mode=mode, ordered=ordered)
        if type(distances) == float:
            return np.nan
        else:
            speeds = [np.nan]
            for (prev_t, prev_distance), (t, distance) in utils.pairwise(zip(timestamp, distances)):
                dt = (t - prev_t).total_seconds()
                delta_distance = distance - prev_distance

                # Handle duplicate timestamps
                if dt == 0 and delta_distance == 0:
                    speeds.append(speeds[-1])
                elif dt == 0:
                    speeds.append(np.sign(delta_distance) * np.inf)
                else:
                    speeds.append(delta_distance / dt)
            return np.array(speeds)

    def get_gps_vel(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the velocity at which the hydrophone is traveling as measured by the GPS at timestamp(s)
        '''
        hydrophone_coords = self.get_hydrophone_coords(name, timestamp, mode=mode, ordered=ordered)
        if len(hydrophone_coords.shape) == 1:
            return np.nan
        else:
            distances = utils.get_geodesic_distance(hydrophone_coords[:, 1:], hydrophone_coords[:, :-1])
            dts = utils.total_seconds(timestamp[1:], timestamp[:-1])
            vels = [np.nan]
            for distance, dt in zip(distances, dts):
                # Handle duplicate timestamps
                if dt == 0 and distance == 0:
                    vels.append(vels[-1])
                elif dt == 0:
                    vels.append(np.inf)
                else:
                    vels.append(distance / dt)
            return np.array(vels)

    def get_gps_theta(self, name, timestamp, mode='interpolate', ordered=True):
        '''
        Get the angle off the horizontal at which the hydrophone is traveling as measured by the GPS at timestamp(s)
        '''
        hydrophone_coords = self.get_hydrophone_xy(name, timestamp, mode=mode, ordered=ordered)
        if len(hydrophone_coords.shape) == 1:
            return np.nan
        else:
            x, y = hydrophone_coords.T
            vecs = np.array([np.diff(x), np.diff(y)]).T
            thetas = utils.angle_between(utils.unit_2d(0), vecs)
            return np.concatenate([[np.nan], thetas])

    def plot(self, save=False, replace=True, show=True, **kwargs):
        def ints():
            i = 0
            while True:
                yield i
                i += 1

        i = ints()
        n = 1 + 2 * len(self.hydrophones)
        seconds = kwargs.get('plot_total_seconds', False)
        ratio = kwargs.get('ratio', 5)
        no_titles = kwargs.get('exclude_titles', False)
        width = kwargs.get('width', 3.5)

        arr = mpl.figure.figaspect((1+ratio)/ratio)
        w, h = width * arr / arr[0]
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [ratio, 1]}, figsize=(w, h))

        # Plot hydrophone coordinates and detections
        all_coords = np.zeros([2, 0])
        for name, data in self.hydrophones.items():
            # Plot total seconds or timestamps
            if seconds:
                coords_times = utils.total_seconds(data.coords.index, dataset.start_time)
                detections_times = utils.total_seconds(data.detections.index, dataset.start_time)
            else:
                coords_times = data.coords.index
                detections_times = data.detections.index

            # Add times
            ax1.scatter(coords_times, [n - next(i)]*len(data.coords))
            ax1.scatter(detections_times, [n - next(i)]*len(data.detections))

            # Plot coordinates
            x, y = data.coords['longitude'], data.coords['latitude']
            all_coords = np.concatenate([all_coords, [x, y]], axis=1)
            dx, dy = np.concatenate([np.diff(x), [0]]), np.concatenate([np.diff(y), [0]])
            l, = ax0.plot(x, y, marker='o', label='{} coords'.format(name))

            # Plot detections
            detection_lats, detection_lons = self.get_hydrophone_coords(name, data.detections.index)
            ax0.plot(detection_lons, detection_lats, marker='.', linestyle='None', label='{} detections'.format(name))
            ax0.quiver(x, y, dx, dy, units='xy', angles='xy', scale_units='xy', scale=1, width=0.00001, color=l.get_color())

        # Plot tag coordinates
        x, y = self.tag.coords['longitude'], self.tag.coords['latitude']
        all_coords = np.concatenate([all_coords, [x, y]], axis=1)
        dx, dy = np.concatenate([np.diff(x), [0]]), np.concatenate([np.diff(y), [0]])
        l, = ax0.plot(x, y, marker='.', label='Tag coords')
        ax0.quiver(x, y, dx, dy, units='xy', angles='xy', scale_units='xy', scale=1, width=0.00001, color=l.get_color())
        if seconds:  # Plot total seconds or timestamps
            tag_times = utils.total_seconds(self.tag.coords.index, dataset.start_time)
        else:
            tag_times = self.tag.coords.index
        if type(self.tag.raw) is np.ndarray:
            ax1.plot(tag_times, [n - next(i)]*2, '-o', color=l.get_color(), label='Tag coords')
        else:
            ax1.scatter(tag_times, [n - next(i)]*len(self.tag.coords), label='Tag coords')

        # Add legend
        # Reorder entries
        num_entries = 2*len(self.hydrophones)+2
        handles, labels = ax0.get_legend_handles_labels()
        actual_entries = len(handles)
        handles = np.array(handles + [None]*(num_entries-actual_entries)).reshape(-1, 2).T.flatten()[:num_entries]
        labels = np.array(labels + [None]*(num_entries-actual_entries)).reshape(-1, 2).T.flatten()[:num_entries]
        if kwargs.get('legend_outside', True):
            ax0.legend(handles, labels, loc='center left', ncol=2, bbox_to_anchor=(1, 0.5))
        else:
            ax0.legend(handles, labels, loc='upper right', ncol=2)

        # Add background map
        if self.origin is not None:
            coord_bounds = ax0.get_xlim(), ax0.get_ylim()
            print(coord_bounds)
            (west, east), (south, north) = coord_bounds
            img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir='../maps/OpenStreetMap/Mapnik')
        ax0.imshow(img, extent=ext)

        # Add titles and labels
        ax0.set_xlabel('Longitude')
        ax0.set_ylabel('Latitude')
        if not no_titles:
            ax0.set_title('Trajectories')
        if seconds:
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xlabel('Time')
        ax1.set_ylabel('Series')
        ax1.set_yticks([])
        ax1.set_yticklabels([])

        # Set axes and figure sizes
        all_x = all_coords[0]
        all_y = all_coords[1]
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]
        (minx, miny), (maxx, maxy) = utils.pad_bounds(((np.min(all_x), np.min(all_y)), (np.max(all_x), np.max(all_y))), f=1.5, square=True)
        ax0.set_xlim((minx, maxx))
        ax0.set_ylim((miny, maxy))
        ax1.set_box_aspect(1/ratio)
        fig.set_tight_layout(True)

        if not no_titles:
            fig.suptitle('{} summary'.format(self.name))
        if show:
            plt.show()
        if save:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/{n}_summary.png'.format(n=self.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath, bbox_inches='tight')
        plt.close()
    
    def plot_tof_vs_gps(self, save=True, replace=True, show=True, **kwargs):
        fig, ax_grid = plt.subplots(3, len(self.hydrophones))
        ax_grid = ax_grid.reshape(3, len(self.hydrophones))
        for i, (name, data) in enumerate(self.hydrophones.items()):
            for j, axs in enumerate(ax_grid.T):
                f = lambda x: x if i == j else None
                g = lambda x: None if i == j else x
                kwargs = {'color': g('grey'), 'linestyle': g('dotted'), 'linewidth': g(1), 'zorder': g(-1)}
                times = data.detections.index

                # Axis 0
                # Plot gps distance
                gps_distances = self.get_gps_distance(name, times)
                l, = axs[0].plot(times, gps_distances, marker=f(','), label=f('gps distance'), **kwargs)

                # Plot tof distance
                tof_distances = np.array(data.detections['total_distance'])
                axs[0].plot(times, tof_distances, label=f('tof distance'), **kwargs)

                # Plot shifted tof distance
                shifted_tof_distances = tof_distances + gps_distances[0] - tof_distances[0]
                axs[0].plot(times, shifted_tof_distances, label=f('shifted tof distance'), **kwargs)

                # Plot error adjusted tof distance
                error_slope = ((gps_distances[-1]-tof_distances[-1]) - (gps_distances[0]-tof_distances[0])) / (times[-1]-times[0]).total_seconds()
                error = lambda t: error_slope * utils.total_seconds(t, times[0]) + (gps_distances[0]-tof_distances[0])
                adjusted_tof_distances = tof_distances + error(times)
                axs[0].plot(times, adjusted_tof_distances, label=f('adjusted tof distance'), **kwargs)

                # Add legend and title
                if i == j:
                    axs[0].scatter(times, gps_distances, marker='.')
                    axs[0].scatter(self.tag.coords.index, self.get_gps_distance(name, self.tag.coords.index), marker='o', color=l.get_color())
                    axs[0].scatter(times, tof_distances, marker='.')
                    axs[0].scatter(times, shifted_tof_distances, marker='.')
                    axs[0].scatter(times, adjusted_tof_distances, marker='.')
                    axs[0].legend()
                    axs[0].set_title(name)
                    axs[0].set_ylabel('meters')

                # Axis 1
                # Plot errors
                n = len(times)
                tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - tof_distances)) / n)
                shifted_tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - shifted_tof_distances)) / n)
                adjusted_tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - adjusted_tof_distances)) / n)
                axs[1].plot(times, [0]*len(times), **kwargs)
                axs[1].plot(times, gps_distances - tof_distances, label=f('tof error, RMS={:.2f}'.format(tof_distance_rms)), **kwargs)
                axs[1].plot(times, gps_distances - shifted_tof_distances, label=f('shifted tof error, RMS={:.2f}'.format(shifted_tof_distance_rms)), **kwargs)
                axs[1].plot(times, gps_distances - adjusted_tof_distances, label=f('adjusted tof error, RMS={:.2f}'.format(adjusted_tof_distance_rms)), **kwargs)
                if i == j:
                    axs[1].scatter(times, [0]*len(times), marker='.')
                    axs[1].scatter(times, gps_distances - tof_distances, marker='.')
                    axs[1].scatter(times, gps_distances - shifted_tof_distances, marker='.')
                    axs[1].scatter(times, gps_distances - adjusted_tof_distances, marker='.')
                    axs[1].legend()
                    axs[1].set_ylabel('meters')

                # Axis 2
                # Plot gps speed
                gps_speeds = self.get_gps_speed(name, times)
                axs[2].plot(times, gps_speeds, label=f('gps speed'), **kwargs)

                # Plot tof speed
                tof_speeds = data.detections['delta_distance'] / data.detections['dt']
                axs[2].plot(times, tof_speeds, label=f('tof speed'), **kwargs)

                # Plot adjusted tof speed
                adjusted_tof_speed = np.array([0]+[d-prev_d for prev_d, d in utils.pairwise(adjusted_tof_distances)]) / data.detections['dt']
                axs[2].plot(times, adjusted_tof_speed, label=f('adjusted tof speed'), **kwargs)
                if i == j:
                    axs[2].scatter(times, gps_speeds, marker='.')
                    axs[2].scatter(self.tag.coords.index, self.get_gps_speed(name, self.tag.coords.index), marker='o', color=l.get_color())
                    axs[2].scatter(times, tof_speeds, marker='.')
                    axs[2].scatter(times, adjusted_tof_speed, marker='.')
                    axs[2].legend()
                    axs[2].set_ylabel('m/s')

        # Set common axes
        for i, axs in enumerate(ax_grid):
            bounds = np.array([ax.get_xlim() + ax.get_ylim() for ax in axs])
            x_min = np.min(bounds[:, 0])
            x_max = np.max(bounds[:, 1])
            y_min = np.min(bounds[:, 2])
            y_max = np.max(bounds[:, 3])
            for ax in axs:
                ax.set_xlim((x_min, x_max))
                ax.set_ylim((y_min, y_max))
            for ax in axs[1:]:
                ax.get_yaxis().set_visible(False)
            
            if i == len(ax_grid) - 1:
                for ax in axs:
                    ax.set_xlabel('Time')

        if not kwargs.get('exclude_titles', False):
            fig.suptitle('{} tof vs gps'.format(self.name))
        fig.subplots_adjust(wspace=0, hspace=0)
        if show:
            plt.show()
        if save:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/{n}_tof_vs_gps.png'.format(n=self.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        plt.close()

    def plot_range_error(self, name, save=False, replace=True, show=True, **kwargs):
        # Get parameters
        no_titles = kwargs.get('exclude_titles', False)
        width = kwargs.get('width', 3.5)
        ratio = kwargs.get('ratio', 0.8)
        seconds = kwargs.get('plot_total_seconds', False)

        # Set up figure and axes
        arr = mpl.figure.figaspect(ratio)
        w, h = width * arr / arr[0]
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(w, h), sharex=True)

        # Get data
        data = self.hydrophones[name]
        times = data.detections.index
        if seconds:
            x = utils.total_seconds(times, times[0])
        else:
            x = times

        # Axis 0
        # Plot gps distance
        gps_distances = self.get_gps_distance(name, times)
        l, = ax0.plot(x, gps_distances, marker=',', label='GPS distance')

        # Plot tof distance
        tof_distances = np.array(data.detections['total_distance'])
        ax0.plot(x, tof_distances, label='TOF distance')

        # Plot shifted tof distance
        shifted_tof_distances = tof_distances + gps_distances[0] - tof_distances[0]
        ax0.plot(x, shifted_tof_distances, label='Shifted TOF distance')

        # Plot error adjusted tof distance
        error_slope = ((gps_distances[-1]-tof_distances[-1]) - (gps_distances[0]-tof_distances[0])) / (times[-1]-times[0]).total_seconds()
        error = lambda t: error_slope * utils.total_seconds(t, times[0]) + (gps_distances[0]-tof_distances[0])
        adjusted_tof_distances = tof_distances + error(times)
        ax0.plot(x, adjusted_tof_distances, label='Adjusted TOF distance')

        # Add legend and title
        ax0.scatter(x, gps_distances, marker='.')
        if seconds:
            ax0.scatter(utils.total_seconds(self.tag.coords.index, times[0]), self.get_gps_distance(name, self.tag.coords.index), marker='o', color=l.get_color())
        else:
            ax0.scatter(self.tag.coords.index, self.get_gps_distance(name, self.tag.coords.index), marker='o', color=l.get_color())
        ax0.scatter(x, tof_distances, marker='.')
        ax0.scatter(x, shifted_tof_distances, marker='.')
        ax0.scatter(x, adjusted_tof_distances, marker='.')
        ax0.legend(loc='upper right')
        if not no_titles:
            ax0.set_title('Hydrophone {}'.format(name))
        ax0.set_ylabel('Distance (m)')
        ax0.set_title('(a)', loc='left', fontsize='medium')

        # Axis 1
        # Plot errors
        n = len(times)
        tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - tof_distances)) / n)
        shifted_tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - shifted_tof_distances)) / n)
        adjusted_tof_distance_rms = np.sqrt(np.sum(np.square(gps_distances - adjusted_tof_distances)) / n)
        ax1.plot(x, [0]*len(times))
        if kwargs.get('error_values', False):
            ax1.plot(x, gps_distances - tof_distances, label='TOF error, RMS={:.2f}'.format(tof_distance_rms))
            ax1.plot(x, gps_distances - shifted_tof_distances, label='Shifted TOF error, RMS={:.2f}'.format(shifted_tof_distance_rms))
            ax1.plot(x, gps_distances - adjusted_tof_distances, label='Adjusted TOF error, RMS={:.2f}'.format(adjusted_tof_distance_rms))
        else:
            ax1.plot(x, gps_distances - tof_distances, label='TOF error'.format(tof_distance_rms))
            ax1.plot(x, gps_distances - shifted_tof_distances, label='Shifted TOF error'.format(shifted_tof_distance_rms))
            ax1.plot(x, gps_distances - adjusted_tof_distances, label='Adjusted TOF error'.format(adjusted_tof_distance_rms))
        ax1.scatter(x, [0]*len(times), marker='.')
        ax1.scatter(x, gps_distances - tof_distances, marker='.')
        ax1.scatter(x, gps_distances - shifted_tof_distances, marker='.')
        ax1.scatter(x, gps_distances - adjusted_tof_distances, marker='.')
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Error (m)')
        if seconds:
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xlabel('Time')
        ax1.set_title('(b)', loc='left', fontsize='medium')

        if not no_titles:
            fig.suptitle('{} tof vs gps'.format(self.name))
        fig.subplots_adjust(wspace=0, hspace=0)
        if show:
            plt.show()
        if save:
            savepath = utils.add_version(
                kwargs.get('savepath', '../datasets/{n}/{n}_tof_vs_gps.png'.format(n=self.name)),
                replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        plt.close()

if __name__ == '__main__':
    dataset = Dataset('tag78_static_two_clusters')
    dataset.plot(save=True, replace=True, exclude_titles=True, plot_total_seconds=True, ratio=7, width=7, legend_outside=False)
    dataset.plot_tof_vs_gps(save=True, replace=True)

    # dataset = Dataset('tag78_shore_2_boat_all_static_test_0')
    # dataset = Dataset('tag78_50m_increment_long_beach_test_0', end_time=datetime.fromisoformat('2022-06-08 09:56'))
    # dataset.plot(save=True, replace=True, exclude_titles=True, plot_total_seconds=True, savepath='../paper/fig6.png', ratio=7, width=7, legend_outside=False)
    # dataset.plot_tof_vs_gps(save=False)

    # dataset.plot_range_error(457049, show=True, save=False, exclude_titles=True, width=10, plot_total_seconds=True, savepath='../paper/fig_stairs.png')