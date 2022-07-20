import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import utils

def scrollable_legend(fig, legend):
    '''From https://stackoverflow.com/questions/55863590/adding-scroll-button-to-matlibplot-axes-legend'''
    # pixels to scroll per mousewheel event
    d = {"down" : 30, "up" : -30}

    def func(evt):
        if legend.contains(evt):
            bbox = legend.get_bbox_to_anchor()
            bbox = Bbox.from_bounds(bbox.x0, bbox.y0+d[evt.button], bbox.width, bbox.height)
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", func)

replace = False
save = True

# Create new dataframe to hold merged data
data = pd.DataFrame([], columns=['source', 'path', 'total_dt', 'signal', 'distance', 'bearing', 'logged_speed', 'gps_speed', 'noise'])

# Define some information about the tags in case distances need to be calculated
tag_id = 65478

# Loop through the files found
files = utils.imerge(
    utils.find_files('../data/07-18-2022', name=r'tag78(?!_swimming)(?!_shore_2_boat_all_static_test_\w{5,6}_0).*VR100.*', extension=r'\.csv'))
    # utils.find_files('../data/07-18-2022', name=r'tag78(?!_swimming).*', extension=r'\.csv'))
    # utils.find_files('../data/06-29-2022', name=r'.*457012.*', extension=r'\.csv'))
    # utils.find_files('../data/06-08-2022', name=r'.*(?:increment|none).*457012_0'))
    # utils.find_files('../data/06-01-2022', name=r'.*457012.*'))
    # utils.find_files('../data/06-01-2022', name=r'.*manual.*'))
    # utils.find_files('../data/05-26-2022', name=r'tag78.*', extension=r'\.csv'),
    # utils.find_files('../data/06-01-2022', extension=r'\.csv'))
    # utils.find_files('../data/05-31-2022', '../data/05-27-2022', extension=r'\.csv'))
    # utils.find_files('../../icex-lair-2021', name=r'data_[\d]+', extension=r'\.csv'))
save_to = '../plots/07-18-2022'

def get_df_column(df, name):
    return np.array(df.get(name, [np.nan] * len(df)))

for i, dir_entry in enumerate(files):
    try:
        path = os.path.normpath(dir_entry.path)
        name = os.path.splitext(dir_entry.name)[0]
        df = pd.read_csv(path)

        # Retrieve signal level measurements
        if 'signal_level' in df.columns:
            signals = np.array(df['signal_level'])
        else:
            signals = np.array(df['Noise-Level (dB)'])

        # Retrieve distance measurements
        if 'gps_distance' in df.columns:
            print('{}: using gps_distance'.format(name))
            distances = np.array(df['gps_distance'])
        elif 'absolute_distance' in df.columns:
            print('{}: using absolute_distance'.format(name))
            distances = np.array(df['absolute_distance'])
        elif 'Distance (m)' in df.columns:
            print('{}: using Distance (m)'.format(name))
            distances = np.array(df['Distance (m)'])
        elif 'total_distance' in df.columns:
            print('{}: using total_distance'.format(name))
            distances = np.array(df['total_distance'])
        else:
            print('{}: calculating distance'.format(name))
            # Calculate distances from timestamps
            avg_dt = utils.avg_dt_dict[tag_id]
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            timestamps = np.array([timestamp for a_tag_id, timestamp in zip(df['Transmitter ID Number'], df['Date/Time']) if a_tag_id == tag_id])
            delta_tof = utils.get_delta_tof(timestamps, avg_dt)
            distances = np.array([0.0] + np.cumsum(delta_tof * 1460).tolist())

        # Append data
        # Columns: ['source', 'path', 'signal', 'distance', 'bearing', 'logged_speed', 'gps_speed', 'noise']
        partial_data = pd.DataFrame(np.array([
            [i] * len(df),
            [path] * len(df),
            get_df_column(df, 'total_dt'),
            signals,
            distances,
            get_df_column(df, 'relative_tag_bearing'),
            get_df_column(df, 'logged_speed'),
            get_df_column(df, 'gps_speed'),
            get_df_column(df, 'noise_level'),
        ], dtype=object).T, columns=data.columns)
        data = pd.concat([data, partial_data])
    except Exception as err:
        print(err)

# Set dtypes
for col_name in ['source', 'total_dt', 'signal', 'distance', 'bearing', 'logged_speed', 'gps_speed', 'noise']:
    data[col_name] = pd.to_numeric(data[col_name])
data.set_index('source', inplace=True)

fig, ax = plt.subplots()

# Fit a line to all of the data
isnan = np.logical_or(np.isnan(data['distance']), np.isnan(data['signal']))
distances = np.array(data['distance'])[~isnan]
signals = np.array(data['signal'])[~isnan]
m, b = utils.fit_line(distances, signals)
r_sqr = np.corrcoef(distances, signals)[0][1] ** 2
ax.plot(distances, m * distances + b, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr), color='black')

# Plot the points for the csv and fit a line to them
# cmap = plt.get_cmap('jet')
for i in data.index.unique():
    # color = cmap(float(i) / len(lengths))
    source_data = data.loc[i]
    name = os.path.splitext(os.path.split(source_data['path'].iloc[0])[1])[0]
    isnan = np.logical_or(np.isnan(source_data['distance']), np.isnan(source_data['signal']))
    distance_subset = np.array(source_data['distance'])[~isnan]
    signal_subset = np.array(source_data['signal'])[~isnan]
    m, b = utils.fit_line(distance_subset, signal_subset)
    r_sqr = np.corrcoef(distance_subset, signal_subset)[0][1] ** 2
    ax.scatter(distance_subset, signal_subset, label=name)
    ax.plot(distance_subset, m * distance_subset + b, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr))

# Label axes
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Signal Level (dB)')
ax.set_title('Signal = m * distance + b')

# Shrink current axis by 20%
shrink_by = 0.3  # %
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * (1 - shrink_by), box.height])

# Set the legend outside of the plot
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# scrollable_legend(fig, legend)
fig = plt.gcf()
plt.show()

# Save plot
if save:
    savepath = utils.add_version(os.path.join(save_to, 'signal_plot.png'), replace=replace)
    print('Saving to {}'.format(savepath))
    utils.savefig(fig, savepath)
plt.close()

# For each source try out different linear models to predict the signal strength
# column_sets = [
#     ['distance'],
#     ['distance', 'bearing'],
#     ['distance', 'gps_speed'],
#     ['distance', 'logged_speed'],
#     ['distance', 'gps_speed', 'logged_speed'],
#     ['distance', 'bearing', 'gps_speed'],
#     ['distance', 'bearing', 'logged_speed'],
#     ['distance', 'bearing', 'gps_speed', 'logged_speed'],
# ]
column_sets = [
    ['distance']
]
for columns in column_sets:
    x = None
    for source_data in utils.imerge([data], map(lambda i: data.loc[i], data.index.unique())):
        is_full_dataset = data is source_data
        if is_full_dataset:
            name = 'All data'
        else:
            path = source_data['path'].iloc[0]
            name = os.path.splitext(os.path.split(path)[1])[0]

        # Dependent variable
        signals = np.array(source_data['signal'])

        # Explanatory variables
        explanatory_vars = source_data[columns].to_numpy()

        # Find nan values
        isnan = np.apply_along_axis(
            np.logical_or.reduce,
            1,
            np.concatenate([np.isnan(explanatory_vars), np.isnan(signals.reshape((-1, 1)))], axis=1))
        
        A = np.concatenate([explanatory_vars, np.ones([len(explanatory_vars), 1])], axis=1)
        if x is None:
            x = np.linalg.lstsq(A[~isnan], signals[~isnan], rcond=None)[0]
        
        predicted_signals = A @ x
        diff = signals - predicted_signals

        r_sqr = 1 - np.var(diff[~isnan]) / np.var(signals[~isnan])
        error = np.sqrt(np.sum(np.square(diff[~isnan])))

        # Plot signals, predicted signals, error, etc.
        fig, axd = plt.subplot_mosaic([['left', 'left', 'right']])
        ax0, ax1 = axd['left'], axd['right']
        ax0.plot(source_data['total_dt'], signals, label='Signal level')
        ax0.plot(source_data['total_dt'], predicted_signals, label='Predicted signal level. R^2={:.6f}, error={:.6f}\nx={}'.format(r_sqr, error, x))
        ax0_twin = ax0.twinx()
        ax0_twin.plot(source_data['total_dt'], diff, color='#2ca02c', label='Error: mean={:.6f}, var={:.6f}'.format(np.mean(diff[~isnan]), np.var(diff[~isnan])))
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Signal level (dB)')
        ax0_twin.set_ylabel('Error (dB)')
        ax0.legend()
        ax0_twin.legend()
        ax0.set_title('Signal vs. time')
        ax1.hist(diff, label='Error: mean={:.6f}, var={:.6f}'.format(np.mean(diff[~isnan]), np.var(diff[~isnan])))
        ax1.set_xlabel('Error (dB)')
        ax1.set_ylabel('Counts')
        ax1.legend()
        ax1.set_title('Error histogram')
        print('Model: {}. R^2={:.6f}, error={:.6f}, x={}'.format(columns, r_sqr, error, x))
        fig.suptitle('{}\nSignal model: {}'.format(name, columns))
        fig.set_tight_layout(tight=True)
        plt.show()

        # Save plot
        if save:
            if is_full_dataset:
                savepath = utils.add_version(os.path.join(save_to, 'all_data_model_{}.png'.format('_'.join(columns))), replace=replace)
            else:
                savepath = utils.get_savepath(path, '_model_{}'.format('_'.join(columns)), replace=replace)
            print('Saving to {}'.format(savepath))
            utils.savefig(fig, savepath)
        plt.close()
        
# for datapath, source, distance_subset, signal_subset in data:
#     dt_distance = np.diff(distance_subset)
#     mask = np.array([True] + (dt_distance >= 0).tolist())
#     plt.scatter(distance_subset[mask], signal_subset[mask], label='Moving away {}'.format(np.sum(mask)))
#     plt.scatter(distance_subset[~mask], signal_subset[~mask], label='Moving towards {}'.format(np.sum(~mask)))
#     plt.xlabel('Distance (m)')
#     plt.ylabel('Signal level (dB)')
#     plt.title(source)
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()

#     savepath = utils.get_savepath(datapath, '_signal_direction', replace=replace)
#     print('Saving to {}'.format(savepath))
#     utils.savefig(fig, savepath)