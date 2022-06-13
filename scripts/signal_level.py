import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from common import imerge, fit_line, find_files, avg_dt_dict, get_delta_tof, add_version, get_savepath, savefig

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

# Create arrays to hold the data extracted
distances = []
signals = []
lengths = []
sources = []
datapaths = []

# Define some information about the tags in case distances need to be calculated
tag_id = 65478

# Loop through the files found
files = imerge(
    find_files('../data/06-08-2022', name=r'.*increment.*457049_0'))
    # find_files('../data/06-01-2022', name=r'.*manual.*'))
    # find_files('../data/05-26-2022', name=r'tag78.*', extension=r'\.csv'),
    # find_files('../data/06-01-2022', extension=r'\.csv'))
    # find_files('../data/05-31-2022', '../data/05-27-2022', extension=r'\.csv'))
    # find_files('../../icex-lair-2021', name=r'data_[\d]+', extension=r'\.csv'))
for i, dir_entry in enumerate(files):
    try:
        source = os.path.splitext(dir_entry.name)[0]
        data = pd.read_csv(dir_entry.path)

        # Retrieve distance measurements
        if 'gps_distance' in data.columns:
            print('{}: using gps_distance'.format(source))
            distance_subset = np.array(data['gps_distance'])
        elif 'absolute_distance' in data.columns:
            print('{}: using absolute_distance'.format(source))
            distance_subset = np.array(data['absolute_distance'])
        elif 'Distance (m)' in data.columns:
            print('{}: using Distance (m)'.format(source))
            distance_subset = np.array(data['Distance (m)'])
        elif 'total_distance' in data.columns:
            print('{}: using total_distance'.format(source))
            distance_subset = np.array(data['total_distance'])
        else:
            print('{}: calculating distance'.format(source))
            # Calculate distances from timestamps
            avg_dt = avg_dt_dict[tag_id]
            data['Date/Time'] = pd.to_datetime(data['Date/Time'])
            timestamps = np.array([timestamp for a_tag_id, timestamp in zip(data['Transmitter ID Number'], data['Date/Time']) if a_tag_id == tag_id])
            delta_tof = get_delta_tof(timestamps, avg_dt)
            distance_subset = np.array([0.0] + np.cumsum(delta_tof * 1460).tolist())

        # Retrieve signal level measurements
        if 'signal_level' in data.columns:
            signal_subset = np.array(data['signal_level'])
        else:
            signal_subset = np.array(data['Noise-Level (dB)'])
        
        # Filter out nan distances
        # signal_subset = np.array([signal for distance, signal in zip(distance_subset, signal_subset) if not np.isnan(distance)])
        signal_subset = signal_subset[~np.isnan(distance_subset)]
        distance_subset = distance_subset[~np.isnan(distance_subset)]

        # Append data
        distances += distance_subset.tolist()
        signals += signal_subset.tolist()
        lengths.append(len(distance_subset))
        sources.append(source)
        datapaths.append(dir_entry.path)
    except Exception as err:
        print(err)
distances = np.array(distances)
signals = np.array(signals)

fig, ax = plt.subplots()

# Fit a line to all of the data
m, b = fit_line(distances, signals)
r_sqr = np.corrcoef(distances, signals)[0][1] ** 2
ax.plot(distances, m * distances + b, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr), color='black')

# Plot the points for the csv and fit a line to them
# cmap = plt.get_cmap('jet')
data = []
for i, (length, acc, source, datapath) in enumerate(zip(lengths, np.cumsum(lengths), sources, datapaths)):
    distance_subset = distances[acc-length:acc]
    signal_subset = signals[acc-length:acc]
    data.append((datapath, source, distance_subset, signal_subset))
    m, b = fit_line(distance_subset, signal_subset)
    r_sqr = np.corrcoef(distance_subset, signal_subset)[0][1] ** 2
    # color = cmap(float(i) / len(lengths))
    ax.scatter(distance_subset, signal_subset, label=source)
    ax.plot(distance_subset, m * distance_subset + b, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr))
    # if i < 14:
    #     ax.scatter(distance_subset, signal_subset, color=color, label=source)
    #     ax.plot(distance_subset, m * distance_subset + b, color=color, label='m={:.6f}, b={:.6f}, R^2={:.6f}'.format(m, b, r_sqr))
    # else:
    #     ax.scatter(distance_subset, signal_subset, color=color)
    #     ax.plot(distance_subset, m * distance_subset + b, color=color)

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

savepath = add_version('../plots/06-08-2022/signal_plot.png')
print('Saving to {}'.format(savepath))
savefig(fig, savepath)

for datapath, source, distance_subset, signal_subset in data:
    dt_distance = np.diff(distance_subset)
    mask = np.array([True] + (dt_distance >= 0).tolist())
    plt.scatter(distance_subset[mask], signal_subset[mask], label='Moving away {}'.format(np.sum(mask)))
    plt.scatter(distance_subset[~mask], signal_subset[~mask], label='Moving towards {}'.format(np.sum(~mask)))
    plt.xlabel('Distance (m)')
    plt.ylabel('Signal level (dB)')
    plt.title(source)
    plt.legend()
    fig = plt.gcf()
    plt.show()

    savepath = get_savepath(datapath, '_signal_direction')
    print('Saving to {}'.format(savepath))
    savefig(fig, savepath)