import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import movingpandas as mpd
import utils
import bisect
import datetime as datetime
import pandas as pd

def plot_df(pf, df, save_to=None, plot_avg=True, msg = '', bbox=None, padding=1.1, show=True, square=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    avg_particle = np.average(pf._history, axis=1)

    num_particles = pf._num_particles
    num_steps = len(pf._history)

    particles, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='particles')

    # Create relevant trajectories

    origin = (10.9262055, -85.7966545)
    # origin = (10.92378733, -85.79437267) # for the 
    groundtruth_traj = mpd.Trajectory(df.set_index('datetime'), 'tag_traj', x='tag_longitude', y='tag_latitude')
    hydrophone_trajs = {}
    for serial_no in df['serial_no'].unique():
        hydrophone_trajs[serial_no] = mpd.Trajectory(df[df['serial_no'] == serial_no].set_index('datetime'), '{}_traj'.format(serial_no), x='longitude', y='latitude')

    # Download map tiles for the background
    cartesian_bounds = np.array(bbox).reshape(2, 2).T
    cartesian_bounds = utils.pad_bounds(cartesian_bounds.T, f=2).T
    if origin is not None:
        coord_bounds = utils.to_coords(cartesian_bounds, origin)
        (south, west), (north, east) = coord_bounds
        # img, ext = utils.bounds2img(west, south, east, north, zoom=17, map_dir='../maps/OpenStreetMap/Mapnik')
        # true_ext = utils.to_cartesian(np.flip(np.array(ext).reshape(2, 2), axis=0), origin).T.flatten()
    # background = ax.imshow(img, extent=true_ext)
    background = ax.imshow([[]])

    groundtruth_path_x = []
    groundtruth_path_y = []
    groundtruth_path, = ax.plot([], [], 'bo', label='true path')

    best_particle_path_x = []
    best_particle_path_y = []
    best_particle_path, = ax.plot([], [], 'r-', label='est path')
    best_particle_path_2, = ax.plot([], [], linestyle='None', marker='.', color='red')

    hydrophones = {}
    hydrophone_circles = {}
    for serial_no in hydrophone_trajs.keys():
        hydrophones[serial_no], = ax.plot([], [], linestyle='None', marker='o', label=str(serial_no))
        hydrophone_circles[serial_no] = mpl.patches.Circle((0, 0), 0, fill=False, linewidth=1)
        ax.add_patch(hydrophone_circles[serial_no])

    steps = ax.text(3, 6, "Step = 0 / " + str(num_steps), horizontalalignment="center", verticalalignment="top")
    ax.legend()

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
        nonlocal msg
        print(msg)
        if msg != '':
            msg += '   '
        ax.set_title(msg + "Num steps: " + str(num_steps) + ", Num particle: " + str(num_particles))
        return artists

    def update(frame):
        print('Frame:', frame, avg_particle[frame, 0], avg_particle[frame, 1])
        print('Path length:', len(best_particle_path_x))

        if frame == 0:
            groundtruth_path_x.clear()
            groundtruth_path_y.clear()
            best_particle_path_x.clear()
            best_particle_path_y.clear()

        # Plot best particle path
        if plot_avg:
            best_particle_path_x.append(avg_particle[frame, 0])
            best_particle_path_y.append(avg_particle[frame, 1])
            best_particle_path.set_data(best_particle_path_x, best_particle_path_y)
            best_particle_path_2.set_data(best_particle_path_x, best_particle_path_y)

        # Plot groundtruth path
        pos = groundtruth_traj.get_position_at(pf._time_history[frame])
        pos = utils.to_cartesian((pos.y, pos.x), origin)
        groundtruth_path_x.append(pos[0])
        groundtruth_path_y.append(pos[1])
        groundtruth_path.set_data(groundtruth_path_x, groundtruth_path_y) # can we set them directly> groundtruth_path[:t,0]

        # Plot other particles poses
        particles.set_data(pf._history[frame, :, 0], pf._history[frame, :, 1])

        # Plot hydrophones
        for serial_no, hydrophone in hydrophones.items():
            hydrophone_traj = hydrophone_trajs[serial_no]
            curr_time = pf._time_history[frame]
            start_time = curr_time - pd.Timedelta(seconds=8)
            end_time = curr_time + pd.Timedelta(seconds=8)

            pos = hydrophone_traj.get_position_at(curr_time) #interpolation is default
            pos = utils.to_cartesian((pos.y, pos.x), origin)

            print("position")
            print(type(pos), pos) # numpy array (2x1) []

            hydrophone.set_data([pos[0]], [pos[1]])
            idx = bisect.bisect(hydrophone_traj.df.index, curr_time) - 1 

            print("idx", idx)
            print(hydrophone_traj.df.index, type(hydrophone_traj.df.index))
            print(curr_time, type(curr_time))

            # hydrophone_traj.df.index is DatetimeIndex
            # curr_time is timestamps pandas

            time_range_array = (start_time <= hydrophone_traj.df.index) & (hydrophone_traj.df.index <=end_time)
            print(time_range_array)
            
            # if the datetime is within 8 seconds of the hydrophone reading
            if any(time_range_array): 
                r = hydrophone_traj.df.iloc[idx]['gps_distance']
                hydrophone_circles[serial_no].set(center=pos, radius=r)
            else: 
                r = 0
                pos = np.array([0,0])
                hydrophone_circles[serial_no].set(center=pos, radius=r)

            print(r, type(r))
            print(hydrophone_circles, type(hydrophone_circles))

        # hydrophone1_x = pf._hydrophone_state_history[frame, 0]
        # hydrophone1_y = pf._hydrophone_state_history[frame, 1]
        # hydrophone2_x = pf._hydrophone_state_history[frame, 4]
        # hydrophone2_y = pf._hydrophone_state_history[frame, 5]
        # hydrophone1.set_data([hydrophone1_x], [hydrophone1_y])
        # hydrophone2.set_data([hydrophone2_x], [hydrophone2_y])
        # hydrophone1_range.set(center=(hydrophone1_x, hydrophone1_y), radius=pf._kf1_history[frame, 0])
        # hydrophone2_range.set(center=(hydrophone2_x, hydrophone2_y), radius=pf._kf2_history[frame, 0])

        # Update title
        if frame/num_steps > .9:
            steps.set_text('')
        else:
            steps.set_text("Step = " + str(frame) + " / " + str(num_steps))

        # Return changed artists?
        return artists

    anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, 10), init_func = init, blit=False, interval = 33, repeat=True)
    
    # Get bounding box or use default values
    if bbox:
        minx, miny, maxx, maxy = bbox
    else:
        minx = 3000
        maxx = 5000
        miny = 5000
        maxy = 6500

    # Add padding to the bounds
    midx = (maxx + minx) / 2
    midy = (maxy + miny) / 2
    deltax = (maxx - minx) * padding
    deltay = (maxy - miny) * padding
    if square:
        deltax = max(deltax, deltay)
        deltay = deltax
    minx = midx - deltax / 2
    maxx = midx + deltax / 2
    miny = midy - deltay / 2
    maxy = midy + deltay / 2
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])

    if show:
        plt.show()

    if save_to is not None:
        print('Saving animation to {}...'.format(save_to))
        f = save_to
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writer=writergif)
        print('Animation Saved!')

def data_to_axis_units(x_data, fig, ax):
    ppd = 72 / fig.dpi
    trans = ax.transData.transform
    return ((trans((1, x_data)) - trans((0, 0))) * ppd)[1]