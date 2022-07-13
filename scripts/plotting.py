import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from filter import Filter

def plot_df(pf, groundtruth_path_data, save_to=None, plot_avg=True, msg = '', bbox=None, padding=1.1, show=True, square=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    avg_particle = np.average(pf._history, axis=1)

    num_particles = pf._num_particles
    num_steps = len(pf._history)

    particles, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='particles')

    groundtruth_path_x = []
    groundtruth_path_y = []
    groundtruth_path, = ax.plot([], [], 'bo', label='true path')

    measurements_x = []
    measurements_y = []
    measurements, = ax.plot([], [], linestyle='None', marker='o', color='darkorange', label='measurements')

    best_particle_path_x = []
    best_particle_path_y = []
    best_particle_path, = ax.plot([], [], 'r-', label='est path')
    best_particle_path_2, = ax.plot([], [], linestyle='None', marker='.', color='red')

    hydrophone1, = ax.plot([], [], linestyle='None', marker='o', label='hydrophone 1')
    hydrophone2, = ax.plot([], [], linestyle='None', marker='o', label='hydrophone 2')
    hydrophone1_range, = ax.plot([], [])
    hydrophone2_range, = ax.plot([], [])

    steps = ax.text(3, 6, "Step = 0 / " + str(num_steps), horizontalalignment="center", verticalalignment="top")
    ax.legend()

    # Artists indexed later are drawn over ones indexed earlier
    artists = [
        hydrophone1_range,
        hydrophone2_range,
        hydrophone1,
        hydrophone2,
        best_particle_path_2,
        best_particle_path,
        measurements,
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
            measurements_x.clear()
            measurements_y.clear()

        # Plot best particle path
        if plot_avg:
            best_particle_path_x.append(avg_particle[frame, 0])
            best_particle_path_y.append(avg_particle[frame, 1])
            best_particle_path.set_data(best_particle_path_x, best_particle_path_y)
            best_particle_path_2.set_data(best_particle_path_x, best_particle_path_y)

        # Plot groundtruth path
        groundtruth_path_x.append(groundtruth_path_data[frame, 0])
        groundtruth_path_y.append(groundtruth_path_data[frame, 1])
        groundtruth_path.set_data(groundtruth_path_x, groundtruth_path_y) # can we set them directly> groundtruth_path[:t,0]

        # Plot measurements
        if groundtruth_path_data[frame, 5] == 1:
            measurements_x.append(groundtruth_path_data[frame, 0])
            measurements_y.append(groundtruth_path_data[frame, 1])
            measurements.set_data(measurements_x, measurements_y)

        # Plot other particles poses
        particles.set_data(pf._history[frame, :, 0], pf._history[frame, :, 1])

        # Plot hydrophones
        hydrophone1_x = pf._hydrophone_state_history[frame, 0]
        hydrophone1_y = pf._hydrophone_state_history[frame, 1]
        hydrophone2_x = pf._hydrophone_state_history[frame, 4]
        hydrophone2_y = pf._hydrophone_state_history[frame, 5]
        hydrophone1.set_data([hydrophone1_x], [hydrophone1_y])
        hydrophone2.set_data([hydrophone2_x], [hydrophone2_y])
        hydrophone1_range.set(center=(hydrophone1_x, hydrophone1_y), radius=pf._kf1_history[frame, 0])
        hydrophone2_range.set(center=(hydrophone2_x, hydrophone2_y), radius=pf._kf2_history[frame, 0])

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

def ring_coords(center, inner, outer, cos=None, sin=None):
    if cos is None or sin is None:
        theta = np.linspace(-np.pi, np.pi, 100)
        cos = np.cos(theta)
        sin = np.sin(theta)