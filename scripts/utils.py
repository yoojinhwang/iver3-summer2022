import os
import numpy as np
import re
import itertools
from functools import partial

avg_dt_dict = {65477: 8.179110, 65478: 8.179071, 65479: 7.958926}

def find_files(*sources, name=None, extension=None):
    '''
    Recursively retrieve files from the given source directories whose names and/or extensions (full)match the given patterns.
    name: string or regex pattern
    extension: string or regex pattern
    Returns a DirEntry generator
    '''
    # Compile regexes if needed
    if name is None:
        name = re.compile(r'.*')
    elif type(name) is not re.Pattern:
        name = re.compile(name)
    if extension is None:
        extension = re.compile(r'.*')
    elif type(extension) is not re.Pattern:
        extension = re.compile(extension)

    # Keep track of the sources already scanned and the files already found
    memo_table = {}

    def find_files_helper(*sources):
        # Search through each source directoty
        for source in sources:
            # Get all of the contents of the source directory and search them
            entries = os.scandir(source)
            for entry in entries:
                # Check if the entry has already been scanned or matched
                normed = os.path.normpath(entry.path)
                if normed not in memo_table:
                    memo_table[normed] = True
                    # If the current entry is itself a directory, search it recursively
                    if entry.is_dir():
                        yield from find_files_helper(entry)

                    # Otherwise yield entries whose name matches the name pattern and whose extension matches the extension pattern
                    else:
                        # Return only entries that have not already been found
                        filename, fileext = os.path.splitext(entry.name)
                        if name.fullmatch(filename) is not None and \
                        extension.fullmatch(fileext) is not None:
                            yield entry
            entries.close()
    return find_files_helper(*sources)

def fit_slope(x, y):
    '''Fits a line with a y-intercept of 0, i.e. it fits a slope.'''
    return np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)[0]

def fit_line(x, y):
    '''Fits a line, i.e. slope and y-intercept'''
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]

def imerge(*its):
    '''Merge iterators end to end'''
    for it in its:
        for i in it:
            yield i

def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...'''
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_delta_tof(timestamps, avg_dt):
    '''Compute delta time of flights from a series of timestamps and an average dt'''
    total_dt = [dt.total_seconds() for dt in (timestamps - timestamps[0])]
    dt = np.diff(total_dt)
    return (dt - avg_dt / 2) % avg_dt - avg_dt / 2

def iir_filter(x, ff=1):
    # First-order IIR filter (infinite impulse response)
    # Essentially acts as a low pass filter with a time constant dependent on
    # the sampling period and the coefficient used (ff):
    # http://www.tsdconseil.fr/tutos/tuto-iir1-en.pdf
    y = [x[0]]
    for i, x_i in enumerate(x[1:]):
        y.append((1 - ff) * y[i] + ff * x_i)
    return np.array(y)

def mkdir(path):
    '''Create a directory if it does not exist'''
    if not os.path.isdir(path):
        parent, _ = os.path.split(path)
        mkdir(parent)
        os.mkdir(path)

def add_version(savepath, replace=False):
    '''Appends a version number to the end of a path if that path already exists with the given name and if replace is False'''
    savefile, saveext = os.path.splitext(savepath)
    if replace or not os.path.exists(savepath):
        return savepath
    else:
        version = 1
        new_savepath = '{} ({}){}'.format(savefile, version, saveext)
        while os.path.exists(new_savepath):
            version += 1
            new_savepath = '{} ({}){}'.format(savefile, version, saveext)
        return new_savepath

def savefig(fig, savepath, *args, **kwargs):
    '''Save a given figure to the given path, creating a directory as needed and making sure not to overwrite files'''
    savedir, _ = os.path.split(savepath)
    mkdir(savedir)
    fig.savefig(savepath, *args, **kwargs)

def split_path(path):
    '''Split a path into all of its components (i.e. at the slashes)'''
    components = []
    first, last = os.path.split(path)
    while first != '':
        components = [last] + components
        first, last = os.path.split(first)
    return [last] + components

def get_savepath(datapath, msg, replace=False):
    components = split_path(datapath)
    for i, component in enumerate(components):
        if component == 'data':
            components[i] = 'plots'
            break
    components[-1] = os.path.splitext(components[-1])[0]
    return add_version('{}{}.png'.format(os.path.join(*components), msg), replace=replace)

def to_cartesian(coords, ref):
    '''
    Gives the x and y difference (in meters) from the given reference coordinates to the first set of coordinates.
    x is aligned to north, y is aligned to east
    '''
    R = 6371009  # Radius of the earth in meters
    coords = np.radians(coords)
    ref = np.radians(ref)

    # Formula, where lat1, lon1 = ref and lat2, lon2 = coords
    # delta_x = R * (lon2 - lon1) * np.cos(lat1)
    # delta_y = R * (lat2 - lat1)

    delta_x = R * (coords[1] - ref[1]) * np.cos(ref[0])
    delta_y = R * (coords[0] - ref[0])
    return np.array([delta_x, delta_y]).T

# def angle_between(ref, vec):
#     '''
#     Calculate the angle between two vectors or list of vectors placed tail to tail in radians.

#     Returns angles between -pi and pi.

#     Use the right hand rule to determine the sign, where ref is the index finger.

#     Can probably handle any dimensional vectors, although I've only tested it on 2d vectors and
#     some special case 3d vectors.

#     Ref and vec must either be a 1d array of coordinates, or a 2d array where the rows give the
#     coordinates in each dimension. In both cases, they must each have the same number of dimensions.
#     If both are 2d arrays, they must each have the same number of vectors/
#     '''
#     # Reshape any 2d arrays. Does not change 1d arrays
#     ref = np.stack(ref, axis=-1)
#     vec = np.stack(vec, axis=-1)

#     # Reshape vectors as needed, repeating if one is a 1d array but the other is 2d
#     if len(ref.shape) == len(vec.shape) == 1:
#         ref = ref.reshape(-1, 1).T
#         vec = vec.reshape(-1, 1).T
#     elif len(ref.shape) == 1:
#         ref = np.repeat(ref.reshape(-1, 1).T, vec.shape[0], axis=0)
#     elif len(vec.shape) == 1:
#         vec = np.repeat(vec.reshape(-1, 1).T, ref.shape[0], axis=0)
#     elif ref.shape[0] != vec.shape[0]:
#         raise ValueError('Both inputs must have the same number of vectors')

#     # Get the dimension of the vectors
#     dim = ref.shape[1]

#     # Compute the magnitude of each vector
#     get_mag = lambda x: np.sqrt(x.dot(x))
#     ref_mag = np.apply_along_axis(get_mag, -1, ref)
#     vec_mag = np.apply_along_axis(get_mag, -1, vec)

#     # Stack each pair of vectors horizontally for easy use with np.apply_along_axis
#     stacked = np.hstack([ref, vec])

#     # Get the dot product of each pair of vectors
#     get_dot = lambda x: x[:dim].dot(x[dim:])
#     dot = np.apply_along_axis(get_dot, -1, stacked)

#     # Get the sign for each pair of vectors
#     get_m = lambda x: np.pad(x.reshape(2, dim).T, pad_width=((0, 0), (0, dim-2)), constant_values=1)
#     m = np.apply_along_axis(get_m, 1, stacked)

#     get_sign = lambda x: np.sign(
#         np.linalg.det(  # Get determinant
#             np.pad(  # after padding
#                 x.reshape(2, dim).T,  # the row into a matrix
#                 pad_width=((0, 0), (0, dim-2)),
#                 constant_values=1)))
#     sign = np.apply_along_axis(get_sign, 1, stacked)

#     # Return the computed angle
#     return sign * np.arccos(dot / (ref_mag * vec_mag))

def angle_between(ref, vec):
    '''
    Calculate the angle between 2d vectors.
    Returns angles between -pi and pi.
    Use the right hand rule to determine the sign, where ref is the index finger.
    '''
    # Convert to numpy arrays in case it hasn't been done yet
    ref = np.array(ref).T
    vec = np.array(vec).T

    # Compute the dot product between the vectors
    dot = vec[0]*ref[0] + vec[1]*ref[1]

    # Get the magnitude of each vector
    get_mag = lambda x: np.sqrt(x.dot(x))
    ref_mag = np.apply_along_axis(get_mag, 0, ref)
    vec_mag = np.apply_along_axis(get_mag, 0, vec)

    # Compute the absolute value of the angle between the vectors using their dot product
    abs_angles = np.arccos(dot / (ref_mag * vec_mag))

    # Compute the sign of the angle between the vectors using their determinant
    sign = np.sign(ref[0]*vec[1] - vec[0]*ref[1])

    # This computation maps -1 -> -1, 0 -> 1, and 1 -> 1.
    # We want 0s to be 1s because the only time the determinant is 0 is when the angle between
    # the two vectors is 0 or it is pi. In the latter case, we want to leave the angle as pi.
    sign = sign - np.abs(sign) + 1

    return (sign * abs_angles).T

def unit_2d(angle):
    '''Get a unit vector from an angle'''
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)

def wrap_to(theta, center=0, range=2*np.pi):
    '''Wrap to center-range/2, center+range/2'''
    return ((theta + center - range/2) % range) + center - range/2

def wrap_to_pi(theta):
    '''Wrap an angle in radians to -pi to pi'''
    return wrap_to(theta, center=0, range=2*np.pi)

def wrap_to_180(theta):
    '''Wrap an angle in degrees to -180 to 180'''
    return wrap_to(theta, center=0, range=360)

def wrap_to_twopi(theta):
    '''Wrap an angle in radians to 0 to 2pi'''
    return wrap_to(theta, center=np.pi, range=2*np.pi)

def wrap_to_360(theta):
    '''Wrap an angle in degrees to 0 to 360'''
    return wrap_to(theta, center=180, range=360)

def convert_heading(heading):
    '''Convert a heading (angle in degrees, measured clockwise from north) to an angle in radians measured counterclockwise from west'''
    return wrap_to_pi(-np.radians(heading) - np.pi)

def combine_axes(arr, axes):
    '''Combines the dimensions between [axes[0], axes[1]) into a single axis at the index of axes[0]'''
    s = arr.shape
    return arr.reshape(s[:axes[0]] + (-1,) + s[axes[1]:])

def apply_along_axes(func, axes, arr, *args, **kwargs):
    '''Applies the given function on slices taken from the given axes'''
    s = arr.shape
    reshape = partial(np.reshape, newshape=s[axes[0]:axes[1]])
    func1d = lambda x: func(reshape(x))
    return np.apply_along_axis(func1d, axes[0], combine_axes(arr, axes), *args, **kwargs)