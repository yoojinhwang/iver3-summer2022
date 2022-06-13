import os
import numpy as np
import re
import itertools

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

    # Search through each source directoty
    for source in sources:
        # Get all of the contents of the source directory and search them
        entries = os.scandir(source)
        for entry in entries:
            # If the current entry is itself a directory, search it recursively
            if entry.is_dir():
                yield from find_files(entry, name=name, extension=extension)

            # Otherwise yield entries whose name matches the name pattern and whose extension matches the extension pattern
            else:
                filename, fileext = os.path.splitext(entry.name)
                if name.fullmatch(filename) is not None and \
                   extension.fullmatch(fileext) is not None:
                    yield entry
        entries.close()

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