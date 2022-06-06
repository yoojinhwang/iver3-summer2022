import os
import numpy as np
import re

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

def fit_line(x, y):
    '''Fits a line, i.e. slope and y-intercept'''
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]

def imerge(*its):
    for it in its:
        for i in it:
            yield i

def get_delta_tof(timestamps, avg_dt):
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