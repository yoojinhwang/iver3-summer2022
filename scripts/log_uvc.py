from uvc import UVC
import sys
import time
import utils
import csv
import os
from datetime import datetime
import numpy as np

# Define a callback to log data
def log_data(uvc):
    knots_per_meter = 1.944
    latitude, longitude = uvc.get_coords(default=('',''))
    x_speed, y_speed = uvc.get_speeds(default=(np.nan, np.nan))
    if np.isnan(x_speed) or np.isnan(y_speed):
        speed = ''
    else:
        print("Logdata variables")
        speed = np.sqrt(x_speed**2 + y_speed**2) * knots_per_meter
        print(latitude, longitude, speed, uvc.get_heading(default=''))
    data = [
        datetime.now(),
        latitude,
        longitude,
        speed,
        uvc.get_heading(default='')
    ]

    # Write to a savefile if one was given and to the console
    if savefile is not None:
        writer.writerow(data)
    print(','.join([str(datum) for datum in data]))
    return data



if __name__ == '__main__':
    # Read in command line arguments: savepath to a file to dump data
    _, *rest = sys.argv
    #rest = ['\Users\iver\Documents\iver3-summer2022\data\']
    knots_per_meter = 1.944

    # Define the column names of the data to be logged
    columns = [
                'datetime',
                'Latitude',
                'Longitude',
                'Vehicle Speed (Kn)',
                'C True Heading'
            ]

    # Open a file to save the data to if a savepath was given
    if len(rest) != 0:
        savepath = rest[0]
        directory, filename = os.path.split(savepath)
        utils.mkdir(directory)
        savefile = open(utils.add_version(savepath), 'w', newline='')

        # Write the csv header
        writer = csv.writer(savefile)
        writer.writerow(columns)
    else:
        savepath = None
        savefile = None

    # Create the hydrophone and GPS objects
    uvc = UVC('COM1', verbosity=2)

    # Define a callback to request data
    def request_data():
        uvc._write_command('OSD','C','G','S','P','Y','D','T','I')
    
    
    # Add the callbacks
    uvc.on_listening_run(request_data)
    uvc.on_listening_run(log_data)

    # Start UVC
    print(','.join(columns))
    uvc.start()
    while uvc.is_listening():
        try:
            uvc.run()
            time.sleep(1)
        except KeyboardInterrupt:
            break

    if not uvc.is_closed():
        # Stop UVC
        uvc.stop()
        uvc.close()
    
    # Close savefile if one was created
    if savefile is not None:
        savefile.close()
