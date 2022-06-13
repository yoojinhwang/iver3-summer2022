from hydrophone import Hydrophone
from gps import GPS
import time
import sys
from common import add_version, mkdir
import os
import csv

def get_detection_callback(savefile=None):
    return detection_callback

if __name__ == '__main__':
    # Read in command line arguments: port, serial_no, savepath to a file to dump data
    _, port, serial_no, *rest = sys.argv

    # Define the column names of the data to be logged
    columns = [
                'serial_no',
                'line_counter',
                'datetime',
                'code_space',
                'tag_id',
                'signal_level',
                'noise_level',
                'channel',
                'latitude',
                'longitude',
                'total_dt',
                'dt',
                'delta_tof',
                'delta_distance',
                'total_distance'
            ]

    # Open a file to save the data to if a savepath was given
    if len(rest) != 0:
        savepath = rest[0]
        directory, filename = os.path.split(savepath)
        mkdir(directory)
        savefile = open(add_version(savepath), 'w', newline='')

        # Write the csv header
        writer = csv.writer(savefile)
        writer.writerow(columns)
    else:
        savepath = None
        savefile = None
    print(','.join(columns))

    # Create the hydrophone and GPS objects
    h1 = Hydrophone(port, int(serial_no))
    gps = GPS('COM5')

    # Define a detection callback
    def detection_callback(detection, tag_info):
        total_dt = (tag_info['current_detection_time'] - tag_info['first_detection_time']).total_seconds()

        # Retrieve latitude and longitude
        latitude = gps.get_latitude()
        if latitude is None:
            latitude = ''
        longitude = gps.get_longitude()
        if longitude is None:
            longitude = ''
        
        # Collect all of the data
        data = [
            detection['serial_no'],
            detection['sequence'],
            detection['datetime'],
            detection['data']['code_space'],
            detection['data']['id'],
            detection['data']['signal_level'],
            detection['data']['noise_level'],
            detection['data']['channel'],
            latitude,
            longitude,
            total_dt,
            tag_info['delta_time'],
            tag_info['delta_tof'],
            tag_info['delta_distance'],
            tag_info['accumulated_distance']
        ]

        # Write to a savefile if one was given and to the console
        if savefile is not None:
            writer.writerow(data)
        print(','.join([str(datum) for datum in data]))
    
    # Add the detection callback to the hydrophone
    h1.on_detection(get_detection_callback(savefile=savefile))

    # Start hydrophone and GPS
    h1.start()
    gps.start()
    while h1.is_starting():
        h1.run()
        gps.run()
        time.sleep(0.01)

    if not h1.is_closed():
        # Listen for pings
        while True:
            try:
                h1.run()
                gps.run()
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        
        # Stop the hydrophone and GPS
        h1.stop()
        gps.stop()
        while h1.is_stopping():
            h1.run()
            time.sleep(0.01)
        h1.close()
        gps.close()