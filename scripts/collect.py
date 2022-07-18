from hydrophone import Hydrophone
from gps import GPS
from datetime import datetime
import time
import sys
import utils
import os
import csv

if __name__ == '__main__':
    # How often to log GPS coordinates. If no detections are received, the GPS coordinates are logged with all tag-related fields blank.
    GPS_RATE = 10  # seconds
    last_gps_time = None

    # Read in command line arguments: port, serial_no, savepath to a file to dump data
    _, port, serial_no, *rest = sys.argv

    # Define the column names of the data to be logged
    columns = [
                'serial_no',
                'line_counter',
                'datetime',
                'code_space',
                'tag_id',
                'sensor_adc',
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
        utils.mkdir(directory)
        savefile = open(utils.add_version(savepath), 'w', newline='')

        # Write the csv header
        writer = csv.writer(savefile)
        writer.writerow(columns)
    else:
        savepath = None
        savefile = None

    # Create the hydrophone and GPS objects
    h1 = Hydrophone(port, int(serial_no), verbosity=2)
    gps = GPS('COM4', verbosity=1)

    # Define a detection callback
    def detection_callback(tag_info={}, detection={}):
        global last_gps_time
        last_gps_time = time.time()

        if 'current_detection_time' and 'first_detection_time' in tag_info:
            total_dt = (tag_info.get('current_detection_time', None) - tag_info.get('first_detection_time', None)).total_seconds()
        else:
            total_dt = ''
        
        latitude, longitude = gps.get_coords(default=('', ''))

        # Collect all of the data
        detection_data = detection.get('data', {})
        data = [
            detection.get('serial_no', ''),
            detection.get('sequence', ''),
            detection.get('datetime', ''),
            detection_data.get('code_space', ''),
            detection_data.get('id', ''),
            detection_data.get('sensor_adc', ''),
            detection_data.get('signal_level', ''),
            detection_data.get('noise_level', ''),
            detection_data.get('channel', ''),
            latitude,
            longitude,
            total_dt,
            tag_info.get('delta_time', ''),
            tag_info.get('delta_tof', ''),
            tag_info.get('delta_distance', ''),
            tag_info.get('accumulated_distance', '')
        ]

        # Write to a savefile if one was given and to the console
        if savefile is not None:
            writer.writerow(data)
        print(','.join([str(datum) for datum in data]))
    
    # Add the detection callback to the hydrophone
    h1.on_detection(detection_callback)

    # Start hydrophone and GPS
    h1.start()
    gps.start()
    while h1.is_starting():
        h1.run()

        # Run GPS, attempting to reopen it if it has closed
        if gps.is_closed():
            gps = GPS('COM4', verbosity=0)
            gps.start()
            gps._verbosity = 1
        else:
            gps.run()
        time.sleep(0.01)

    # Listen for pings
    print(','.join(columns))
    last_gps_time = time.time()
    while h1.is_listening():
        try:
            h1.run()

            # Run GPS, attempting to reopen it if it has closed
            if gps.is_closed():
                gps = GPS('COM4', verbosity=0)
                gps.start()
                gps._verbosity = 1
            else:
                gps.run()
            
            # Log GPS coords if too much time has passed
            if time.time() - last_gps_time > GPS_RATE:
                detection_callback(detection={'datetime': datetime.now()})

            time.sleep(0.01)
        except KeyboardInterrupt:
                break

    if not h1.is_closed():
        # Stop the hydrophone and GPS
        h1.stop()
        gps.stop()
        while h1.is_stopping():
            h1.run()
            time.sleep(0.01)
        h1.close()
        gps.close()
    
    # Close savefile if one was created
    if savefile is not None:
        savefile.close()