from hydrophone import Hydrophone
import time
import sys
import utils
import os
import csv

def get_detection_callback(savefile=None):
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
    
    # Write the csv header
    if savefile is not None:
        writer = csv.writer(savefile)
        writer.writerow(columns)
    print(','.join(columns))

    # Define a detection callback
    def detection_callback(detection, tag_info):
        total_dt = (tag_info['current_detection_time'] - tag_info['first_detection_time']).total_seconds()
        data = [
            detection['serial_no'],
            detection['sequence'],
            detection['datetime'],
            detection['data']['code_space'],
            detection['data']['id'],
            detection['data']['signal_level'],
            detection['data']['noise_level'],
            detection['data']['channel'],
            '',
            '',
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

    return detection_callback

if __name__ == '__main__':
    # Read in command line arguments: port, serial_no, savepath to a file to dump data
    _, port, serial_no, *rest = sys.argv
    if len(rest) != 0:
        savepath = rest[0]
    else:
        savepath = None
    
    # Open a file to save the data to if a savepath was given
    if savepath is not None:
        directory, filename = os.path.split(savepath)
        utils.mkdir(directory)
        savefile = open(utils.add_version(savepath), 'w', newline='')
    else:
        savefile = None

    # Create the hydrophone object
    h1 = Hydrophone(port, int(serial_no))
    h1.on_detection(get_detection_callback(savefile=savefile))

    # Start hydrophone
    h1.start()
    while h1.is_starting():
        h1.run()
        time.sleep(0.01)

    if not h1.is_closed():
        # Listen for pings
        while True:
            try:
                h1.run()
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        
        # Stop the hydrophone
        h1.stop()
        while h1.is_stopping():
            h1.run()
            time.sleep(0.01)
        h1.close()