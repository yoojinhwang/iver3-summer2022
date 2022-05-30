"""
    Goal of script is to write sensor readings into a json file
    for a given time period
"""

import serial
import re
import time
import csv
from datetime import datetime

# Example compass output: $C175.2P1.8R47.9T29.6D0.00

def compass_decode(input):
    ''' Input is the compass raw reading and outputs a string
    of decoded values according to datasheet OS5000 Compass Manual'''
    
    heading = re.search('%s(.*)%s' % ('C', 'P'), line).group(1)
    pitch = re.search('%s(.*)%s' % ('P', 'R'), line).group(1)
    roll = re.search('%s(.*)%s' % ('R', 'T'), line).group(1)
    return heading, pitch, roll
    
if __name__ == '__main__':
    print(type(datetime))

    with serial.Serial('COM14', 19200) as ser:
        filename = "compass_readings_" + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv'
        with open(filename, "w") as new_file:
            csv_writer = csv.writer(new_file)
        
            while True:
                line = ser.readline().decode()
                h, p, r = compass_decode(line)
                print(h, p, r)
                csv_writer.writerow([h,p,r])
