# Goal of script is to write sensor readings into csv file

import serial

# Example: $C175.2P1.8R47.9T29.6D0.00

# def compass_decode(): 
#     ''' Input is the compass raw reading and outputs a string
#     of decoded values according to datasheet OS5000 Compass Manual''' 

if __name__ == '__main__':
    try:
        ser = serial.Serial('COM14',19200) 
        line = ser.readline()
        print(line)
    except: 
        print('Failed to read compass value')
