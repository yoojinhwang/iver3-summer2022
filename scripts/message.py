import operator
import serial
from functools import reduce

def command_message(*args):
    msg = str(args[0])
    for arg in args[1:]:
        msg += ',' + str(arg)
    return '${}*{}\r\n'.format(msg, get_check_sum(msg))

def get_check_sum(msg):
    return hex(reduce(operator.xor, bytes(msg, 'utf-8')))[2:]

if __name__ == '__main__':
    msg = command_message('OPK', 34.10840978412025, -117.71277897934034, 3, 2)
    try:
        ser = serial.Serial('COM1')
        ser.write(bytes(msg, 'utf-8'))
        ser.close()
        print('Sent message: {}'.format(msg))
    except:
        print('Failed to send message: {}'.format(msg))