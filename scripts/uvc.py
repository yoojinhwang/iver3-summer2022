import utils
from serial_device import SerialDevice, SerialState
from gps import GPS
import time
from functools import reduce
from datetime import datetime
import operator
import re
import numpy as np
import math

class UVCState(SerialState):
    '''Abstract class representing a state that the UVC can be in.'''
    def __init__(self, robot):
        super().__init__(robot)
        self._robot = robot

class UVC(SerialDevice):
    # Exract information from response lines
    # https://regex101.com/r/JP9wEw/1
    _RESPONSE_REGEX = re.compile(r'^\$([A-Z]+),?(.*)\*([0-9A-F]{2})$')
    _COMPASS_REGEX = re.compile(r'^([-\d\.]+)P([-\d\.]+)R([-\d\.]+)T([-\d\.]+)D([-\d\.]+)$')

    def __init__(self, port, verbosity=0):
        super().__init__(port, verbosity=verbosity, event_types=[
            'on_command',
            'on_compass',
            'on_gps',
            'on_dvl',
            'on_ack'
        ], baudrate=57600)
        self._latitude = None
        self._longitude = None
        self._heading = None
        self._depth = None
        self._x_speed = None
        self._y_speed = None
        self.on_line(self._process_line)
        self.on_compass(self._update_heading)
        self.on_compass(self._update_depth)
        self.on_gps(self._update_coords)
        self.on_dvl(self._update_speeds)

    def __repr__(self):
        return '<UVC({}) at {}>'.format(self._port, hex(id(self)))
    
    def __str__(self):
        return 'UVC({})'.format(self._port)

    def _parse_line(self, line):
        '''Extract the contents of a line sent by the GPS into a dictionary.'''
        contents = {'raw': line}
        match = UVC._RESPONSE_REGEX.match(line)
        groups = match.groups()

        # Format values and add them to the contents dictionary
        contents['command'] = groups[0]
        contents['data_raw'] = groups[1]
        contents['checksum'] = '0x' + groups[2]

        # Create another dictionary for the information contained in the data fields to be added to the contents dictionary
        data_dict = {}
        data = contents['data_raw'].split(',')

        # Extract data of different types depending on the command type
        if contents['command'] == 'C':
            # Compass command
            match = UVC._COMPASS_REGEX.match(data[0])
            groups = match.groups()
            data_dict['heading'] = float(groups[0])
            data_dict['pitch'] = float(groups[1])
            data_dict['roll'] = float(groups[2])
            data_dict['temperature'] = float(groups[3])
            data_dict['depth'] = float(groups[4])
        if contents['command'] == 'GPRMC':
            # GPS command
            # Extract datetime
            if data[0] == '':
                data_dict['datetime'] = None
            else:
                date_str = '{}{}'.format(datetime.now().strftime('%Y%m%d'), data[0])
                data_dict['datetime'] = datetime.strptime(date_str, '%Y%m%d%H%M%S.%f')
            data_dict['status'] = data[1]

            # Extract lat and lon
            data_dict['latitude'] = GPS._parse_latitude(data[2], data[3])
            data_dict['longitude'] = GPS._parse_longitude(data[4], data[5])
        if contents['command'] == 'DVL':
            # DVL command
            # Values of 0 for the beam ranges or 
            verify = lambda x: x if x > -3200 else np.nan
            data_dict['x_speed'] = verify(float(data[0]))
            data_dict['y_speed'] = verify(float(data[1]))
        if contents['command'] == 'ACK':
            # Acknowledgement command
            data_dict['message_type'] = int(data[0])
            data_dict['status'] = int(data[1])
            data_dict['error_no'] = int(data[2])
            if len(data) > 3:
                data_dict['setting'] = data[3]
                data_dict['num_values'] = data[4]
                data_dict['values'] = data[5:]

        contents['data'] = data_dict
        return contents

    def _get_check_sum(self, msg):
        return self.to_hex(reduce(operator.xor, bytes(msg, 'utf-8')))

    def _command_message(self, *command_args):
        msg = str(command_args[0])
        for arg in command_args[1:]:
            msg += ',' + str(arg)
        return '${}*{}\r\n'.format(msg, self._get_check_sum(msg))
    
    def _write_command(self, *command_args):
        '''Format a command message and write it.'''
        command_msg = self._command_message(*command_args)
        self._write(bytes(command_msg, 'utf-8'))
        self._dispatch('on_command', command_args=command_args, command_msg=command_msg)
        return command_msg
    
    def _process_line(self, line):
        contents = self._parse_line(line)
        if contents['command'] == 'C':
            self._dispatch('on_compass', contents=contents)
        elif contents['command'] == 'GPRMC':
            self._dispatch('on_gps', contents=contents)
        elif contents['command'] == 'DVL':
            self._dispatch('on_dvl', contents=contents)
        elif contents['command'] == 'ACK':
            self._dispatch('on_ack', contents=contents)

    def _update_coords(self, contents):
        self._latitude = contents['data']['latitude']
        self._longitude = contents['data']['longitude']

    def _update_heading(self, contents):
        self._heading = contents['data']['heading']
    
    def _update_depth(self, contents):
        self._depth = contents['data']['depth']
    
    def _update_speeds(self, contents):
        self._x_speed = contents['data']['x_speed']
        self._y_speed = contents['data']['y_speed']

    def get_heading(self, default=None):
        if self.is_closed() or self._heading is None:
            return default
        else:
            return self._heading
    
    def get_depth(self, default=None):
        if self.is_closed() or self._depth is None:
            return default
        else:
            return self._depth

    def get_coords(self, default=(None, None)):
        if self.is_closed() or self._latitude is None or self._longitude is None:
            return default
        else:
            return (self._latitude, self._longitude)
    
    def get_speeds(self, default=(None, None)):
        if self.is_closed() or self._x_speed is None or self._y_speed is None:
            return default
        else:
            return (self._x_speed, self._y_speed)

    def to_hex(self, value):
        '''
        Convert an int between 0 and 255 to a hex string without the 0x prefix
        between 00 and FF.
        '''
        return hex(value)[2:].upper()
    
    def from_hex(self, value):
        '''
        Convert a hex string without the 0x prefix between 00 and FF to an int
        between 0 and 255.
        '''
        return int('0x' + value, base=16)

    def on_command(self, callback):
        self._register('on_command', callback)
    
    def on_compass(self, callback):
        self._register('on_compass', callback)
    
    def on_gps(self, callback):
        self._register('on_gps', callback)
    
    def on_dvl(self, callback):
        self._register('on_dvl', callback)
    
    def on_ack(self, callback):
        self._register('on_ack', callback)

if __name__ == '__main__':
    uvc = UVC('COM1', verbosity=1)
    uvc.start()
    # uvc.on_command(lambda command_args, command_msg: print(command_msg))
    # uvc._write_command('OSD','C','G','S','P','Y','D','T','I')
    # uvc._write_command('OMP', '0080808080', '00', '10')
    t = 0
    while True:
        try:
            angle = math.floor(127.5 * math.sin(t*(2*math.pi)/5) + 127.5)
            neg_angle = 255 - angle
            print(angle)
            angle = uvc.to_hex(angle)
            neg_angle = uvc.to_hex(neg_angle)
            uvc._write_command('OMP', '{}{}{}{}80'.format(angle, neg_angle, neg_angle, angle), '00', '10')
            t += 0.01
            uvc.run()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break