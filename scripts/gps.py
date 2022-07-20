from serial_device import SerialDevice, SerialState
import re
import time
from datetime import datetime

class GPS(SerialDevice):
    '''
    GPS state machine.

    States
    ------
    Idle:
        Flush all input and output and monitor for any interruptions in the
        serial communication. If the communication is interrupted, the
        GPS switches to the Closed state.
    Listening:
        The GPS collects all lines received and records and processes
        coordinates.
    Closed:
        The serial port is closed permanently.
    '''
    
    # Extract information from NMEA formatted lines
    # https://regex101.com/r/3Yt5gH/1
    _NMEA_REGEX = re.compile(r'^\$(\w{2})(\w{3}),(.*)\*([0-9A-F]{2})$')

    def __init__(self, port, verbosity=0):
        '''
        Start serial communcation. The GPS starts in the Idle state.

        Parameters
        ----------
        port : str
            The port the GPS is on.
        verbosity : int
            Controls how much the GPS prints. 0 = nothing, 1 = some status
            messages, >2 = everything else.
        '''
        super().__init__(port, verbosity=verbosity, event_types=['on_coords'])
        self._latitude = None
        self._longitude = None
        self.on_line(self._update_coords)
    
    def __repr__(self):
        return '<GPS({}) at {}>'.format(self._port, hex(id(self)))
    
    def __str__(self):
        return 'GPS({})'.format(self._port)

    def _parse_line(self, line):
        '''Extract the contents of a line sent by the GPS into a dictionary.'''
        contents = {'raw': line}
        match = GPS._NMEA_REGEX.match(line)
        groups = match.groups()

        # Format values and add them to the contents dictionary
        contents['talker'] = groups[0]
        contents['message'] = groups[1]
        contents['data_raw'] = groups[2]
        contents['hex_sum'] = '0x' + groups[3]

        # Create another dictionary for the information contained in the data fields to be added to the contents dictionary
        data_dict = {}
        data = contents['data_raw'].split(',')

        if contents['message'] == 'GLL':
            date_str = '{}{}'.format(datetime.now().strftime('%Y%m%d'), data[4])
            data_dict['datetime'] = datetime.strptime(date_str, '%Y%m%d%H%M%S.%f')
            data_dict['latitude'] = GPS._parse_latitude(data[0], data[1])
            data_dict['longitude'] = GPS._parse_longitude(data[2], data[3])

        contents['data'] = data_dict
        return contents

    def _parse_latitude(latitude, latitude_dir='N'):
        if latitude == '' or latitude_dir == '':
            return None

        degrees = latitude[:2]
        minutes = latitude[2:]
        latitude = float(degrees) + float(minutes) / 60
        if latitude_dir == 'S':
            latitude *= -1
        return latitude
    
    def _parse_longitude(longitude, longitude_dir='E'):
        if longitude == '' or longitude_dir == '':
            return None

        degrees = longitude[:3]

        minutes = longitude[3:]
        longitude = float(degrees) + float(minutes) / 60
        if longitude_dir == 'W':
            longitude *= -1
        return longitude

    def _update_coords(self, line):
        '''Process the lines received by the GPS.'''
        try:
            contents = self._parse_line(line)
            if contents['message'] in ['GLL']:
                self._latitude = contents['data']['latitude']
                self._longitude = contents['data']['longitude']
                self._dispatch('on_coords', coords=(self._latitude, self._longitude), timestamp=contents['data']['datetime'], contents=contents)
        except Exception as err:
            self._print('Exception while processing line:', err, v=2)

    def get_coords(self, default=(None, None)):
        '''
        Returns the most updated coordinates.

        Parameters
        ----------
        default :
            If the GPS is closed or latitude and/or longitude are not
            available, return default.
        
        Returns
        -------
        latitude : float
        longitude : float
        '''
        if self.is_closed() or self._latitude is None or self._longitude is None:
            return default
        else:
            return (self._latitude, self._longitude)
    
    def on_coords(self, callback):
        '''
        Register a callback to run when the gps receives new coordinates.

        Parameters
        ----------
        callback : callable
            Parameters
            ----------
            coords : (float, float) tuple
                The first float is latitude and the second is longitude.
            timestamp : datetime
                Time the coordinates were received at.
            contents : dict
                The parsed contents of the line from which the coordinates and
                timestamp were retrieved.
        '''
        self._register('on_coords', callback)

if __name__ == '__main__':
    gps = GPS('COM4', verbosity=2)

    def coords_callback(coords, timestamp, **_):
        print('{}: Latitude={}, Longitude={}'.format(timestamp, coords[0], coords[1]))
    gps.on_coords(coords_callback)

    gps.start()
    while gps.is_listening():
        try:
            gps.run()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break
    
    gps.close()