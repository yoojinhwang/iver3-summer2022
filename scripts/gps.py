from serial_device import SerialDevice, SerialState, Idle, Listening, Closed
import re
import time

class GPSState(SerialState):
    '''Abstract class representing a state that the GPS can be in'''
    def __init__(self, gps):
        super().__init__(gps)
        self._gps = gps

class GPS(SerialDevice):
    '''Takes care of the serial communication with the GPS'''
    
    # Extract information from NMEA formatted lines
    # https://regex101.com/r/3Yt5gH/1
    _NMEA_REGEX = re.compile(r'^\$(\w{2})(\w{3}),(.*)\*([0-9A-F]{2})$')

    def __init__(self, com_port, verbosity=0):
        '''Requires the com port that the GPS is on'''
        super().__init__(com_port, verbosity=verbosity)
        self._latitude = None
        self._longitude = None
    
    def __repr__(self):
        return '<GPS({}) at {}>'.format(self._com_port, hex(id(self)))
    
    def __str__(self):
        return 'GPS({})'.format(self._com_port)

    def _parse_line(self, line):
        '''Extract the contents of a line sent by the GPS into a dictionary'''
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
            data_dict['latitude'] = self._parse_latitude(data[0], data[1])
            data_dict['longitude'] = self._parse_longitude(data[2], data[3])

        contents['data'] = data_dict
        return contents

    def _parse_latitude(self, latitude, latitude_dir):
        degrees = latitude[:2]
        minutes = latitude[2:]
        latitude = float(degrees) + float(minutes) / 60
        if latitude_dir == 'S':
            latitude *= -1
        return latitude
    
    def _parse_longitude(self, longitude, longitude_dir):
        degrees = longitude[:3]
        minutes = longitude[3:]
        longitude = float(degrees) + float(minutes) / 60
        if longitude_dir == 'W':
            longitude *= -1
        return longitude

    def _process_lines(self, lines):
        '''Process the lines received by the GPS'''
        for line in lines:
            try:
                contents = self._parse_line(line)
                if contents['message'] in ['GLL']:
                    self._latitude = contents['data']['latitude']
                    self._longitude = contents['data']['longitude']
            except Exception as err:
                self._print('Exception while processing line:', err, v=2)
    
    def start(self):
        '''Set the GPS to the listening state'''
        if type(self._state) != Closed:
            self._print('Starting', v=1)
            self._state = Listening(self)
            return True
        return False
    
    def stop(self):
        '''Set the GPS to the idle state'''
        if type(self._state) != Closed:
            self._print('Stopping', v=1)
            self._state = Idle(self)
            return True
        return False
    
    def is_starting(self):
        '''Check whether the GPS is currently starting'''
        return False
    
    def is_stopping(self):
        '''Check whether the GPS is currently stopping'''
        return False

    def get_latitude(self, default=None):
        if self.is_closed() or self._latitude is None:
            return default
        else:
            return self._latitude
    
    def get_longitude(self, default=None):
        if self.is_closed() or self._latitude is None:
            return default
        else:
            return self._longitude

if __name__ == '__main__':
    gps = GPS('COM5', verbosity=2)
    gps.start()

    while gps.is_listening():
        try:
            gps.run()
            print('Latitude: {} Longitude: {}'.format(gps.get_latitude(), gps.get_longitude()))
            time.sleep(0.01)
        except KeyboardInterrupt:
            break
    
    # gps.close()