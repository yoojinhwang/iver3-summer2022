import serial
from abc import ABC, abstractmethod
import re

class GPSState(ABC):
    '''Abstract class representing a state that the GPS can be in'''
    def __init__(self, gps):
        self._gps = gps
    
    @abstractmethod
    def run(self):
        pass

class Idle(GPSState):
    '''Does nothing'''
    def __init__(self, gps):
        super().__init__(gps)
    
    def run(self):
        pass

class Listening(GPSState):
    '''Processes GPS data'''
    def __init__(self, gps):
        super().__init__(gps)
        self._gps._ser.flush()
    
    def run(self):
        # Read all available lines and keep track of the ones that include position information
        messages = []

        line = self._gps._readline()
        while len(line) != 0:
            # Check whether a line is a detection line and if so, store it
            try:
                contents = self._gps._parse_line(line)
                if contents['message'] in ['GLL']:
                    messages.append(contents)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as err:
                print(err)

            # Try to read another line
            line = self._gps._readline()
        
        self._gps._process_messages(messages)

class Closed(GPSState):
    '''Does nothing except close the serial communication on init'''
    def __init__(self, gps):
        super().__init__(gps)
        if self._gps._ser is not None:
            print('Closing serial port')
            self._gps._ser.close()
    
    def run(self):
        pass

class GPS:
    '''Takes care of the serial communication with the GPS'''
    
    # Extract information from NMEA formatted lines
    # https://regex101.com/r/3Yt5gH/1
    _NMEA_REGEX = re.compile(r'^\$(\w{2})(\w{3}),(.*)\*([0-9A-F]{2})$')

    def __init__(self, com_port):
        '''Requires the com port that the GPS is on'''
        self._com_port = com_port
        self._state = Idle(self)
        self._line_buffer = b''
        self._latitude = None
        self._longitude = None

        # Attempt to open a serial port for the gps. If this fails, switch to the closed state
        self._ser = None
        try:
            self._ser = serial.Serial(
                self._com_port,
                baudrate=9600,
                timeout=0)
        except serial.SerialException:
            print('Unable to open port {}'.format(com_port))
            self._state = Closed(self)
    
    def _readline(self):
        '''Wrapper over serial.Serial.readline with some extra functionality to handle being called repeatedly in a while loop'''
        
        # If no data is waiting to be read, return an empty string
        if self._ser.in_waiting:
            # Read a line from the serial port. Since this is being called in a loop repeatedly, the result of readline may not be
            # an entire line. Only the first part of a line might be available. The result is appended to the hydrophone's read
            # buffer for use when an entire line is ready.
            partial_line = self._ser.readline()
            self._line_buffer += partial_line

            # Check if there is an entire line in the read buffer and if so, remove it from the buffer and return it
            if b'\r\n' in partial_line:
                line, rest = self._line_buffer.split(b'\r\n', maxsplit=1)
                self._line_buffer = rest

                # Sometimes the hydrophone sends trash data that can't be converted to a string?? If so, ignore it.
                try:
                    return line.decode('utf-8')
                except UnicodeDecodeError as err:
                    print('Discarding line: ', err)
                    return ''
        return ''

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

    def _process_messages(self, messages):
        '''Process the lines received by the GPS'''
        for message in messages:
            self._latitude = message['data']['latitude']
            self._longitude = message['data']['longitude']

    def run(self):
        '''Should be run in a loop without much delay between calls'''
        self._state.run()
    
    def close(self):
        '''Close the serial connection. After this is called, the GPS object becomes unusable'''
        if type(self._state) != Closed:
            self._state = Closed(self)
            return True
        return False
    
    def start(self):
        '''Set the GPS to the listening state'''
        if type(self._state) != Closed:
            self._state = Listening(self)
            return True
        return False
    
    def stop(self):
        '''Set the GPS to the idle state'''
        if type(self._state) != Closed:
            self._state = Idle(self)
            return True
        return False
    
    def is_idle(self):
        '''Check whether the GPS is currently idle'''
        return type(self._state) == Idle
    
    def is_listening(self):
        '''Check whether the GPS is currently listening'''
        return type(self._state) == Listening
    
    def is_closed(self):
        '''Check whether the GPS is currently closed'''
        return type(self._state) == Closed

    def get_latitude(self):
        return self._latitude
    
    def get_longitude(self):
        return self._longitude

if __name__ == '__main__':
    gps = GPS('COM5')
    gps.start()

    while True:
        try:
            gps.run()
        except KeyboardInterrupt:
            break
    
    gps.close()