from serial_device import SerialDevice, SerialState, Idle, Listening, Closed
from datetime import datetime, timedelta
import re
import time
import copy

class HydrophoneState(SerialState):
    '''Abstract class representing a state that the hydrophone can be in'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone)
        self._hydrophone = hydrophone

class CommandSequence(HydrophoneState):
    '''Sends a sequence of commands to the hydrophone'''

    # Amount of time to wait before resending a command
    RESEND_TIME = 1  # seconds

    def __init__(self, hydrophone, command_sequence=[], next_state=Idle, timeout_state=Idle):
        super().__init__(hydrophone)
        self._command_idx = 0
        self._awaiting_response = False
        self._start_time = time.time()
        self._last_command_time = time.time()
        self._command_sequence = command_sequence
        self._next_state = next_state
        self._timeout_state = timeout_state

    def run(self):
        '''Handles sending the sequence of commands'''

        # Check to see if we timed out waiting for a response from the hydrophone
        current_time = time.time()
        if self._hydrophone._timeout != 0 and (current_time - self._start_time) > self._hydrophone._timeout:
            # If so switch to the timeout state
            self._print('Command sequence timed out', v=1)
            self._hydrophone._state = self._timeout_state(self._hydrophone)
        else:
            # Otherwise, either send a command, or read the serial port for a response to the pending command
            if not self._awaiting_response:
                # Send the next command on the list
                current_command = self._command_sequence[self._command_idx]

                # If the current command is the time command, use the hydrophone's _set_time function
                if current_command == 'TIME':
                    command_msg = self._hydrophone._set_time()
                else:
                    command_msg = self._hydrophone._write_command(current_command)
                self._print('Sending command \'{}\': {}'.format(current_command, command_msg), v=1)
                
                # We are now awaiting a response from the hydrophone
                self._last_command_time = current_time
                self._awaiting_response = True
            else:
                # Read all available lines checking for a response
                line = self._hydrophone._readline()
                while len(line) != 0:
                    # If we're still waiting for a response, keep checking lines
                    if self._awaiting_response:
                        # Attempt to parse the current line
                        try:
                            contents = self._hydrophone._parse_line(line)

                            # If its a response, we are no longer awaiting a response
                            if contents['type'] == 'response':
                                self._print('Response received with status: {}'.format(contents['status']), v=1)
                                # If the response status is OK, we can continue to the next command
                                if contents['status'] == 'OK':
                                    self._command_idx += 1
                                self._awaiting_response = False
                        except Exception as err:
                            self._print('Exception while listening:', err, v=1)

                    # Get the next line
                    line = self._hydrophone._readline()
                
                # If we don't receive a response within a certain amount of time, try resending the command
                if current_time - self._last_command_time > CommandSequence.RESEND_TIME:
                    self._print('Response is taking too long, resending command', v=1)
                    self._awaiting_response = False
        
        # If all start commands have been sent and responded to, the hydrophone can move on to the next state
        if self._command_idx >= len(self._command_sequence) and not self._awaiting_response:
            self._print('Command sequence completed', v=1)
            self._hydrophone._state = self._next_state(self._hydrophone)

class Starting(CommandSequence):
    '''Sends start sequence to the hydrophone'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone, 
            command_sequence=['STOP', 'ERASE', 'TIME', 'START', 'RTMPROFILE=0,SI=0', 'RTMNOW'],
            next_state=Listening,
            timeout_state=Idle)
        self._print('Starting', v=1)

class Stopping(CommandSequence):
    '''Sends stop sequence to the hydrophone'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone,
            command_sequence=['STOP', 'STORAGE', 'QUIT'],
            next_state=Idle,
            timeout_state=Idle)
        self._print('Stopping', v=1)

# class Listening(HydrophoneState):
#     '''Processes pinger detections'''
#     def __init__(self, hydrophone):
#         super().__init__(hydrophone)
#         self._hydrophone._ser.flush()

#     def run(self):
#         # Read all available lines and keep track of the detection lines
#         detections = []

#         line = self._hydrophone._readline()
#         while len(line) != 0:
#             # Check whether a line is a detection line and if so, store it
#             try:
#                 contents = self._hydrophone._parse_line(line)
#                 if contents['type'] == 'output' and contents['data']['type'] == 'pinger':
#                     detections.append(contents)
#             except KeyboardInterrupt:
#                 raise KeyboardInterrupt
#             except Exception as err:
#                 self._print(err, v=1)
            
#             # Try to read another line
#             line = self._hydrophone._readline()
        
#         # Process the detections
#         self._hydrophone._process_detections(detections)

class Hydrophone(SerialDevice):
    '''Takes care of the serial communication with the hydrophone'''

    # Format string for datetime.now
    _TIME_FORMAT = '%Y-%m-%d %X'

    # Exract information from response lines
    # https://regex101.com/r/Prf6OM/1
    _RESPONSE_REGEX = re.compile(r'^\*(\d{6})\.(0)#(\d{2})\[(\d{4})\],(?:(.*),)?(OK|FAILURE|INVALID),#([0-9A-F]{2})$')

    # Extract information from output lines
    # https://regex101.com/r/29xl6s/1
    _OUTPUT_REGEX = re.compile(r'^(\d{6}),(\d{3}),(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}),(.*),#([0-9A-F]{2})$')

    _AVG_DT_DICT = {65477: 8.179110, 65478: 8.179071, 65479: 7.958926}

    _SPEED_OF_SOUND = 1460  # m/s

    def __init__(self, com_port, serial_no, timeout=0, verbosity=0):
        '''Requires the com port that the hydrophone is on and the hydrophone's serial number in order to begin talking to it'''
        self._serial_no = serial_no
        super().__init__(com_port, verbosity=verbosity)
        self._timeout = timeout
        self._tags = {}
        self._detection_callback = None
    
    def __repr__(self):
        return '<Hydrophone({}, {}) at {}>'.format(self._com_port, self._serial_no, hex(id(self)))
    
    def __str__(self):
        return 'Hydrophone({}, {})'.format(self._com_port, self._serial_no)

    def _parse_line(self, line):
        '''Extract the contents of a line sent by the hydrophone into a dictionary'''
        contents = {'raw': line}

        # Indicates a response to a command
        if line[0] == '*':
            # Use a regex to extract values
            match = Hydrophone._RESPONSE_REGEX.match(line)
            groups = match.groups()

            # Format values and add them to the contents dictionary
            contents['type'] = 'response'
            contents['serial_no'] = int(groups[0])
            contents['port'] = int(groups[1])
            contents['decimal_sum'] = int(groups[2])
            contents['num_bytes'] = int(groups[3])
            contents['response_raw'] = groups[4]
            contents['status'] = groups[5]
            contents['hex_sum'] = '0x' + groups[6]
        
        # Otherwise, its an output line
        else:
            # Use a regex to extract values
            match = Hydrophone._OUTPUT_REGEX.match(line)
            groups = match.groups()

            # Format values and add them to the contents dictionary
            contents['type'] = 'output'
            contents['serial_no'] = int(groups[0])
            contents['sequence'] = int(groups[1])
            contents['datetime'] = datetime.fromisoformat(groups[2])
            contents['data_raw'] = groups[3]
            contents['hex_sum'] = '0x' + groups[4]

            # Create another dictionary for the information contained in the data fields to be added to the contents dictionary
            data_dict = {}

            # Remove key from key value pairs in data
            data = contents['data_raw'].split(',')
            for i, pair in enumerate(data):
                values = pair.split('=')
                if len(values) == 2:
                    data[i] = values[1]
            
            # If the first value is 'STS', the line is a status line. Otherwise, its a detection line.
            if data[0] == 'STS':
                data_dict['type'] = 'status'
                data_dict['detection_count'] = int(data[1])
                data_dict['ping_count'] = int(data[2])
                data_dict['line_voltage'] = float(data[3])
                data_dict['temperature'] = float(data[4])
                data_dict['detection_memory'] = float(data[5])
                data_dict['raw_memory'] = float(data[6])
                data_dict['tilt'] = {['x', 'y', 'z'][i]: float(num) for i, num in enumerate(data[7].split(':'))}
                data_dict['output_noise'] = data[8]
                data_dict['output_ppm_noise'] = data[9]
            else:
                # Pinger detections have 5 values, whereas sensor detections have 6
                if len(data) == 5:
                    data_dict['type'] = 'pinger'
                    data_dict['code_space'] = data[0]
                    data_dict['id'] = int(data[1])
                    data_dict['signal_level'] = float(data[2])
                    data_dict['noise_level'] = float(data[3])
                    data_dict['channel'] = int(data[4])
                else:
                    data_dict['type'] = 'sensor'
                    data_dict['code_space'] = data[0]
                    data_dict['id'] = int(data[1])
                    data_dict['sensor_adc'] = data[2]
                    data_dict['signal_level'] = float(data[3])
                    data_dict['noise_level'] = float(data[4])
                    data_dict['channel'] = int(data[5])
                    # raise ValueError('I haven\'t implemented sensor tags sorry :(')

            contents['data'] = data_dict
        return contents
    
    def _command_message(self, command):
        '''Format a command message to be sent to the hydrophone'''

        # According to the documentation, the port is always 0
        port = 0

        # Compute the decimal sum of the serial number + the port
        decimal_sum = 0
        for char in str(self._serial_no) + str(port):
            decimal_sum += int(char)
        
        # Format the command and return it as a string
        return '*{}.{}#{},{}\r'.format(
            self._serial_no,
            port,
            decimal_sum,
            command
        )
    
    def _write_command(self, command):
        '''Format a command message and write it to the hydrophone'''
        command_msg = self._command_message(command)
        self._write(bytes(command_msg, 'utf-8'))
        return command_msg
    
    def _set_time(self):
        '''Send a command to the hydrophone instructing it to set its clock to the current time'''
        return self._write_command('TIME={}'.format(datetime.now().strftime(Hydrophone._TIME_FORMAT)))

    def _process_lines(self, lines):
        '''Process the lines received by the Hydrophone'''
        detections = []
        for line in lines:
            try:
                contents = self._parse_line(line)
                if contents['type'] == 'output' and contents['data']['type'] == 'pinger':
                    detections.append(contents)
            except Exception as err:
                self._print('Exception while processing line:', err, v=2)
        self._process_detections(detections)

    def _process_detections(self, detections):
        # Loop through detections and compute time of flight for each pinger found
        for detection in detections:
            tag_id = detection['data']['id']
            detection_time = detection['datetime']

            # If the tag has not been encountered before, add it to the dictionary of tags
            if tag_id not in self._tags:
                self._tags[tag_id] = {
                    'avg_dt': Hydrophone._AVG_DT_DICT.get(tag_id, 8.179071),
                    'first_detection_time': detection_time,
                    'previous_detection_time': detection_time,
                    'current_detection_time': detection_time,
                    'delta_time': 0,
                    'delta_tof': 0,
                    'delta_distance': 0,
                    'accumulated_distance': 0
                }

            # Otherwise, use the information from its last detection to calculate the time of flight
            else:
                tag_info = self._tags[tag_id]

                # Calculate information about the current detection
                avg_dt = tag_info['avg_dt']
                dt = (detection_time - self._tags[tag_id]['previous_detection_time']).total_seconds()
                delta_tof = (dt - avg_dt / 2) % avg_dt - avg_dt / 2
                delta_distance = delta_tof * Hydrophone._SPEED_OF_SOUND

                # Update current tag information
                tag_info['current_detection_time'] = detection_time
                tag_info['delta_time'] = dt
                tag_info['delta_tof'] = delta_tof
                tag_info['delta_distance'] = delta_distance
                tag_info['accumulated_distance'] += delta_distance

            if callable(self._detection_callback):
                self._detection_callback(detection, copy.deepcopy(self._tags[tag_id]))

            self._tags[tag_id]['previous_detection_time'] = detection_time
    
    def start(self):
        '''Set the hydrophone to its start state. This means it will get ready to send the start sequence of commands'''
        if type(self._state) != Closed:
            self._state = Starting(self)
            return True
        return False
    
    def stop(self):
        '''Set the hydrophone to its stop state. This means it will get ready to send the stop sequence of commands'''
        if type(self._state) != Closed:
            self._state = Stopping(self)
            return True
        return False
    
    def is_starting(self):
        '''Check whether the hydrophone is currently starting'''
        return type(self._state) == Starting
    
    def is_stopping(self):
        '''Check whether the hydrophone is currently stopping'''
        return type(self._state) == Stopping
    
    def get_tag_info(self):
        '''Return the hydrophone's internal information about each tag it has detected'''
        return copy.deepcopy(self._tags)

    def on_detection(self, callback):
        '''Register a callback function to be run everytime the hydrophone receives a detection'''
        self._detection_callback = callback

# class Hydrophone:
#     '''Takes care of the serial communication with the hydrophone'''

#     # Format string for datetime.now
#     _TIME_FORMAT = '%Y-%m-%d %X'

#     # Exract information from response lines
#     # https://regex101.com/r/Prf6OM/1
#     _RESPONSE_REGEX = re.compile(r'^\*(\d{6})\.(0)#(\d{2})\[(\d{4})\],(?:(.*),)?(OK|FAILURE|INVALID),#([0-9A-F]{2})$')

#     # Extract information from output lines
#     # https://regex101.com/r/29xl6s/1
#     _OUTPUT_REGEX = re.compile(r'^(\d{6}),(\d{3}),(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}),(.*),#([0-9A-F]{2})$')

#     _AVG_DT_DICT = {65477: 8.179110, 65478: 8.179071, 65479: 7.958926}

#     _SPEED_OF_SOUND = 1460  # m/s

#     def __init__(self, com_port, serial_no, timeout=0, verbosity=0):
#         '''Requires the com port that the hydrophone is on and the hydrophone's serial number in order to begin talking to it'''
#         self._com_port = com_port
#         self._serial_no = serial_no
#         self._timeout = timeout
#         self._state = Idle(self)
#         self._line_buffer = b''
#         self._tags = {}
#         self._detection_callback = None
#         self._verbosity = verbosity

#         # Attempt to open a serial port for the hydrophone. If this fails, switch to the closed state
#         self._ser = None
#         try:
#             self._ser = serial.Serial(
#                 self._com_port,
#                 baudrate=9600,
#                 timeout=0,
#                 stopbits=serial.STOPBITS_ONE,
#                 parity=serial.PARITY_NONE,
#                 bytesize=serial.EIGHTBITS)
#         except serial.SerialException:
#             self._print('Unable to open port {}'.format(com_port), v=1)
#             self._state = Closed(self)
    
#     def __repr__(self):
#         return '<Hydrophone({}, {}) at {}>'.format(self._com_port, self._serial_no, hex(id(self)))
    
#     def __str__(self):
#         return 'Hydrophone({}, {})'.format(self._com_port, self._serial_no)
    
#     def _print(self, *args, v=0, **kwargs):
#         if self._verbosity >= v:
#             print(self, *args, **kwargs)

#     def _readline(self):
#         '''Wrapper over serial.Serial.readline with some extra functionality to handle being called repeatedly in a while loop'''

#         # If no data is waiting to be read, return an empty string
#         if self._ser.in_waiting:
#             # Read a line from the serial port. Since this is being called in a loop repeatedly, the result of readline may not be
#             # an entire line. Only the first part of a line might be available. The result is appended to the hydrophone's read
#             # buffer for use when an entire line is ready.
#             partial_line = self._ser.readline()
#             self._line_buffer += partial_line

#         # Check if there is an entire line in the read buffer and if so, remove it from the buffer and return it
#         if b'\r\n' in self._line_buffer:
#             line, rest = self._line_buffer.split(b'\r\n', maxsplit=1)
#             self._line_buffer = rest

#             # Sometimes the hydrophone sends trash data that can't be converted to a string?? If so, ignore it.
#             try:
#                 return line.decode('utf-8')
#             except UnicodeDecodeError as err:
#                 self._print('Discarding line: ', err, v=1)
#                 return ''
#         return ''

#     def _parse_line(self, line):
#         '''Extract the contents of a line sent by the hydrophone into a dictionary'''
#         contents = {'raw': line}

#         # Indicates a response to a command
#         if line[0] == '*':
#             # Use a regex to extract values
#             match = Hydrophone._RESPONSE_REGEX.match(line)
#             groups = match.groups()

#             # Format values and add them to the contents dictionary
#             contents['type'] = 'response'
#             contents['serial_no'] = int(groups[0])
#             contents['port'] = int(groups[1])
#             contents['decimal_sum'] = int(groups[2])
#             contents['num_bytes'] = int(groups[3])
#             contents['response_raw'] = groups[4]
#             contents['status'] = groups[5]
#             contents['hex_sum'] = '0x' + groups[6]
        
#         # Otherwise, its an output line
#         else:
#             # Use a regex to extract values
#             match = Hydrophone._OUTPUT_REGEX.match(line)
#             groups = match.groups()

#             # Format values and add them to the contents dictionary
#             contents['type'] = 'output'
#             contents['serial_no'] = int(groups[0])
#             contents['sequence'] = int(groups[1])
#             contents['datetime'] = datetime.fromisoformat(groups[2])
#             contents['data_raw'] = groups[3]
#             contents['hex_sum'] = '0x' + groups[4]

#             # Create another dictionary for the information contained in the data fields to be added to the contents dictionary
#             data_dict = {}

#             # Remove key from key value pairs in data
#             data = contents['data_raw'].split(',')
#             for i, pair in enumerate(data):
#                 values = pair.split('=')
#                 if len(values) == 2:
#                     data[i] = values[1]
            
#             # If the first value is 'STS', the line is a status line. Otherwise, its a detection line.
#             if data[0] == 'STS':
#                 data_dict['type'] = 'status'
#                 data_dict['detection_count'] = int(data[1])
#                 data_dict['ping_count'] = int(data[2])
#                 data_dict['line_voltage'] = float(data[3])
#                 data_dict['temperature'] = float(data[4])
#                 data_dict['detection_memory'] = float(data[5])
#                 data_dict['raw_memory'] = float(data[6])
#                 data_dict['tilt'] = {['x', 'y', 'z'][i]: float(num) for i, num in enumerate(data[7].split(':'))}
#                 data_dict['output_noise'] = data[8]
#                 data_dict['output_ppm_noise'] = data[9]
#             else:
#                 # Pinger detections have 5 values, whereas sensor detections have 6
#                 if len(data) == 5:
#                     data_dict['type'] = 'pinger'
#                     data_dict['code_space'] = data[0]
#                     data_dict['id'] = int(data[1])
#                     data_dict['signal_level'] = float(data[2])
#                     data_dict['noise_level'] = float(data[3])
#                     data_dict['channel'] = int(data[4])
#                 else:
#                     data_dict['type'] = 'sensor'
#                     data_dict['code_space'] = data[0]
#                     data_dict['id'] = int(data[1])
#                     data_dict['sensor_adc'] = data[2]
#                     data_dict['signal_level'] = float(data[3])
#                     data_dict['noise_level'] = float(data[4])
#                     data_dict['channel'] = int(data[5])
#                     # raise ValueError('I haven\'t implemented sensor tags sorry :(')

#             contents['data'] = data_dict
#         return contents
    
#     def _command_message(self, command):
#         '''Format a command message to be sent to the hydrophone'''

#         # According to the documentation, the port is always 0
#         port = 0

#         # Compute the decimal sum of the serial number + the port
#         decimal_sum = 0
#         for char in str(self._serial_no) + str(port):
#             decimal_sum += int(char)
        
#         # Format the command and return it as a string
#         return '*{}.{}#{},{}\r'.format(
#             self._serial_no,
#             port,
#             decimal_sum,
#             command
#         )
    
#     def _write_command(self, command):
#         '''Format a command message and write it to the hydrophone'''
#         command_msg = self._command_message(command)
#         self._ser.write(bytes(command_msg, 'utf-8'))
#         return command_msg
    
#     def _set_time(self):
#         '''Send a command to the hydrophone instructing it to set its clock to the current time'''
#         return self._write_command('TIME={}'.format(datetime.now().strftime(Hydrophone._TIME_FORMAT)))

#     def _process_detections(self, detections):
#         # Loop through detections and compute time of flight for each pinger found
#         for detection in detections:
#             tag_id = detection['data']['id']
#             detection_time = detection['datetime']

#             # If the tag has not been encountered before, add it to the dictionary of tags
#             if tag_id not in self._tags:
#                 self._tags[tag_id] = {
#                     'avg_dt': Hydrophone._AVG_DT_DICT.get(tag_id, 8.179071),
#                     'first_detection_time': detection_time,
#                     'previous_detection_time': detection_time,
#                     'current_detection_time': detection_time,
#                     'delta_time': 0,
#                     'delta_tof': 0,
#                     'delta_distance': 0,
#                     'accumulated_distance': 0
#                 }

#             # Otherwise, use the information from its last detection to calculate the time of flight
#             else:
#                 tag_info = self._tags[tag_id]

#                 # Calculate information about the current detection
#                 avg_dt = tag_info['avg_dt']
#                 dt = (detection_time - self._tags[tag_id]['previous_detection_time']).total_seconds()
#                 delta_tof = (dt - avg_dt / 2) % avg_dt - avg_dt / 2
#                 delta_distance = delta_tof * Hydrophone._SPEED_OF_SOUND

#                 # Update current tag information
#                 tag_info['current_detection_time'] = detection_time
#                 tag_info['delta_time'] = dt
#                 tag_info['delta_tof'] = delta_tof
#                 tag_info['delta_distance'] = delta_distance
#                 tag_info['accumulated_distance'] += delta_distance

#             if callable(self._detection_callback):
#                 self._detection_callback(detection, copy.deepcopy(self._tags[tag_id]))

#             self._tags[tag_id]['previous_detection_time'] = detection_time

#     def run(self):
#         '''Should be run in a loop without much delay between calls'''
#         self._state.run()
    
#     def close(self):
#         '''Close the serial connection. After this is called, the hydrophone object becomes unusable'''
#         if type(self._state) != Closed:
#             self._state = Closed(self)
#             return True
#         return False
    
#     def start(self):
#         '''Set the hydrophone to its start state. This means it will get ready to send the start sequence of commands'''
#         if type(self._state) != Closed:
#             self._state = Starting(self)
#             return True
#         return False
    
#     def stop(self):
#         '''Set the hydrophone to its stop state. This means it will get ready to send the stop sequence of commands'''
#         if type(self._state) != Closed:
#             self._state = Stopping(self)
#             return True
#         return False
    
#     def is_idle(self):
#         '''Check whether the hydrophone is currently idle'''
#         return type(self._state) == Idle
    
#     def is_starting(self):
#         '''Check whether the hydrophone is currently starting'''
#         return type(self._state) == Starting
    
#     def is_listening(self):
#         '''Check whether the hydrophone is currently listening'''
#         return type(self._state) == Listening
    
#     def is_stopping(self):
#         '''Check whether the hydrophone is currently stopping'''
#         return type(self._state) == Stopping
    
#     def is_closed(self):
#         '''Check whether the hydrophone is currently closed'''
#         return type(self._state) == Closed
    
#     def get_tag_info(self):
#         '''Return the hydrophone's internal information about each tag it has detected'''
#         return copy.deepcopy(self._tags)

#     def on_detection(self, callback):
#         '''Register a callback function to be run everytime the hydrophone receives a detection'''
#         self._detection_callback = callback