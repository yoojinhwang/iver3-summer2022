from serial_device import SerialDevice, SerialState, Idle, Listening, Closed
from datetime import datetime
import re
import time
import copy

class HydrophoneState(SerialState):
    '''Abstract class representing a state that the hydrophone can be in.'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone)
        self._hydrophone = hydrophone

class CommandSequence(HydrophoneState):
    '''Sends a sequence of commands to the hydrophone.'''

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
        '''Handles sending the sequence of commands.'''

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
    '''Sends start sequence to the hydrophone.'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone, 
            command_sequence=['STOP', 'ERASE', 'TIME', 'START', 'RTMPROFILE=0,SI=0', 'RTMNOW'],
            next_state=Listening,
            timeout_state=Idle)
        self._print('Starting', v=1)
        self._hydrophone._dispatch('on_starting_init')
    
    def run(self):
        self._hydrophone._dispatch('on_starting_run')
        super().run()
    
    def end(self):
        self._hydrophone._dispatch('on_starting_end')

class Stopping(CommandSequence):
    '''Sends stop sequence to the hydrophone.'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone,
            command_sequence=['STOP', 'STORAGE', 'QUIT'],
            next_state=Idle,
            timeout_state=Idle)
        self._print('Stopping', v=1)
        self._hydrophone._dispatch('on_stopping_init')
    
    def run(self):
        self._hydrophone._dispatch('on_stopping_run')
        super().run()
    
    def end(self):
        self._hydrophone._dispatch('on_stopping_end')

class Hydrophone(SerialDevice):
    '''    
    Hydrophone state machine.

    States
    ------
    Idle :
        Flush all input and output and monitor for any interruptions in the
        serial communication. If the communication is interrupted, the
        hydrophone switches to the Closed state.
    Starting :
        A sequence of commands are sent to the hydrophone board to prepare it
        to listen for tag detections. If the commands timeout, the hydrophone
        switches to the Idle state.
    Listening :
        The hydrophone collects all lines received and records and processes
        dectections.
    Stopping : 
        A sequence of commands are sent to the hydrophone board to stop
        listening for tag detections and to sleep to consume minimum power. If
        the commands timeout, the hydrophone switches to the Idle state.
    Closed :
        The serial port is closed permanently.
    '''

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

    def __init__(self, port, serial_no, timeout=0, verbosity=0):
        '''
        Start serial communcation. The hydrophone starts in the Idle state.

        Parameters
        ----------
        port : str
            The port the hydrophone is on.
        verbosity : int
            Controls how much the hydrophone prints. 0 = nothing, 1 = some
            status messages, >2 = everything else.
        '''
        # TODO: Add keyword arguments for each of the hydrophone's timeouts
        self._serial_no = serial_no
        super().__init__(port, verbosity=verbosity, event_types=[
            'on_starting_init',
            'on_starting_run',
            'on_starting_end',
            'on_stopping_init',
            'on_stopping_run',
            'on_stopping_end',
            'on_detection'
        ])
        self._timeout = timeout
        self._tags = {}
        self._detection_callback = None
        self.on_line(self._process_line)
    
    def __repr__(self):
        return '<Hydrophone({}, {}) at {}>'.format(self._port, self._serial_no, hex(id(self)))
    
    def __str__(self):
        return 'Hydrophone({}, {})'.format(self._port, self._serial_no)

    def _parse_line(self, line):
        '''Extract the contents of a line sent by the hydrophone into a dictionary.'''
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
        '''Format a command message to be sent to the hydrophone.'''

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
        '''Format a command message and write it to the hydrophone.'''
        command_msg = self._command_message(command)
        self._write(bytes(command_msg, 'utf-8'))
        return command_msg
    
    def _set_time(self):
        '''Send a command to the hydrophone instructing it to set its clock to the current time.'''
        return self._write_command('TIME={}'.format(datetime.now().strftime(Hydrophone._TIME_FORMAT)))

    def _process_line(self, line):
        '''Parse line and check whether it is a tag detection.'''
        # Parse the line
        contents = self._parse_line(line)

        # Process the line further if it is a detection line
        if contents['type'] == 'output' and contents['data']['type'] == 'pinger':
            tag_id = contents['data']['id']
            detection_time = contents['datetime']

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

            # Dispatch callbacks and update tag info
            self._dispatch('on_detection', tag_info=copy.deepcopy(self._tags[tag_id]), detection=contents)
            self._tags[tag_id]['previous_detection_time'] = detection_time
    
    def start(self):
        '''
        Set the hydrophone to the Starting state in order to execute its start
        command sequence.

        Returns
        -------
        bool
            True if the hydrophone has begun its start sequence succesfully,
            False otherwise.
        '''
        if type(self._state) != Closed:
            self._state = Starting(self)
            return True
        return False
    
    def stop(self):
        '''
        Set the hydrophone to the Stopping state in order to execute its stop
        command sequence.

        Returns
        -------
        bool
            True if the hydrophone has begun its stop sequence succesfully,
            False otherwise.
        '''
        if type(self._state) != Closed:
            self._state = Stopping(self)
            return True
        return False
    
    def is_starting(self):
        '''
        Check whether the hydrophone is currently in the Starting state.

        Returns
        -------
        bool
        '''
        return type(self._state) == Starting
    
    def is_stopping(self):
        '''
        Check whether the hydrophone is currently in the Stopping state.

        Returns
        -------
        bool
        '''
        return type(self._state) == Stopping
    
    def get_tag_info(self):
        '''
        Return the hydrophone's internal information about each tag it has
        detected.

        Returns
        -------
        tags : dict
            Keys
            ----
            tag_id : int
                The ID of each tag that has been detected.

            Values
            ------
            tag_info : dict
                Contains information about a tag.

                Keys
                ----
                'avg_dt' : float
                    The tag's average time between pings.
                'first_detection_time' : datetime
                    The time at which the tag was first detected.
                'previous_detection_time' : datetime
                    The time at which the tag was last detected.
                'current_detection_time' : datetime
                    The time of the current detection.
                'delta_time' : float
                    Time in seconds between the current and previous detection
                    times.
                'delta_tof' : float
                    The change in the tag's time of flight (tof) between the
                    current and previous detections.
                'delta_distance' : float
                    The change in the tag's distance between the current and
                    previous detections as estimated from time of flight.
                'accumulated_distance' : float
                    The change in the tag's distance between the current and
                    first detections as estimated from time of flight.
        '''
        return copy.deepcopy(self._tags)

    def on_detection(self, callback):
        '''
        Register a callback to run when the hydrophone receives a detection.

        Parameters
        ----------
        callback : callable
            Parameters
            ----------
            tag_info : dict
                Dictionary containing the hydrophone's internal information
                about the tag detected. See Hydrophone.get_tag_info for the
                dictionary's contents.
            detection : dict
                The parsed contents of the line from which the detection's
                information was received.
        '''
        self._register('on_detection', callback)