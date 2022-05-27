import serial
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import re
import time

class HydrophoneState(ABC):
    '''Abstract class representing a state that the hydrophone can be in at any given time'''
    def __init__(self, hydrophone):
        self._hydrophone = hydrophone

    @abstractmethod
    def run(self):
        pass

class Idle(HydrophoneState):
    '''Does nothing'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone)

    def run(self):
        pass

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
            print('Hydrophone command sequence timed out')
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
                print('Sending command \'{}\': {}'.format(current_command, command_msg))
                
                # We are now awaiting a response from the hydrophone
                self._last_command_time = current_time
                self._awaiting_response = True
            else:
                # print('Awaiting response')
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
                                print('Response received with status: {}'.format(contents['status']))
                                # If the response status is OK, we can continue to the next command
                                if contents['status'] == 'OK':
                                    self._command_idx += 1
                                self._awaiting_response = False
                        except Exception as err:
                            print(err)

                    # Get the next line
                    line = self._hydrophone._readline()
                
                # If we don't receive a response within a certain amount of time, try resending the command
                if current_time - self._last_command_time > CommandSequence.RESEND_TIME:
                    print('Response is taking too long, resending command')
                    self._awaiting_response = False
        
        # If all start commands have been sent and responded to, the hydrophone can move on to the next state
        if self._command_idx >= len(self._command_sequence) and not self._awaiting_response:
            print('Hydrophone command sequence completed')
            self._hydrophone._state = self._next_state(self._hydrophone)

class Starting(CommandSequence):
    '''Sends start sequence to the hydrophone'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone, 
            command_sequence=['STOP', 'ERASE', 'TIME', 'START', 'RTMPROFILE=0,SI=0', 'RTMNOW'],
            next_state=Listening,
            timeout_state=Idle)

class Listening(HydrophoneState):
    '''Processes pinger detections'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone)

    def run(self):
        # Read all available lines and keep track of the detection lines
        detections = []

        line = self._hydrophone._readline()
        while len(line) != 0:
            # print(line)

            # Check whether a line is a detection line and if so, store it
            contents = self._hydrophone._parse_line(line)
            if contents['type'] == 'output' and contents['info']['type'] == 'pinger':
                detections.append(contents)
            
            # Try to read another line
            line = self._hydrophone._readline()
        
        # Process the detections
        self._hydrophone._process_detections(detections)

class Stopping(CommandSequence):
    '''Sends stop sequence to the hydrophone'''
    def __init__(self, hydrophone):
        super().__init__(hydrophone,
            command_sequence=['STOP', 'STORAGE', 'QUIT'],
            next_state=Idle,
            timeout_state=Idle)

class Closed(HydrophoneState):
    '''Does nothing except close the serial communcation on init'''

    def __init__(self, hydrophone):
        super().__init__(hydrophone)
        if self._hydrophone._ser is not None:
            print('Closing serial port')
            self._hydrophone._ser.close()

    def run(self):
        pass

class Hydrophone:
    '''Takes care of the serial communication with the hydrophone'''

    # Format string for datetime.now
    _TIME_FORMAT = '%Y-%m-%d %X'

    # Exract infomration from response lines
    # https://regex101.com/r/Prf6OM/1
    _RESPONSE_REGEX = re.compile(r'^\*(\d{6})\.(0)#(\d{2})\[(\d{4})\],(?:(.*),)?(OK|FAILURE|INVALID),#([0-9A-F]{2})$')

    # Extract information from output lines
    # https://regex101.com/r/29xl6s/1
    _OUTPUT_REGEX = re.compile(r'^(\d{6}),(\d{3}),(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}),(.*),#([0-9A-F]{2})$')

    # _AVG_DT = 8.17907142857143  # s
    # _AVG_DT = 8.179
    _AVG_DT = 8.179071

    _SPEED_OF_SOUND = 343  # m/s

    def __init__(self, com_port, serial_no, timeout=0):
        '''Requires the com_port that the hydrophone is on and the hydrophone's serial number in order to begin talking to it'''
        self._serial_no = serial_no
        self._com_port = com_port
        self._timeout = timeout
        self._state = Idle(self)
        self._line_buffer = b''
        self._tags = {}

        # Attempt to open a serial port for the hydrophone. If this fails, switch to the closed state
        self._ser = None
        try:
            self._ser = serial.Serial(
                self._com_port,
                baudrate=9600,
                timeout=0,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                bytesize=serial.EIGHTBITS)
        except serial.SerialException:
            print('Unable to open port {}'.format(com_port))
            self._state = Closed(self)
    
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
            contents['info_raw'] = groups[3]
            contents['hex_sum'] = '0x' + groups[4]

            # Create another dictionary for the information contained in the info fields to be added to the contents dictionary
            info_dict = {}

            # Remove key from key value pairs in info
            info = contents['info_raw'].split(',')
            for i, pair in enumerate(info):
                values = pair.split('=')
                if len(values) == 2:
                    info[i] = values[1]
            
            # If the first value is 'STS', the line is a status line. Otherwise, its a detection line.
            if info[0] == 'STS':
                info_dict['type'] = 'status'
                info_dict['detection_count'] = int(info[1])
                info_dict['ping_count'] = int(info[2])
                info_dict['line_voltage'] = float(info[3])
                info_dict['temperature'] = float(info[4])
                info_dict['detection_memory'] = float(info[5])
                info_dict['raw_memory'] = float(info[6])
                info_dict['tilt'] = {['x', 'y', 'z'][i]: float(num) for i, num in enumerate(info[7].split(':'))}
                info_dict['output_noise'] = info[8]
                info_dict['output_ppm_noise'] = info[9]
            else:
                # Pinger detections have 5 values, whereas sensor detections have 6
                if len(info) == 5:
                    info_dict['type'] = 'pinger'
                    info_dict['code_space'] = info[0]
                    info_dict['id'] = info[1]
                    info_dict['signal_level'] = float(info[2])
                    info_dict['noise_level'] = float(info[3])
                    info_dict['channel'] = int(info[4])
                else:
                    raise ValueError('I haven\'t implemented sensor tags sorry :(')

            contents['info'] = info_dict
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
        self._ser.write(bytes(command_msg, 'utf-8'))
        return command_msg
    
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

    def _set_time(self):
        '''Send a command to the hydrophone instructing it to set its clock to the current time'''
        return self._write_command('TIME={}'.format(datetime.now().strftime(Hydrophone._TIME_FORMAT)))

    def _process_detections(self, detections):
        # Loop through detections and compute time of flight for each pinger found
        for detection in detections:
            tag_id = detection['info']['id']
            detection_time = detection['datetime']

            # If the tag has not been encountered before, add it to the dictionary of tags
            if tag_id not in self._tags:
                self._tags[tag_id] = {'first_detection_time': detection_time, 'last_detection_time': detection_time, 'first_time_of_flight': 0, 'time_of_flight': 0}
                print('{}: dt=0, modded_time=0, tof=0, tof_0=0'.format(tag_id))
            
            # Otherwise, use the information from its first detection to calculate the time of flight
            else:
                total_dt = (detection_time - self._tags[tag_id]['first_detection_time']).total_seconds()
                relative_tof = (total_dt - Hydrophone._AVG_DT / 2) % Hydrophone._AVG_DT - Hydrophone._AVG_DT / 2
                relative_distance = relative_tof * Hydrophone._SPEED_OF_SOUND

                dt = (detection_time - self._tags[tag_id]['last_detection_time']).total_seconds()
                delta_tof = (dt - Hydrophone._AVG_DT / 2) % Hydrophone._AVG_DT - Hydrophone._AVG_DT / 2
                delta_distance = delta_tof * Hydrophone._SPEED_OF_SOUND
                self._tags[tag_id]['time_of_flight'] += delta_tof
                total_tof = self._tags[tag_id]['time_of_flight']
                total_distance = total_tof * Hydrophone._SPEED_OF_SOUND
                # print('{}: total_dt={:.6f}, relative_tof={:.6f}, relative_distance={:.6f}, dt={:.6f}, delta_tof={:.6f}, delta_distance={:.6f} total_tof={:.6f}, total_distance={:.6f}'.format(tag_id, total_dt, relative_tof, relative_distance, dt, delta_tof, delta_distance, total_tof, total_distance))
                print('{}: total_dt={:.6f}, delta_tof={:.6f}, delta_distance={:.6f}, total_distance={:.6f}, signal_level={:.6f}'.format(tag_id, total_dt, delta_tof, delta_distance, total_distance, detection['info']['signal_level']))

                self._tags[tag_id]['last_detection_time'] = detection_time

                # tof_0 = self._tags[tag_id]['first_time_of_flight']
                # dt = (detection_time - self._tags[tag_id]['first_detection_time']).total_seconds()
                # modded_time = dt % Hydrophone._AVG_DT

                # # We don't expect time of flights much larger than 2 seconds, so if we see one larger than half the average
                # # timedelta, it is more likely that the time of flight of the original detection was greater than expected.
                # if modded_time > Hydrophone._AVG_DT * 0.5:
                #     # Update first time of flight
                #     tof_0 = max(tof_0, Hydrophone._AVG_DT - modded_time)
                #     self._tags[tag_id]['first_time_of_flight'] = tof_0
                
                # # Compute time of flight and record it
                # time_of_flight = (dt + tof_0) % Hydrophone._AVG_DT

                # # Adjust for floating point errors
                # if abs(time_of_flight - Hydrophone._AVG_DT) > 0.000001:
                #     time_of_flight = 0.0

                # self._tags[tag_id]['time_of_flight'] = time_of_flight
                # print('{}: total_dt={}, total_tof={}, total_distance={}'.format(tag_id, dt, time_of_flight, time_of_flight * Hydrophone._SPEED_OF_SOUND))

    def run(self):
        '''Should be run in a loop without much delay between calls'''
        self._state.run()
    
    def close(self):
        '''Close the serial connection. After this is called, the hydrophone object becomes unusable'''
        self._state = Closed(self)
        return True
    
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
    
    def is_idle(self):
        '''Check whether the hydrophone is currently idle'''
        return type(self._state) == Idle
    
    def is_starting(self):
        '''Check whether the hydrophone is currently starting'''
        return type(self._state) == Starting
    
    def is_listening(self):
        '''Check whether the hydrophone is currently listening'''
        return type(self._state) == Listening
    
    def is_stopping(self):
        '''Check whether the hydrophone is currently stopping'''
        return type(self._state) == Stopping
    
    def is_closed(self):
        '''Check whether the hydrophone is currently closed'''
        return type(self._state) == Closed

if __name__ == '__main__':
    h1 = Hydrophone('COM3', 457049)
    h1.start()
    while h1.is_starting():
        h1.run()
        time.sleep(0.01)
    while True:
        try:
            h1.run()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break
    h1.stop()
    while h1.is_stopping():
        h1.run()
        time.sleep(0.01)
    h1.close()