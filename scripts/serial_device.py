import serial
from abc import ABC, abstractmethod

class SerialState(ABC):
    '''Abstract class representing a state that a serial device can be in'''
    def __init__(self, device):
        self._device = device
    
    @abstractmethod
    def run(self):
        pass

    def _print(self, *args, **kwargs):
        self._device._print(*args, **kwargs)

class Idle(SerialState):
    '''Reset IO and monitor serial port'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Idle', v=1)
    
    def run(self):
        self._device._flush()

class Listening(SerialState):
    '''Listen for data from the device'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Listening', v=1)
    
    def run(self):
        try:
            # Read all available lines and process them
            lines = []
            line = self._device._readline()
            while len(line) != 0:
                lines.append(line)

                # Try to read another line
                line = self._device._readline()
            # Process the messages received
            self._device._process_lines(lines)
        except KeyboardInterrupt:
            # Propagate KeyboardInterrupts
            raise KeyboardInterrupt
        except serial.SerialException as err:
            # If there is a serial error, switch to the closed state
            self._print('Serial exception while listening:', err, v=1)
            self._device._state = Closed(self._device)
        except Exception as err:
            self._print('Exception while listening:', err, v=1)

class Closed(SerialState):
    '''Does nothing except close the serial communication on init'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Closed', v=1)
        if self._device._ser is not None:
            self._print('Closing serial port', v=1)
            self._device._ser.close()
    
    def run(self):
        pass

class SerialDevice(ABC):
    '''
    Abstract class to set up some methods for handling devices that
    communicate over serial.
    '''

    def __init__(self, com_port, verbosity=0, **kwargs):
        self._com_port = com_port
        self._verbosity = verbosity
        self._state = Idle(self)
        self._line_buffer = b''

        # Attempt to open a serial port. If this fails, switch to the closed
        # state.
        self._ser = None
        try:
            self._ser = serial.Serial(
                self._com_port,
                baudrate=kwargs.get('baudrate', 9600),
                timeout=kwargs.get('timeout', 0),
                stopbits=kwargs.get('stopbits', serial.STOPBITS_ONE),
                parity=kwargs.get('parity', serial.PARITY_NONE),
                bytesize=kwargs.get('bytesize', serial.EIGHTBITS))
        except serial.SerialException:
            self._print('Unable to open port {}'.format(com_port), v=1)
            self._state = Closed(self)
    
    def __repr__(self):
        return '<SerialDevice({}) at {}>'.format(self._com_port, hex(id(self)))
    
    def __str__(self):
        return 'SerialDevice({})'.format(self._com_port)

    def _print(self, *args, v=0, **kwargs):
        '''Print if the device's verbosity is greater than or equal to v'''
        if self._verbosity >= v:
            print(self, *args, **kwargs)
    
    def _flush(self):
        '''Flush any input or output that has not been processed'''
        try:
            if self._ser.in_waiting:
                self._ser.reset_input_buffer()
            if self._ser.out_waiting:
                self._ser.reset_output_buffer()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except serial.SerialException as err:
            self._print('Serial exception:', err, v=1)
            self._state = Closed(self)
        except Exception as err:
            self._print('Exception:', err, v=1)
    
    def _readline(self):
        '''
        Wrapper over serial.Serial.readline with some extra functionality to
        handle being called repeatedly in a while loop.
        '''

        # If no data is waiting to be read, return an empty string.
        if self._ser.in_waiting:
            # Read a line from the serial port. Since this is being called in
            # a loop repeatedly, the result of readline may not be an entire
            # line. Only the first part of a line might be available. The
            # result is appended to the device's read buffer for use when an
            # entire line is ready.
            partial_line = self._ser.readline()
            self._line_buffer += partial_line

        # Check if there is an entire line in the read buffer and if so,
        # remove it from the buffer and return it.
        if b'\r\n' in self._line_buffer:
            line, rest = self._line_buffer.split(b'\r\n', maxsplit=1)
            self._line_buffer = rest

            # Sometimes the device sends trash data that can't be converted to
            # a string?? If so, ignore it.
            try:
                return line.decode('utf-8')
            except UnicodeDecodeError as err:
                self._print('Discarding line: ', err, v=1)
                return ''
        return ''
    
    def _write(self, *args, **kwargs):
        return self._ser.write(*args, **kwargs)

    @abstractmethod
    def _process_lines(self, lines):
        '''Process the lines received in the listening state'''
        pass

    def run(self):
        '''Should be run in a loop without much delay between calls'''
        self._state.run()
    
    def close(self):
        '''
        Close the serial connection. After this is called, the serial device
        object becomes unusable.
        '''
        if type(self._state) != Closed:
            self._state = Closed(self)
            return True
        return False
    
    @abstractmethod
    def start(self):
        '''Get the device into the Listening state'''
        pass

    @abstractmethod
    def stop(self):
        '''Get the device into the Idle state'''
        pass

    def is_idle(self):
        '''Check whether the device is currently idle'''
        return type(self._state) == Idle
    
    def is_listening(self):
        '''Check whether the device is currently listening'''
        return type(self._state) == Listening
    
    def is_closed(self):
        '''Check whether the device is currently closed'''
        return type(self._state) == Closed
    
    @abstractmethod
    def is_starting(self):
        '''Check whether the device is currently starting'''
        pass

    @abstractmethod
    def is_stopping(self):
        '''Check whether the device is currently stopping'''
        pass