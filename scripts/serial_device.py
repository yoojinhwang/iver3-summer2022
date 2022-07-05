import serial
from abc import ABC, abstractmethod
from events import EventDispatcher

class SerialState(ABC):
    '''Abstract class representing a state that a serial device can be in.'''
    def __init__(self, device):
        self._device = device
        if self._device._state is not None:
            self._device._state.end()
    
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def end(self):
        pass

    def _print(self, *args, **kwargs):
        self._device._print(*args, **kwargs)

class Idle(SerialState):
    '''Reset IO and monitor serial port.'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Idle', v=1)
        self._device._dispatch('on_idle_init')
    
    def run(self):
        self._device._dispatch('on_idle_run')
        self._device._flush()
    
    def end(self):
        self._device._dispatch('on_idle_end')

class Listening(SerialState):
    '''Listen for data from the device.'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Listening', v=1)
        self._device._dispatch('on_listening_init')
    
    def run(self):
        self._device._dispatch('on_listening_run')
        try:
            # Read all available lines and process them
            lines = []
            line = self._device._readline()
            while len(line) != 0:
                lines.append(line)

                # Try to read another line
                line = self._device._readline()
        except KeyboardInterrupt:
            # Propagate KeyboardInterrupts
            raise KeyboardInterrupt
        except serial.SerialException as err:
            # If there is a serial error, switch to the closed state
            self._print('Serial exception while listening:', err, v=1)
            self._device._state = Closed(self._device)
        
        # Process the messages received
        self._device._process_lines(lines)
    
    def end(self):
        self._device._dispatch('on_listening_end')

class Closed(SerialState):
    '''Does nothing except close the serial communication on init.'''
    def __init__(self, device):
        super().__init__(device)
        self._print('Closed', v=1)
        self._device._dispatch('on_closed_init')
        if self._device._ser is not None:
            self._print('Closing serial port', v=1)
            self._device._ser.close()
    
    def run(self):
        self._device._dispatch('on_closed_run')
    
    def end(self):
        self._device._dispatch('on_closed_end')

class SerialDevice(EventDispatcher):
    '''
    A base class containing methods for handling devices that communicate over
    a serial port.

    The device is modeled as a state machine with states including Idle,
    Listening, and Closed. In the Idle state, the device flushes all input and
    output and monitors for any interruptions in the serial communication. In
    the Listening state, the device collects all lines received and triggers
    callbacks for each. In the Closed state, the serial port is closed
    permanently.

    Built off of pyserial's serial.Serial class.
    '''

    def __init__(self, port, verbosity=0, event_types=[], **kwargs):
        '''
        Start serial communcation. The device starts in the Idle state.

        Parameters
        ----------
        port : str
            The port the device is on.
        verbosity : int
            Controls how much the device prints. 0 = noting, 1 = some status
            messages, >2 = defined by subclasses. In general, at higher
            verbosity levels the device will print everything it would have
            printed at lower levels and more.
        events : list
            A list of event names to register in addition to the default ones.
        '''
        super().__init__([
            'on_idle_init',
            'on_idle_run',
            'on_idle_end',
            'on_listening_init',
            'on_listening_run',
            'on_listening_end',
            'on_closed_init',
            'on_closed_run',
            'on_closed_end',
            'on_line'
        ] + event_types)
        self._port = port
        self._verbosity = verbosity
        self._state = None
        self._state = Idle(self)
        self._line_buffer = b''

        # Attempt to open a serial port. If this fails, switch to the closed
        # state.
        self._ser = None
        try:
            self._ser = serial.Serial(
                self._port,
                baudrate=kwargs.get('baudrate', 9600),
                timeout=kwargs.get('timeout', 0),
                stopbits=kwargs.get('stopbits', serial.STOPBITS_ONE),
                parity=kwargs.get('parity', serial.PARITY_NONE),
                bytesize=kwargs.get('bytesize', serial.EIGHTBITS))
        except serial.SerialException:
            self._print('Unable to open port {}'.format(self._port), v=1)
            self._state = Closed(self)
    
    def __repr__(self):
        return '<SerialDevice({}) at {}>'.format(self._port, hex(id(self)))
    
    def __str__(self):
        return 'SerialDevice({})'.format(self._port)

    def _print(self, *args, v=1, **kwargs):
        '''Print if the device's verbosity is greater than or equal to v.'''
        if v <= 0:
            raise ValueError('v should always be greater than 0.')
        if self._verbosity >= v:
            print(self, *args, **kwargs)
    
    def _flush(self):
        '''Flush any input or output that has not been processed.'''
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
        '''Write to the serial port. Wrapper over serial.Serial.write.'''
        return self._ser.write(*args, **kwargs)

    def _process_lines(self, lines):
        '''Process the lines received in the listening state.'''
        for line in lines:
            self._dispatch('on_line', line)

    def run(self):
        '''Should be run in a loop without much delay between calls.'''
        self._state.run()
    
    def close(self):
        '''
        Close the serial connection. After this is called, the serial device
        object becomes unusable.

        Returns
        -------
        bool
            True if the device succesfully closed, False otherwise, including
            if the device is already closed.
        '''
        if type(self._state) != Closed:
            self._state = Closed(self)
            return True
        return False
    
    def start(self):
        '''
        Set the device to the Listening state from the Idle state.

        Returns
        -------
        bool
            True if the device succesfully started, False otherwise.
        '''
        if type(self._state) == Idle:
            self._state = Listening(self)
            return True
        return False

    def stop(self):
        '''
        Set the device to the Idle state from the Listening state.

        Returns
        -------
        bool
            True if the device succesfully stopped, False otherwise.
        '''
        if type(self._state) == Listening:
            self._state = Idle(self)
            return True
        return False

    def is_idle(self):
        '''
        Check whether the device is currently in the Idle state.
        
        Returns
        -------
        bool
        '''
        return type(self._state) == Idle
    
    def is_listening(self):
        '''
        Check whether the device is currently in the Listening state.

        Returns
        -------
        bool
        '''
        return type(self._state) == Listening
    
    def is_closed(self):
        '''
        Check whether the device is currently in the Closed state.
        
        Returns
        -------
        bool
        '''
        return type(self._state) == Closed
    
    def on_idle_init(self, callback):
        '''
        Register a callback to run when the device enters the Idle state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_idle_init', callback)
    
    def on_idle_run(self, callback):
        '''
        Register a callback to run while the device is in the Idle state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_idle_run', callback)
    
    def on_idle_end(self, callback):
        '''
        Register a callback to run when the device leaves the Idle state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_idle_end', callback)
    
    def on_listening_init(self, callback):
        '''
        Register a callback to run when the device enters the Listening state.

        Parameters
        ----------
        callback : callable
            No parameters are passed in        
        '''
        self._register('on_listening_init', callback)
    
    def on_listening_run(self, callback):
        '''
        Register a callback to run while the device is in the Listening state.

        Parameters
        ----------
        callback : callable
            No parameters are passed in        
        '''
        self._register('on_listening_run', callback)
    
    def on_listening_end(self, callback):
        '''
        Register a callback to run when the device leaves the Listening state.

        Parameters
        ----------
        callback : callable
            No parameters are passed in        
        '''
        self._register('on_listening_end', callback)
    
    def on_closed_init(self, callback):
        '''
        Register a callback to run when the device enters the Closed state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_closed_init', callback)
    
    def on_closed_run(self, callback):
        '''
        Register a callback to run while the device is in the Closed state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_closed_run', callback)

    def on_closed_end(self, callback):
        '''
        Register a callback to run when the device leaves the Closed state.
        
        Parameters
        ----------
        callback : callable
            No parameters are passed in
        '''
        self._register('on_closed_end', callback)
    
    def on_line(self, callback):
        '''
        Register a callback to run when the device receives a line.
        
        Parameters
        ----------
        callback : callable
            Parameters
            ----------
            line : str
                The line received from the serial device
        '''
        self._register('on_line', callback)