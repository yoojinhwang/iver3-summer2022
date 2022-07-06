import operator
from functools import partial
from abc import ABC

class EventDispatcher(ABC):
    '''
    An EventDispatcher registers a list of events (hashables) and stores
    a list of callbacks for each one. It provides methods for adding callbacks
    for each event name and dispatching events, i.e. calling the callbacks for
    a given event name. This class is meant to be subclassed.
    '''
    def __init__(self, event_types):
        '''Initialize with the given event types.'''
        self.__event_types = event_types
        self.__callback_dict = {event_type: [] for event_type in self.__event_types}

    def _add_event_types(self, event_types):
        '''Add a list of event types.'''
        for event_type in event_types:
            if event_type not in self.__event_types:
                self.__event_types.append(event_type)
                self.__callback_dict[event_type] = []

    def _set_callbacks(self, event_type, callbacks):
        '''Set the list of callbacks for a given event type.'''
        # Raise an error if the given event_type is not valid
        if event_type not in self.__event_types:
            raise ValueError('{} is not a registered event type.'.format(event_type))
        else:
            self.__callback_dict[event_type] = callbacks

    def _get_callbacks(self, event_type):
        '''Get the list of callbacks for a given event type.'''
        # Raise an error if the given event_type name is not valid
        if event_type not in self.__event_types:
            raise ValueError('{} is not a registered event type.'.format(event_type))
        else:
            return self.__callback_dict[event_type]            

    def _register(self, event_type, callback):
        '''Register a callback for a given type.'''
        # Make sure the provided callback is callable
        if not callable(callback):
            raise ValueError('{} is not callable'.format(callback))

        # Append the callback
        self._get_callbacks(event_type).append(callback)
    
    def _remove_callback(self, event_type, callback):
        '''Remove a callback for a given type.'''
        # Filter out all instances of the given callback
        self._set_callbacks(
            event_type,
            list(
                filter(
                    partial(operator.is_not, callback),
                    self._get_callbacks(event_type))))

    def _dispatch(self, event_type, *args, **kwargs):
        '''Dispatch callbacks for a given type.'''
        for callback in self._get_callbacks(event_type):
            callback(*args, **kwargs)