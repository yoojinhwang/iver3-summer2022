from events import EventDispatcher
from abc import abstractmethod
import bisect

class Filter(EventDispatcher):
    PREDICTION = 0
    CORRECTION = 1

    def _insort_key(x):
        return x[1][0]

    def __init__(self):
        super().__init__(event_types=['on_prediction', 'on_correction'])
        self.reset()

    def reset(self):
        self._queue = []
        self._current_time = None
        self._last_step_time = None
        self._last_prediction_time = None
        self._last_correction_time = None

    def queue_prediction(self, timestamp, data):
        step = (Filter.PREDICTION, (timestamp, data))
        bisect.insort(self._queue, step, key=Filter._insort_key)
    
    def queue_correction(self, timestamp, data):
        step = (Filter.CORRECTION, (timestamp, data))
        bisect.insort(self._queue, step, key=Filter._insort_key)
    
    @abstractmethod
    def _prediction_step(self, timestamp, data, dt):
        self._dispatch('on_prediction', timestamp, data, dt)

    @abstractmethod
    def _correction_step(self, timestamp, data, dt):
        self._dispatch('on_correction', timestamp, data, dt)

    def iterate(self):
        step_type, (timestamp, data) = self._queue.pop(0)

        # Initialize _last_step_time if necessary
        if self._last_step_time is None:
            self._last_step_time = timestamp

        # Update _current_time
        self._current_time = timestamp

        # Run prediction and correction steps
        if step_type == Filter.PREDICTION:
            # Initialize _last_prediction_time if necessary
            if self._last_prediction_time is None:
                self._last_prediction_time = timestamp

            # Run the prediction step
            dt = (timestamp - self._last_prediction_time).total_seconds()
            self._prediction_step(timestamp, data, dt)

            # Update _last_prediction_time if necessary
            self._last_prediction_time = timestamp
        elif step_type == Filter.CORRECTION:
            # Initialize _last_correction_time if necessary
            if self._last_correction_time is None:
                self._last_correction_time = timestamp
            
            # Run the correction step
            dt = (timestamp - self._last_correction_time).total_seconds()
            self._correction_step(timestamp, data, dt)

            # Update _last_correction_time if necessary
            self._last_correction_time = timestamp

        # Update _last_step_time
        self._last_step_time = timestamp

    def run(self):
        while len(self._queue) > 0:
            self.iterate()
    
    def on_prediction(self, callback):
        self._register('on_prediction', callback)
    
    def on_correction(self, callback):
        self._register('on_correction', callback)