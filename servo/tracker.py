import logging
import time
import numpy as np
import traceback
import collections
from datetime import datetime

from . import core
from . import config
from . import utils
from .fsm import Transition

log = logging.getLogger(__name__)


class Record():
    """Class to record tracker the data obtained by the tracker class.

    This class contains numpy arrays to store the history of OPD, tip,
    tilt, and time values. It provides methods to append new data and
    save the recorded data to a pandas csv file.

    For fast append of new data the class uses a numpy arrays with a
    fixed size (config.TRACKER_RECORD_SIZE) and a pointer to the
    current position. When the buffer is full, the data is
    automatically saved and it wraps around and starts overwriting the
    oldest data. The save method saves the data in a pandas DataFrame
    and then to a csv file, including only the valid data (up to the
    current position if the buffer is not full, or the entire buffer
    if it is full).
    """
    def __init__(self):        
        self.opd = np.zeros(config.TRACKER_RECORD_SIZE)
        self.tip = np.zeros(config.TRACKER_RECORD_SIZE)
        self.tilt = np.zeros(config.TRACKER_RECORD_SIZE)
        self.time = np.zeros(config.TRACKER_RECORD_SIZE)
        self.velocity = np.zeros(config.TRACKER_RECORD_SIZE)
        self.position = 0
        self.full = False

    def append(self, opd, tip, tilt, time, velocity):
        self.opd[self.position] = opd
        self.tip[self.position] = tip
        self.tilt[self.position] = tilt
        self.time[self.position] = time
        self.velocity[self.position] = velocity
        self.position += 1
        if self.position >= config.TRACKER_RECORD_SIZE:
            self.full = True
            self.position = 0
            self.save()

    def save(self):
        """Save the recorded data to a pandas csv file.
        The file is named with the current timestamp."""
        import pandas as pd
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'servo_tracker_record_{timestamp}.csv'
        if self.full:
            data = {
                'opd': self.opd,
                'tip': self.tip,
                'tilt': self.tilt,
                'velocity': self.velocity,
                'time': self.time,
            }
        else:
            data = {
                'opd': self.opd[:self.position],
                'tip': self.tip[:self.position],
                'tilt': self.tilt[:self.position],
                'time': self.time[:self.position],
                'velocity': self.velocity[:self.position],
            }
        df = pd.DataFrame(data)
        
        # lors du save
        df["time_iso"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.to_csv(filename, index=False)
        log.info(f'Tracker data saved to {filename}')
        
        

class Tracker(core.Worker):

    def __init__(self, data, events):
        super().__init__(data, events)

        self.table |= {
            (self.State.RUNNING, self.Event.START_RECORDING): Transition(
                self.State.RUNNING, action=self._start_recording),
            (self.State.RUNNING, self.Event.STOP_RECORDING): Transition(
                self.State.RUNNING, action=self._stop_recording),
        }

        self.frequencies = [int(ifreq) for ifreq in config.TRACKER_STATS_FREQUENCIES]
        if config.TRACKER_FREQUENCY not in self.frequencies:
            self.frequencies.append(config.TRACKER_FREQUENCY)
            
        log.info(f'Tracker stats frequencies: {self.frequencies} Hz')

        self.opd_buffer = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)
        self.tip_buffer = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)
        self.tilt_buffer = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)
        self.time_buffer = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)
        
        self.last_opds = dict()
        for ifreq in self.frequencies:
            self.last_opds[ifreq] = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)
            
        self.last_times = dict()
        for ifreq in self.frequencies:
            self.last_times[ifreq] = utils.RingBuffer(config.TRACKER_BUFFER_SIZE)

        self.window_sizes = dict()
        for ifreq in self.frequencies:
            self.window_sizes[ifreq] = min(max(1, config.TRACKER_FREQUENCY // ifreq), config.TRACKER_BUFFER_SIZE)
        log.info(f'Tracker window sizes: {self.window_sizes}')

        self.record = Record()
        self.is_recording = False

        self.start_perf = time.perf_counter()
        self.start_wall = time.time()

        self._last_velocity = 0.0

        log.info('Tracker initialized')


    def _compute_stats(self, frequency):
        window_size = self.window_sizes[frequency]
        meanopd = self.opd_buffer.mean_last(window_size)
        meantip = self.tip_buffer.mean_last(window_size)
        meantilt = self.tilt_buffer.mean_last(window_size)
        meantime = self.time_buffer.mean_last(window_size)
        
        self.data[f'Tracker.opd_{frequency}'][0] = float(meanopd)
        self.data[f'Tracker.opd_std_{frequency}'][0] = float(self.opd_buffer.std_last(window_size))
        self.data[f'Tracker.tip_{frequency}'][0] = float(meantip)
        self.data[f'Tracker.tilt_{frequency}'][0] = float(meantilt)

        self.last_opds[frequency].append(meanopd)
        self.last_times[frequency].append(meantime)

        if len(self.last_opds[frequency]) > window_size:
            _time = (meantime - self.last_times[frequency][-window_size-1])
            if np.isfinite(_time) and _time > 0:
                velocity = (meanopd - self.last_opds[frequency][-window_size-1]) / _time
            else:
                velocity = self._last_velocity
                log.warning(f'could not compute velocity @ {frequency}: invalid time difference: {_time}')
            if not np.isfinite(velocity):
                velocity = self._last_velocity
                log.warning(f'could not compute velocity @ {frequency}: non-finite value: {velocity}')
                
            self._last_velocity = float(velocity)
            self.data[f'Tracker.velocity_{frequency}'][0] = float(velocity)
                
        else:
            velocity = np.nan

        if self.is_recording and frequency == config.TRACKER_RECORD_FREQUENCY:
            # meantime comes from time.perf_counter() and must be
            # converted to a real timestamp by adding the start time of
            # the tracker
            meantimestamp = self.start_wall + (meantime - self.start_perf)
            self.record.append(meanopd, meantip, meantilt, meantimestamp, velocity)
            

    def _start_recording(self, _):
        log.info('Starting tracker recording')
        self.is_recording = True

    def _stop_recording(self, _):
        log.info('Stopping tracker recording')
        self.is_recording = False
        self.record.save()
        
    def loop_once(self):
        
        self.frame_time = time.perf_counter()
        
        self.opd_buffer.append(np.mean(self.data['IRCamera.mean_opd'][0]))
        self.tip_buffer.append(np.mean(self.data['IRCamera.tip'][0]))
        self.tilt_buffer.append(np.mean(self.data['IRCamera.tilt'][0]))
        self.time_buffer.append(self.frame_time)

        for ifreq in self.frequencies:
            self._compute_stats(ifreq)

        loop_end_time = time.perf_counter()
        time.sleep(max(0, 1./config.TRACKER_FREQUENCY - (loop_end_time - self.frame_time)))
        self.data['Tracker.frequency'][0] = 1/(time.perf_counter() - self.frame_time)

    def cleanup(self):
        pass


