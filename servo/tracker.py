import logging
import time
import numpy as np
import traceback
import collections

from . import core
from . import config
from . import utils

log = logging.getLogger(__name__)


class Tracker(core.Worker):

    def __init__(self, data, events):
        super().__init__(data, events)

        self.frequencies = [int(ifreq) for ifreq in config.TRACKER_STATS_FREQUENCIES]
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
            velocity = (meanopd - self.last_opds[frequency][-window_size]) / (self.frame_time - self.last_times[frequency][-window_size])
            self.data[f'Tracker.velocity_{frequency}'][0] = float(velocity)
            

        
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


