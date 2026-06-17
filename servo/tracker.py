import logging
import time
import numpy as np
import traceback
import collections
from datetime import datetime

from . import core
from . import config
from . import utils
from . import faster
from .fsm import Transition

log = logging.getLogger(__name__)


class Record():
    """Class to record data obtained by tracker class.

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
    def __init__(self, ir_image_shape):
        self.opd = np.zeros(config.TRACKER_RECORD_SIZE)
        self.tip = np.zeros(config.TRACKER_RECORD_SIZE)
        self.tilt = np.zeros(config.TRACKER_RECORD_SIZE)
        self.time = np.zeros(config.TRACKER_RECORD_SIZE)
        self.velocity = np.zeros(config.TRACKER_RECORD_SIZE)
        self.ir_image_intensity = np.zeros(config.TRACKER_RECORD_SIZE)
        self.ir_image_shape = ir_image_shape
        self._init_ir_images()
        self.position = 0
        self.full = False

    def _init_ir_images(self):
        self.ir_images = np.zeros((config.TRACKER_RECORD_SIZE, *self.ir_image_shape), dtype=config.FRAME_DTYPE)
        self.ir_images_normalized = np.zeros_like(self.ir_images)        
        

    def append(self, opd, tip, tilt, time, velocity, ir_image, ir_image_nomalized, ir_image_intensity):
        self.opd[self.position] = opd
        self.tip[self.position] = tip
        self.tilt[self.position] = tilt
        self.time[self.position] = time
        self.velocity[self.position] = velocity
        self.ir_images[self.position] = ir_image
        self.ir_images_normalized[self.position] = ir_image_nomalized
        self.ir_image_intensity[self.position] = ir_image_intensity
        
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
        ircube_filename = f'servo_tracker_ircube_{timestamp}.npy'
        ircube_normalized_filename = f'servo_tracker_ircube_normalized_{timestamp}.npy'
        
        if self.full:
            data = {
                'opd': self.opd,
                'tip': self.tip,
                'tilt': self.tilt,
                'velocity': self.velocity,
                'time': self.time,
                'ir_image_intensity': self.ir_image_intensity,
            }
            ir_cube = self.ir_images.copy()
            ir_cube_normalized = self.ir_images_normalized.copy()
        else:
            data = {
                'opd': self.opd[:self.position],
                'tip': self.tip[:self.position],
                'tilt': self.tilt[:self.position],
                'time': self.time[:self.position],
                'velocity': self.velocity[:self.position],
                'ir_image_intensity': self.ir_image_intensity[:self.position],
            }
            ir_cube = self.ir_images[:self.position].copy()
            ir_cube_normalized = self.ir_images_normalized[:self.position].copy()
            
        # saving date
        df = pd.DataFrame(data)
        df["time_iso"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.to_csv(filename, index=False)
        log.info(f'Tracker data saved to {filename}')

        # saving ir cube
        np.save(ircube_filename, ir_cube)
        log.info(f'Tracker ir cube saved to {ircube_filename}')
        np.save(ircube_normalized_filename, ir_cube_normalized)
        log.info(f'Tracker normalized ir cube saved to {ircube_normalized_filename}')

    def get_normalization_coeffs(self, width):

        if self.position < 2000:
            log.warning(f'Not enough data to compute normalization coefficients: only {self.position} frames recorded')
            return None, None
        
        ircube = np.ascontiguousarray(self.ir_images[:self.position].copy())

        # save ircube used for normalization as an npy file for later analysis with a timestamp in the path
        np.save(f'servo_tracker_normalization_ircube_{time.strftime("%Y%m%d-%H%M%S")}.npy', ircube)
        
        # recompute normalization maps
        # ix, iy = np.array(ircube.shape[1:])//2
        # profile_len = ircube.shape[1]
        # hprofiles = np.empty((ircube.shape[0], ircube.shape[1]), dtype=np.float32)
        # vprofiles = np.empty_like(hprofiles)
        # for iz, iframe in enumerate(ircube):
        #     ivprof, ihprof, _ = faster.compute_profiles_local_f32(iframe.astype(np.float32, copy=False).T,
        #                                                           int(iy), int(ix), int(width),
        #                                                           int(profile_len), True)
        #     hprofiles[iz, :] = ihprof.copy()
        #     vprofiles[iz, :] = ivprof.copy()

        try:
            log.info('Computing normalization coefficients with a cube of shape: ' + str(ircube.shape))
            roinorm_min, roinorm_max = utils.get_roi_normalization_coeffs(ircube)
        except Exception as e:
            log.error(f'Error computing normalization coefficients')
            return None, None

        return roinorm_min.T, roinorm_max.T
        
        

class Tracker(core.Worker):

    def __init__(self, data, events):
        super().__init__(data, events)

        self.table |= {
            (self.State.RUNNING, self.Event.START_RECORDING): Transition(
                self.State.RUNNING, action=self._start_recording),
            (self.State.RUNNING, self.Event.STOP_RECORDING): Transition(
                self.State.RUNNING, action=self._stop_recording),
            (self.State.RUNNING, self.Event.NORMALIZE): Transition(
                self.State.RUNNING, action=self._normalize),
        }
        
        self.frequencies = [ifreq for ifreq in config.TRACKER_STATS_FREQUENCIES]
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
            self.window_sizes[ifreq] = min(max(1, config.TRACKER_FREQUENCY // ifreq),
                                           config.TRACKER_BUFFER_SIZE)
        log.info(f'Tracker window sizes: {self.window_sizes}')

        self._init_ir_image_specs()
        
        self.record = Record(self.ir_image_shape)
        self.is_recording = False

        self.start_perf = time.perf_counter()
        self.start_wall = time.time()

        self._last_velocity = 0.0

        log.info('Tracker initialized')

    def _init_ir_image_specs(self):
        self.profile_len = int(self.data['IRCamera.profile_len'][0])
        self.ir_image_size = self.profile_len**2
        self.ir_image_shape = (self.profile_len, self.profile_len)

        


    def _compute_stats(self, frequency):
        window_size = self.window_sizes[frequency]
        meanopd = self.opd_buffer.mean_last(window_size)
        meantip = self.tip_buffer.mean_last(window_size)
        meantilt = self.tilt_buffer.mean_last(window_size)
        meantime = self.time_buffer.mean_last(window_size)

        frequency_str = f"{int(frequency)}" if frequency % 1 == 0 else f"{frequency:.1f}"

        self.data[f'Tracker.opd_{frequency_str}'][0] = float(meanopd)
        self.data[f'Tracker.opd_std_{frequency_str}'][0] = float(self.opd_buffer.std_last(window_size))
        self.data[f'Tracker.tip_{frequency_str}'][0] = float(meantip)
        self.data[f'Tracker.tilt_{frequency_str}'][0] = float(meantilt)

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
                log.warning(f'could not compute velocity @ {frequency}: non-finite value: {velocity}')
                velocity = self._last_velocity

                
            self._last_velocity = float(velocity)
            self.data[f'Tracker.velocity_{frequency_str}'][0] = float(velocity)
                
        else:
            velocity = np.nan

        # meantime comes from time.perf_counter() and must be
        # converted to a real timestamp by adding the start time of
        # the tracker
        meantimestamp = self.start_wall + (meantime - self.start_perf)
        ir_image = self.data['IRCamera.roi'][:self.ir_image_size].copy()
        ir_image = ir_image.reshape(self.ir_image_shape).T

        ir_image_normalized = self.data['IRCamera.roi_normalized'][:self.ir_image_size].copy()
        ir_image_normalized = ir_image_normalized.reshape(self.ir_image_shape).T

        ir_image_intensity = np.sum(ir_image)#np.sum(ir_image[ir_image > np.percentile(ir_image, 99)])
        self.data[f'Tracker.ir_image_intensity_{frequency_str}'][0] = ir_image_intensity

        # detect roi size change
        _profile_len = int(self.data['IRCamera.profile_len'][0])
        if _profile_len != self.profile_len:
            log.warning(f'IR image size changed from {self.ir_image_size} to {_profile_len}. Reinitializing IR image buffers.')
            self._init_ir_image_specs()
            self.record._init_ir_images()

        

        if self.is_recording and frequency == config.TRACKER_RECORD_FREQUENCY:
            self.record.append(meanopd, meantip, meantilt, meantimestamp, velocity,
                               ir_image, ir_image_normalized, ir_image_intensity)
            

    def _start_recording(self, _):
        log.info('Starting tracker recording')
        if self.is_recording:
            log.warning('Tracker recording is already running. Saving current record before starting a new one.')
            self.record.save()
        else:
            self.is_recording = True
            self.data['Tracker.is_recording'][0] = True

    def _stop_recording(self, _):
        log.info('Stopping tracker recording')
        self.is_recording = False
        self.data['Tracker.is_recording'][0] = False
        self.record.save()

    def _normalize(self, _):
        log.info('Normalizing tracker data')
        width = int(self.data['params.PROFILE_WIDTH'][0])

        
        import inspect
        print("STACK DEPTH:", len(inspect.stack()))

        roinorm_min, roinorm_max = self.record.get_normalization_coeffs(width)
        if roinorm_min is None or roinorm_max is None:
            log.warning('Could not compute normalization coefficients.')
            return
        
        self.data['Servo.roinorm_min'][:self.profile_len**2] = roinorm_min.astype(config.FRAME_DTYPE).flatten()
        self.data['Servo.roinorm_max'][:self.profile_len**2] = roinorm_max.astype(config.FRAME_DTYPE).flatten()
        
        
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


