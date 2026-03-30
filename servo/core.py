import numpy as np
import logging
from multiprocessing import shared_memory
from enum import IntEnum, auto
import traceback
import time

from . import config
from . import params
from . import state_store

from .fsm import WorkerState, StateMachine, Transition

log = logging.getLogger(__name__)


class Worker(StateMachine):

    def __init__(self, data, events, State=None):
        self.data = data
        self.events = events
        self.classname = self.__class__.__name__
        
        if State is None:
            self.State = WorkerState
        else:
            self.State = State

        # construct event class
        events_dict = {event.upper(): auto() for event in getattr(config, f'{self.__class__.__name__.upper()}_EVENTS', [])}
        events_dict['START'] = auto()
        events_dict['STOP'] = auto()

        self.stop_event = self.events[f'{self.classname}.' + 'stop']
                
        self.Event = IntEnum('Event', events_dict)

        table = {
            (self.State.IDLE, self.Event.START): Transition(
                self.State.RUNNING, action=self._start),
            (self.State.IDLE, self.Event.STOP): Transition(
                self.State.STOPPED, action=self._stop),
            (self.State.RUNNING, self.Event.STOP): Transition(
                self.State.STOPPED, action=self._stop),
        }
    
        super().__init__(self.State.IDLE, table)

    def poll(self):    
        evs = self.events
        classname = self.__class__.__name__
        events_dict = list(getattr(config, f'{classname.upper()}_EVENTS', []))
        if 'start' not in events_dict:
            events_dict.append('start')
        if 'stop' not in events_dict:
            events_dict.append('stop')
        
        for iname in events_dict:
            if evs.get(f'{classname}.' + iname) and evs[f'{classname}.' + iname].is_set():
                evs[f'{classname}.' + iname].clear()
                self.dispatch(getattr(self.Event, iname.upper()), payload=None)

    def on_enter_running(self, _):
        log.info(f"{self.classname} enter running")

        # running loop
        while True:
            try:
                self.poll()
                
                if self.state is self.State.STOPPED:
                    break
                if self.stop_event.is_set():
                    break
                
                
                if hasattr(self, 'loop_once'):
                    try:
                        self.loop_once()
                    except Exception as e:
                        log.error('Exception at loop_once:\n' + traceback.format_exc())
                        self.dispatch(self.Event.STOP)
                
                time.sleep(1e-2)
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.dispatch(self.Event.STOP)
            except Exception as e:
                log.error('Exception at running:\n' + traceback.format_exc())
                self.dispatch(self.Event.STOP)
                
                
    def on_exit_running(self, _):
        log.info(f"{self.classname} exit running")
        
    def cleanup(self):
        """Optional: close hardware / files / buffers."""
        pass

    def _start(self, _):
        log.info(f'Starting {self.classname}')
        
    def _stop(self, _):
        self.stop()

    def _publish_state(self, state=None):
        if state is None:
            state = self.state
        
        super()._publish_state(state=state)
        
        try:
            self.data[f'{self.classname}.state'][0] = float(state.value)
        except Exception as e:
            log.warning(f'could not publish state for {self.classname}, error: {e}, traceback: {traceback.format_exc()}')

    def stop(self):
        """Always called when shutting down."""
        log.info('Stopping worker')
        self.cleanup()


class SharedData(object):
    def __init__(self):
        self.shms = dict()
        self.arrs = dict()
        self.stored_names = list()
        self.state = state_store.StateStore(
            app_name="servo",
            filename="state.json",
            schema_version=1,
            autosave_interval=10.0,
            only_main_process_writes=True
        )

        self.add_array('IRCamera.last_frame', np.zeros(config.FULL_FRAME_SIZE, dtype=config.FRAME_DTYPE))
        self.add_value('IRCamera.frame_size', int(config.FULL_FRAME_SIZE))
        self.add_value('IRCamera.frame_dimx', int(config.FULL_FRAME_SHAPE[0]))
        self.add_value('IRCamera.frame_dimy', int(config.FULL_FRAME_SHAPE[1]))
        self.add_value('IRCamera.initialized', False)

        self.add_array('IRCamera.hprofile', np.zeros(config.FULL_FRAME_SHAPE[0], dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile', np.zeros(config.FULL_FRAME_SHAPE[1], dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.hprofile_normalized', np.zeros(config.FULL_FRAME_SHAPE[0], dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile_normalized', np.zeros(config.FULL_FRAME_SHAPE[0], dtype=config.FRAME_DTYPE))

        self.add_array('IRCamera.hprofile_levels', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile_levels', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.hprofile_levels_pos', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile_levels_pos', np.zeros(3, dtype=config.FRAME_DTYPE))

        self.add_value('IRCamera.profile_x', int(config.DEFAULT_PROFILE_POSITION[0]), stored=True)
        self.add_value('IRCamera.profile_y', int(config.DEFAULT_PROFILE_POSITION[1]), stored=True)
        self.add_value('IRCamera.profile_len', int(config.DEFAULT_PROFILE_LEN), stored=True)
        self.add_value('IRCamera.profile_width', int(config.DEFAULT_PROFILE_WIDTH), stored=True)

        self.add_array('IRCamera.roi', np.zeros(config.FULL_FRAME_SIZE, dtype=config.FRAME_DTYPE))

        self.add_array('IRCamera.angles', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('IRCamera.last_angles', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('IRCamera.opds', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_value('IRCamera.mean_opd', float(0.))
        self.add_value('IRCamera.mean_opd_offset', float(0.), stored=True)
        self.add_value('IRCamera.tip', float(0.), stored=True)
        self.add_value('IRCamera.tilt', float(0.), stored=True)

        self.add_value('IRCamera.mean_sampling_time', float(np.nan))
        self.add_value('IRCamera.fps', float(np.nan))
        self.add_value('IRCamera.loop_time', float(np.nan))
        self.add_value('IRCamera.loop_fps', float(np.nan))
        self.add_value('IRCamera.lost_frames', int(0))
        self.add_value('IRCamera.state', float(0), stored=False)
        self.add_value('IRCamera.full_output', float(0), stored=False)

        # selected pixels: 0:none, 1:side, 2:center
        self.add_value('Servo.state', float(0), stored=False)
        
        self.add_array('Servo.pixels_x', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                  dtype=int), stored=True)
        self.add_array('Servo.pixels_y', np.zeros(config.FULL_FRAME_SHAPE[1],
                                                  dtype=int), stored=True)
        
        
        self.add_array('Servo.roinorm_min', np.zeros(config.FULL_FRAME_SIZE,
                                                     dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.roinorm_max', np.ones(config.FULL_FRAME_SIZE,
                                                    dtype=config.FRAME_DTYPE), stored=True)

        self.add_array('Servo.hnorm_min', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                      dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.hnorm_max', np.ones(config.FULL_FRAME_SHAPE[0],
                                                     dtype=config.FRAME_DTYPE), stored=True) 
        self.add_array('Servo.vnorm_min', np.zeros(config.FULL_FRAME_SHAPE[1],
                                                      dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.vnorm_max', np.ones(config.FULL_FRAME_SHAPE[1],
                                                     dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.hellipse_norm_coeffs',
                       np.ones(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.vellipse_norm_coeffs',
                       np.ones(4, dtype=config.FRAME_DTYPE), stored=True)

        
        self.add_value('Servo.opd_target', float(np.nan), stored=True)
        self.add_value('Servo.tip_target', float(0.), stored=True)
        self.add_value('Servo.tilt_target', float(0.), stored=True)
        self.add_value('Servo.velocity_target', float(config.NEXLINE_MOVING_VELOCITY), stored=True)

        self.add_value('Servo.is_lost', True, stored=False)
        self.add_value('Servo.track_loop_time', float(np.nan), stored=False)
        self.add_value('Servo.walk_loop_time', float(np.nan), stored=False)
        
        self.add_value('Servo.walk_velocity', float(np.nan), stored=False)
        
        self.add_array('DAQ.piezos_level',
                       np.zeros(3, dtype=config.DAQ_PIEZO_LEVELS_DTYPE),
                       stored=True)
        
        self.add_array('DAQ.piezos_level_actual',
                       np.zeros(3, dtype=config.DAQ_PIEZO_LEVELS_DTYPE),
                       stored=True)
        self.add_value('DAQ.state', float(0), stored=False)
        self.add_value('DAQ.loop_time', float(0), stored=False)
        self.add_value('DAQ.frequency', float(0), stored=False)
        
        
        self.add_value('Nexline.state', float(0), stored=False)
        self.add_value('Nexline.moving_velocity', float(config.NEXLINE_MOVING_VELOCITY), stored=False)
        
        self.add_value('Nexline.positive_velocity_calibration_factor',
                       float(config.NEXLINE_POS_CALIB_FACTOR), stored=True)
        self.add_value('Nexline.negative_velocity_calibration_factor',
                       float(config.NEXLINE_NEG_CALIB_FACTOR), stored=True)
        self.add_value('Nexline.position', float(np.nan), stored=True)
        
        self.add_value('SerialComm.state', float(0), stored=False)
        self.add_value('Viewer.state', float(0), stored=False)

        self.add_array('SerialComm.last_status_frame', np.zeros(config.SERIAL_STATUS_FRAME_SIZE, dtype=np.uint8), stored=False)

        self.add_value('Tracker.state', float(0), stored=False)
        self.add_value('Tracker.frequency', float(np.nan), stored=False)
        frequencies = [int(ifreq) for ifreq in config.TRACKER_STATS_FREQUENCIES]
        for ifreq in frequencies:
            self.add_value(f'Tracker.opd_{ifreq}', float(np.nan), stored=False)
            self.add_value(f'Tracker.tip_{ifreq}', float(np.nan), stored=False)
            self.add_value(f'Tracker.tilt_{ifreq}', float(np.nan), stored=False)
            self.add_value(f'Tracker.opd_std_{ifreq}', float(np.nan), stored=False)
            self.add_value(f'Tracker.velocity_{ifreq}', float(np.nan), stored=False)

        # FIR metrics
        self.add_value('FIR.short.tracking.e_opd', float(0), stored=False)
        self.add_value('FIR.short.tracking.e_tip', float(0), stored=False)
        self.add_value('FIR.short.tracking.e_tilt', float(0), stored=False)
        for iprefix in ['OPD', 'DA1', 'DA2']:
            self.add_value(f'FIR.short.tracking.{iprefix}.u_raw', float(np.nan), stored=False)
            self.add_value(f'FIR.short.tracking.{iprefix}.u', float(np.nan), stored=False)
            self.add_value(f'FIR.short.tracking.{iprefix}.w_norm', float(np.nan), stored=False)
            
            
            
        # record also config values
        for ikey in [k for k in dir(params) if k.isupper()]:
            _attr = getattr(params, ikey)
            if isinstance(_attr, str):
                continue
            if isinstance(_attr, type):
                continue
            
            try:
                iter(_attr)
            except TypeError:
                self.add_value(f'params.{ikey}', _attr, stored=True)
            else:
                self.add_array(f'params.{ikey}', np.array(_attr), stored=True)
        
        

    def add_array(self, name, array, stored=False):
        if stored:
            self.stored_names.append(name)
            if self.state.get(name) is None:
                log.warning(f'{name} could not be loaded from saved states')
            else:
                init = np.array(self.state.get(name), dtype=array.dtype)
                if np.any(np.isnan(init)):
                    log.warning(f'{name} could not be loaded from saved states')
                else:
                    array = init
        try:
            self.shms[name] = shared_memory.SharedMemory(create=True, name=name, size=array.nbytes)
        except FileExistsError:
            self.shms[name] = shared_memory.SharedMemory(name=name)

        self.arrs[name] = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shms[name].buf)
        self.arrs[name][:] = array  # fast vectorized fill
        log.info(f'added shared array {name} {array.shape}')

    def add_value(self, name, val, stored=False):
        self.add_array(name, np.array([val,]), stored=stored)

    def __getitem__(self, name):
        return self.arrs[name]

    def __setitem__(self, name, value):
        self.arrs[name] = value

    def keys(self):
        return self.arrs.keys()

    def get_velocity_calibration_factor(self, direction):
        if direction > 0:
            calibration_factor = abs(self['Nexline.positive_velocity_calibration_factor'][0])
        else:
            calibration_factor = abs(self['Nexline.negative_velocity_calibration_factor'][0])
            
        if (calibration_factor == 0) or (np.isnan(calibration_factor)):
            log.warning('bad calibration factor, using 1 instead')
            calibration_factor = 1
            
        return calibration_factor

    def stop(self):
        # Save states (unchanged)
        log.info('saving states')
        for iname in self.stored_names:
            idata = list(self[iname][:])
            self.state.set(iname, idata)
            if len(idata) < 10:
                log.info(f' {iname} : {idata}')
            else:
                log.info(f' {iname} : {idata[:5]}...{idata[-5:]}')
        self.state.save()

        # Free shared memory
        for ikey in self.shms:
            self.shms[ikey].close()
            try:
                self.shms[ikey].unlink()
                log.info(f'removed shared memory {ikey}')
            except FileNotFoundError:
                log.warning(f'shared memory {ikey} could not be unlinked')


