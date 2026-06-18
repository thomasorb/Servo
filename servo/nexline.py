import time
from enum import IntEnum, Enum, auto
import numpy as np
import logging
import traceback

import pipython
import pipython.pitools

from . import config
from . import core
from .fsm import StateMachine, Transition, NexlineState

log = logging.getLogger(__name__)


class NexModes(IntEnum):
    FULL_STEP = 0
    NANO_STEP = 1
    ANALOG = 2
    
class Nexline(core.Worker):
    
    def __init__(self, data, events):
        
        super().__init__(data, events, State=NexlineState)
        
        self.table = {
            (NexlineState.IDLE, self.Event.START): Transition(
                NexlineState.RUNNING, action=self._start),
            (NexlineState.IDLE, self.Event.STOP): Transition(
                NexlineState.STOPPED, action=self._stop),
            (NexlineState.RUNNING, self.Event.STOP): Transition(
                NexlineState.STOPPED, action=self._stop),
            (NexlineState.RUNNING, self.Event.MOVE): Transition(
                NexlineState.MOVING, action=self._move),
            (NexlineState.MOVING, self.Event.STOP_MOVING): Transition(
                NexlineState.RUNNING, action=self._stop_moving),
            (NexlineState.MOVING, self.Event.STOP): Transition(
                NexlineState.STOPPED, action=self._stop),
        }

        self._mode = None
        self._move_active = False
        self._move_start_time = 0.0

    def loop_once(self):
        if self._mode == "moving":
            self._loop_moving()

    def _start(self, _):
        log.info('Starting Nexline')
        
        self.pidevice = pipython.GCSDevice('E-712')
        try:
            devices = self.pidevice.EnumerateUSB()
            for i, device in enumerate(devices):
                log.info('{} - {}'.format(i, device))
            self.pidevice.ConnectUSB(serialnum='120009499')
            
            if not self.pidevice.connected:
                log.error('error at usb connection')
                self.dispatch(self.Event.STOP)

            pipython.pitools.waitonready(self.pidevice)
    
            log.info(f'Nexline ID: {self.pidevice.qIDN()}')
            log.info(f'Nexline Operating mode: {self.pidevice.qSVO(config.NEXLINE_CHANNEL)[config.NEXLINE_CHANNEL]}')
            self.set_driving_mode(NexModes.NANO_STEP)
            self.set_velocity(config.NEXLINE_MOVING_VELOCITY)
            log.info('init stepping')
            self.step(1)
            self.step(-1)
            log.info('Nexline initialized')
            
        except Exception as e:
            log.error(f'error during init: {e}, traceback: {traceback.format_exc()}')
            self.dispatch(self.Event.STOP)

    def to_opd(self, mpd):
        """convert mpd to opd"""
        return mpd * 2 / np.cos(np.deg2rad(config.LASER_ANGLE))

    def to_mpd(self, opd):
        """convert opd to mpd"""
        return opd / 2 * np.cos(np.deg2rad(config.LASER_ANGLE))
        
    def _stop(self, _):
        log.info('Stopping Nexline')
        self.stop()
        
    def stop(self):
        try:
            self.pidevice.HLT(config.NEXLINE_CHANNEL)
            pipython.pitools.stopall(self.pidevice)
        except Exception as e:
            log.error(f'Exception at Nexline halt: {e}')

        try:
            self.pidevice.RNP(config.NEXLINE_CHANNEL, 0) # relax nexline
            self.pidevice.close()
        except Exception as e:
            log.error(f'error at closing device: {e}')
        log.info('Nexline stopped')

    def _getval(self, answer):
        try:
            return next(iter(answer[config.NEXLINE_CHANNEL].values()))
        except Exception as e:
            log.error(f'badly formatted answer: {e}')

    def get_velocity(self):
        """
        return optical velocity in um/s
        """
        return self.to_opd(self._getval(self.pidevice.qSPA(config.NEXLINE_CHANNEL, 0x07000204)))

    def print_velocity(self):
        log.info(f'actual optical velocity: {self.get_velocity()} um/s')
    
    def set_velocity(self, velocity):
        """
        set optical velocity in um/s
        """
        calibration_factor = self.data.get_velocity_calibration_factor(np.sign(velocity))

        velocity = abs(velocity) * calibration_factor
        
        log.info(f'setting velocity to {velocity} um/s (calibration factor: {calibration_factor})')
        try:
            self.print_velocity()
            self.pidevice.CCL(1, 'advanced') # switch to high command level
            self.pidevice.SPA(config.NEXLINE_CHANNEL, 0x07000204, float(self.to_mpd(
                abs(velocity))))
            self.pidevice.CCL(0) # switch to low command level
            self.print_velocity()
            
        except Exception as e:
            log.error(f'error when setting velocity: {e}')


    # piezo wal step size: 0x07011700
    
    
    def set_driving_mode(self, driving_mode: NexModes):
        log.info(f'switching driving mode to {driving_mode}')
        try:
            log.info(f'actual driving mode {self._getval(self.pidevice.qSPA(config.NEXLINE_CHANNEL, 0x7001a00))}')
            self.pidevice.CCL(1, 'advanced') # switch to high command level
            self.pidevice.SPA(config.NEXLINE_CHANNEL, 0x7001a00, int(driving_mode)) # piezowalk driving mode
            self.pidevice.CCL(0) # switch to low command level
            log.info(f'new driving mode {self._getval(self.pidevice.qSPA(config.NEXLINE_CHANNEL, 0x7001a00))}')
            
        except Exception as e:
            log.error(f'error setting driving mode: {e}')

    def _move(self, _):
        pass

    def step(self, step_nb):
        try:
            self.pidevice.OSM(config.NEXLINE_CHANNEL, step_nb)
        except Exception as e:
            log.warning(f'error at OSM command: {e}')

        startt = time.time()
        while True:

            self.poll()

            if self.events['Servo.stop'].is_set():
                self.dispatch(self.Event.STOP)
                break

            try:
                if not self.pidevice.qOSN(config.NEXLINE_CHANNEL)[1]:
                    break
            except Exception as e:
                log.warning(f'error at OSN query: {e}')
                break
            
            if (time.time() - startt) > config.NEXLINE_TIMEOUT:
                log.error('Nexline move timeout')
                break

            time.sleep(0.01)
    
    def on_enter_moving(self, _):
        self._mode = "moving"
        self._move_active = True

        opd = self.data['Servo.opd_target'][0] - self.data['Tracker.opd_3'][0]

        self._move_opd = opd
        self._move_start_time = time.perf_counter()

        self.set_velocity(self.data['Nexline.moving_velocity'][0] * np.sign(opd))

        if np.isnan(opd):
            log.error(f'bad relative opd: {opd}')
            self.dispatch(self.Event.STOP_MOVING)
            return

        self._step_nb = self.to_mpd(opd) / 1e3 / self.to_mpd(config.NEXLINE_STEP_SIZE)

        log.info(f'move planned: opd={opd} nm, steps={self._step_nb}')

        try:
            self.pidevice.OSM(config.NEXLINE_CHANNEL, self._step_nb)
        except Exception as e:
            log.error(f'OSM error: {e}')
            self.dispatch(self.Event.STOP_MOVING)

    def _loop_moving(self):
        if not self._move_active:
            return

        # STOP demandé
        if self.events['Servo.stop'].is_set():
            self.dispatch(self.Event.STOP)
            return

        # STOP_MOVING externe
        if self.events['Nexline.stop_moving'].is_set():
            self.dispatch(self.Event.STOP_MOVING)
            return

        try:
            moving = self.pidevice.qOSN(config.NEXLINE_CHANNEL)[1]
        except Exception as e:
            log.warning(f'qOSN error: {e}')
            self.dispatch(self.Event.STOP_MOVING)
            return

        # fin du mouvement
        if not moving:
            log.info("Nexline move finished")
            self.dispatch(self.Event.STOP_MOVING)
            return

        # timeout safety
        if (time.perf_counter() - self._move_start_time) > config.NEXLINE_TIMEOUT:
            log.error("Nexline timeout")
            self.dispatch(self.Event.STOP_MOVING)

    def on_exit_moving(self, _):
        self._mode = None
        self._move_active = False
    
    def _stop_moving(self, _):
        log.info('Nexline move stopping')

        try:
            if self.pidevice.qOSN(config.NEXLINE_CHANNEL)[1]:
                self.pidevice.HLT(config.NEXLINE_CHANNEL)
                pipython.pitools.stopall(self.pidevice)
                log.info('Nexline move halted')
        except Exception as e:
            log.error(f'Exception at Nexline halt: {e}')

        

    
