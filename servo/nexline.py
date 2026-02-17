import time
from enum import IntEnum, Enum, auto
import numpy as np
import logging

import pipython
import pipython.pitools

from . import config
from . import core
from .fsm import StateMachine, Transition, NexlineState, NexlineEvent

log = logging.getLogger(__name__)


class NexModes(IntEnum):
    FULL_STEP = 0
    NANO_STEP = 1
    ANALOG = 2
    
class Nexline(core.Worker, StateMachine):
    
    def __init__(self, data, events):

        table = {
            (NexlineState.IDLE,    NexlineEvent.START): Transition(
                NexlineState.RUNNING, action=self._start),
            (NexlineState.RUNNING, NexlineEvent.STOP): Transition(
                NexlineState.STOPPED, action=self._stop),
            (NexlineState.RUNNING, NexlineEvent.MOVE): Transition(
                NexlineState.MOVING, action=self._move),
            (NexlineState.MOVING, NexlineEvent.STOP_MOVING): Transition(
                NexlineState.RUNNING, action=self._stop_moving),
        }

        core.Worker.__init__(self, data, events)
        StateMachine.__init__(self, NexlineState.IDLE, table)

        self.dispatch(NexlineEvent.START)

    def _start(self, _):
        log.info('Starting Nexline')
        
        self.pidevice = pipython.GCSDevice('E-712')
        try:
            #devices = self.pidevice.EnumerateUSB()
            #for i, device in enumerate(devices):
            #    print('{} - {}'.format(i, device))
            self.pidevice.ConnectUSB(serialnum='120009499')
            
            if not self.pidevice.connected:
                log.error('error at usb connection')
                self.stop()

            pipython.pitools.waitonready(self.pidevice)
    
            log.info(f'Nexline ID: {self.pidevice.qIDN()}')
            log.info(f'Nexline Operating mode: {self.pidevice.qSVO(config.NEXLINE_CHANNEL)[config.NEXLINE_CHANNEL]}')
            self.print_pos()
        except Exception as e:
            log.error(f'error during init: {e}')
            self.stop()

    # Hooks (facultatif)
    def on_enter_running(self, _):
        log.info(">> RUNNING")
        # running loop
        # while True:
        #     try:
        #         self.poll()
                
        #         if self.state is FsmState.STOPPED:
        #             break
        #         time.sleep(0.1)
        #     except KeyboardInterrupt:
        #         log.error('Keyboard interrupt')
        #         self.events('Servo.stop').set()
                
        
    def on_exit_running(self, _):
        log.info("<< RUNNING")

    def _publish_state(self, state=None):
        super()._publish_state(state=state)
        try:
            self.data['Nexline.state'][0] = float(state.value)
        except Exception: pass

    def to_opd(self, mpd):
        """convert mpd to opd"""
        return mpd * 2 / np.cos(np.deg2rad(config.LASER_ANGLE))

    def to_mpd(self, opd):
        """convert opd to mpd"""
        return opd / 2 * np.cos(np.deg2rad(config.LASER_ANGLE))
        
    def loop_once(self):
        try:
            self.poll()
            
            #if self.state is NexlineState.STOPPED:
            #    return
            time.sleep(0.1)
        except KeyboardInterrupt:
            log.error('Keyboard interrupt')
            self.events('Nexline.stop').set()

    def _stop(self, _):
        log.info('Stopping Nexline')
        self.stop()
        
    def stop(self):
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

    def poll(self):
        evs = self.events

        for iname in config.NEXLINE_EVENTS:
            if evs.get('Nexline.' + iname) and evs['Nexline.' + iname].is_set():
                evs['Nexline.' + iname].clear()
                self.dispatch(getattr(NexlineEvent, iname.upper()), payload=None)


    def get_pos(self):
        """return the Nexline encoded position
        """
        return self.pidevice.qPOS(config.NEXLINE_CHANNEL)[config.NEXLINE_CHANNEL]

    def print_pos(self):
        log.info(f'actual pos {self.get_pos()} um')
    
    def get_velocity(self):
        """
        return optical velocity in um/s
        """
        return self.to_opd(self._getval(self.pidevice.qSPA(config.NEXLINE_CHANNEL, 0x07000204)))

    def print_velocity(self):
        log.info(f'actual optical velocity: {self.get_velocity()} um/s')
    
    def set_velocity(self, velocity=250):
        """
        set optical velocity in um/s
        """
        try:
            self.print_velocity()
            self.pidevice.CCL(1, 'advanced') # switch to high command level
            self.pidevice.SPA(config.NEXLINE_CHANNEL, 0x07000204, float(self.to_mpd(velocity)))
            self.pidevice.CCL(0) # switch to low command level
            self.print_velocity()
            
        except Exception as e:
            log.error(f'error when setting velocity: {e}')

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
    
    def on_enter_moving(self, _):
        log.info('centering OPD piezo')
        
        self.data['DAQ.piezos_level'][0] = 5
        while True:
            if np.abs(self.data['DAQ.piezos_level_actual'][0] - 5) < 0.1:
                break
                      
        log.info('Moving Nexline')
        self.set_velocity(50)

        opd = self.data['Servo.opd_target'][0] - self.data['IRCamera.mean_opd'][0]
        
        if not np.isnan(opd):        
            velocity = self.get_velocity() # um/s

            step_nb = self.to_mpd(opd) / 1e3 / config.NEXLINE_STEP_SIZE

            log.info(f'moving at {velocity} um/s with a step size of {self.to_opd(config.NEXLINE_STEP_SIZE)} for {opd} nm ({step_nb} steps) (optical)')
            #return
            self.print_pos()

            log.info(f'{self.pidevice.qSSA(config.NEXLINE_CHANNEL)}')
            startt = time.time()
            self.pidevice.OSM(config.NEXLINE_CHANNEL, step_nb)

            while True:
                if not self.pidevice.qOSN(config.NEXLINE_CHANNEL)[1]:
                    break

                self.poll()

                if self.events['Servo.stop'].is_set():
                    try:
                        self.pidevice.HLT(config.NEXLINE_CHANNEL)
                        pipython.pitools.stopall(self.pidevice)
                    except Exception as e:
                        log.error(f'Exception at Nexline halt: {e}')
                    log.error('Nexline halted')
                    break
                
                if (time.time() - startt) > config.NEXLINE_TIMEOUT:
                    log.error('Nexline move timeout')
                    break
            
            self.print_pos()
            log.info(f'finished stepping in {time.time() - startt} s')
        else:
            log.error(f'bad relative opd: {opd}')

        self.dispatch(NexlineEvent.STOP_MOVING)

    def _stop_moving(self, _):
        log.info('Nexline move stopping')
        #self.pidevice.RNP(config.NEXLINE_CHANNEL, 0)


        

    
