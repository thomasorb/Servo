import multiprocessing
import time
import traceback
import psutil
import logging
import numpy as np

from . import ircam
from . import core
from . import viewer
from . import logger
from . import piezo
from . import config
from . import utils
from . import pid
from . import nexline
from . import com

from .fsm import StateMachine, Transition, ServoState, ServoEvent, NexlineState


log = logging.getLogger(__name__)


def worker_process(queue, data, WorkerClass, events, priority=None, kwargs=None):
    """
    Subprocess that:
      - configures logging
      - instantiates and runs the worker
      - guarantees worker.stop() is called
    """
    if kwargs is None:
        kwargs = {}
    
    # Optionally adjust process priority
    if priority is not None:
        try:
            psutil.Process().nice(priority)
        except psutil.AccessDenied:
            pass

    # Configure logging for workers
    logger.configure_worker_logging(queue=queue, redirect_std=True)
    log = logging.getLogger(f"servo.worker.{WorkerClass.__name__}")

    log.info("Worker started")

    worker = None
    
    try:
        # Instantiate user worker
        worker = WorkerClass(data, events, **kwargs)
        worker.run()

    except Exception:
        log.error(f"Unhandled exception inside worker:\n {traceback.format_exc()}")

    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                log.error(f"Error in worker.stop():\n {traceback.format_exc()}")

        log.info("Worker terminated cleanly")



class Servo(StateMachine):
    def __init__(self, mode='calib', noviewer=False, nocam=False):
        self.mode = mode
        self.noviewer = noviewer
        self.nocam = nocam
        
        table = {
            (ServoState.IDLE, ServoEvent.START): Transition(
                ServoState.RUNNING, action=self._start),
            (ServoState.RUNNING, ServoEvent.NORMALIZE): Transition(
                ServoState.RUNNING, action=self._normalize),
            (ServoState.RUNNING, ServoEvent.STOP): Transition(
                ServoState.STOPPED, action=self._stop),
            (ServoState.RUNNING, ServoEvent.MOVE_TO_OPD): Transition(
                ServoState.RUNNING, action=self._move_to_opd),
            (ServoState.RUNNING, ServoEvent.CLOSE_LOOP): Transition(
                ServoState.STAY_AT_OPD, action=self._close_loop),         
            (ServoState.STAY_AT_OPD, ServoEvent.OPEN_LOOP): Transition(
                ServoState.RUNNING, action=self._open_loop),          
            (ServoState.RUNNING, ServoEvent.ROI_MODE): Transition(
                ServoState.RUNNING, action=self._roi_mode),
            (ServoState.RUNNING, ServoEvent.FULL_FRAME_MODE): Transition(
                ServoState.RUNNING, action=self._full_frame_mode),
            (ServoState.RUNNING, ServoEvent.RESET_ZPD): Transition(
                ServoState.RUNNING, action=self._reset_zpd),
            
        }
        super().__init__(ServoState.IDLE, table)

        self.event_manager = multiprocessing.Manager()
        self.events = self.event_manager.dict()

        for iname in config.SERVO_EVENTS:
            self.events['Servo.' + iname] = self.event_manager.Event()

        for iname in config.NEXLINE_EVENTS:
            self.events['Nexline.' + iname] = self.event_manager.Event()


        self.queue = logger.get_logging_queue()

        self.data = core.SharedData()

        
    def poll(self):
        evs = self.events

        for iname in config.SERVO_EVENTS:
            if evs.get('Servo.' + iname) and evs['Servo.' + iname].is_set():
                evs['Servo.' + iname].clear()
                self.dispatch(getattr(ServoEvent, iname.upper()), payload=None)

    # Hooks (facultatif)
    def on_enter_running(self, _):
        log.info(">> RUNNING")
        # running loop
        while True:
            try:
                self.poll()
                
                if self.state is ServoState.STOPPED:
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.events('Servo.stop').set()
                
        
    def on_exit_running(self, _):
        log.info("<< RUNNING")

    def get_pid_control(self, name):
        dt = config.OPD_LOOP_TIME
        
        pid_coeffs = self.data[f'Servo.PID_{name}'][:3]
        
        pid_control = pid.PID(
            pid.PIDConfig(
                kp=pid_coeffs[0],
                ki=pid_coeffs[1],
                kd=pid_coeffs[2],
                dt=dt,
                out_min=config.PIEZO_V_MIN,
                out_max=config.PIEZO_V_MAX,
                deriv_filter_hz=1/dt/1000., kaw=5.0, deadband=0.0
            ))
        return pid_control

    def on_enter_stay_at_opd(self, _):
        log.info(">> STAY_AT_OPD")
        
        opd_target = self.data['Servo.opd_target'][0]
        tip_target = self.data['Servo.tip_target'][0]
        tilt_target = self.data['Servo.tilt_target'][0]
        
        log.info(f"   OPD target: {opd_target} nm")
        log.info(f"   TIP target: {tip_target} radians")
        log.info(f"   TILT target: {tilt_target} radians")
        
        
        while True:
            try:
                pid_opd_control = self.get_pid_control('OPD')
                pid_da_control = self.get_pid_control('DA')

                opd = np.mean(self.data['IRCamera.mean_opd_buffer'][:10])

                tip = np.mean(self.data['IRCamera.tip_buffer'][:10])

                tilt = np.mean(self.data['IRCamera.tilt_buffer'][:10])
                
                if not np.isnan(opd):

                    self.data['DAQ.piezos_level'][0] = pid_opd_control.update(
                        control=self.data['DAQ.piezos_level'][0],
                        setpoint=opd_target,
                        measurement=opd)

                    self.data['DAQ.piezos_level'][1] = pid_da_control.update(
                        control=self.data['DAQ.piezos_level'][1],
                        setpoint=tip_target,
                        measurement=tip)

                    self.data['DAQ.piezos_level'][2] = pid_da_control.update(
                        control=self.data['DAQ.piezos_level'][2],
                        setpoint=tilt_target,
                        measurement=tilt)

                    
                else:
                    self.events['Servo.open_loop'].set()
                    break

                self.poll()

                if self.events['Servo.stop'].is_set():
                      break

                if self.events['Servo.open_loop'].is_set():
                      break

                time.sleep(config.OPD_LOOP_TIME)
                
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.events('Servo.stop').set()
                break

    def on_exit_stay_at_opd(self, _):
        log.info("<< STAY_AT_OPD")


    def start_worker(self, WorkerClass, priority, **kwargs):
        stop_event_name = f"{WorkerClass.__name__}.stop"
        self.events[stop_event_name] = self.event_manager.Event()
        
        worker = multiprocessing.Process(target=worker_process,
                                         args=(self.queue, self.data, WorkerClass,
                                               self.events, priority, kwargs))
        self.workers.append((worker, stop_event_name))
        worker.start()

        
    # Actions
    def _start(self, _):

        log.info("Starting Servo")
        ## start all threads
        self.workers = list()

        # start ir camera
        if not self.nocam:
            self.start_worker(ircam.IRCamera, -20)

        # start nexline
        self.start_worker(nexline.Nexline, 0)
        
        # start piezos
        self.start_worker(piezo.DAQ, 0)

        # start com
        self.start_worker(
            com.SerialComm, 0,
            port=config.SERIAL_PORT,
            baudrate=config.SERIAL_BAUDRATE,
            status_rate_hz=config.SERIAL_STATUS_RATE,
        )
        
        # start viewer
        if not self.noviewer:
            if not self.nocam:
                while True:
                    time.sleep(0.1)
                    if self.data['IRCamera.initialized'][0]: break
            self.start_worker(viewer.Viewer, 10)


    def _publish_state(self, state=None):
        super()._publish_state(state=state)
        try:
            self.data['Servo.state'][0] = float(state.value)
        except Exception: pass
        
    def _normalize(self, _):
        log.info("Normalizing")
        
        start_value = 3
        end_value = 7
        recall_value = self.data['DAQ.piezos_level'][0]

        rec_hprofiles = list()
        rec_vprofiles = list()
        profile_len = self.data['IRCamera.profile_len'][0]

        rec_rois = list()
        dimx = self.data['IRCamera.frame_dimx'][0]
        dimy = self.data['IRCamera.frame_dimy'][0]
        frame_size = self.data['IRCamera.frame_size'][0]
        
        def piezo_goto(val, rec=False):
        
            self.data['DAQ.piezos_level'][0] = np.array(
                val, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)

            while True:
                self.poll()

                if self.data['DAQ.piezos_level_actual'][0] == val:
                    break
                if self.events['Servo.stop'].is_set():
                    break
                if rec:
                    rec_hprofiles.append(np.copy(self.data['IRCamera.hprofile'][:profile_len]))
                    rec_vprofiles.append(np.copy(self.data['IRCamera.vprofile'][:profile_len]))
                    rec_rois.append(np.copy(self.data['IRCamera.roi'][:profile_len**2]).reshape((
                        profile_len, profile_len)))
                
            log.info(f"OPD piezo at {self.data['DAQ.piezos_level_actual'][0]}")

        piezo_goto(start_value)

        piezo_goto(end_value, rec=True)

        piezo_goto(recall_value)

        rec_hprofiles = np.array(rec_hprofiles)
        rec_vprofiles = np.array(rec_vprofiles)
        
        hnorm = utils.get_normalization_coeffs(rec_hprofiles).astype(config.FRAME_DTYPE)
        self.data['Servo.hnorm_min'][:profile_len] = hnorm[:,0]
        self.data['Servo.hnorm_max'][:profile_len] = hnorm[:,1]

        vnorm = utils.get_normalization_coeffs(rec_vprofiles).astype(config.FRAME_DTYPE)
        self.data['Servo.vnorm_min'][:profile_len] = vnorm[:,0]
        self.data['Servo.vnorm_max'][:profile_len] = vnorm[:,1]

        roinorm_min, roinorm_max = utils.get_roi_normalization_coeffs(
            np.array(rec_rois))
        self.data['Servo.roinorm_min'][:profile_len**2] = roinorm_min.astype(config.FRAME_DTYPE).flatten()
        self.data['Servo.roinorm_max'][:profile_len**2] = roinorm_max.astype(config.FRAME_DTYPE).flatten()

        # compute ellipse normalization coeffs
        hpixels_states = self.data['Servo.pixels_x']
        vpixels_states = self.data['Servo.pixels_y']        
        hpixels_lists = utils.get_pixels_lists(hpixels_states)
        vpixels_lists = utils.get_pixels_lists(vpixels_states)
        
        hellipse_norm_coeffs = utils.get_ellipse_normalization_coeffs(
            rec_hprofiles, hnorm, hpixels_lists)
        vellipse_norm_coeffs = utils.get_ellipse_normalization_coeffs(
            rec_vprofiles, vnorm, vpixels_lists)

        self.data['Servo.hellipse_norm_coeffs'][:4] = hellipse_norm_coeffs.astype(
            config.DATA_DTYPE)
        
        self.data['Servo.vellipse_norm_coeffs'][:4] = vellipse_norm_coeffs.astype(
            config.DATA_DTYPE)
        
        np.save('hprofiles.npy', np.array(rec_hprofiles))
        np.save('vprofiles.npy', np.array(rec_vprofiles))
        np.save('rois.npy', np.array(rec_rois))

    def _reset_zpd(self, _):
        log.info("ZPD Reset")
        self.data['IRCamera.angles'][:4] = np.zeros(4, dtype=config.FRAME_DTYPE)
        self.data['IRCamera.last_angles'][:4] = np.zeros(4, dtype=config.FRAME_DTYPE)
        self.data['IRCamera.mean_opd_offset'][0] = self.data['IRCamera.mean_opd'][0]
        
        
    def _move_to_opd(self, _):
        log.info("Moving to OPD")

        opd_target = self.data['Servo.opd_target'][0]
        
        log.info(f"   OPD target: {opd_target} nm")
        log.info('moving with Nexline')
        
        self.events['Nexline.move'].set()
        
        time.sleep(0.3) # wait for event dispatch
        while True:
            if NexlineState(self.data['Nexline.state']).name != 'MOVING':
                break
            time.sleep(0.1)

        opd_diff = np.abs(self.data['Servo.opd_target'][0] - self.data['IRCamera.mean_opd'][0])
        if  opd_diff > config.PIEZO_MAX_OPD_DIFF:
            log.error(f'opd difference too large for piezo reach: {opd_diff}')
            return
        log.info('moving with piezos')
            

        while True:

            pid_opd_control = self.get_pid_control('OPD')
            
            opd = np.mean(self.data['IRCamera.mean_opd_buffer'][:10])
            if not np.isnan(opd):
            
                self.data['DAQ.piezos_level'][0] = pid_opd_control.update(
                    control=self.data['DAQ.piezos_level'][0],
                    setpoint=opd_target,
                    measurement=opd)

            self.poll()

            if np.abs(opd - opd_target) < config.OPD_TOLERANCE:
                log.info(f"OPD target reached: {opd} nm")
                break

            if self.events['Servo.stop'].is_set():
                  break
                                  
            time.sleep(config.OPD_LOOP_TIME)


        log.info(f"OPD piezo at {opd}")

    def _close_loop(self, _):
        log.info("Closing loop on target OPD")

    def _open_loop(self, _):
        log.info("Opening loop")

    def _roi_mode(self, _):
        if self.nocam:
            log.warning("Cannot switch to ROI mode: no camera")
            return

        log.info("Switching to ROI mode")
        self.events['IRCamera.stop'].set()

        w = self.data['IRCamera.profile_len'][0]
        x = self.data['IRCamera.profile_x'][0]
        y = self.data['IRCamera.profile_y'][0]
        self.start_worker(ircam.IRCamera, -20,
                          frame_shape=(w, w),
                          frame_center=(x, y),
                          roi_mode=True)
        

    def _full_frame_mode(self, _):
        if self.nocam:
            log.warning("Cannot switch to full frame mode: no camera")
            return

        log.info("Switching to full frame mode")
        self.events['IRCamera.stop'].set()

        self.start_worker(ircam.IRCamera, -20)
        
        
    def _stop(self, _):
        log.info("Stopping Servo")

        # Graceful stop
        for p, ev in self.workers:
            time.sleep(1)
            self.events[ev].set()

        # Wait for clean exit
        for p, _ in self.workers:
            p.join(timeout=5)

        self.data.stop() # must be done before any exit kill
        del self.event_manager

        # Fallback kill if needed
        for p, _ in self.workers:
            if p.is_alive():
                log.error(f"{p.name} stuck, forcing terminate()")
                p.terminate()
                p.join()

        self.queue.put_nowait(None)



