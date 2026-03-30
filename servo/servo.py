import collections
import multiprocessing
import time
import traceback
import psutil
import logging
import numpy as np
import signal

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
from . import tracker
from . import eventmanager

from .fsm import StateMachine, Transition, ServoState, NexlineState

log = logging.getLogger(__name__)

import subprocess

def worker_process(queue, data, WorkerClass, events, priority=None, kwargs=None):
    """
    Subprocess worker bootstrap:
      - configure logging
      - apply scheduling (if IRCamera)
      - instantiate and run worker
      - guarantee worker.stop() is called
    """
    if kwargs is None:
        kwargs = {}

    logger.configure_worker_logging(queue=queue, redirect_std=True)
    log = logging.getLogger(f"servo.worker.{WorkerClass.__name__}")
    
    niceness = None
    cpus = None

    if isinstance(priority, dict):
        niceness = priority.get("niceness", config.SERVO_DEFAULT_NICENESS)
        cpus = priority.get("cpus", None)

    # CPU affinity 
    if cpus is not None:
        try:
            p = psutil.Process()
            p.cpu_affinity(cpus)
            log.info(f"Set CPU affinity to CPUs {cpus} for {WorkerClass.__name__}")
        except Exception as e:
            log.warning(f"Failed to set CPU affinity to {cpus} for {WorkerClass.__name__}: {e}")
        
    # Process priority (nice level)
    if niceness is not None:
        try:
            p = psutil.Process()
            p.nice(niceness)
            log.info(f"Set nice level to {niceness} for {WorkerClass.__name__}")
        except Exception as e:
            log.warning(f"Failed to set nice level {niceness} for {WorkerClass.__name__}: {e}")
    
    # Instantiate and run the worker
    worker = None
    try:
        worker = WorkerClass(data, events, **kwargs)
        worker.dispatch(worker.Event.START)
    except BaseException as be:
        log.error(f"Worker crashed: {type(be).__name__}: {be}")
        log.error("Traceback:\n%s", traceback.format_exc())
        raise
    finally:
        if worker is not None:
            try:
                worker.dispatch(worker.Event.STOP)
            except Exception:
                log.error("Error in worker.stop()", exc_info=True)

        log.info("Worker terminated cleanly")
        
class Servo(core.Worker):
    def __init__(self, mode='calib', noviewer=False, nocam=False):
        self.mode = mode
        self.noviewer = noviewer
        self.nocam = nocam
        
        self.event_manager = eventmanager.SharedMemoryEventManager()
        events = self.event_manager.dict()
        events['Servo.start'] = self.event_manager.Event()
        events['Servo.stop'] = self.event_manager.Event()

        for iname in config.SERVO_EVENTS:
            events['Servo.' + iname] = self.event_manager.Event()

        for iname in config.NEXLINE_EVENTS:
            events['Nexline.' + iname] = self.event_manager.Event()

        for iname in config.TRACKER_EVENTS:
            events['Tracker.' + iname] = self.event_manager.Event()

        events['Servo.velocity_calibration_completed'] = self.event_manager.Event()

        self.queue = logger.get_logging_queue()

        data = core.SharedData()

        super().__init__(data, events, State=ServoState)

        self.table = {
            (ServoState.IDLE, self.Event.START): Transition(
                ServoState.RUNNING, action=self._start),
            (ServoState.RUNNING, self.Event.NORMALIZE): Transition(
                ServoState.RUNNING, action=self._normalize),
            (ServoState.TRACKING, self.Event.NORMALIZE): Transition(
                ServoState.TRACKING, action=self._normalize),            
            (ServoState.RUNNING, self.Event.STOP): Transition(
                ServoState.STOPPED, action=self._stop),
            (ServoState.TRACKING, self.Event.MOVE_TO_OPD): Transition(
                ServoState.TRACKING, action=self._move_to_opd),
            (ServoState.RUNNING, self.Event.CLOSE_LOOP): Transition(
                ServoState.TRACKING, action=self._close_loop),         
            (ServoState.TRACKING, self.Event.OPEN_LOOP): Transition(
                ServoState.RUNNING, action=self._open_loop),          
            (ServoState.RUNNING, self.Event.ROI_MODE): Transition(
                ServoState.RUNNING, action=self._roi_mode),
            (ServoState.RUNNING, self.Event.FULL_FRAME_MODE): Transition(
                ServoState.RUNNING, action=self._full_frame_mode),
            (ServoState.TRACKING, self.Event.RESET_ZPD): Transition(
                ServoState.TRACKING, action=self._reset_zpd),
            (ServoState.TRACKING, self.Event.WALK_TO_OPD): Transition(
                ServoState.WALKING, action=self._walk_to_opd),
            (ServoState.WALKING, self.Event.STOP_WALKING): Transition(
                ServoState.TRACKING, action=self._stop_walking),
            (ServoState.RUNNING, self.Event.CALIBRATE_VELOCITY): Transition(
                ServoState.RUNNING, action=self._calibrate_velocity),                
        }
        
        self.fir_short_mimo = None

    def install_signal_handlers(self):
        """
        Install SIGINT/SIGTERM handlers to trigger a clean shutdown via _stop transition.
        """
        def _handle_signal(signum, frame):
            log.warning(f"Received signal {signum}, initiating graceful stop...")
            try:
                # Set the FSM event that triggers the STOP transition
                self.dispatch(self.Event.STOP)
            except Exception:
                log.error("Error dispatching STOP on signal", exc_info=True)

        # Parent process only: install handlers
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    # Hooks (facultatif)
    def on_enter_running(self, _):
        self.install_signal_handlers()
        super().on_enter_running(_)
        

    def get_pid_control(self, name,
                        out_min=config.PIEZO_V_MIN,
                        out_max=config.PIEZO_V_MAX):
        
        dt = config.OPD_LOOP_TIME

        pid_control = pid.PiezoPID(
            self.data,
            f'params.PID_{name}',
            dt=dt,
            out_min=out_min,
            out_max=out_max,
            deriv_filter_hz=1/dt/1000., kaw=5.0, deadband=0.0
        )
        
        return pid_control

    def on_enter_tracking(self, _):
        log.info(">> TRACKING")

        self.data['Servo.is_lost'][0] = float(False)
        
        opd_target = self.data['Servo.opd_target'][0]
        tip_target = self.data['Servo.tip_target'][0] # associated with DA1
        tilt_target = self.data['Servo.tilt_target'][0] # associated with DA2
        
        log.info(f"   OPD target: {opd_target} nm")
        log.info(f"   TIP target: {tip_target} radians")
        log.info(f"   TILT target: {tilt_target} radians")

        da1_level_orig = self.data['DAQ.piezos_level'][1]
        da2_level_orig = self.data['DAQ.piezos_level'][2]
        
        last_update_time = time.time()
        da1_buffer = list()
        da2_buffer = list()
        
        pid_opd_control = self.get_pid_control('TRACK_OPD')
        
        max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]

        da1_min = da1_level_orig - max_v_diff
        da1_max = da1_level_orig + max_v_diff
        da2_min = da2_level_orig - max_v_diff
        da2_max = da2_level_orig + max_v_diff
        
        pid_da1_control = self.get_pid_control(
            'TRACK_DA1', out_min=da1_min, out_max=da1_max)
        pid_da2_control = self.get_pid_control(
            'TRACK_DA2', out_min=da2_min, out_max=da2_max)

        if self.fir_short_mimo is None:
            log.info("Initializing adaptive FIR short MIMO controller")
            self.fir_short_mimo = pid.AdaptiveFIRShortMIMO(
                name="tracking",
                shared_data=self.data,
                n_taps=20,      # ~0.2 s @ 100 Hz
                u_max=0.2,
            )
            
        else:
            log.info("Adaptive FIR short MIMO controller already initialized, keeping existing state")
                    
        refresh_startt = time.perf_counter()
        while True:
            try:
                loop_startt = time.perf_counter()
                da1_buffer.append(self.data['DAQ.piezos_level'][1])
                da2_buffer.append(self.data['DAQ.piezos_level'][2])
                if time.time() - last_update_time > self.data['params.PIEZO_DA_LOOP_UPDATE_TIME'][0]:
                    da1_level_orig = np.median(da1_buffer)
                    da2_level_orig = np.median(da2_buffer)
                    da1_buffer.clear()
                    da2_buffer.clear()
                    last_update_time = time.time()
                        
                opd = self.data['Tracker.opd_100'][0]
                tip = self.data['Tracker.tip_30'][0]
                tilt = self.data['Tracker.tilt_30'][0]

                if not np.isnan(opd):
                    u_pid_opd = pid_opd_control.update(
                        control=self.data['DAQ.piezos_level'][0],
                        setpoint=opd_target,
                        measurement=opd)

                    u_pid_da1 = pid_da1_control.update(
                        control=self.data['DAQ.piezos_level'][1],
                        setpoint=tip_target,
                        measurement=tip)

                    u_pid_da2 = pid_da2_control.update(
                        control=self.data['DAQ.piezos_level'][2],
                        setpoint=tilt_target,
                        measurement=tilt)

                    e_opd  = opd_target  - opd
                    e_tip  = tip_target  - tip
                    e_tilt = tilt_target - tilt

                    u_ff = self.fir_short_mimo.update(e_opd, e_tip, e_tilt)

                    self.data['DAQ.piezos_level'][0] = np.clip(
                        u_pid_opd + u_ff["OPD"], config.PIEZO_V_MIN, config.PIEZO_V_MAX)

                    self.data['DAQ.piezos_level'][1] = np.clip(
                        u_pid_da1 + u_ff["DA1"], da1_min, da1_max)

                    self.data['DAQ.piezos_level'][2] = np.clip(
                        u_pid_da2 + u_ff["DA2"], da2_min, da2_max)

                    
                else:
                    log.error('bad opd value, lost tracking')
                    self.events['Servo.open_loop'].set()
                    break


                # non-critic processing
                if time.perf_counter() - refresh_startt > config.SERVO_NONCRITIC_REFRESH_TIME:
                    pid_opd_control.update_coeffs()

                    max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]
                    da1_min = da1_level_orig - max_v_diff
                    da1_max = da1_level_orig + max_v_diff
                    da2_min = da2_level_orig - max_v_diff
                    da2_max = da2_level_orig + max_v_diff
        
                    pid_da1_control.update_config(out_min=da1_min, out_max=da1_max)
                    pid_da2_control.update_config(out_min=da2_min, out_max=da2_max)
                    
                    self.poll()
                    if self.events['Servo.stop'].is_set():
                        return
                    if self.events['Servo.open_loop'].is_set():
                        return
                    if self.data['Servo.is_lost'][0] == float(True):
                        return
                    
                    refresh_startt = time.perf_counter()


                process_time = time.perf_counter() - loop_startt
                if process_time < config.OPD_LOOP_TIME:
                    time.sleep(config.OPD_LOOP_TIME - process_time)
                    
                self.data['Servo.track_loop_time'][0] = time.perf_counter() - loop_startt
                
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.events['Servo.stop'].set()
                return

            except Exception as e:
                log.error(f"Exception on tracking: {type(e).__name__}: {e}")
                log.error("Traceback:\n%s", traceback.format_exc())
                self.events['Servo.open_loop'].set()
                return


    def on_exit_tracking(self, _):
        log.info("<< TRACKING")


    def start_worker(self, WorkerClass, priority, **kwargs):
        stop_event_name = f"{WorkerClass.__name__}.stop"
        self.events[stop_event_name] = self.event_manager.Event()

        if priority is None:
            log.info(f"Starting worker {WorkerClass.__name__} with default priority")
            priority={"niceness": config.SERVO_DEFAULT_NICENESS, "cpus": [1]}
            
        worker = multiprocessing.Process(
            target=worker_process,
            args=(self.queue, self.data, WorkerClass, self.events, priority, kwargs),
            name=f"worker.{WorkerClass.__name__}"
        )
        self.workers.append((worker, stop_event_name))
        worker.start()
        
    def _start(self, _):

        log.info("Starting Servo")
        ## start all threads
        self.workers = list()

        # start ir camera
        if not self.nocam:
            self._start_worker_roi_mode()
            
        # start nexline
        self.start_worker(nexline.Nexline, None)
        
        # start piezos
        self.start_worker(piezo.DAQ, None)

        # start tracker
        self.start_worker(tracker.Tracker, None)
        
        # start com
        self.start_worker(
            com.SerialComm, 10,
            port=config.SERIAL_PORT,
            baudrate=config.SERIAL_BAUDRATE,
            status_rate_hz=config.SERIAL_STATUS_RATE,
        )
        
        # start viewer
        if not self.noviewer:
            if not self.nocam:
                timeout_start = time.time()
                while True:
                    time.sleep(0.1)
                    if self.data['IRCamera.initialized'][0]: break
                    if time.time() - timeout_start > 5:
                        log.error("Timeout waiting for IRCamera initialization; starting viewer anyway")
                        break
            self.start_worker(viewer.Viewer, priority={"niceness": config.SERVO_LOW_NICENESS, "cpus": [0]})

        
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

            if not rec:
                self.data['DAQ.piezos_level'][0] = np.array(
                    val, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
            else:
                self.data['IRCamera.full_output'][0] = float(1.0) # force full output for normalization recording
                _goto_start_time = time.time()
                _goto_start_level = self.data['DAQ.piezos_level_actual'][0]
                levels = np.linspace(_goto_start_level, val, config.SERVO_NORMALIZE_REC_SIZE)
                for ilevel in levels:
                    
                    self.data['DAQ.piezos_level'][0] = np.array(
                        ilevel, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
                    rec_hprofiles.append(np.copy(self.data['IRCamera.hprofile'][:profile_len]))
                    rec_vprofiles.append(np.copy(self.data['IRCamera.vprofile'][:profile_len]))
                    rec_rois.append(np.copy(self.data['IRCamera.roi'][:profile_len**2]).reshape((
                        profile_len, profile_len)))
                    
                    time.sleep(config.SERVO_NORMALIZE_REC_TIME / config.SERVO_NORMALIZE_REC_SIZE)

                self.data['IRCamera.full_output'][0] = float(0.)
                
                self.data['DAQ.piezos_level'][0] = np.array(
                    val, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)

                
            
            while True:
                self.poll()
                if self.data['DAQ.piezos_level_actual'][0] == val:
                    break
                if self.events['Servo.stop'].is_set():
                    break
                
                
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

        try:
            np.save('hprofiles.npy', np.array(rec_hprofiles))
            np.save('vprofiles.npy', np.array(rec_vprofiles))
            np.save('rois.npy', np.array(rec_rois))
        except Exception as e:
            log.error(f"Failed to save normalization data: {e}")

    def _reset_zpd(self, _):
        log.info("ZPD Reset")
        self.data['IRCamera.angles'][:4] = np.zeros(4, dtype=config.FRAME_DTYPE)
        self.data['IRCamera.last_angles'][:4] = np.zeros(4, dtype=config.FRAME_DTYPE)
        self.data['IRCamera.mean_opd_offset'][0] = float(0)
        

    def _center_piezos(self):
        # centering OPD piezo at mid range
        log.info('centering OPD piezo')
                
        self.data['DAQ.piezos_level'][0] = 5

        startt = time.perf_counter()
        while True:
            if np.abs(self.data['DAQ.piezos_level_actual'][0] - 5) < 0.1:
                break
            if time.perf_counter() - startt > config.SERVO_OPD_TIMEOUT:
                log.error("Timeout while centering OPD piezo")
                return False
        return True

    def _move_with_nexline(self):
        self.events['Nexline.move'].set()
        
        time.sleep(0.1) # wait for event dispatch
        while True: # wait for nexline move to finish
            if NexlineState(int(self.data['Nexline.state'])).name != 'MOVING':
                break
            time.sleep(0.1)
            
    def _walk_to_opd(self, _):
        pass
    
    def on_enter_walking(self, _):
        log.info("Walking to OPD")

        final_opd_target = float(self.data['Servo.opd_target'][0]) # nm
        opd_start = float(self.data['Tracker.opd_1'][0]) # nm
        
        direction = float(np.sign(final_opd_target - opd_start))
        velocity_target = direction * abs(float(self.data['Servo.velocity_target'][0])) # um/s
        tip_target = float(self.data['Servo.tip_target'][0])
        tilt_target = float(self.data['Servo.tilt_target'][0])
        
        log.info(f"   Final OPD target: {final_opd_target} nm")
        log.info(f"   Velocity target: {velocity_target} um/s")
        log.info(f"   TIP target: {tip_target} radians")
        log.info(f"   TILT target: {tilt_target} radians")

        da1_level_orig = float(self.data['DAQ.piezos_level'][1])
        da2_level_orig = float(self.data['DAQ.piezos_level'][2])

        last_da_update_time = time.time()
        da1_buffer = list()
        da2_buffer = list()
        
        #if not self._center_piezos():
        #   log.error("Failed to center piezos, cannot walk to OPD")
        #   return
        
        # --- inside Servo._walk_to_opd(), where the velocity PID is created ---
        dt = config.OPD_LOOP_TIME  # 0.01 s (100 Hz)

        #vel_controller = self.get_pid_control('WALK_OPD')
        pid_opd_control = self.get_pid_control('WALK_OPD')
        
        max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]

        da1_min = da1_level_orig - max_v_diff
        da1_max = da1_level_orig + max_v_diff
        da2_min = da2_level_orig - max_v_diff
        da2_max = da2_level_orig + max_v_diff

        pid_da1_control = self.get_pid_control(
            'TRACK_DA1', out_min=da1_min, out_max=da1_max)
        pid_da2_control = self.get_pid_control(
            'TRACK_DA2', out_min=da2_min, out_max=da2_max)
        
        # check if piezos are in a correct range
        if not (4 < self.data['DAQ.piezos_level'][0] < 6):
            log.error('piezo off range, start with OPD piezo near the central position')
            self.data['Servo.opd_target'][0] = float(self.data['Tracker.opd_1'][0]) # nm
            self.dispatch(self.Event.STOP_WALKING)
            time.sleep(1) # wait for event dispatch
            return

        move_start_opd = float(self.data['Tracker.opd_1'][0]) # nm
        move_startt = time.perf_counter()
        refresh_startt = time.perf_counter()
        last_update_time = time.perf_counter()

        step_velocity_adjustment_factor = 1.0 # nexline velocity adjustment factor for each step to compensate for piezo contribution, will be updated at each step end based on the estimated velocity error ratio

        self.events['Tracker.start_recording'].set() # start recording tracker data

        stop_walk = False
        while True: # move step by step until reaching the target
            
            if stop_walk: break

            step = direction * self.data['params.NEXLINE_OPD_UPDATE'][0]

            itarget = float(self.data['Tracker.opd_3'][0]) + step
            
            self.data['Servo.opd_target'][0] = float(itarget)
            self.data['Nexline.moving_velocity'][0] = abs(velocity_target * step_velocity_adjustment_factor) # um/s
            self.events['Nexline.move'].set()
        
            step_startt = time.perf_counter()
            step_start_opd = float(self.data['Tracker.opd_3'][0])
            step_start_piezo_level = float(self.data['DAQ.piezos_level'][0])

            while True: # wait for nexline step to finish
                if stop_walk: break
                try:
                    loop_startt = time.perf_counter()

                    opd = float(self.data['Tracker.opd_100'][0]) # nm
                    tip = float(self.data['Tracker.tip_30'][0]) 
                    tilt = float(self.data['Tracker.tilt_30'][0])

                    nexline_is_moving = NexlineState(int(self.data['Nexline.state'])).name == 'MOVING'
                    if time.perf_counter() - step_startt > 0.5: # wait for event dispatch
                        if not nexline_is_moving:
                            break

                    #velocity = float(self.data['Tracker.velocity_30'][0]) / 1000. # um/s
                    #self.data['DAQ.piezos_level'][0] = vel_controller.update(
                    #    control=self.data['DAQ.piezos_level'][0],
                    #    setpoint=velocity_target,
                    #    measurement=velocity)

                    #estimated_opd = step_start_opd + velocity_target * (time.perf_counter() - step_startt) * 1000 # nm
                    estimated_opd = move_start_opd + velocity_target * (time.perf_counter() - move_startt) * 1000 # nm
                    self.data['DAQ.piezos_level'][0] = pid_opd_control.update(
                        control=self.data['DAQ.piezos_level'][0],
                        setpoint=estimated_opd,
                        measurement=opd)

                    da1_buffer.append(self.data['DAQ.piezos_level'][1])
                    da2_buffer.append(self.data['DAQ.piezos_level'][2])
                    
                    if time.time() - last_update_time > self.data['params.PIEZO_DA_LOOP_UPDATE_TIME'][0]:
                        da1_level_orig = np.median(da1_buffer)
                        da2_level_orig = np.median(da2_buffer)
                        da1_buffer.clear()
                        da2_buffer.clear()
                        last_update_time = time.perf_counter()

                    self.data['DAQ.piezos_level'][1] = pid_da1_control.update(
                        control=self.data['DAQ.piezos_level'][1],
                        setpoint=tip_target,
                        measurement=tip)

                    self.data['DAQ.piezos_level'][2] = pid_da2_control.update(
                        control=self.data['DAQ.piezos_level'][2],
                        setpoint=tilt_target,
                        measurement=tilt)

                    if time.perf_counter() - refresh_startt > config.SERVO_NONCRITIC_REFRESH_TIME:

                        #vel_controller.update_coeffs()
                        pid_opd_control.update_coeffs()
                        
                        max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]
                        da1_min = da1_level_orig - max_v_diff
                        da1_max = da1_level_orig + max_v_diff
                        da2_min = da2_level_orig - max_v_diff
                        da2_max = da2_level_orig + max_v_diff

                        pid_da1_control.update_config(out_min=da1_min, out_max=da1_max)
                        pid_da2_control.update_config(out_min=da2_min, out_max=da2_max)

                        refresh_startt = time.perf_counter()


                    if ((np.abs(opd - final_opd_target) < config.OPD_TOLERANCE)
                        or (direction > 0 and opd >= final_opd_target)
                        or (direction < 0 and opd <= final_opd_target)):
                        log.info(f"OPD target reached: {opd} nm")
                        self.data['Servo.opd_target'][0] = float(final_opd_target)
                        stop_walk = True
                        break

                    self.poll()
                    if self.events['Servo.stop'].is_set():
                        stop_walk = True
                        break

                    if self.events['Servo.open_loop'].is_set():
                        stop_walk = True
                        break

                    if self.data['Servo.is_lost'][0] == float(True):
                        stop_walk = True
                        break

                    process_time = time.perf_counter() - loop_startt
                    if process_time < config.OPD_LOOP_TIME:
                        time.sleep(config.OPD_LOOP_TIME - process_time)

                    loop_time = time.perf_counter() - loop_startt
                    self.data['Servo.walk_loop_time'][0] = loop_time
                    
                except KeyboardInterrupt:
                    log.error('Keyboard interrupt')
                    self.events['Servo.stop'].set()
                    stop_walk = True
                    break

                except Exception as e:
                    log.error('Exception at walk_to_opd:\n' + traceback.format_exc())
                    self.events['Servo.stop'].set()
                    stop_walk = True
                    break
                
            # step ending, adjust nexline velocity to replace the piezo near the center
            if stop_walk: break
            
            # estimate of the nexline velocity
            step_opd_walked = float(self.data['Tracker.opd_3'][0]) - step_start_opd
            if abs(step_opd_walked) < 0.5 * abs(step): # sanity check to avoid wrong velocity estimation due to small stepping
                log.warning(f"Step OPD walked {step_opd_walked} nm is too small, skipping velocity adjustment")
                continue
            
            step_piezo_end = float(self.data['DAQ.piezos_level'][0])
            step_piezo_diff = (step_piezo_end - 5)
            step_piezo_walked = step_piezo_diff * config.DAQ_PIEZO_OPD_PER_LEVEL
            step_time = time.perf_counter() - step_startt
            step_nexline_velocity = (step_opd_walked - step_piezo_walked) / step_time / 1000. # um/s
            velocity_ratio = abs(step_nexline_velocity / velocity_target)
            log.info(f"Walked {step_opd_walked:.2f} nm (piezo contribution {step_piezo_walked:.2f} nm) in {step_time:.1f} s, estimated Nexline velocity {step_nexline_velocity:.2f} um/s, ratio {velocity_ratio:.2%}")
            step_velocity_adjustment_factor /= velocity_ratio # corrected to obtain the right velocity for the nexline
            log.info(f"Uncompensated velocity adjustment factor: {step_velocity_adjustment_factor:.3f}")
            step_velocity_adjustment_factor += (step_piezo_end - 5) / 5 * self.data['params.NEXLINE_VELOCITY_ADJUSTMENT_GAIN'][0]
            step_velocity_adjustment_factor_clipped = max(0.1, min(1.9, step_velocity_adjustment_factor)) # limit adjustment factor to avoid too much compensation which may cause instability
            if step_velocity_adjustment_factor != step_velocity_adjustment_factor_clipped:
                log.warning(f"Velocity adjustment factor {step_velocity_adjustment_factor:.3f} out of bounds, clipped to {step_velocity_adjustment_factor_clipped:.3f}")
                step_velocity_adjustment_factor = step_velocity_adjustment_factor_clipped
            log.info(f"Velocity adjustment factor compensated to keep piezo in central zone: {step_velocity_adjustment_factor:.3f}")

        log.info("Finished walking to OPD")
        self.dispatch(self.Event.STOP_WALKING)

    def on_exit_walking(self, _):
        log.info("Exiting walking state, stopping Nexline and tracker recording")
        self.events['Nexline.stop_moving'].set()
        self.events['Tracker.stop_recording'].set() # stop recording tracker data
        time.sleep(1.) # avoid too fast transition to close loop after walk, which may cause instability
        
    def _stop_walking(self, _):
        pass                

    def calibrate_velocity(self, opd_start, velocity_target, direction):
        
        opd_end = opd_start + np.sign(direction) * abs(float(self.data['params.NEXLINE_CALIB_OPD'][0]))

        calibration_factor = self.data.get_velocity_calibration_factor(direction)
        
        step = direction * self.data['params.NEXLINE_OPD_UPDATE'][0]
        targets = np.arange(opd_start, opd_end, step)
        targets = list(targets[1:]) + [opd_end]

        calibration_buffer = list()
        move_startt = time.perf_counter()
        for i, itarget in enumerate(targets):
            
            self.data['Servo.opd_target'][0] = float(itarget)
            self.data['Nexline.moving_velocity'][0] = abs(velocity_target / calibration_factor) # um/s

            self.events['Nexline.move'].set()
        
            step_startt = time.perf_counter()

            while True: # wait for nexline move to finish
                loop_startt = time.perf_counter()

                nexline_is_moving = NexlineState(int(self.data['Nexline.state'])).name == 'MOVING'
                if time.perf_counter() - step_startt > 0.5: # wait for event dispatch
                    if not nexline_is_moving:
                        break
                calibration_buffer.append([
                    time.perf_counter() - move_startt,
                    self.data['Tracker.opd_100'][0],
                    nexline_is_moving,
                    i])

                time.sleep(max(0, config.IRCAM_SERVO_OUTPUT_TIME - (time.perf_counter() - loop_startt)))

                
        dir_str = 'positive' if direction > 0 else 'negative'
        t, opd, state, step_nb = np.array(calibration_buffer).T
        p = np.polyfit(t, opd, deg=1)
        opd_fit = np.polyval(p, t)
        calibration_buffer = np.array([t, opd, state, step_nb, opd_fit]).T
        np.save(f'velocity_calibration_buffer_{dir_str}.npy', calibration_buffer)
        
        measured_velocity = abs(p[0] / 1000) # um/s
        
        calibration_factor = velocity_target / measured_velocity
        
        log.info(f"Measured velocity: {measured_velocity} um/s, calibration factor: {calibration_factor}")
        if direction > 0:
            self.data['Nexline.positive_velocity_calibration_factor'][0] = calibration_factor
        else:
            self.data['Nexline.negative_velocity_calibration_factor'][0] = calibration_factor
        
    def _calibrate_velocity(self, _):
        log.info("Calibrating velocity")

        velocity_target = abs(float(self.data['Servo.velocity_target'][0])) # um/s
        
        # moving with Nexline for a few seconds in both directions and
        # measure resulting OPD velocity to calibrate the conversion
        # factor
        opd_start = self.data['Tracker.opd_1'][0]
        self.calibrate_velocity(opd_start, velocity_target, direction=1)
        opd_start = self.data['Tracker.opd_1'][0]
        self.calibrate_velocity(opd_start, velocity_target, direction=-1)
        self.events['Servo.velocity_calibration_completed'].set()
        log.info("Velocity calibration completed")

            
    def _move_to_opd(self, _):
        log.info("Moving to OPD")

        startt = time.time()
        
        opd_target = self.data['Servo.opd_target'][0]
        log.info(f"   OPD target: {opd_target} nm")

        # set nexline velocity at default moving velocity
        self.data['Nexline.moving_velocity'][0] = config.NEXLINE_MOVING_VELOCITY

        # centering OPD piezo at mid range
        if not self._center_piezos():
            log.error("Failed to center piezos, cannot move to OPD")
            return

        # moving with Nexline until close enough for piezo reach
        while True:
            opd_diff = np.abs(self.data['Servo.opd_target'][0] - self.data['IRCamera.mean_opd'][0])
            if opd_diff > config.PIEZO_MAX_OPD_DIFF/2.:
                log.info(f'opd difference {opd_diff} too large for piezo reach, moving with Nexline')
        
                self._move_with_nexline()
            else:
                break

            if self.events['Servo.stop'].is_set():
                return

        opd_diff = np.abs(self.data['Servo.opd_target'][0] - self.data['IRCamera.mean_opd'][0])
        if opd_diff > config.PIEZO_MAX_OPD_DIFF:
            log.error(f'opd difference too large for piezo reach: {opd_diff}')
            return
        
        log.info('moving with piezos')
            
        # Loop until OPD target is reached within tolerance, or stop event is set
        while True:

            pid_opd_control = self.get_pid_control('TRACK_OPD')
            
            opd = self.data['Tracker.opd_10'][0]
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
                  return
                                  
            time.sleep(config.OPD_LOOP_TIME)
            

        log.info(f'finished moving in {time.time() - startt} s')
        log.info(f"OPD piezo at {opd}")

    def _close_loop(self, _):
        log.info("Closing loop on target OPD")
        self.data['Servo.opd_target'][0] = self.data['IRCamera.mean_opd'][0]
        log.info(f"OPD target: {self.data['Servo.opd_target'][0]} nm")

    def _open_loop(self, _):
        log.info("Opening loop")

    def _roi_mode(self, _):
        if self.nocam:
            log.warning("Cannot switch to ROI mode: no camera")
            return

        log.info("Switching to ROI mode")
        self.events['IRCamera.stop'].set()
        time.sleep(0.5) # wait for camera to stop
        self._start_worker_roi_mode()
        

    def _start_worker_roi_mode(self):
        w = self.data['IRCamera.profile_len'][0]
        x = self.data['IRCamera.profile_x'][0]
        y = self.data['IRCamera.profile_y'][0]
        self.start_worker(ircam.IRCamera,
                          priority={"niceness": config.SERVO_MAX_NICENESS, "cpus": [2]},
                          frame_shape=(w, w),
                          frame_center=(x, y),
                          roi_mode=True)
        

    def _full_frame_mode(self, _):
        if self.nocam:
            log.warning("Cannot switch to full frame mode: no camera")
            return

        log.info("Switching to full frame mode")
        self.events['IRCamera.stop'].set()
        time.sleep(0.5) # wait for camera to stop
        self.start_worker(ircam.IRCamera, -20, roi_mode=False)
                
    
    def _stop(self, _):
        log.info("Stopping Servo")

        if self.fir_short_mimo is not None:
            self.fir_short_mimo.save()
            log.info("Saved FIR short MIMO weights")
    
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



