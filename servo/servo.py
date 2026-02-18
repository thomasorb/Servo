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

from .fsm import StateMachine, Transition, ServoState, ServoEvent, NexlineState

log = logging.getLogger(__name__)

import subprocess

def _elevate_ircam_rt(pid: int, cpu: int = 2, prio: int = 80):
    """
    Elevate only the given PID to SCHED_FIFO (prio) and pin it to a specific CPU.
    Requires sudo privileges for chrt & taskset. No effect on parent process.
    """
    # Pin to CPU (logical index; CPU 2 means the 3rd logical core)
    try:
        subprocess.run(["sudo", "taskset", "-pc", str(cpu), str(pid)], check=True)
    except Exception as e:
        log.error(f"Failed to set CPU affinity for IRCam PID {pid}: {e}")

    # Set real-time scheduling policy to SCHED_FIFO with priority 'prio'
    try:
        subprocess.run(["sudo", "chrt", "-f", "-p", str(prio), str(pid)], check=True)
    except Exception as e:
        log.error(f"Failed to set SCHED_FIFO for IRCam PID {pid}: {e}")
        
        
def worker_process(queue, data, WorkerClass, events, priority=None, kwargs=None):
    """
    Subprocess that:
    - configures logging
    - instantiates and runs the worker
    - guarantees worker.stop() is called
    """
    
    if kwargs is None:
        kwargs = {}

    # (existing) optional nice
    
    # if priority is not None:
    #     try:
    #         psutil.Process().nice(priority)
    #     except:
    #         log.error(f"Failed to set nice level {priority} for {WorkerClass.__name__}: {e}")

    
    # --- Hardening for IRCam only (no sudo needed here) ---
    # if WorkerClass.__name__ == "IRCamera":
    #     # 1) Lock pages in RAM to avoid major page faults during callbacks
    #     try:
    #         import ctypes, gc
    #         libc = ctypes.CDLL("libc.so.6", use_errno=True)
    #         MCL_CURRENT, MCL_FUTURE = 1, 2
    #         libc.mlockall(MCL_CURRENT | MCL_FUTURE)
    #         # 2) Disable GC in the RT-ish child; do collections off-RT if needed
    #         gc.disable()
    #     except Exception:
    #         pass

    _stop_requested = {"flag": False}

    def _child_handle_signal(signum, frame):
        # Avoid re-entrancy
        if not _stop_requested["flag"]:
            _stop_requested["flag"] = True
            # Best-effort: set the worker stop event known by parent
            try:
                stop_event_name = f"{WorkerClass.__name__}.stop"
                if events.get(stop_event_name):
                    events[stop_event_name].set()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _child_handle_signal)
    signal.signal(signal.SIGTERM, _child_handle_signal)

    logger.configure_worker_logging(queue=queue, redirect_std=True)
    log = logging.getLogger(f"servo.worker.{WorkerClass.__name__}")
    log.info("Worker started")

    worker = None
    try:
        worker = WorkerClass(data, events, **(kwargs or {}))
        worker.run()           # <- may raise BaseException on signal/KeyboardInterrupt
    except BaseException as be:
        # Catch KeyboardInterrupt and any termination to ensure stop() is called
        log.error(f"Worker terminating due to {type(be).__name__}: {be}")
        log.error("Traceback:\n" + traceback.format_exc())
        # Re-raise after stop() in finally if you want parent to see non-zero exit
        raise
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                log.error("Error in worker.stop()", exc_info=True)
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
            (ServoState.STAY_AT_OPD, ServoEvent.MOVE_TO_OPD): Transition(
                ServoState.STAY_AT_OPD, action=self._move_to_opd),
            (ServoState.RUNNING, ServoEvent.CLOSE_LOOP): Transition(
                ServoState.STAY_AT_OPD, action=self._close_loop),         
            (ServoState.STAY_AT_OPD, ServoEvent.OPEN_LOOP): Transition(
                ServoState.RUNNING, action=self._open_loop),          
            (ServoState.RUNNING, ServoEvent.ROI_MODE): Transition(
                ServoState.RUNNING, action=self._roi_mode),
            (ServoState.RUNNING, ServoEvent.FULL_FRAME_MODE): Transition(
                ServoState.RUNNING, action=self._full_frame_mode),
            (ServoState.STAY_AT_OPD, ServoEvent.RESET_ZPD): Transition(
                ServoState.STAY_AT_OPD, action=self._reset_zpd),
            
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


    def install_signal_handlers(self):
        """
        Install SIGINT/SIGTERM handlers to trigger a clean shutdown via _stop transition.
        """
        def _handle_signal(signum, frame):
            log.warning(f"Received signal {signum}, initiating graceful stop...")
            try:
                # Set the FSM event that triggers the STOP transition
                self.dispatch(ServoEvent.STOP)
            except Exception:
                log.error("Error dispatching STOP on signal", exc_info=True)

        # Parent process only: install handlers
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        
    def poll(self):
        evs = self.events

        for iname in config.SERVO_EVENTS:
            if evs.get('Servo.' + iname) and evs['Servo.' + iname].is_set():
                evs['Servo.' + iname].clear()
                self.dispatch(getattr(ServoEvent, iname.upper()), payload=None)

    # Hooks (facultatif)
    def on_enter_running(self, _):
        self.install_signal_handlers()
        log.info(">> RUNNING")
        # running loop
        while True:
            try:
                self.poll()
                
                if self.state is ServoState.STOPPED:
                    break
                time.sleep(0.1)
            except Exception as e:
                log.error('Exception at running:\n' + traceback.format_exc())
                self.events['Servo.stop'].set()
                
        
    def on_exit_running(self, _):
        log.info("<< RUNNING")

    def get_pid_control(self, name,
                        out_min=config.PIEZO_V_MIN,
                        out_max=config.PIEZO_V_MAX):

        dt = config.OPD_LOOP_TIME
        
        pid_control = pid.PiezoPID(
            self.data,
            f'Servo.PID_{name}',
            dt=dt,
            out_min=out_min,
            out_max=out_max,
            deriv_filter_hz=1/dt/1000., kaw=5.0, deadband=0.0
            )
        
        return pid_control

    def on_enter_stay_at_opd(self, _):
        log.info(">> STAY_AT_OPD")

        self.data['Servo.is_lost'][0] = float(False)
        
        opd_target = self.data['Servo.opd_target'][0]
        tip_target = self.data['Servo.tip_target'][0]
        tilt_target = self.data['Servo.tilt_target'][0]
        
        log.info(f"   OPD target: {opd_target} nm")
        log.info(f"   TIP target: {tip_target} radians")
        log.info(f"   TILT target: {tilt_target} radians")

        da1_level_orig = self.data['DAQ.piezos_level'][1]
        da2_level_orig = self.data['DAQ.piezos_level'][2]
        
        last_update_time = time.time()
        da1_buffer = list()
        da2_buffer = list()
        
        pid_opd_control = self.get_pid_control('OPD')
        pid_da1_control = self.get_pid_control(
            'DA',
            out_min=da1_level_orig - config.PIEZO_DA_LOOP_MAX_V_DIFF,
            out_max=da1_level_orig + config.PIEZO_DA_LOOP_MAX_V_DIFF)
        pid_da2_control = self.get_pid_control(
            'DA',
            out_min=da2_level_orig - config.PIEZO_DA_LOOP_MAX_V_DIFF,
            out_max=da2_level_orig + config.PIEZO_DA_LOOP_MAX_V_DIFF)
        
        while True:
            try:
                da1_buffer.append(self.data['DAQ.piezos_level'][1])
                da2_buffer.append(self.data['DAQ.piezos_level'][2])
                if time.time() - last_update_time > config.PIEZO_DA_LOOP_UPDATE_TIME:
                    da1_level_orig = np.median(da1_buffer)
                    da2_level_orig = np.median(da2_buffer)
                    da1_buffer.clear()
                    da2_buffer.clear()
                    last_update_time = time.time()
                
        
                opd = np.median(self.data['IRCamera.mean_opd_buffer'][:10])

                tip = np.median(self.data['IRCamera.tip_buffer'][:10])

                tilt = np.median(self.data['IRCamera.tilt_buffer'][:10])
                
                if not np.isnan(opd):

                    pid_opd_control.update_coeffs()

                    pid_da1_control.update_config(
                        out_min=da1_level_orig - config.PIEZO_DA_LOOP_MAX_V_DIFF,
                        out_max=da1_level_orig + config.PIEZO_DA_LOOP_MAX_V_DIFF)
                    
                    pid_da2_control.update_config(
                        out_min=da2_level_orig - config.PIEZO_DA_LOOP_MAX_V_DIFF,
                        out_max=da2_level_orig + config.PIEZO_DA_LOOP_MAX_V_DIFF)

                    self.data['DAQ.piezos_level'][0] = pid_opd_control.update(
                        control=self.data['DAQ.piezos_level'][0],
                        setpoint=opd_target,
                        measurement=opd)

                    self.data['DAQ.piezos_level'][1] = pid_da1_control.update(
                        control=self.data['DAQ.piezos_level'][1],
                        setpoint=tip_target,
                        measurement=tip)

                    self.data['DAQ.piezos_level'][2] = pid_da2_control.update(
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

                if self.data['Servo.is_lost'][0] == float(True):
                    break

                time.sleep(config.OPD_LOOP_TIME)
                
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.events['Servo.stop'].set()
                break

    def on_exit_stay_at_opd(self, _):
        log.info("<< STAY_AT_OPD")


    def start_worker(self, WorkerClass, priority, **kwargs):
        stop_event_name = f"{WorkerClass.__name__}.stop"
        self.events[stop_event_name] = self.event_manager.Event()
    
        worker = multiprocessing.Process(
            target=worker_process,
            args=(self.queue, self.data, WorkerClass, self.events, priority, kwargs),
            name=f"worker.{WorkerClass.__name__}"
        )
        self.workers.append((worker, stop_event_name))
        worker.start()

        # # >>> Only for IRCam: elevate child to RT on CPU 2 <<<
        # if WorkerClass is ircam.IRCamera:
        #     if worker.pid is not None:
        #         _elevate_ircam_rt(worker.pid, cpu=2, prio=80)
        #     else:
        #         log.error("IRCam worker PID unavailable; cannot elevate to RT")
        
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
        self.data['IRCamera.mean_opd_offset'][0] = float(0)#-self.data['IRCamera.mean_opd'][0]
        
        
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



