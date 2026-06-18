import collections
import multiprocessing
import time
import traceback
import psutil
import logging
import numpy as np
import signal
import scipy.interpolate

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
from . import faster

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

    logger.configure_worker_logging(queue=queue)
    log = logging.getLogger(f"servo.worker.{WorkerClass.__name__}")

    log.info("Worker process started with PID %d", multiprocessing.current_process().pid)
    
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
        worker.run()
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
            (ServoState.RUNNING, self.Event.CALIBRATE_TIP_TILT): Transition(
                ServoState.RUNNING, action=self._calibrate_tip_tilt),
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
            (ServoState.RUNNING, self.Event.RESET_ZPD): Transition(
                ServoState.RUNNING, action=self._reset_zpd),
            (ServoState.TRACKING, self.Event.WALK_TO_OPD): Transition(
                ServoState.WALKING, action=self._walk_to_opd),
            (ServoState.WALKING, self.Event.STOP_WALKING): Transition(
                ServoState.TRACKING, action=self._stop_walking),
            (ServoState.RUNNING, self.Event.CALIBRATE_VELOCITY): Transition(
                ServoState.RUNNING, action=self._calibrate_velocity),
            (ServoState.RUNNING, self.Event.START_WAITING): Transition(
                ServoState.WAITING, action=self._start_waiting),
            (ServoState.WAITING, self.Event.STOP_WAITING): Transition(
                ServoState.TRACKING, action=self._stop_waiting),
        }
        
        self.fir_short_mimo = None
        self.tip_model = None
        self.tilt_model = None

        self._mode = None

        # tracking state
        self._tracking_last_update = 0.0
        self._tracking_refresh = 0.0
        self._tracking_da1_buffer = []
        self._tracking_da2_buffer = []

        self._pid_opd = None
        self._pid_da1 = None
        self._pid_da2 = None

        # waiting / walking flags
        self._waiting_active = False
        self._walking_active = False

        # internal walking states
        self._walking_state = None   # "init", "step", "wait_step", "done"
        self._walking_active = False


    def loop_once(self):
        if self._mode == "tracking":
            self._loop_tracking()
        elif self._mode == "waiting":
            self._loop_waiting()
        elif self._mode == "walking":
            self._loop_walking()

        
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
        self._mode = None
        self.install_signal_handlers()
        super().on_enter_running(_)
        

    def on_enter_tracking(self, _):
        log.info(">> TRACKING")

        if self.tip_model is None or self.tilt_model is None:
            log.error("tip/tilt models not calibrated")
            self.events['Servo.open_loop'].set()
            return

        self.data['Servo.is_lost'][0] = float(False)

        self._mode = "tracking"

        self._pid_opd = pid.get_pid_control(self.data, 'TRACK_OPD')
        self._pid_da1 = pid.get_pid_control(self.data, 'TRACK_DA1')
        self._pid_da2 = pid.get_pid_control(self.data, 'TRACK_DA2')

        self._tracking_refresh = time.perf_counter()
        self._tracking_last_update = time.perf_counter()

        self._tracking_da1_buffer.clear()
        self._tracking_da2_buffer.clear()
        
        self._tracking_opd_target = self.data['Servo.opd_target'][0]
        

    def _loop_tracking(self):
        loop_startt = time.perf_counter()

        opd = float(self.data['Tracker.opd_100'][0])
        tip = float(self.data['Tracker.tip_10'][0])
        tilt = float(self.data['Tracker.tilt_10'][0])

        if np.isnan(opd):
            log.error("Tracking lost")
            self.events['Servo.open_loop'].set()
            return

        phase = utils.opd2phase(opd)
        tip_target = self.tip_model(phase)
        tilt_target = self.tilt_model(phase)

        self.data['Servo.tip_target'][0] = float(tip_target)
        self.data['Servo.tilt_target'][0] = float(tilt_target)

        u_opd = self._pid_opd.update(
            control=self.data['DAQ.piezos_level'][0],
            setpoint=self._tracking_opd_target,
            measurement=opd)

        u_da1 = self._pid_da1.update(
            control=self.data['DAQ.piezos_level'][1],
            setpoint=tilt_target,
            measurement=tilt)

        u_da2 = self._pid_da2.update(
            control=self.data['DAQ.piezos_level'][2],
            setpoint=tip_target,
            measurement=tip)

        self.data['Servo.e_opd'][0] = float(opd - self._tracking_opd_target)
        self.data['Servo.e_tip'][0] = float(tip - tip_target)
        self.data['Servo.e_tilt'][0] = float(tilt - tilt_target)

        self.data['DAQ.piezos_level'][0] = u_opd
        self.data['DAQ.piezos_level'][1] = u_da1
        self.data['DAQ.piezos_level'][2] = u_da2

        # periodic refresh
        if time.perf_counter() - self._tracking_refresh > config.SERVO_NONCRITIC_REFRESH_TIME:
            self._pid_opd.update_coeffs()
            self._tracking_refresh = time.perf_counter()

        # respect loop timing
        dt = time.perf_counter() - loop_startt
        if dt < config.OPD_LOOP_TIME:
            time.sleep(config.OPD_LOOP_TIME - dt)

        self.data['Servo.track_loop_time'][0] = time.perf_counter() - loop_startt

    def on_exit_tracking(self, _):
        self._mode = None
        log.info("<< TRACKING")


    def start_worker(self, WorkerClass, priority, **kwargs):
        stop_event_name = f"{WorkerClass.__name__}.stop"
        self.events[stop_event_name] = self.event_manager.Event()

        if priority is None:
            log.info(f"Starting worker {WorkerClass.__name__} with default priority")
            priority={"niceness": config.SERVO_DEFAULT_NICENESS,
                      "cpus": config.SERVO_CPU_DEFAULT}
            
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
            self.start_worker(viewer.Viewer, priority={"niceness": config.SERVO_DEFAULT_NICENESS, "cpus": config.SERVO_CPU_VIEWER})


    def piezo_goto(self, val, rec=False, record_keys='roi'):

        if record_keys == 'roi':
            profile_len = self.data['IRCamera.profile_len'][0]
        else:
            assert type(record_keys) == list and len(record_keys) > 0, "record_keys should be a list of data keys to record"
            
        if not rec:
            self.data['DAQ.piezos_level'][0] = np.array(
                val, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
        else:
            rec_values = list()
            self.data['IRCamera.full_output'][0] = float(1.0) # force full output for normalization recording
            _goto_start_time = time.time()
            _goto_start_level = self.data['DAQ.piezos_level_actual'][0]
            levels = np.linspace(_goto_start_level, val, config.SERVO_NORMALIZE_REC_SIZE)
            for ilevel in levels:

                self.data['DAQ.piezos_level'][0] = np.array(
                    ilevel, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
                if record_keys == 'roi':
                    rec_values.append(np.copy(self.data['IRCamera.roi'][:profile_len**2]).reshape((
                        profile_len, profile_len)))
                else:
                    rec_values.append([float(self.data[ikey][0]) for ikey in record_keys])
                        
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

        if rec:
            return np.array(rec_values)


    def _calibrate_tip_tilt(self, _):
        log.info("Calibrating Tip/Tilt")
        
        start_value = 3
        end_value = 7
        recall_value = self.data['DAQ.piezos_level'][0]

        opd_init = self.data['IRCamera.mean_opd'][0]

        self.piezo_goto(start_value)

        tiptilt = self.piezo_goto(end_value, rec=True, record_keys=['IRCamera.mean_opd', 'IRCamera.tip', 'IRCamera.tilt'])

        self.piezo_goto(recall_value)

        try:
            np.save('tiptilt.npy', np.array(tiptilt))
        except Exception as e:
            log.error(f"Failed to save tip-tilt calibration data: {e}")


        opd = tiptilt[:,0]
        tip = tiptilt[:,1]
        tilt = tiptilt[:,2]
        phase = utils.opd2phase(opd)

        def modelize_periodic(x, y, n=50, coeff=1):
            xbins = np.linspace(0, 2*np.pi, n)
            w = xbins[1] - xbins[0]
            ybins = list()
            for ibin in xbins:
                dist = np.mod(x - ibin+coeff*w/2, 2*np.pi)
                weights = np.exp(-dist/w)
                weights.fill(1)
                ok = dist < w*coeff
                ybins.append(np.sum(y[ok] * weights[ok])/np.sum(weights[ok]))
            ymodel = scipy.interpolate.interp1d(xbins, ybins)
            return ymodel
    
        self.tip_model = modelize_periodic(phase, tip)
        self.tilt_model = modelize_periodic(phase, tilt)
        
        try:
            import matplotlib.pyplot as plt
            
            plt.scatter(phase, tip, alpha=0.01)
            plt.scatter(phase, tilt, alpha=0.01)
            x = np.linspace(0, 2*np.pi, 1000)
            plt.plot(x, self.tip_model(x), c='red')
            plt.plot(x, self.tilt_model(x), c='red')
            plt.show()
        except Exception as e:
            log.error(f"Failed to plot tip-tilt calibration data: {e}")

        # recall opd init in case changed were done
        self.data['Servo.opd_target'][0] = opd_init
        
    def _normalize(self, _):
        log.info("Normalizing")
        
        start_value = 3
        end_value = 7
        recall_value = self.data['DAQ.piezos_level'][0]

        opd_init = self.data['IRCamera.mean_opd'][0]

        profile_len = self.data['IRCamera.profile_len'][0]

        dimx = self.data['IRCamera.frame_dimx'][0]
        dimy = self.data['IRCamera.frame_dimy'][0]
        frame_size = self.data['IRCamera.frame_size'][0]
        
        self.piezo_goto(start_value)

        rec_rois = self.piezo_goto(end_value, rec=True)

        self.piezo_goto(recall_value)

        roinorm_min, roinorm_max = utils.get_roi_normalization_coeffs(
            np.array(rec_rois))

        self.data['Servo.roinorm_min'][:profile_len**2] = roinorm_min.astype(config.FRAME_DTYPE).flatten()
        self.data['Servo.roinorm_max'][:profile_len**2] = roinorm_max.astype(config.FRAME_DTYPE).flatten()

        roinorm_amp = roinorm_max - roinorm_min
        ix_profile = profile_len//2 + int(self.data['IRCamera.profile_h_shift'][0])
        iy_profile = profile_len//2 + int(self.data['IRCamera.profile_v_shift'][0])
        iwid = int(self.data['params.PROFILE_WIDTH'][0])
        mask = roinorm_max.astype(np.float32, copy=False) if roinorm_max is not None else None

        rec_hprofiles = list()
        rec_vprofiles = list()
        for iroi in rec_rois:
            iroi_norm = utils.normalize_roi(iroi, roinorm_min, roinorm_amp)
            ivprofile_norm, ihprofile_norm = faster.compute_profiles_local_f32(
                iroi_norm.astype(np.float32, copy=False),
                int(iy_profile), int(ix_profile), int(iwid),
                int(profile_len), get_roi=False, mask=mask)
            rec_hprofiles.append(ihprofile_norm)
            rec_vprofiles.append(ivprofile_norm)
                    
        rec_hprofiles = np.array(rec_hprofiles)
        rec_vprofiles = np.array(rec_vprofiles)
        
        high_values = np.nanpercentile(roinorm_max, 99)
        if  high_values > 0.75:
            log.warning("High ROI normalization max value detected, may cause saturation: 99th percentile={:.3f}".format(high_values))
            
        # compute ellipse normalization coeffs
        hpixels_states = self.data['Servo.pixels_x']
        vpixels_states = self.data['Servo.pixels_y']        
        hpixels_lists = utils.get_pixels_lists(hpixels_states)
        vpixels_lists = utils.get_pixels_lists(vpixels_states)
        
        hellipse_norm_coeffs = utils.get_ellipse_normalization_coeffs(
            rec_hprofiles, hpixels_lists)
        vellipse_norm_coeffs = utils.get_ellipse_normalization_coeffs(
            rec_vprofiles, vpixels_lists)

        self.data['Servo.hellipse_norm_coeffs'][:6] = hellipse_norm_coeffs.astype(
            config.DATA_DTYPE)
        
        self.data['Servo.vellipse_norm_coeffs'][:6] = vellipse_norm_coeffs.astype(
            config.DATA_DTYPE)

        try:
            np.save('hprofiles.npy', np.array(rec_hprofiles))
            np.save('vprofiles.npy', np.array(rec_vprofiles))
            np.save('rois.npy', np.array(rec_rois))
        except Exception as e:
            log.error(f"Failed to save normalization data: {e}")

        # recall opd init in case changed were done during normalization
        self.data['Servo.opd_target'][0] = opd_init

        try:
            # show levels before and after normalization for visual check
            import matplotlib.pyplot as plt
            def show_ellipse(ax, levels, title):
                ax.scatter(levels[:,1], levels[:,0], alpha=0.1)
                ax.axis('equal')
                ax.grid()
                ax.set_title(title)

            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            hlevels = utils.compute_profiles_levels(np.array(rec_hprofiles), None, hpixels_lists)
            show_ellipse(axes[0,0], hlevels, "V profiles levels before normalization")
            hlevels_norm = utils.normalize_ellipses(hlevels, hellipse_norm_coeffs)
            show_ellipse(axes[0,1], hlevels_norm, "V profiles levels after normalization")

            vlevels = utils.compute_profiles_levels(np.array(rec_vprofiles), None, vpixels_lists)
            show_ellipse(axes[1,0], vlevels, "H profiles levels before normalization")
            vlevels_norm = utils.normalize_ellipses(vlevels, vellipse_norm_coeffs)
            show_ellipse(axes[1,1], vlevels_norm, "H profiles levels after normalization")
            plt.show()
            
        except Exception as e:
            log.error(f"Failed to show normalization check plots: {e}")


        

    def _reset_zpd(self, _):
        log.info("ZPD Reset")
        actual_opd = self.data['IRCamera.mean_opd'][0]
        new_opd = self.data['Servo.opd_target'][0]
        actual_offset = self.data['IRCamera.mean_opd_offset'][0]
        new_offset = actual_opd + actual_offset - new_opd
        log.info(f"   Actual OPD: {actual_opd} nm | New OPD target: {new_opd} nm | Previous offset: {actual_offset} nm | internal OPD: {actual_opd + actual_offset} |  Applying offset: {new_offset} nm")
        self.data['IRCamera.mean_opd_offset'][0] = float(new_offset)
        

    def _center_piezos(self):
        # centering OPD piezo at mid range
        log.info('centering OPD piezo')
                
        self.data['DAQ.piezos_level'][0] = 5

        time.sleep(1) # wait for piezo to move
        
    def _move_with_nexline(self):
        self.events['Nexline.move'].set()
        
        time.sleep(0.1) # wait for event dispatch
        while True: # wait for nexline move to finish
            if NexlineState(int(self.data['Nexline.state'][0])).name != 'MOVING':
                break
            time.sleep(0.1)
            
    def _walk_to_opd(self, _):
        pass


    def _start_waiting(self, _):
        pass

    def on_enter_waiting(self, _):
        log.info("Waiting")
        self._mode = "waiting"
        self._waiting_active = True

        self._waiting_start = time.perf_counter()
        self._waiting_last_norm = time.perf_counter()

        self._dac = DAController(self.data, self.tip_model, self.tilt_model, mode='walking')

        self.events['Tracker.start_recording'].set()

    def _loop_waiting(self):
        if not self._waiting_active:
            return

        t = time.perf_counter() - self._waiting_start

        piezo_position = 5. + utils.triangle(
            t,
            config.WAITING_PIEZO_AMPLITUDE,
            self.data['params.WAITING_PIEZO_PERIOD'])

        self.data['DAQ.piezos_level'][0] = piezo_position

        if time.perf_counter() - self._waiting_last_norm > self.data['params.SERVO_WAIT_NORMALIZE_TIME'][0]:
            self.events['Tracker.normalize'].set()
            self._waiting_last_norm = time.perf_counter()

        self._dac.update()

    def on_exit_waiting(self, _):
        self.events['Tracker.stop_recording'].set() # stop recording tracker data

        self._center_piezos()
        time.sleep(3) # wait for piezo to stabilize
        self.data['Servo.opd_target'][0] = float(self.data['Tracker.opd_1'][0]) # set OPD target to current value to avoid jumps when starting tracking
        
        log.info('Waiting ended, piezos centered, ready to walk')


    def _stop_waiting(self, _):
        pass
    
    def on_enter_walking(self, _):
        log.info("Walking to OPD")

        self._mode = "walking"
        self._walking_active = True
        self._walking_state = "init"

        self._dac = DAController(self.data, self.tip_model, self.tilt_model, mode='walking')

        self._walking_start_time = time.perf_counter()
        self._walking_refresh = time.perf_counter()
        self._walking_last_norm = time.perf_counter()

        self._walk_direction = np.sign(
            self.data['Servo.opd_target'][0] - self.data['Tracker.opd_1'][0]
        )

        self._velocity_target = (
            self._walk_direction * abs(self.data['Servo.velocity_target'][0])
        )

        self._step_velocity_adjustment = 1.0

        self._final_target = float(self.data['Servo.opd_target'][0])

        self._pid_walk_opd = pid.get_pid_control(self.data, 'WALK_OPD')
        
        self.events['Tracker.start_recording'].set()


    def _loop_walking(self):
        if not self._walking_active:
            return

        if self._walking_state == "init":
            self._walking_init_step()

        elif self._walking_state == "step":
            self._walking_start_step()

        elif self._walking_state == "wait_step":
            self._walking_update_step()

        elif self._walking_state == "final":
            self._walking_finalize()
            
    def _walking_init_step(self):
        self._move_start_opd = float(self.data['Tracker.opd_1'][0])
        self._move_start_time = time.perf_counter()

        self._walking_state = "step"

    def _walking_start_step(self):
        step = self._walk_direction * self.data['params.NEXLINE_OPD_UPDATE'][0]

        self._step_target = float(self.data['Tracker.opd_3'][0]) + step

        self.data['Servo.opd_target'][0] = self._step_target
        self.data['Nexline.moving_velocity'][0] = abs(
            self._velocity_target * self._step_velocity_adjustment
        )

        self.events['Nexline.move'].set()

        self._step_start_time = time.perf_counter()
        self._step_start_opd = float(self.data['Tracker.opd_3'][0])
        self._step_start_piezo = float(self.data['DAQ.piezos_level'][0])

        self._walking_state = "wait_step"

    def _walking_update_step(self):
        loop_startt = time.perf_counter()

        opd = float(self.data['Tracker.opd_100'][0])

        # stop handling
        if self.events['Servo.stop'].is_set():
            self.dispatch(self.Event.STOP_WALKING)
            return

        nexline_moving = (
            NexlineState(int(self.data['Nexline.state'][0])).name == 'MOVING'
        )

        if (time.perf_counter() - self._step_start_time > 0.5) and (not nexline_moving):
            self._walking_end_step()
            return

        estimated_opd = (
            self._move_start_opd +
            self._velocity_target * (time.perf_counter() - self._move_start_time) * 1000
        )

        self.data['Servo.e_opd'][0] = float(opd - estimated_opd)
        self.data['Servo.e_velocity'][0] = float((float(self.data['Tracker.velocity_3'][0]) / 1000)
                                                 - self._velocity_target)

        self.data['DAQ.piezos_level'][0] = self._pid_walk_opd.update(  
            control=self.data['DAQ.piezos_level'][0],
            setpoint=estimated_opd,
            measurement=opd,
        )

        self._dac.update()

        if time.perf_counter() - self._walking_last_norm > self.data['params.SERVO_WALK_NORMALIZE_TIME'][0]:
            self.events['Tracker.normalize'].set()
            self._walking_last_norm = time.perf_counter()

        final_target = self._final_target 

        # check if final target got further than final_target
        if abs(opd - final_target) < config.OPD_TOLERANCE:
            log.info(f"OPD target reached: {opd}")
            self._walking_state = "final"
            return
        elif self._walk_direction > 0 and opd > final_target:
            log.info(f"OPD target reached (overshoot): {opd} > {final_target}")
            self._walking_state = "final"
            return
        elif self._walk_direction < 0 and opd < final_target:
            log.info(f"OPD target reached (overshoot): {opd} < {final_target}")
            self._walking_state = "final"
            return

        dt = time.perf_counter() - loop_startt
        if dt < config.OPD_LOOP_TIME:
            time.sleep(config.OPD_LOOP_TIME - dt)
        
    def _walking_end_step(self):
        step_opd_walked = float(self.data['Tracker.opd_3'][0]) - self._step_start_opd
        step_expected = self._walk_direction * self.data['params.NEXLINE_OPD_UPDATE'][0]

        if abs(step_opd_walked) < 0.5 * abs(step_expected):
            self._walking_state = "step"
            return

        step_time = time.perf_counter() - self._step_start_time
        step_velocity = step_opd_walked / step_time / 1000.0

        ratio = abs(step_velocity / self._velocity_target)

        self._step_velocity_adjustment /= ratio

        self._step_velocity_adjustment = np.clip(
            self._step_velocity_adjustment, 0.1, 1.9
        )

        log.info(f"Velocity adjustment: {self._step_velocity_adjustment:.3f}")

        self._walking_state = "step"
        
    def _walking_finalize(self):
        self._walking_active = False
        self.dispatch(self.Event.STOP_WALKING)
        return

    def on_exit_walking(self, _):
        self._mode = None 
        self._walking_active = False
        log.info("Exiting walking state, stopping Nexline and tracker recording")

        self.events['Nexline.stop_moving'].set()
        self.events['Tracker.stop_recording'].set()
        time.sleep(1.)
        
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

                nexline_is_moving = NexlineState(int(self.data['Nexline.state'][0])).name == 'MOVING'
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
        self._center_piezos()
        
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

            pid_opd_control = pid.get_pid_control(self.data, 'TRACK_OPD')
            
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
                          priority={"niceness": config.SERVO_MAX_NICENESS,
                                    "cpus": config.SERVO_CPU_IRCAM},
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
        self.start_worker(ircam.IRCamera,
                          priority={"niceness": config.SERVO_MAX_NICENESS,
                                    "cpus": config.SERVO_CPU_IRCAM},
                          roi_mode=False)
                
    
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
                log.warning(f"{p.name} stuck, forcing terminate() SIGTERM")
                p.terminate()
                p.join(timeout=2)
                
            if p.is_alive():
                log.error(f"{p.name} really stuck, forcing terminate() SIGKILL")
                p.kill()
                p.join(timeout=2)

                

        self.queue.put_nowait(None)



class DAController(object):
    def __init__(self, data, tip_model, tilt_model, mode='walking'):

        self.tip_model = tip_model
        self.tilt_model = tilt_model

        if mode == 'walking':
            params_name = 'WALK'
        elif mode == 'tracking':
            params_name = 'TRACK'
        else:
            raise ValueError(f"Invalid mode {mode} for DAController")

        self.data = data
        
        self.da1_level_orig = float(self.data['DAQ.piezos_level'][1])
        self.da2_level_orig = float(self.data['DAQ.piezos_level'][2])

        self.last_da_update_time = time.perf_counter()
        self.da1_buffer = list()
        self.da2_buffer = list()

        dt = config.OPD_LOOP_TIME  # 0.01 s (100 Hz)

        max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]

        da1_min = self.da1_level_orig - max_v_diff
        da1_max = self.da1_level_orig + max_v_diff
        da2_min = self.da2_level_orig - max_v_diff
        da2_max = self.da2_level_orig + max_v_diff

        self.pid_da1_control = pid.get_pid_control(
            self.data,
            f'{params_name}_DA1', out_min=da1_min, out_max=da1_max,
            deadband=self.data['params.PID_DA_DEADBAND'],
            kaw=self.data['params.PID_DA_KAW'])
        self.pid_da2_control = pid.get_pid_control(
            self.data,
            f'{params_name}_DA2', out_min=da2_min, out_max=da2_max,
            deadband=self.data['params.PID_DA_DEADBAND'],
            kaw=self.data['params.PID_DA_KAW'])

        self.last_update_time = time.perf_counter()
        self.last_refresh_time = time.perf_counter()
        
    def update(self):
        opd = float(self.data['Tracker.opd_100'][0]) # nm
        tip = float(self.data['Tracker.tip_10'][0]) 
        tilt = float(self.data['Tracker.tilt_10'][0])

        phase = utils.opd2phase(opd)
        tip_target = self.tip_model(phase)
        tilt_target = self.tilt_model(phase)
        self.data['Servo.tip_target'][0] = float(tip_target)
        self.data['Servo.tilt_target'][0] = float(tilt_target)

        e_tip  = tip_target  - tip
        e_tilt = tilt_target - tilt
        self.data['Servo.e_tip'][0] = float(e_tip)
        self.data['Servo.e_tilt'][0] = float(e_tilt)
                
        self.data['DAQ.piezos_level'][1] = self.pid_da1_control.update(
            control=self.data['DAQ.piezos_level'][1],
            setpoint=tilt_target,
            measurement=tilt)
        
        self.data['DAQ.piezos_level'][2] = self.pid_da2_control.update(
            control=self.data['DAQ.piezos_level'][2],
            setpoint=tip_target,
            measurement=tip)
        
        self.da1_buffer.append(self.data['DAQ.piezos_level'][1])
        self.da2_buffer.append(self.data['DAQ.piezos_level'][2])

        if time.perf_counter() - self.last_update_time > self.data['params.PIEZO_DA_LOOP_UPDATE_TIME'][0]:
            self.da1_level_orig = np.median(self.da1_buffer)
            self.da2_level_orig = np.median(self.da2_buffer)
            self.da1_buffer.clear()
            self.da2_buffer.clear()
            self.last_update_time = time.perf_counter()


        if time.perf_counter() - self.last_refresh_time > config.SERVO_NONCRITIC_REFRESH_TIME:

            max_v_diff = self.data['params.PIEZO_DA_LOOP_MAX_V_DIFF'][0]
            da1_min = self.da1_level_orig - max_v_diff
            da1_max = self.da1_level_orig + max_v_diff
            da2_min = self.da2_level_orig - max_v_diff
            da2_max = self.da2_level_orig + max_v_diff
            
            self.pid_da1_control.update_config(out_min=da1_min, out_max=da1_max,
                                               deadband=self.data['params.PID_DA_DEADBAND'],
                                               kaw=self.data['params.PID_DA_KAW'])
            self.pid_da2_control.update_config(out_min=da2_min, out_max=da2_max,
                                               deadband=self.data['params.PID_DA_DEADBAND'],
                                               kaw=self.data['params.PID_DA_KAW'])

            self.last_refresh_time = time.perf_counter()





