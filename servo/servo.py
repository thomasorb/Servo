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
from .fsm import StateMachine, FsmEvent, FsmState, Transition


log = logging.getLogger(__name__)


def worker_process(queue, data, WorkerClass, events, priority=None):
    """
    Subprocess that:
      - configures logging
      - instantiates and runs the worker
      - guarantees worker.stop() is called
    """

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
        worker = WorkerClass(data, events)
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
            (FsmState.IDLE,    FsmEvent.START): Transition(FsmState.RUNNING, action=self._start),
            (FsmState.RUNNING, FsmEvent.NORMALIZE): Transition(FsmState.RUNNING, action=self._normalize),
            (FsmState.RUNNING, FsmEvent.ERROR): Transition(FsmState.FAILED),
            (FsmState.RUNNING, FsmEvent.STOP):  Transition(FsmState.STOPPED, action=self._stop),
            (FsmState.FAILED,  FsmEvent.RESET): Transition(FsmState.IDLE),
            (FsmState.STOPPED, FsmEvent.RESET): Transition(FsmState.IDLE),
        }
        super().__init__(FsmState.IDLE, table)

        self.event_manager = multiprocessing.Manager()
        self.events = self.event_manager.dict()

        for iname in config.SERVO_EVENTS:
            self.events[iname] = self.event_manager.Event()

        self.queue = logger.get_logging_queue()

        self.data = core.SharedData()

        
    def poll(self):
        evs = self.events

        for iname in config.SERVO_EVENTS:
            if evs.get(iname) and evs[iname].is_set():
                evs[iname].clear()
                self.dispatch(getattr(FsmEvent, iname.upper()), payload=None)

    # Hooks (facultatif)
    def on_enter_running(self, _):
        log.info(">> RUNNING")
        # running loop
        while True:
            try:
                self.poll()
                if self.state is FsmState.STOPPED:
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                log.error('Keyboard interrupt')
                self.events('stop').set()
                

        
    def on_exit_running(self, _):
        log.info("<< RUNNING")

    # Actions
    def _start(self, _):

        log.info("Starting Servo")
        ## start all threads
        self.workers = list()

        def start_worker(WorkerClass, priority):
            stop_event_name = f"{WorkerClass.__name__}.stop"
            self.events[stop_event_name] = self.event_manager.Event()

            worker = multiprocessing.Process(target=worker_process,
                                             args=(self.queue, self.data, WorkerClass,
                                                   self.events, priority))
            self.workers.append((worker, stop_event_name))
            worker.start()


        # start ir camera
        if not self.nocam:
            start_worker(ircam.IRCamera, -20)

        # start piezos
        start_worker(piezo.DAQ, 0)

        # start viewer
        if not self.noviewer:
            if not self.nocam:
                while True:
                    time.sleep(0.1)
                    if self.data['IRCamera.initialized'][0]: break
            start_worker(viewer.Viewer, 10)


    def _normalize(self, _):
        log.info("Normalizing")

        start_value = 3
        end_value = 7
        recall_value = self.data['DAQ.piezos_level'][0]

        rec_hprofiles = list()
        rec_vprofiles = list()
        profile_len = self.data['IRCamera.profile_len'][0]
        
        def piezo_goto(val, rec=False):
        
            self.data['DAQ.piezos_level'][0] = np.array(
                val, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)

            while True:
                if self.data['DAQ.piezos_level_actual'][0] == val:
                    break
                if self.events['stop'].is_set():
                    break
                if rec:
                    rec_hprofiles.append(np.copy(self.data['IRCamera.hprofile'][:profile_len]))
                    rec_vprofiles.append(np.copy(self.data['IRCamera.vprofile'][:profile_len]))

            log.info(f"OPD piezo at {self.data['DAQ.piezos_level_actual'][0]}")

        piezo_goto(start_value)

        piezo_goto(end_value, rec=True)

        piezo_goto(recall_value)

        hnorm = utils.get_normalization_coeffs(np.array(rec_hprofiles)).astype(config.FRAME_DTYPE)
        self.data['IRCamera.hnorm_min'][:profile_len] = hnorm[:,0]
        self.data['IRCamera.hnorm_max'][:profile_len] = hnorm[:,1]

        vnorm = utils.get_normalization_coeffs(np.array(rec_vprofiles)).astype(config.FRAME_DTYPE)
        self.data['IRCamera.vnorm_min'][:profile_len] = vnorm[:,0]
        self.data['IRCamera.vnorm_max'][:profile_len] = vnorm[:,1]
        
        #np.save('hprofiles.npy', np.array(rec_hprofiles))
        #np.save('vprofiles.npy', np.array(rec_vprofiles))

            
            
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



