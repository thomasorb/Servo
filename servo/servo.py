import logging
import multiprocessing
import time
import traceback
import psutil

from . import ircam
from . import core
from . import viewer
from . import logger
from . import piezo
from . import config
from . import fsm

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

    
def run(mode='loop', nocam=False, noviewer=False):


    event_manager = multiprocessing.Manager()
    events = event_manager.dict()

    for iname in config.SERVO_EVENTS:
        events[iname] = event_manager.Event()

    try:
        psutil.Process().nice(0)  # privilégier uniquement ce worker
    except psutil.AccessDenied:
        log.warning('priority of the process could not be changed (Acces Denied)')
        
    queue = logger.get_logging_queue()

    data = core.SharedData()


    ## start Servo FSM
    servo_fsm = fsm.ServoFSM(data, events)
    
    ## start all threads
    workers = list()

    def start_worker(WorkerClass, priority):
        stop_event_name = f"{WorkerClass.__name__}.stop"
        events[stop_event_name] = event_manager.Event()
        
        worker = multiprocessing.Process(target=worker_process,
                                         args=(queue, data, WorkerClass,
                                               events, priority))
        workers.append((worker, stop_event_name))
        worker.start()


    # start ir camera
    if not nocam:
        start_worker(ircam.IRCamera, -20)
    
    # start piezos
    start_worker(piezo.DAQ, 0)
        
    # start viewer
    if not noviewer:
        if not nocam:
            while True:
                time.sleep(0.1)
                if data['IRCamera.initialized'][0]: break
        start_worker(viewer.Viewer, 10)

    # start servo
    events['start'].set()
    
    # running loop
    while True:
        try:
            servo_fsm.poll()
            if servo_fsm.state is fsm.FsmState.STOPPED:
                break
            time.sleep(0.1)
        except KeyboardInterrupt:
            log.error('Keyboard interrupt')
            break
        
    # Graceful stop
    for p, ev in workers:
        time.sleep(1)
        events[ev].set()

    # Wait for clean exit
    for p, _ in workers:
        p.join(timeout=5)

    data.stop() # must be done before any exit kill
    del event_manager
    
    # Fallback kill if needed
    for p, _ in workers:
        if p.is_alive():
            log.error(f"{p.name} stuck, forcing terminate()")
            p.terminate()
            p.join()

    queue.put_nowait(None)

if __name__ == '__main__':
    main()

