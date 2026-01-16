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

log = logging.getLogger(__name__)
    


def worker_process(queue, data, WorkerClass, stop_event, priority=None):
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
        worker = WorkerClass(data, stop_event)
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
    try:
        psutil.Process().nice(0)  # privilégier uniquement ce worker
    except psutil.AccessDenied:
        log.warning('priority of the process could not be changed (Acces Denied)')
        
    queue = logger.get_logging_queue()

    data = core.SharedData()

    
    ## start all threads
    workers = list()

    # start ir camera
    if not nocam:
        ircam_stop_event = multiprocessing.Event()
        ircam_worker = multiprocessing.Process(target=worker_process,
                                               args=(queue, data, ircam.IRCamera,
                                                     ircam_stop_event, -20))
        workers.append((ircam_worker, ircam_stop_event))
        ircam_worker.start()
    
    # start piezos
    piezo_stop_event = multiprocessing.Event()
    piezo_worker = multiprocessing.Process(target=worker_process,
                                           args=(queue, data, piezo.DAQ,
                                                 piezo_stop_event, 0))
    workers.append((piezo_worker, piezo_stop_event))
    piezo_worker.start()
    
    
        
    # start viewer
    if not noviewer:
        if not nocam:
            while True:
                time.sleep(0.1)
                if data['IRCamera.initialized'][0]: break
        viewer_stop_event = multiprocessing.Event()
        viewer_worker = multiprocessing.Process(target=worker_process,
                                                args=(queue, data, viewer.Viewer,
                                                      viewer_stop_event, 10))
        workers.append((viewer_worker, viewer_stop_event))
        viewer_worker.start()


    input('press any key to exit\n')

    try:
        # Graceful stop
        for p, ev in workers:
            time.sleep(1)
            ev.set()

        # Wait for clean exit
        for p, _ in workers:
            p.join(timeout=5)

        # Fallback kill if needed
        for p, _ in workers:
            if p.is_alive():
                log.error(f"{p.name} stuck, forcing terminate()")
                p.terminate()
                p.join()
    except Exception() as e:
        log.error(f'Error at exit: {e}')

    finally:
        data.stop()

        queue.put_nowait(None)


if __name__ == '__main__':
    main()

