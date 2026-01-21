import logging
import psutil

from .servo import Servo
from . import fsm

log = logging.getLogger(__name__)
    
    
def run(**kwargs):

    try:
        psutil.Process().nice(0)  # privilégier uniquement ce worker
    except psutil.AccessDenied:
        log.warning('priority of the process could not be changed (Acces Denied)')

    servo = Servo(**kwargs)

    # start servo
    servo.dispatch(fsm.FsmEvent.START)
    
        


