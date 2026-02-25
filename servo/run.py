import logging
import psutil
import traceback

from .servo import Servo, ServoEvent
log = logging.getLogger(__name__)
    
    
def run(**kwargs):

    try:
        servo = Servo(**kwargs)
        # start servo
        servo.dispatch(ServoEvent.START)

    except KeyboardInterrupt:
        log.error(f'error during run: {traceback.format_exc()}')

        


