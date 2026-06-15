import atexit
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import traceback
from typing import Optional

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

_DEFAULT_LOG_FILE = os.path.expanduser("~/.local/state/servo/servo.log")
LOG_FILE = os.getenv("SERVO_LOG_FILE", _DEFAULT_LOG_FILE)

MAX_BYTES = int(os.getenv("SERVO_LOG_MAX_BYTES", str(30 * 1024 * 1024)))
BACKUP_COUNT = int(os.getenv("SERVO_LOG_BACKUP_COUNT", "3"))

# Keep real streams (avoid recursion)
_REAL_STDOUT = sys.__stdout__
_REAL_STDERR = sys.__stderr__

# -------------------------------------------------------------------
# COLORS (pastel theme)
# -------------------------------------------------------------------

RESET = "\033[0m"

COLORS = {
    "DEBUG": "\033[38;2;100;180;255m",   # clear blue
    "INFO": "\033[38;2;80;220;120m",     # fresh green
    "WARNING": "\033[38;2;255;200;80m",  # warm yellow/orange
    "ERROR": "\033[38;2;255;80;80m",     # clear red
    "CRITICAL": "\033[38;2;255;100;255m" # magenta
}

class ColorFormatter(logging.Formatter):
    """Colored formatter with improved contrast."""

    
    LEVEL_MAP = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    def format(self, record):
        color = COLORS.get(record.levelname, "")
        
        original_levelname = record.levelname
        record.levelname = self.LEVEL_MAP.get(original_levelname, original_levelname)

        level = f"\033[1m{record.levelname}\033[0m"  # bold level
        record.levelname = level

        message = super().format(record)
        return f"{color}{message}{RESET}"

# -------------------------------------------------------------------
# GLOBAL (main process only)
# -------------------------------------------------------------------

_queue: Optional[mp.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_initialized = False


# -------------------------------------------------------------------
# SAFE QUEUE HANDLER
# -------------------------------------------------------------------

class SafeQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that never recurses."""
    def handleError(self, record):
        try:
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                traceback.print_exception(*exc_info, file=_REAL_STDERR)
        except Exception:
            pass


# -------------------------------------------------------------------
# INIT (MAIN PROCESS ONLY)
# -------------------------------------------------------------------

def init_logging(level="INFO", log_file=None):
    global _queue, _listener, _initialized

    if _initialized:
        return

    logging.raiseExceptions = False

    # multiprocessing-safe queue
    ctx = mp.get_context()
    _queue = ctx.Queue()

    log_file = log_file or LOG_FILE
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # file handler (no colors)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )

    # console handler (WITH COLORS)
    console_handler = logging.StreamHandler(_REAL_STDOUT)

    base_format = "%(asctime)s│%(levelname)s│(%(processName)s)│%(name)s│ %(message)s"
    
    formatter = logging.Formatter(
        base_format,
        datefmt="%H:%M:%S"
    )
    
    color_formatter = ColorFormatter(
        base_format,
        datefmt="%H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(color_formatter)

    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    _listener = logging.handlers.QueueListener(
        _queue,
        file_handler,
        console_handler,
        respect_handler_level=True
    )
    _listener.start()

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    for h in list(root.handlers):
        root.removeHandler(h)

    root.addHandler(SafeQueueHandler(_queue))

    _initialized = True
    atexit.register(shutdown_logging)


# -------------------------------------------------------------------
# WORKER CONFIG
# -------------------------------------------------------------------

def configure_worker_logging(level=None, queue=None):
    if queue is None:
        raise RuntimeError("Logging queue must be provided to worker")

    logging.raiseExceptions = False

    root = logging.getLogger()

    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    handler = SafeQueueHandler(queue)
    root.addHandler(handler)

    # ensure propagation for all module loggers
    for name, logger_obj in logging.root.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.handlers.clear()
            logger_obj.propagate = True


# -------------------------------------------------------------------
# SHUTDOWN
# -------------------------------------------------------------------

def shutdown_logging():
    global _listener, _initialized

    try:
        if _listener:
            _listener.stop()
    except Exception:
        pass

    _initialized = False


# -------------------------------------------------------------------
# ACCESSOR
# -------------------------------------------------------------------

def get_logging_queue():
    return _queue
