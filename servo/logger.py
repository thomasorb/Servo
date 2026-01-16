import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import uuid
import atexit
import time
from typing import Optional

# --------- Default log file location ----------
_DEFAULT_LOG_FILE = os.path.expanduser("~/.local/state/servo/servo.log")
LOG_FILE = os.getenv("SERVO_LOG_FILE", _DEFAULT_LOG_FILE)

# Global objects used by the logging system
_queue: Optional[mp.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_initialized: bool = False
_session_id: Optional[str] = None
_session_start_ns: Optional[int] = None

# --------- ANSI colors for console output ----------

import os
import sys

def _supports_color():
    """
    Returns True if the current output supports ANSI colors.
    - TTY required
    - On Windows, requires colorama or WT/conhost with VT enabled
    - Disables color in common CI environments unless FORCE_COLOR is set
    """
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    if os.name == "nt":
        # Try to enable ANSI on Windows with colorama if available
        try:
            import colorama
            colorama.just_fix_windows_console()
            return True
        except Exception:
            # Windows 10+ often supports VT sequences in modern terminals
            return os.getenv("WT_SESSION") is not None
    return True


def _rgb(r, g, b):
    """Return an ANSI 24-bit color sequence for foreground RGB."""
    return f"\033[38;2;{r};{g};{b}m"


RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


# Pastel palettes (RGB)
_THEMES = {
    "pastel": {
        "DEBUG":  (137, 165, 255),  # soft periwinkle blue
        "INFO":   (144, 238, 196),  # mint green
        "WARNING":(255, 236, 158),  # buttery yellow
        "ERROR":  (255, 183, 178),  # soft coral
        "CRITICAL": (218, 160, 255) # lavender
    },
    "soft": {
        "DEBUG":  (164, 196, 244),  # baby blue
        "INFO":   (180, 234, 204),  # seafoam
        "WARNING":(252, 222, 156),  # peachy sand
        "ERROR":  (250, 170, 160),  # light salmon
        "CRITICAL": (230, 180, 255) # lilac
    },
    "mono": {
        # No color; mapped to None → disables coloring
        "DEBUG":   None,
        "INFO":    None,
        "WARNING": None,
        "ERROR":   None,
        "CRITICAL":None,
    }
}


class ColorFormatter(logging.Formatter):
    """
    Console formatter with selectable pastel themes.
    Uses TrueColor ANSI when available, otherwise gracefully degrades.
    If color is disabled, falls back to plain text formatting.
    """
    def __init__(self, fmt, theme="pastel", use_color=None):
        super().__init__(fmt)
        self.theme = theme if theme in _THEMES else "pastel"
        self.palette = _THEMES[self.theme]
        self.use_color = _supports_color() if use_color is None else use_color

        # Fallback to mono if colors not supported
        if not self.use_color or self.theme == "mono":
            self.palette = _THEMES["mono"]

    def _colorize_level(self, levelname, text):
        color_rgb = self.palette.get(levelname)
        if not color_rgb:
            return text  # mono or unknown
        r, g, b = color_rgb
        return f"{_rgb(r, g, b)}{text}{RESET}"

    def format(self, record):
        # Format base message
        base = super().format(record)

        if not self.palette.get(record.levelname):
            return base  # mono

        # Subtle styling: color entire line softly + dim the metadata brackets
        # Keep format: "[LEVEL] message" → colorize the level tag, leave message uncolored or slightly colored
        try:
            # Colorize the [LEVEL] part only for readability
            if base.startswith("[") and "] " in base:
                left, rest = base.split("] ", 1)
                level = left.strip("[]")
                left_col = f"[{self._colorize_level(level, level)}]"
                return f"{left_col} {rest}"
            else:
                # If format differs, softly color the whole line
                return self._colorize_level(record.levelname, base)
        except Exception:
            # Never break logging on formatting
            return base


class _StreamToLogger:
    """
    Redirects print() and raw stdout/stderr writes into the logging system.
    This ensures all output ends up in the same log file and console stream.
    """
    def __init__(self, logger_name, level):
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self._logger.log(self._level, message)

    def flush(self):
        pass


def _to_level(level):
    """Converts strings like 'INFO' to logging.INFO integers."""
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(level.upper(), logging.INFO)


# -------------------------------------------------------------
# Shutdown handler MUST be defined before init_logging()
# -------------------------------------------------------------
def shutdown_logging():
    """
    Called automatically when the process exits.
    Ensures the QueueListener is stopped cleanly and logs the session end.
    """
    global _listener, _session_start_ns

    if _listener is not None:
        try:
            root = logging.getLogger()
            if _session_start_ns is not None:
                elapsed_s = (time.time_ns() - _session_start_ns) / 1e9
                root.info(f"=== Servo Session {_session_id} ended (elapsed: {elapsed_s:.3f}s) ===")
            else:
                root.info(f"=== Servo Session {_session_id} ended ===")
        except Exception:
            pass

        try:
            _listener.stop()
        except Exception:
            pass

        _listener = None


# -------------------------------------------------------------
# Main initialization function
# -------------------------------------------------------------
def init_logging(level="INFO", log_file=None, redirect_std=True, theme='pastel'):
    """
    Initialize a robust multi-process logging system.
    - Creates a unique session ID and timestamps execution.
    - Centralizes logs from all processes via a QueueListener.
    - Writes to both a file and a colorized console stream.
    - Optionally redirects print() and raw stdout to logging.
    """
    global _queue, _listener, _initialized, _session_id, _session_start_ns

    if _initialized:
        return

    lvl = _to_level(level)
    log_path = log_file or LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(lvl)

    # File handler (no colors, long format)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(processName)s) (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_fmt)

    # Console handler (__stdout__ avoids loops if sys.stdout is redirected)
    console_handler = logging.StreamHandler(stream=getattr(sys, "__stdout__", sys.stdout))
    console_handler.setFormatter(ColorFormatter("[%(levelname)s] (%(name)s) %(message)s"))

    # Shared queue and listener for all logs (main + workers)
    _queue = mp.Queue()
    _listener = logging.handlers.QueueListener(
        _queue, file_handler, console_handler, respect_handler_level=True
    )
    _listener.start()

    # Replace root handlers with a QueueHandler
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(_queue))

    # Session bookkeeping
    _session_id = uuid.uuid4().hex[:8]
    _session_start_ns = time.time_ns()
    root.info(f"=== Servo Session {_session_id} started === (log: {log_path})")

    # Optional: redirect print() and stderr
    if redirect_std:
        sys.stdout = _StreamToLogger("stdout", logging.INFO)
        sys.stderr = _StreamToLogger("stderr", logging.ERROR)

    # Register graceful shutdown
    atexit.register(shutdown_logging)

    _initialized = True


# -------------------------------------------------------------
# Worker process initialization
# -------------------------------------------------------------
def configure_worker_logging(level=None, redirect_std=True, queue=None):
    """
    Must be called at the beginning of each multiprocessing worker.
    Ensures the worker uses the same QueueHandler as the main process.
    """
    global _queue

    root = logging.getLogger()

    if level is not None:
        root.setLevel(_to_level(level))

    if queue is not None:
        _queue = queue

    # Each worker has only a QueueHandler
    root.handlers = []
    if _queue is None:
        # Fallback — should never happen unless called incorrectly
        stream = logging.StreamHandler(stream=getattr(sys, "__stdout__", sys.stdout))
        fmt = ColorFormatter("[%(levelname)s] %(name)s: %(message)s")
        stream.setFormatter(fmt)
        root.addHandler(stream)
    else:
        root.addHandler(logging.handlers.QueueHandler(_queue))

    if redirect_std:
        sys.stdout = _StreamToLogger("stdout", logging.INFO)
        sys.stderr = _StreamToLogger("stderr", logging.ERROR)



def get_logging_queue():
    """
    Returns the logging queue so it can be explicitly passed
    to multiprocessing workers.
    """
    return _queue
