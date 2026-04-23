import atexit
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import time
import uuid
from typing import Optional


# --------- Default log file location ----------
_DEFAULT_LOG_FILE = os.path.expanduser("~/.local/state/servo/servo.log")
LOG_FILE = os.getenv("SERVO_LOG_FILE", _DEFAULT_LOG_FILE)

# Approximate size limit (~30 MB) with rotation
_DEFAULT_MAX_BYTES = 30 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3

LOG_MAX_BYTES = int(os.getenv("SERVO_LOG_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))
LOG_BACKUP_COUNT = int(os.getenv("SERVO_LOG_BACKUP_COUNT", str(_DEFAULT_BACKUP_COUNT)))

# Keep original streams to avoid logging recursion loops
_REAL_STDOUT = getattr(sys, "__stdout__", sys.stdout)
_REAL_STDERR = getattr(sys, "__stderr__", sys.stderr)

# Global objects used by the logging system
_queue: Optional[mp.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_initialized: bool = False
_session_id: Optional[str] = None
_session_start_ns: Optional[int] = None


# --------- ANSI colors for console output ----------
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"

_THEMES = {
    "pastel": {
        "DEBUG": (137, 165, 255),
        "INFO": (144, 238, 196),
        "WARNING": (255, 236, 158),
        "ERROR": (255, 183, 178),
        "CRITICAL": (218, 160, 255),
    },
    "soft": {
        "DEBUG": (164, 196, 244),
        "INFO": (180, 234, 204),
        "WARNING": (252, 222, 156),
        "ERROR": (250, 170, 160),
        "CRITICAL": (230, 180, 255),
    },
    "mono": {
        "DEBUG": None,
        "INFO": None,
        "WARNING": None,
        "ERROR": None,
        "CRITICAL": None,
    },
}


def _supports_color(stream=None) -> bool:
    """Return True if the given output stream supports ANSI colors."""
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True

    stream = stream or _REAL_STDOUT
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False

    if os.name == "nt":
        try:
            import colorama  # type: ignore

            colorama.just_fix_windows_console()
            return True
        except Exception:
            return os.getenv("WT_SESSION") is not None

    return True


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


class ColorFormatter(logging.Formatter):
    """Console formatter with optional pastel coloring."""

    def __init__(self, fmt: str, theme: str = "pastel", use_color: Optional[bool] = None):
        super().__init__(fmt)
        self.theme = theme if theme in _THEMES else "pastel"
        self.palette = _THEMES[self.theme]
        if use_color is None:
            use_color = _supports_color(_REAL_STDOUT)
        self.use_color = use_color

        if not self.use_color or self.theme == "mono":
            self.palette = _THEMES["mono"]

    def _colorize_level(self, levelname: str, text: str) -> str:
        color_rgb = self.palette.get(levelname)
        if not color_rgb:
            return text
        r, g, b = color_rgb
        return f"{_rgb(r, g, b)}{text}{RESET}"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.use_color:
            return base

        try:
            if base.startswith("[") and "] " in base:
                left, rest = base.split("] ", 1)
                level = left.strip("[]")
                left_col = self._colorize_level(level, f"[{level}]")
                return f"{left_col} {rest}"
            return self._colorize_level(record.levelname, base)
        except Exception:
            return base


class _StreamToLogger:
    """
    Redirect raw stdout/stderr writes into logging, while preventing recursion.
    """

    def __init__(self, logger_name: str, level: int, fallback_stream=None):
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._fallback = fallback_stream or _REAL_STDERR
        self._in_write = False

    def write(self, message: str) -> None:
        if not message:
            return
        if message.isspace():
            return

        # Prevent recursive logging when logging itself writes to stderr (handleError()).
        if self._in_write:
            try:
                self._fallback.write(message)
                self._fallback.flush()
            except Exception:
                pass
            return

        # Bypass logging's internal error banners/tracebacks to the real stderr.
        if (
            message.startswith("--- Logging error ---")
            or message.startswith("Traceback (most recent call last):")
            or message.startswith("Call stack:")
        ):
            try:
                self._fallback.write(message)
                self._fallback.flush()
            except Exception:
                pass
            return

        try:
            self._in_write = True
            msg = message.rstrip("\n")
            if msg:
                self._logger.log(self._level, msg)
        finally:
            self._in_write = False

    def flush(self) -> None:
        try:
            self._fallback.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self._fallback, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._fallback, "encoding", "utf-8")


def _to_level(level):
    """Convert strings like 'INFO' to logging.INFO integers."""
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(str(level).upper(), logging.INFO)


def shutdown_logging():
    """Stop QueueListener cleanly and restore original std streams."""
    global _listener, _session_start_ns, _initialized

    root = logging.getLogger()

    try:
        if _session_start_ns is not None:
            elapsed_s = (time.time_ns() - _session_start_ns) / 1e9
            root.info(f"=== Servo Session {_session_id} ended (elapsed: {elapsed_s:.3f}s) ===")
        else:
            root.info(f"=== Servo Session {_session_id} ended ===")
    except Exception:
        pass

    try:
        if _listener is not None:
            _listener.stop()
    except Exception:
        pass

    _listener = None

    # Restore real streams (avoid surprises after shutdown)
    try:
        sys.stdout = _REAL_STDOUT
        sys.stderr = _REAL_STDERR
    except Exception:
        pass

    _initialized = False


def init_logging(level="INFO", log_file=None, redirect_std=True, theme="pastel"):
    """
    Initialize multiprocessing-safe logging through a central QueueListener.
    All stdout/stderr can be redirected to logging.
    """
    global _queue, _listener, _initialized, _session_id, _session_start_ns

    if _initialized:
        return

    lvl = _to_level(level)
    log_path = log_file or LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(lvl)

    # File handler with rotation (~30 MB)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(processName)s) (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        mode="a",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(lvl)
    file_handler.setFormatter(file_fmt)

    # Console handler (use real stdout to avoid loops)
    console_stream = _REAL_STDOUT
    console_handler = logging.StreamHandler(stream=console_stream)
    console_handler.setLevel(lvl)
    console_handler.setFormatter(ColorFormatter("[%(levelname)s] (%(name)s) %(message)s", theme=theme))

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

    if redirect_std:
        sys.stdout = _StreamToLogger("stdout", logging.INFO, fallback_stream=_REAL_STDOUT)
        sys.stderr = _StreamToLogger("stderr", logging.ERROR, fallback_stream=_REAL_STDERR)

    atexit.register(shutdown_logging)
    _initialized = True


def configure_worker_logging(level=None, redirect_std=True, queue=None):
    """
    Configure logging for a multiprocessing worker to send logs to the main listener.
    """
    global _queue

    root = logging.getLogger()

    if level is not None:
        root.setLevel(_to_level(level))

    if queue is not None:
        _queue = queue

    root.handlers = []
    if _queue is None:
        stream = logging.StreamHandler(stream=_REAL_STDOUT)
        stream.setFormatter(ColorFormatter("[%(levelname)s] (%(name)s) %(message)s", theme="mono"))
        root.addHandler(stream)
    else:
        root.addHandler(logging.handlers.QueueHandler(_queue))

    if redirect_std:
        sys.stdout = _StreamToLogger("stdout", logging.INFO, fallback_stream=_REAL_STDOUT)
        sys.stderr = _StreamToLogger("stderr", logging.ERROR, fallback_stream=_REAL_STDERR)


def get_logging_queue():
    """Return the logging queue to pass it explicitly to worker processes."""
    return _queue
