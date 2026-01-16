import json
import os
import sys
import atexit
import signal
import tempfile
import threading
import time
import errno
from pathlib import Path
from typing import Any, Dict, Optional
import math
import numpy as np

# ---------------------------------------------------------------------------
# Cross‑platform file lock (Unix = fcntl, Windows = msvcrt)
# ---------------------------------------------------------------------------

try:
    import fcntl  # Linux / macOS
    def _acquire_lock(fd):
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release_lock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)
except ImportError:
    # Windows implementation
    import msvcrt
    def _acquire_lock(fd):
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _release_lock(fd):
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


# ---------------------------------------------------------------------------
# Helper functions for dotted access
# ---------------------------------------------------------------------------

def _deep_get(d: Dict[str, Any], dotted: str, default=None):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _deep_set(d: Dict[str, Any], dotted: str, value: Any):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Atomic JSON write with multiprocess file lock
# ---------------------------------------------------------------------------

def _atomic_json_dump(obj: Dict[str, Any], path: Path) -> None:
    """
    Atomically write JSON file with multiprocess file lock.
    Safe against race conditions between workers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Lock the final target file
    with open(path, "a+b") as lock_fd:
        _acquire_lock(lock_fd)
        try:
            # Write to a temp file
            tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            os.replace(tmp_name, path)

        finally:
            _release_lock(lock_fd)

        # Cleanup any orphan temp file on exception
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass

def _sanitize_for_json(obj, *, replace_non_finite=True):
    """
    Recursively convert objects to JSON-serializable types:
      - numpy scalars -> Python scalars
      - numpy arrays  -> lists
      - sets/tuples   -> lists
      - dict keys     -> strings (JSON requires string keys)
      - NaN/Inf       -> None (if replace_non_finite=True)
    """
    # NumPy scalars
    if isinstance(obj, (np.floating, )):
        val = float(obj)
        if replace_non_finite and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        # Convert to list of sanitized elements
        return [_sanitize_for_json(x, replace_non_finite=replace_non_finite) for x in obj.tolist()]

    # Builtins
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        if replace_non_finite and isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # Mappings
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # JSON keys must be strings
            key = str(k)
            out[key] = _sanitize_for_json(v, replace_non_finite=replace_non_finite)
        return out

    # Iterables (list/tuple/set/…)
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x, replace_non_finite=replace_non_finite) for x in obj]

    # Fallback: last resort, make it a string
    return str(obj)

# ---------------------------------------------------------------------------
# FULL StateStore WITH periodic auto-save + multiprocess locking
# ---------------------------------------------------------------------------

class StateStore:
    """
    Persistent key-value store with:
      - automatic loading at startup
      - atomic saving at shutdown
      - periodic autosave every N seconds
      - multiprocess-safe file locking
      - dotted key access ("piezos.OPD")

    Intended for: scientific instruments, GUIs, multiprocess systems
    """

    def __init__(
        self,
        app_name: str = "servo",
        filename: str = "state.json",
        schema_version: int = 1,
        autosave_interval: float = 10.0,   # seconds
        only_main_process_writes: bool = True,
        use_config_dir: bool = False,
        directory: Optional[Path] = None,
    ):
        self.schema_version = schema_version
        self.only_main_process_writes = only_main_process_writes
        self.autosave_interval = autosave_interval
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._autosave_thread = None
        self._autosave_stop = threading.Event()

        # Determine location
        if directory is not None:
            base = Path(directory)
        else:
            if use_config_dir:
                base = Path.home() / ".config" / app_name
            else:
                base = Path.home() / ".local" / "state" / app_name

        base.mkdir(parents=True, exist_ok=True)
        self.path = base / filename

        # Load existing state
        self._load()

        # Install shutdown handlers
        atexit.register(self._on_exit)
        self._install_signal_handlers_once()

        # Start autosave thread (main process only)
        if os.getpid() == self._pid:
            self._start_autosave()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        with self._lock:
            return _deep_get(self._data, key, default)

    def set(self, key: str, value):
        with self._lock:
            _deep_set(self._data, key, value)

    def as_dict(self):
        with self._lock:
            return dict(self._data)


    def save(self) -> None:
        """Persist current state to disk (atomic replace)."""
        if self.only_main_process_writes and os.getpid() != self._pid:
            return
        with self._lock:
            payload = {
                "schema_version": self.schema_version,
                "data": self._data,
            }
            # Sanitize here:
            safe_payload = _sanitize_for_json(payload, replace_non_finite=True)
            _atomic_json_dump(safe_payload, self.path)

    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------

    def _load(self):
        if not self.path.exists():
            self._data = {}
            return

        try:
            raw = json.loads(self.path.read_text("utf-8"))
            version = raw.get("schema_version", 1)
            payload = raw.get("data", {})

            if version != self.schema_version:
                payload = self._migrate(payload, version, self.schema_version)

            if not isinstance(payload, dict):
                payload = {}

            self._data = payload

        except Exception:
            # Corrupted file → reset
            self._data = {}

    def _migrate(self, payload, from_ver, to_ver):
        # Simple migration scaffold
        migrated = dict(payload)
        # Add migrations here if schema evolves
        return migrated

    # ------------------------------------------------------------------
    # Autosave system (thread)
    # ------------------------------------------------------------------

    def _autosave_loop(self):
        while not self._autosave_stop.wait(self.autosave_interval):
            try:
                self.save()
            except Exception:
                pass

    def _start_autosave(self):
        self._autosave_thread = threading.Thread(
            target=self._autosave_loop,
            name="StateStoreAutosave",
            daemon=True
        )
        self._autosave_thread.start()

    # ------------------------------------------------------------------
    # Shutdown logic
    # ------------------------------------------------------------------

    def _on_exit(self):
        try:
            self._autosave_stop.set()
            if self._autosave_thread:
                self._autosave_thread.join(timeout=1.0)
            self.save()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Signal handling — ensure saving on Ctrl+C / kill
    # ------------------------------------------------------------------

    _sig_installed = False
    _sig_lock = threading.Lock()

    @classmethod
    def _install_signal_handlers_once(cls):
        with cls._sig_lock:
            if cls._sig_installed:
                return
            cls._sig_installed = True

            def handler(sig, frame):
                try:
                    atexit._run_exitfuncs()
                except Exception:
                    pass
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)

            if threading.current_thread() is threading.main_thread():
                for s in (signal.SIGINT, signal.SIGTERM):
                    try:
                        signal.signal(s, handler)
                    except Exception:
                        pass
