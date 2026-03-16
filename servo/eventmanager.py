"""
Fast shared-memory based replacement for multiprocessing.Manager().Event().

This provides:
    - SharedMemoryEvent: a lightweight, cross-process boolean flag
    - SharedMemoryEventManager: drop-in style manager with Event() and dict()
"""

from multiprocessing import shared_memory
import uuid


class SharedMemoryEvent:
    """
    A minimal, cross-process event implemented using shared_memory.
    API-compatible subset of multiprocessing.Event:
        - set()
        - clear()
        - is_set()
        - wait(timeout)  -> optional, simple polling
    """

    def __init__(self, name=None, create=True):
        # We store a single byte: 0 or 1
        if create:
            if name is None:
                # Generate a unique shared memory name
                name = f"evt_{uuid.uuid4().hex}"
            self.shm = shared_memory.SharedMemory(create=True, size=1, name=name)
            self.shm.buf[0] = 0
            self._owner = True
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self._owner = False

    @property
    def name(self):
        return self.shm.name

    def set(self):
        """Set event to True."""
        self.shm.buf[0] = 1

    def clear(self):
        """Set event to False."""
        self.shm.buf[0] = 0

    def is_set(self):
        """Return True if event is set."""
        return self.shm.buf[0] == 1

    def wait(self, timeout=None):
        """
        Very basic wait() function.
        Only provided for API completeness.
        Since Thomas does not use wait(), this is kept minimal.
        """
        import time
        start = time.time()
        if self.is_set():
            return True
        while True:
            if self.is_set():
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.0001)

    def close(self):
        self.shm.close()

    def unlink(self):
        if self._owner:
            self.shm.unlink()


class SharedMemoryEventManager:
    """
    Drop-in style replacement for multiprocessing.Manager(),
    but specialized for Event() objects implemented via shared_memory.

    Supports:
        .Event()  -> returns SharedMemoryEvent
        .dict()   -> simple Python dict (no proxy needed)
        .shutdown()
    """

    def __init__(self):
        self._events = []

    def Event(self):
        evt = SharedMemoryEvent()
        self._events.append(evt)
        return evt

    def dict(self):
        """
        multiprocessing.Manager().dict() returns a proxy dict,
        but here events are already shareable across processes.

        So we return a normal dict, no proxy required.
        """
        return {}

    def shutdown(self):
        """Cleanup all shared memory blocks."""
        for evt in self._events:
            evt.close()
            evt.unlink()
        self._events.clear()
