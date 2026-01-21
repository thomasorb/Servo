from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, Mapping
from enum import Enum, auto
import logging

from . import config

log = logging.getLogger(__name__)

Guard = Callable[['StateMachine', Any], bool]
Action = Callable[['StateMachine', Any], None]

class FsmEvent(Enum):
    START = auto()
    STOP = auto()
    NORMALIZE = auto()
    ERROR = auto()
    RESET = auto()

class FsmState(Enum):
    IDLE = auto()
    RUNNING = auto()
    FAILED = auto()
    STOPPED = auto()

@dataclass(frozen=True)
class Transition:
    target: FsmState
    guard: Optional[Guard] = None
    action: Optional[Action] = None

class TransitionError(Exception):
    pass

class StateMachine:
    """FSM synchrone à transitions déclaratives."""
    def __init__(self, initial: FsmState, table: Dict[Tuple[FsmState, FsmEvent], Transition]):
        self.state = initial
        self.table = table
        log.info(f"FSM init -> {self.state.name}")
        self._call_hook(f"on_enter_{self.state.name.lower()}", None)

    def _call_hook(self, name: str, payload: Any):
        fn = getattr(self, name, None)
        if callable(fn):
            fn(payload)

    def dispatch(self, event: FsmEvent, payload: Any = None) -> bool:
        tr = self.table.get((self.state, event))
        if tr is None:
            raise TransitionError(f"Invalid transition {self.state.name} --{event.name}--> ?")
        if tr.guard and not tr.guard(self, payload):
            return False
        self._call_hook(f"on_exit_{self.state.name.lower()}", payload)
        if tr.action:
            tr.action(payload)
        old = self.state
        self.state = tr.target
        log.info(f"{old.name} --{event.name}--> {self.state.name}")
        self._call_hook(f"on_enter_{self.state.name.lower()}", payload)
        return True


class ServoFSM(StateMachine):
    def __init__(self):
        table = {
            (FsmState.IDLE,    FsmEvent.START): Transition(FsmState.RUNNING, action=self._start),
            (FsmState.RUNNING, FsmEvent.NORMALIZE): Transition(FsmState.RUNNING, action=self._normalize),
            (FsmState.RUNNING, FsmEvent.ERROR): Transition(FsmState.FAILED),
            (FsmState.RUNNING, FsmEvent.STOP):  Transition(FsmState.STOPPED, action=self._stop),
            (FsmState.FAILED,  FsmEvent.RESET): Transition(FsmState.IDLE),
            (FsmState.STOPPED, FsmEvent.RESET): Transition(FsmState.IDLE),
        }
        super().__init__(FsmState.IDLE, table)

    # Hooks (facultatif)
    def on_enter_running(self, _): log.debug(">> RUNNING")
    def on_exit_running(self, _):  log.debug("<< RUNNING")

    # Actions
    def _start(self, _):
        log.info("Starting Servo")
    def _normalize(self, _):
        log.info("Normalizing")
    def _stop(self, _):
        log.info("Stopping Servo")


class MPEventBridge:
    """
    Non-blocking poll of multiprocessing events
    events : dict-like with keys -> Event()
      - 'start_evt'
      - 'stop_evt'
      - 'error_evt'
      - 'reset_evt'  (optional)
    """
    def __init__(self, fsm, events: Mapping):
        self.fsm = fsm
        self.events = events
        self.poll()

    def poll(self):
        evs = self.events

        for iname in config.SERVO_EVENTS:
            if evs.get(iname) and evs[iname].is_set():
                evs[iname].clear()
                self.fsm.dispatch(getattr(FsmEvent, iname.upper()), payload=None)
