from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, Mapping
from enum import Enum, auto

import logging

log = logging.getLogger(__name__)

Guard = Callable[['StateMachine', Any], bool]
Action = Callable[['StateMachine', Any], None]

class FsmEvent(Enum):
    START = auto()
    STOP = auto()
    NORMALIZE = auto()
    ERROR = auto()
    RESET = auto()
    MOVE_TO_OPD = auto()
    OPEN_LOOP = auto()
    CLOSE_LOOP = auto()
    ROI_MODE = auto()
    FULL_FRAME_MODE = auto()

class FsmState(Enum):
    IDLE = auto()
    RUNNING = auto()
    FAILED = auto()
    STOPPED = auto()
    STAY_AT_OPD = auto()

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
            #raise TransitionError(f"Invalid transition {self.state.name} --{event.name}--> ?")
            log.warning(f"Invalid transition {self.state.name} --{event.name}--> ?")
            return False
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


