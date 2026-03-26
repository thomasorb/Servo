from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, Mapping
from enum import IntEnum, auto

import logging

from . import config

log = logging.getLogger(__name__)

Guard = Callable[['StateMachine', Any], bool]
Action = Callable[['StateMachine', Any], None]

class NexlineState(IntEnum):
    IDLE = auto()
    RUNNING = auto()
    MOVING = auto()
    STOPPED = auto()

class ServoState(IntEnum):
    IDLE = auto()
    RUNNING = auto()
    STOPPED = auto()
    TRACKING = auto()

class WorkerState(IntEnum):
    IDLE = auto()
    RUNNING = auto()
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
    def __init__(self, initial, table):
        self.state = initial
        self.table = table
    
        log.info(f"FSM init -> {self.state.name}")
        self._call_hook(f"on_enter_{self.state.name.lower()}", None)

    def _call_hook(self, name: str, payload: Any):
        fn = getattr(self, name, None)
        if callable(fn):
            fn(payload)

    def _publish_state(self, state=None):
        if state is None:
            state = self.state
        log.info(f'{self.__class__.__name__} entering state {state.name}')

    def dispatch(self, event, payload: Any = None) -> bool:
        if self.table is None:
            log.error("FSM transition table is not defined. self.table is None.")
            return False
            
        tr = self.table.get((self.state, event))
        if tr is None:
            #raise TransitionError(f"Invalid transition {self.state.name} --{event.name}--> ?")
            log.warning(f"Invalid transition {self.state.name} --{event.name}--> ?")
            return False
        if tr.guard and not tr.guard(self, payload):
            return False
        self._publish_state(tr.target)
        self._call_hook(f"on_exit_{self.state.name.lower()}", payload)
        if tr.action:
            tr.action(payload)
        old = self.state
        self.state = tr.target
        log.info(f"{old.name} --{event.name}--> {self.state.name}")
        self._publish_state()
        self._call_hook(f"on_enter_{self.state.name.lower()}", payload)
        return True


