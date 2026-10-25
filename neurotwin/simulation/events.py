"""Future event list for the discrete-event engine.

Ordering is (time, kind, sequence number). Events at the same instant are processed in a fixed
order by kind (staffing first, then bed-freeing events, then new demand), and FIFO by
insertion within a kind. A fixed, documented tie-break is required for reproducibility and for
all policies to face identical event orders under CRN.
"""

from __future__ import annotations

import heapq
import itertools
from enum import IntEnum
from typing import Any


class EventKind(IntEnum):
    # Lower value = processed first among events at the same time.
    SHIFT_CHANGE = 0          # staffing changes before anything else at a shift boundary
    DISCHARGE = 1             # frees a bed
    ICU_PHASE_END = 2         # ICU stay over -> step-down request
    DETERIORATION = 3         # general-ward patient -> escalation request
    ED_TREATMENT_END = 4      # ED treatment over -> admission request
    EMERGENCY_ARRIVAL = 5
    ELECTIVE_ARRIVAL = 6
    PERIODIC_REVIEW = 7       # 4-hourly review (RESEARCH.md §6.2)
    TRACE = 8                 # observation only (occupancy trace); never a decision trigger


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Any]] = []
        self._counter = itertools.count()

    def push(self, time: float, kind: EventKind, payload: Any = None) -> None:
        heapq.heappush(self._heap, (time, int(kind), next(self._counter), payload))

    def pop(self) -> tuple[float, EventKind, Any]:
        time, kind, _, payload = heapq.heappop(self._heap)
        return time, EventKind(kind), payload

    def peek_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
