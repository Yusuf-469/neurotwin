"""Future event list for the discrete-event engine.

Ordering is (time, kind, sequence number). Events at the same instant are processed in a fixed
order by kind (staffing first, then bed-freeing events, then new demand), and FIFO by
insertion within a kind. A fixed, documented tie-break is required for reproducibility and for
all policies to face identical event orders under CRN.

The sequence counter is a plain integer (not itertools.count) so the queue can be snapshotted
and restored exactly (digital twin, ARCHITECTURE.md §3.3).
"""

from __future__ import annotations

import heapq
from enum import IntEnum
from typing import Any, Callable


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
        self._next_seq = 0

    def push(self, time: float, kind: EventKind, payload: Any = None) -> None:
        heapq.heappush(self._heap, (time, int(kind), self._next_seq, payload))
        self._next_seq += 1

    def pop(self) -> tuple[float, EventKind, Any]:
        time, kind, _, payload = heapq.heappop(self._heap)
        return time, EventKind(kind), payload

    def peek_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------ snapshot support

    def entries(self) -> list[tuple[float, EventKind, int, Any]]:
        """All pending events in heap-array order (time, kind, seq, payload)."""
        return [(t, EventKind(k), s, p) for t, k, s, p in self._heap]

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def replace_all(self, entries: list[tuple[float, EventKind, int, Any]], next_seq: int) -> None:
        """Replace the contents (restore or rescheduling). Re-heapifies, so order is preserved
        by (time, kind, seq) regardless of the input order."""
        self._heap = [(t, int(k), s, p) for t, k, s, p in entries]
        heapq.heapify(self._heap)
        self._next_seq = next_seq

    def map_payloads(self, fn: Callable[[Any], Any]) -> None:
        self._heap = [(t, k, s, fn(p)) for t, k, s, p in self._heap]
