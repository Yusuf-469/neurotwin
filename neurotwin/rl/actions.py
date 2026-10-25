"""The 7 allocation-level actions (RESEARCH.md v2 §6.3).

Each action maps the current observable state to at most one placement (request, unit) via the
engine's public queries; HOLD maps to None. The mask marks exactly the actions that map to a
placement, so a masked policy can never choose an impossible action.

"ED general" / "ED ICU" requests in the spec are the ADMIT_GENERAL / ADMIT_ICU request types,
which include elective admissions.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..policies.base import other_ward
from ..simulation.entities import RequestType, RequestView, Unit
from ..simulation.hospital import HospitalSimulation


class Action(IntEnum):
    """The 7 allocation-level actions of RESEARCH.md §6.3."""

    HOLD = 0
    ADMIT_PRIMARY_PRIORITY = 1
    ADMIT_PRIMARY_LONGEST_WAIT = 2
    ADMIT_OFF_SERVICE = 3
    ADMIT_ICU = 4
    STEP_DOWN = 5
    ESCALATE = 6


N_ACTIONS = len(Action)


def _by_priority(r: RequestView) -> tuple:
    return (r.priority, r.created_at, r.rid)


def _by_wait(r: RequestView) -> tuple:
    return (r.created_at, r.rid)


def action_to_placement(
    sim: HospitalSimulation, action: int, requests: list[RequestView] | None = None
) -> tuple[int, Unit] | None:
    """Placement selected by ``action``, or None if HOLD or infeasible."""
    action = Action(action)
    if action is Action.HOLD:
        return None
    requests = sim.waiting_requests() if requests is None else requests
    free = {u: sim.free_staffed(u) for u in Unit}

    if action in (Action.ADMIT_PRIMARY_PRIORITY, Action.ADMIT_PRIMARY_LONGEST_WAIT):
        cands = [
            r for r in requests if r.rtype is RequestType.ADMIT_GENERAL and free[r.primary_unit] > 0
        ]
        if not cands:
            return None
        key = _by_priority if action is Action.ADMIT_PRIMARY_PRIORITY else _by_wait
        r = min(cands, key=key)
        return r.rid, r.primary_unit
    if action is Action.ADMIT_OFF_SERVICE:
        cands = [
            r
            for r in requests
            if r.rtype is RequestType.ADMIT_GENERAL
            and free[r.primary_unit] == 0
            and free[other_ward(r.primary_unit)] > 0
        ]
        if not cands:
            return None
        r = min(cands, key=_by_priority)
        return r.rid, other_ward(r.primary_unit)
    if action is Action.ADMIT_ICU:
        cands = [r for r in requests if r.rtype is RequestType.ADMIT_ICU]
        if not cands or free[Unit.ICU] == 0:
            return None
        return min(cands, key=_by_priority).rid, Unit.ICU
    if action is Action.STEP_DOWN:
        cands = [r for r in requests if r.rtype is RequestType.STEP_DOWN]
        if not cands:
            return None
        r = min(cands, key=_by_wait)
        if free[r.primary_unit] > 0:
            return r.rid, r.primary_unit
        other = other_ward(r.primary_unit)
        return (r.rid, other) if free[other] > 0 else None
    if action is Action.ESCALATE:
        cands = [r for r in requests if r.rtype is RequestType.ESCALATION]
        if not cands or free[Unit.ICU] == 0:
            return None
        return min(cands, key=_by_wait).rid, Unit.ICU
    raise AssertionError(action)


def action_mask(sim: HospitalSimulation) -> np.ndarray:
    """Boolean mask over the 7 actions; HOLD is always valid."""
    requests = sim.waiting_requests()
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[Action.HOLD] = True
    for a in Action:
        if a is not Action.HOLD:
            mask[a] = action_to_placement(sim, a, requests) is not None
    return mask
