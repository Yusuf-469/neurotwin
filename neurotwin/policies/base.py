"""Common policy interface and the rule-based core (RESEARCH.md v2 §6.8).

Contract for every policy
-------------------------
* ``decide(sim)`` is called at an open decision epoch and returns one placement
  ``(request_id, unit)`` or ``None`` (HOLD, which ends the epoch).
* Policies may only use the engine's public query methods (``now``, ``waiting_requests()``,
  ``free_staffed``, ``staffed_capacity``, ``occupied``, ``config``). They never see hidden
  patient attributes, which ``RequestView`` does not contain.
* Policies are deterministic given the observable state, so CRN comparisons isolate the
  effect of the policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..simulation.entities import RequestView, Unit
from ..simulation.hospital import HospitalSimulation


def other_ward(unit: Unit) -> Unit:
    """The other general ward (MED <-> SURG)."""
    if unit is Unit.MED:
        return Unit.SURG
    if unit is Unit.SURG:
        return Unit.MED
    raise ValueError("ICU has no alternative ward")


def staffed_occupancy(sim: HospitalSimulation, unit: Unit) -> float:
    """Occupied / staffed capacity; an unstaffed unit counts as full."""
    staffed = sim.staffed_capacity(unit)
    return sim.occupied(unit) / staffed if staffed > 0 else 1.0


class AllocationPolicy(ABC):
    """Base class: decide(sim) returns (request id, unit) or None for HOLD."""

    name: str = "policy"

    @abstractmethod
    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None: ...

    def params(self) -> dict[str, Any]:
        return {}


class RuleBasedPolicy(AllocationPolicy):
    """Serve requests in a fixed order; place the first request that has an acceptable unit.

    A request that cannot be placed does not block requests for *other* units (no
    head-of-line blocking across bed types). Subclasses define the order and the unit choice.
    """

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        now = sim.now
        for request in sorted(sim.waiting_requests(), key=lambda r: self.order_key(r, now)):
            unit = self.target_unit(request, sim)
            if unit is not None:
                return request.rid, unit
        return None

    @abstractmethod
    def order_key(self, request: RequestView, now: float) -> tuple: ...

    @abstractmethod
    def target_unit(self, request: RequestView, sim: HospitalSimulation) -> Unit | None: ...


def fifo_key(request: RequestView, now: float) -> tuple:
    """Sort key: oldest request first."""
    return (request.created_at, request.rid)


def priority_key(request: RequestView, now: float) -> tuple:
    """Sort key: most urgent first, then oldest."""
    return (request.priority, request.created_at, request.rid)


def primary_then_off_service_after_wait(
    request: RequestView, sim: HospitalSimulation, off_service_wait_hours: float
) -> Unit | None:
    """Common hospital practice: primary unit if a staffed bed is free; for general requests,
    the other ward once the request has waited at least ``off_service_wait_hours``."""
    if sim.free_staffed(request.primary_unit) > 0:
        return request.primary_unit
    if request.rtype.needs_icu:
        return None
    other = other_ward(request.primary_unit)
    if sim.now - request.created_at >= off_service_wait_hours and sim.free_staffed(other) > 0:
        return other
    return None
