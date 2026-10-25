"""Heuristic baselines (RESEARCH.md v2 §6.8)."""

from __future__ import annotations

from typing import Any, Mapping

from ..simulation.entities import RequestView, Unit
from ..simulation.hospital import HospitalSimulation
from .base import (RuleBasedPolicy, fifo_key, other_ward, primary_then_off_service_after_wait,
                   priority_key, staffed_occupancy)


class FCFSAllocator(RuleBasedPolicy):
    """Oldest request first; off-service only after waiting ``off_service_wait_hours``."""

    name = "fcfs"

    def __init__(self, off_service_wait_hours: float = 4.0) -> None:
        if off_service_wait_hours < 0:
            raise ValueError("off_service_wait_hours must be non-negative")
        self.off_service_wait_hours = off_service_wait_hours

    def order_key(self, request: RequestView, now: float) -> tuple:
        return fifo_key(request, now)

    def target_unit(self, request: RequestView, sim: HospitalSimulation) -> Unit | None:
        return primary_then_off_service_after_wait(request, sim, self.off_service_wait_hours)

    def params(self) -> dict[str, Any]:
        return {"off_service_wait_hours": self.off_service_wait_hours}


class PriorityAllocator(FCFSAllocator):
    """Most urgent first (priority 1), ties by waiting time; same off-service rule as FCFS."""

    name = "priority"

    def order_key(self, request: RequestView, now: float) -> tuple:
        return priority_key(request, now)


class LoadBalancingAllocator(RuleBasedPolicy):
    """Priority order; primary ward unless its staffed occupancy exceeds the other ward's by
    more than ``imbalance_threshold``, in which case the patient is placed off-service."""

    name = "load_balancing"

    def __init__(self, imbalance_threshold: float = 0.2) -> None:
        if not 0 <= imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be in [0, 1]")
        self.imbalance_threshold = imbalance_threshold

    def order_key(self, request: RequestView, now: float) -> tuple:
        return priority_key(request, now)

    def target_unit(self, request: RequestView, sim: HospitalSimulation) -> Unit | None:
        primary = request.primary_unit
        if request.rtype.needs_icu:
            return primary if sim.free_staffed(primary) > 0 else None
        other = other_ward(primary)
        imbalance = staffed_occupancy(sim, primary) - staffed_occupancy(sim, other)
        if imbalance > self.imbalance_threshold and sim.free_staffed(other) > 0:
            return other
        return primary if sim.free_staffed(primary) > 0 else None

    def params(self) -> dict[str, Any]:
        return {"imbalance_threshold": self.imbalance_threshold}


class TrunkReservationAllocator(RuleBasedPolicy):
    """Priority order with bed reservation (Miller, 1969).

    A low-priority request (priority >= 3, which includes step-downs) is placed in unit k only
    if more than ``reserve[k]`` staffed beds are free there; P1-2 may take any free bed. The
    reserved beds are kept for urgent arrivals. Off-service placement follows the same waiting
    rule as Priority, and the reservation also applies to the off-service ward.
    """

    name = "trunk_reservation"
    LOW_PRIORITY_FROM = 3

    def __init__(self, reserve: Mapping[str, int] | None = None,
                 off_service_wait_hours: float = 4.0) -> None:
        reserve = dict(reserve or {"MED": 1, "SURG": 1, "ICU": 1})
        if set(reserve) != {u.value for u in Unit} or any(v < 0 for v in reserve.values()):
            raise ValueError("reserve needs non-negative levels for MED, SURG and ICU")
        self.reserve = {Unit(k): int(v) for k, v in reserve.items()}
        self.off_service_wait_hours = off_service_wait_hours

    def order_key(self, request: RequestView, now: float) -> tuple:
        return priority_key(request, now)

    def _admissible(self, request: RequestView, sim: HospitalSimulation, unit: Unit) -> bool:
        threshold = self.reserve[unit] if request.priority >= self.LOW_PRIORITY_FROM else 0
        return sim.free_staffed(unit) > threshold

    def target_unit(self, request: RequestView, sim: HospitalSimulation) -> Unit | None:
        primary = request.primary_unit
        if self._admissible(request, sim, primary):
            return primary
        if request.rtype.needs_icu:
            return None
        other = other_ward(primary)
        if (sim.now - request.created_at >= self.off_service_wait_hours
                and self._admissible(request, sim, other)):
            return other
        return None

    def params(self) -> dict[str, Any]:
        return {"reserve": {u.value: r for u, r in self.reserve.items()},
                "off_service_wait_hours": self.off_service_wait_hours}
