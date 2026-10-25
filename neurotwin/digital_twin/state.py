"""Observable hospital state (ARCHITECTURE.md §6.1; RESEARCH.md v2 §6.2, §6.5).

Contains only information available at decision time. Hidden patient attributes (realised
LOS, treatment times, deterioration times) are never included; this is enforced by building
the state only from the engine's public query methods and RequestView objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..simulation.entities import RequestView, Unit
from ..simulation.hospital import HospitalSimulation


@dataclass(frozen=True)
class HospitalState:
    """Policy-visible hospital state at one moment (no hidden patient data)."""

    time: float
    hour_of_week: float
    physical_beds: dict[str, int]
    staffed_capacity: dict[str, int]
    occupied: dict[str, int]
    nurses_on_shift: dict[str, int]
    nominal_nurses: dict[str, int]
    doctors_on_shift: int
    nominal_doctors: int
    ed_waiting_for_doctor: int
    ed_in_treatment: int
    ed_count: int
    waiting_requests: tuple[RequestView, ...]
    off_service_in_beds: int
    recent_arrivals: tuple[float, ...]
    recent_discharges: tuple[float, ...]
    recent_diversions: tuple[float, ...]
    cumulative_placements: int
    cumulative_diversions: int
    cumulative_off_service: int

    def free_staffed(self, unit: str) -> int:
        return max(0, self.staffed_capacity[unit] - self.occupied[unit])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (for history export, the API and the dashboard)."""
        d = {k: v for k, v in vars(self).items() if k != "waiting_requests"}
        d["waiting_requests"] = [
            {
                "rid": r.rid,
                "type": r.rtype.value,
                "priority": r.priority,
                "created_at": r.created_at,
                "primary_unit": r.primary_unit.value,
                "eligible_units": [u.value for u in r.eligible_units],
                "from_ed": r.from_ed,
            }
            for r in self.waiting_requests
        ]
        for key in ("recent_arrivals", "recent_discharges", "recent_diversions"):
            d[key] = list(d[key])
        return d


def observe(sim: HospitalSimulation) -> HospitalState:
    """Build the observable state from public engine queries only."""
    nominal_nurses, nominal_doctors = sim.nominal_staff
    nurses = sim.nurses_on_shift
    return HospitalState(
        time=sim.now,
        hour_of_week=sim.now % 168.0,
        physical_beds={u.value: sim.physical_beds[u] for u in Unit},
        staffed_capacity={u.value: sim.staffed_capacity(u) for u in Unit},
        occupied={u.value: sim.occupied(u) for u in Unit},
        nurses_on_shift={u.value: nurses[u] for u in Unit},
        nominal_nurses={u.value: nominal_nurses[u] for u in Unit},
        doctors_on_shift=sim.doctors_on_shift,
        nominal_doctors=nominal_doctors,
        ed_waiting_for_doctor=sim.doctor_queue_length,
        ed_in_treatment=sim.doctors_busy,
        ed_count=sim.ed_count,
        waiting_requests=tuple(sim.waiting_requests()),
        off_service_in_beds=sim.off_service_in_beds,
        recent_arrivals=sim.recent_event_times("arrivals"),
        recent_discharges=sim.recent_event_times("discharges"),
        recent_diversions=sim.recent_event_times("diversions"),
        cumulative_placements=sim.acc.placements,
        cumulative_diversions=sim.acc.diversions,
        cumulative_off_service=sim.acc.off_service_placements,
    )
