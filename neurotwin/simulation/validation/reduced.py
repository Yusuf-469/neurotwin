"""Reduced configurations of the real engine for analytical validation (RESEARCH.md v2 §6.17).

The validation runs the *production* engine; it only changes parameters, so passing the checks
is evidence about the code that produces the study results, not about a separate toy model.

Reduction to M/M/c/K:
* homogeneous Poisson emergency arrivals (flat profiles), no electives
* one unit (MED) with c beds; SURG and ICU have no beds; nobody needs the ICU
* staffing never binds (one nurse per bed, ratio 1, no absences)
* no ED treatment stage (an arrival immediately requests a bed)
* exponential LOS, no deterioration, no off-service LOS effect
* identical priority for every patient (so a non-idling policy is FCFS)
* diversion when the ED (= boarding queue) holds Q_max patients, so K = c + Q_max
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SimulationConfig, load_default_config
from ..entities import Unit
from ..hospital import HospitalSimulation


def mmck_config(
    *,
    servers: int,
    arrival_rate: float,
    mean_service: float,
    ed_capacity: int | None,
    horizon: float,
    warmup: float,
) -> SimulationConfig:
    """Engine configuration equivalent to M/M/c/K with K = c + ed_capacity (or M/M/c if None)."""
    return load_default_config().with_overrides(
        horizon_hours=horizon,
        warmup_hours=warmup,
        emergency_base_rate=arrival_rate,
        elective_base_rate=0.0,
        hourly_profile=[1.0] * 24,
        weekly_profile=[1.0] * 7,
        priority_probs_emergency=[0.0, 0.0, 1.0, 0.0, 0.0],
        icu_first_prob=[0.0] * 5,
        specialty_prob_med_emergency=1.0,
        beds={"MED": servers, "SURG": 0, "ICU": 0},
        nurses={"MED": servers, "SURG": 0, "ICU": 0},
        patients_per_nurse={"MED": 1, "SURG": 1, "ICU": 1},
        staff_absence_prob=0.0,
        ed_capacity=ed_capacity,
        ed_treatment=None,
        general_los={"dist": "exponential", "mean": mean_service},
        deterioration_hazard_per_day=0.0,
        off_service_los_multiplier=1.0,
    )


class GreedyPlacement:
    """Non-idling placement: highest priority, then oldest; primary unit, else any free unit.

    Used for engine validation only. It is not one of the study baselines (RESEARCH.md §6.8).
    """

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        best = None
        for req in sim.waiting_requests():
            units = [req.primary_unit] + [
                u for u in req.eligible_units if u is not req.primary_unit
            ]
            unit = next((u for u in units if sim.free_staffed(u) > 0), None)
            if unit is None:
                continue
            key = (req.priority, req.created_at, req.rid)
            if best is None or key < best[0]:
                best = (key, req.rid, unit)
        return None if best is None else (best[1], best[2])


@dataclass(frozen=True)
class QueueEstimates:
    """Per-replication estimates over the measurement window."""

    mean_wait: float
    wait_probability: float
    blocking: float


def estimate_replication(config: SimulationConfig, seed: int) -> QueueEstimates:
    """Run one replication with the greedy policy and return queue estimates.

    Waits are those of requests created in the measurement window and placed before the
    horizon; censoring at the horizon is negligible for horizons much longer than the mean wait.
    """
    from ..hospital import run_episode

    result = run_episode(HospitalSimulation(config, seed=seed), GreedyPlacement())
    waits = result.boarding_times
    return QueueEstimates(
        mean_wait=result.mean_boarding_hours,
        wait_probability=sum(w > 0 for w in waits) / len(waits),
        blocking=result.diversions / result.emergency_arrivals,
    )
