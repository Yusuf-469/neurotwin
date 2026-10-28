"""Live policy race: the what-if comparison advanced step by step (RESEARCH.md §6.21).

The present state is copied once, its measurement window set to [now, now + horizon], and one
branch per competitor is created with the *same* branch seed: every competitor faces identical
future arrivals, stays and staffing (common random numbers), so differences come from decisions
alone. ``step`` advances all branches to the same simulated time and reports each one's running
cost J (same weights as the study) and operational counters, so a client can animate the race.

One race is a single sample of the future; the study's conclusions rest on many paired runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..experiments.objective import CostWeights
from ..simulation.config import ScenarioConfig
from ..simulation.hospital import HospitalSimulation, run_until


@dataclass
class _Lane:
    name: str
    sim: HospitalSimulation
    policy: Any
    trace: list[tuple[float, float]] = field(default_factory=list)  # (hours since start, J so far)
    compute_ms: float = 0.0


class Race:
    """Competitors branched from one state with common random numbers, advanced together."""

    def __init__(
        self,
        base: HospitalSimulation,
        competitors: dict[str, Callable[[], Any]],
        horizon_hours: float,
        seed: int,
        scenario: ScenarioConfig | None = None,
    ) -> None:
        if len(competitors) < 2:
            raise ValueError("a race needs at least two competitors")
        if base.finished:
            raise ValueError("the state must be before the end of the simulation")
        self.start = base.now
        self.end = base.now + horizon_hours
        prepared = base.clone()
        prepared.set_measurement_window(self.start, self.end)
        self.weights = CostWeights.load()
        self.lanes = [
            _Lane(name, prepared.branch(seed, scenario=scenario), factory(), [(0.0, 0.0)])
            for name, factory in competitors.items()
        ]

    @property
    def now(self) -> float:
        return self.lanes[0].sim.now

    @property
    def finished(self) -> bool:
        return all(lane.sim.finished or lane.sim.now >= self.end for lane in self.lanes)

    def step(self, hours: float) -> dict[str, Any]:
        """Advance every competitor to the same simulated time and report standings."""
        target = min(self.now + hours, self.end)
        for lane in self.lanes:
            t0 = time.perf_counter()
            run_until(lane.sim, lane.policy, until=target)
            lane.compute_ms = 1000 * (time.perf_counter() - t0)
            lane.trace.append((lane.sim.now - self.start, self.weights.cost(lane.sim.measured)))
        return self.standings()

    def standings(self) -> dict[str, Any]:
        """Running results; ``rank`` 1 = lowest cost so far."""
        rows = []
        for lane in self.lanes:
            acc = lane.sim.measured
            elapsed_days = max(1e-9, (lane.sim.now - self.start) / 24.0)
            cost = self.weights.cost(acc)
            rows.append(
                {
                    "name": lane.name,
                    "cost": cost,
                    "cost_per_day": cost / elapsed_days if lane.sim.now > self.start else 0.0,
                    "turned_away": acc.diversions,
                    "wrong_ward": acc.off_service_placements,
                    "placements": acc.placements,
                    "waiting_now": len(lane.sim.waiting_requests()),
                    "compute_ms": round(lane.compute_ms, 1),
                    "trace": lane.trace,
                }
            )
        for rank, row in enumerate(sorted(rows, key=lambda r: r["cost"]), start=1):
            row["rank"] = rank
        return {
            "elapsed_hours": self.now - self.start,
            "horizon_hours": self.end - self.start,
            "finished": self.finished,
            "lanes": rows,
        }
