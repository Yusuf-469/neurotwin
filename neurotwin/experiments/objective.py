"""Episode cost shared by RL training and baseline tuning (RESEARCH.md v2 §6.10).

J = alpha * (priority-weighted request-hours) + beta * diversions + kappa * off-service
placements, over the measurement window. Summed over an episode, the RL reward equals -J over
the same window. Tuning every baseline on J (rather than on, say, mean boarding alone) gives
each policy the same objective as the RL agent, which keeps the comparison fair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..simulation.hospital import Accumulators, HospitalSimulation

REWARD_CONFIG = Path(__file__).resolve().parents[2] / "config" / "reward_default.json"


@dataclass(frozen=True)
class CostWeights:
    alpha: float
    beta: float
    kappa: float

    @classmethod
    def load(cls, path: str | Path = REWARD_CONFIG) -> "CostWeights":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})

    def cost(self, acc: Accumulators) -> float:
        return (self.alpha * acc.weighted_request_hours + self.beta * acc.diversions
                + self.kappa * acc.off_service_placements)


def measured_cost(sim: HospitalSimulation, weights: CostWeights) -> float:
    """J over the measurement window of a finished episode, per measured day."""
    days = (sim.config.horizon_hours - sim.config.warmup_hours) / 24.0
    return weights.cost(sim.measured) / days
