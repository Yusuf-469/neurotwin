"""λ-conditioned PPO with online dual ascent (RESEARCH.md v2.6 §6.19).

At deployment the diversion multiplier λ is not fixed: every UPDATE_HOURS of simulated time
the controller compares the diversion rate over the last 24 h with the budget and moves λ by
projected dual ascent. The policy, trained across λ ∈ [0, LAMBDA_MAX], receives λ / LAMBDA_MAX
as its 22nd input, so a higher λ shifts it toward avoiding diversions while it runs.

The controller uses only observable information (recent arrival and diversion times), like
every other policy. ``lambda_trace`` records (time, λ) for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..rl.actions import Action, action_mask, action_to_placement
from ..rl.constrained import LAMBDA0, LAMBDA_MAX, UPDATE_HOURS, dual_ascent_step
from ..rl.features import FeatureScales, features
from ..rl.ppo.agent import PPOAgent
from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation
from .base import AllocationPolicy


class ConstrainedPPOPolicy(AllocationPolicy):
    """λ-conditioned PPO agent plus an online dual-ascent controller for λ."""

    name = "ppo_lagrangian"

    def __init__(self, agent: PPOAgent, lam0: float = LAMBDA0) -> None:
        self.agent = agent
        self.lam = lam0
        self._next_update: float | None = None
        self._scales: FeatureScales | None = None
        self.lambda_trace: list[tuple[float, float]] = []

    @classmethod
    def load(cls, path: str | Path) -> "ConstrainedPPOPolicy":
        """Load a λ-conditioned agent checkpoint."""
        return cls(PPOAgent.load(path))

    def _update_lambda(self, sim: HospitalSimulation) -> None:
        if self._next_update is None:
            self._next_update = sim.now + UPDATE_HOURS
            self.lambda_trace.append((sim.now, self.lam))
        while sim.now >= self._next_update:
            arrivals = len(sim.recent_event_times("arrivals"))
            diversions = len(sim.recent_event_times("diversions"))
            rate = 100.0 * diversions / arrivals if arrivals else 0.0
            self.lam = dual_ascent_step(self.lam, rate)
            self._next_update += UPDATE_HOURS
            self.lambda_trace.append((sim.now, self.lam))

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        """Update λ if due, then act with the λ-conditioned policy (deterministic)."""
        if self._scales is None:
            self._scales = FeatureScales.from_config(sim.config)
        self._update_lambda(sim)
        obs = np.append(features(sim, self._scales), np.float32(self.lam / LAMBDA_MAX))
        action = self.agent.act(obs, action_mask(sim), deterministic=True)
        if action == Action.HOLD:
            return None
        return action_to_placement(sim, action)

    def params(self) -> dict[str, Any]:
        """Current λ and the agent's metadata."""
        return {"lambda": self.lam, **self.agent.meta}
