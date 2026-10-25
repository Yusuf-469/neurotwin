"""PPO as an AllocationPolicy (RESEARCH.md v2 §6.8), for evaluation under the same protocol as
the baselines (run_episode, CRN seeds, decision logging, what-if branches).

Decisions use the deterministic (argmax) action of the masked policy, with the observation
normaliser frozen, so the evaluated policy is a fixed function of the observable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..rl.actions import Action, action_mask, action_to_placement
from ..rl.features import FeatureScales, features
from ..rl.ppo.agent import PPOAgent
from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation
from .base import AllocationPolicy


class PPOPolicy(AllocationPolicy):
    name = "ppo"

    def __init__(self, agent: PPOAgent, deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = deterministic
        self._scales: FeatureScales | None = None

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOPolicy":
        return cls(PPOAgent.load(path), deterministic)

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        if self._scales is None:
            # Scales come from the configuration (scenario-independent), see rl/features.py.
            self._scales = FeatureScales.from_config(sim.config)
        mask = action_mask(sim)
        action = self.agent.act(features(sim, self._scales), mask, self.deterministic)
        if action == Action.HOLD:
            return None
        return action_to_placement(sim, action)

    def params(self) -> dict[str, Any]:
        return {"deterministic": self.deterministic, **self.agent.meta}
