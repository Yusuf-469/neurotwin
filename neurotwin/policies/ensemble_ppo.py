"""Soft-vote ensemble of PPO agents (RESEARCH.md v2.8 §6.26).

Each agent scores the valid actions; the ensemble takes the action with the highest mean
probability. Averaging independently trained agents reduces the influence of any single agent's
idiosyncrasies (for example agent 0's collapse mode, Stage 6 §5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..rl.actions import Action, action_mask, action_to_placement
from ..rl.features import FeatureScales, features
from ..rl.ppo.agent import PPOAgent
from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation
from .base import AllocationPolicy


class EnsemblePPOPolicy(AllocationPolicy):
    """Mean of the agents' masked action probabilities; argmax decides."""

    name = "ppo_ensemble"

    def __init__(self, agents: list[PPOAgent]) -> None:
        if not agents:
            raise ValueError("ensemble needs at least one agent")
        self.agents = agents
        self._scales: FeatureScales | None = None

    @classmethod
    def load(cls, paths: list[str | Path]) -> "EnsemblePPOPolicy":
        return cls([PPOAgent.load(p) for p in paths])

    def probabilities(self, sim: HospitalSimulation) -> tuple[np.ndarray, np.ndarray]:
        if self._scales is None:
            self._scales = FeatureScales.from_config(sim.config)
        mask = action_mask(sim)
        obs = features(sim, self._scales)
        self._last_obs = obs
        probs = np.mean([a.action_probabilities(obs, mask) for a in self.agents], axis=0)
        return probs, mask

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        probs, mask = self.probabilities(sim)
        action = int(np.argmax(np.where(mask, probs, -1.0)))
        # Exact decision-time inputs, kept for the dashboard's "inside the agent" view.
        self.last_obs, self.last_mask = self._last_obs, mask
        self.last_details = {
            "chosen_action": Action(action).name,
            "action_probabilities": {
                Action(a).name: round(float(p), 4) for a, p in enumerate(probs) if mask[a]
            },
        }
        if action == Action.HOLD:
            return None
        return action_to_placement(sim, action)

    def params(self) -> dict[str, Any]:
        return {"n_agents": len(self.agents)}
