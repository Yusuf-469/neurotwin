"""PPO as an AllocationPolicy (RESEARCH.md v2 §6.8), for evaluation under the same protocol as
the baselines (run_episode, CRN seeds, decision logging, what-if branches).

Decisions use the deterministic (argmax) action of the masked policy, with the observation
normaliser frozen, so the evaluated policy is a fixed function of the observable state.

Optional safety shield (RESEARCH.md v2.7 §6.25): while the ED is at capacity, so arrivals are being
diverted, HOLD is removed from the mask whenever a placement is feasible, and the agent takes its
own best remaining action. Holding a bed then cannot protect a future arrival (future arrivals
are turned away), and repeated holding in that state is the observed collapse mode.
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
    """Trained PPO agent wrapped as an AllocationPolicy (deterministic by default)."""

    name = "ppo"

    def __init__(self, agent: PPOAgent, deterministic: bool = True, shield: bool = False) -> None:
        self.agent = agent
        self.deterministic = deterministic
        self.shield = shield
        self.shield_activations = 0
        if shield:
            self.name = "ppo_shielded"
        self._scales: FeatureScales | None = None

    @classmethod
    def load(
        cls, path: str | Path, deterministic: bool = True, shield: bool = False
    ) -> "PPOPolicy":
        return cls(PPOAgent.load(path), deterministic, shield)

    @staticmethod
    def shield_mask(sim: HospitalSimulation, mask):
        """Mask with HOLD removed if the ED is full and a placement is feasible (else unchanged)."""
        cap = sim.config.ed_capacity
        if cap is None or sim.ed_count < cap or not mask[1:].any():
            return mask, False
        shielded = mask.copy()
        shielded[Action.HOLD] = False
        return shielded, True

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        if self._scales is None:
            # Scales come from the configuration (scenario-independent), see rl/features.py.
            self._scales = FeatureScales.from_config(sim.config)
        mask = action_mask(sim)
        applied = False
        if self.shield:
            mask, applied = self.shield_mask(sim, mask)
            self.shield_activations += applied
        obs = features(sim, self._scales)
        dropped = self.agent.meta.get("drop_features", [])
        if dropped:
            obs[dropped] = 0.0  # same feature ablation the agent was trained with
        action = self.agent.act(obs, mask, self.deterministic)
        # Decision-time details for the decision log / explanation layer (read-only).
        probs = self.agent.action_probabilities(obs, mask)
        self.last_details = {
            "chosen_action": Action(action).name,
            "shield_applied": applied,
            "action_probabilities": {
                Action(a).name: round(float(p), 4) for a, p in enumerate(probs) if mask[a]
            },
        }
        if action == Action.HOLD:
            return None
        return action_to_placement(sim, action)

    def params(self) -> dict[str, Any]:
        return {"deterministic": self.deterministic, "shield": self.shield, **self.agent.meta}
