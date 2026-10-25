"""Gymnasium environment for hospital bed allocation (RESEARCH.md v2 §6.2-§6.10, §6.18).

Semantics
---------
* One step = one atomic decision at an open epoch. A placement keeps the epoch open (Δt = 0)
  while further placements are feasible; HOLD, or exhausting the feasible placements, advances
  the simulation to the next epoch. The step's duration ``info["dt_hours"]`` makes the process
  semi-Markov; the PPO discounts with gamma ** dt.
* Reward = -(alpha * ΔW + beta * ΔD + kappa * ΔO) * reward_scale, where ΔW is the exact
  priority-weighted request-hours accrued during the step, ΔD diversions and ΔO off-service
  placements. Summed over an episode this equals -J over the same period (tested), so the
  return is aligned with the evaluation metric.
* Episodes are *truncated* at the horizon (never terminated): the PPO bootstraps from V(s_T).
* ``info["action_mask"]`` and ``action_masks()`` give the valid actions for the *next* step.
  Choosing a masked action raises an error, because a masked policy can never do that and a
  silent fallback would hide bugs.

Start modes
-----------
* ``"full"``: empty hospital, full 49-day protocol (evaluation; the agent also acts during
  the unmeasured warm-up).
* ``"warm"``: branch a random state from a WarmStartPool with the episode seed (training only;
  RESEARCH.md §6.11). Seeds must come from the TRAIN range in this mode (asserted).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..digital_twin.warm_start import WarmStartPool
from ..experiments.objective import CostWeights
from ..simulation.config import SCENARIOS, ScenarioConfig, SimulationConfig, load_default_config
from ..simulation.hospital import HospitalSimulation
from .actions import N_ACTIONS, action_mask, action_to_placement
from .features import N_FEATURES, FeatureScales, features


class HospitalAllocationEnv(gym.Env):
    """Gymnasium environment: atomic, masked, semi-Markov bed-allocation decisions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        scenario: ScenarioConfig | None = None,
        start_mode: str = "full",
        pool: WarmStartPool | None = None,
        weights: CostWeights | None = None,
        reward_scale: float = 0.1,
        scenario_sampler=None,
        drop_features: tuple[int, ...] = (),
    ) -> None:
        """``scenario_sampler(rng) -> ScenarioConfig`` optionally draws a scenario per episode
        (PPO-DR, RESEARCH.md §6.23); it must only produce training-range scenarios.

        ``drop_features`` (0-based indices) are set to 0 in every observation (ablations,
        RESEARCH.md §6.16). A constant input carries no information, so this is equivalent to
        removing the feature while keeping the network shape fixed."""
        super().__init__()
        if start_mode not in ("full", "warm"):
            raise ValueError("start_mode must be 'full' or 'warm'")
        if start_mode == "warm" and pool is None:
            raise ValueError("warm start needs a WarmStartPool")
        if start_mode == "warm" and scenario_sampler is not None:
            # Pool states were generated under scenario A; re-scenario-ing a branched state is
            # not implemented yet (needed only for the PPO-DR extension).
            raise NotImplementedError("scenario sampling with warm starts is not supported yet")
        self.config = config or load_default_config()
        self.scenario = scenario or SCENARIOS["A"]
        self.start_mode = start_mode
        self.pool = pool
        self.weights = weights or CostWeights.load()
        self.reward_scale = reward_scale
        self.scenario_sampler = scenario_sampler
        self.drop_features = tuple(drop_features)
        if any(not 0 <= i < N_FEATURES for i in self.drop_features):
            raise ValueError("drop_features out of range")
        self.scales = FeatureScales.from_config(self.config)
        # Count and wait features have no natural upper bound (e.g. mean wait / 4 h under
        # stress), so the space is [-1, inf); PPO normalises observations with running stats.
        self.observation_space = gym.spaces.Box(-1.0, np.inf, shape=(N_FEATURES,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.sim: HospitalSimulation | None = None
        self._mask = np.ones(N_ACTIONS, dtype=bool)
        self._pool_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 100_000))  # TRAIN range by default
        scenario = self.scenario
        if self.scenario_sampler is not None:
            scenario = self.scenario_sampler(np.random.default_rng(seed))
        if self.start_mode == "warm":
            self._pool_rng = np.random.default_rng(seed)
            self.sim = self.pool.start_episode(seed, self._pool_rng)
        else:
            self.sim = HospitalSimulation(self.config, scenario, seed=seed)
        self._advance_to_decision()
        return self._obs(), self._info(dt=0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.sim is None:
            raise RuntimeError("call reset() first")
        if not self._mask[int(action)]:
            raise ValueError(f"action {action} is masked (invalid) at this state")
        sim = self.sim
        before, t0 = sim.acc.copy(), sim.now
        placement = action_to_placement(sim, int(action))
        if placement is not None:
            sim.place(*placement)
            if not sim.has_feasible_placement():
                self._advance_to_decision()
        else:
            self._advance_to_decision()
        after = sim.acc
        cost = (
            self.weights.alpha * (after.weighted_request_hours - before.weighted_request_hours)
            + self.weights.beta * (after.diversions - before.diversions)
            + self.weights.kappa * (after.off_service_placements - before.off_service_placements)
        )
        truncated = sim.finished
        return self._obs(), -cost * self.reward_scale, False, truncated, self._info(sim.now - t0)

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO interface (used only for reference validation)."""
        return self._mask.copy()

    # ------------------------------------------------------------------ helpers

    def _advance_to_decision(self) -> None:
        sim = self.sim
        if not sim.next_epoch():
            self._mask = np.zeros(N_ACTIONS, dtype=bool)
            self._mask[0] = True  # finished: nothing to decide
            return
        self._mask = action_mask(sim)

    def _obs(self) -> np.ndarray:
        obs = features(self.sim, self.scales)
        if self.drop_features:
            obs[list(self.drop_features)] = 0.0
        return obs

    def _info(self, dt: float) -> dict[str, Any]:
        return {"action_mask": self._mask.copy(), "dt_hours": dt, "time": self.sim.now}
