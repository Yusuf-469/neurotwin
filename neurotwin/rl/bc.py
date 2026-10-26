"""Behaviour cloning of a baseline into the PPO actor network (RESEARCH.md v2.8 §6.26).

Data: full protocol episodes of the teacher policy on TRAIN seeds (scenario A). At each of the
teacher's decisions we record the agent's observation, the action mask and the action that
reproduces the teacher's placement exactly. Decisions no valid action reproduces (about 3% for
Load-Balancing) are skipped rather than approximated.

Training: masked cross-entropy on the actor; the observation normaliser is fitted on the
collected observations and then frozen. The critic is left untrained (PPO fine-tuning warms it up
first, see PPOConfig.critic_warmup_updates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..experiments.seeds import require_split
from ..simulation.config import SCENARIOS, SimulationConfig
from ..simulation.hospital import HospitalSimulation, run_episode
from .actions import N_ACTIONS, Action, action_mask, action_to_placement
from .features import FeatureScales, features
from .ppo.agent import PPOAgent
from .ppo.masking import masked_distribution
from .ppo.networks import ActorCritic
from .ppo.normalize import RunningMeanStd


@dataclass
class Demonstrations:
    obs: np.ndarray  # (n, obs_dim) raw observations
    mask: np.ndarray  # (n, N_ACTIONS) bool
    action: np.ndarray  # (n,) int
    skipped: int  # teacher decisions no action reproduces


class _Recorder:
    """Wraps the teacher: records (obs, mask, matching action), then returns its decision."""

    name = "bc_recorder"

    def __init__(self, teacher, scales: FeatureScales) -> None:
        self.teacher, self.scales = teacher, scales
        self.obs, self.mask, self.action = [], [], []
        self.skipped = 0

    def decide(self, sim: HospitalSimulation):
        choice = self.teacher.decide(sim)
        mask = action_mask(sim)
        if choice is None:
            match = Action.HOLD if mask[Action.HOLD] else None
        else:
            match = next(
                (
                    a
                    for a in Action
                    if a != Action.HOLD and mask[a] and action_to_placement(sim, a) == choice
                ),
                None,
            )
        if match is None:
            self.skipped += 1
        else:
            self.obs.append(features(sim, self.scales))
            self.mask.append(mask)
            self.action.append(int(match))
        return choice

    def params(self) -> dict:
        return {}


def collect(teacher, config: SimulationConfig, seeds) -> Demonstrations:
    """Run the teacher on TRAIN seeds (scenario A) and record demonstrations."""
    seeds = list(seeds)
    require_split(seeds, "train")
    rec = _Recorder(teacher, FeatureScales.from_config(config))
    for s in seeds:
        run_episode(HospitalSimulation(config, SCENARIOS["A"], seed=s), rec)
    return Demonstrations(
        np.asarray(rec.obs, dtype=np.float32),
        np.asarray(rec.mask, dtype=bool),
        np.asarray(rec.action, dtype=np.int64),
        rec.skipped,
    )


def clone(
    demos: Demonstrations,
    seed: int = 0,
    hidden=(64, 64),
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 512,
    holdout: float = 0.1,
) -> tuple[PPOAgent, dict]:
    """Fit the actor to the demonstrations. Returns the agent and train/holdout accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(demos.action)
    perm = rng.permutation(n)
    n_hold = int(n * holdout)
    hold, train_idx = perm[:n_hold], perm[n_hold:]
    rms = RunningMeanStd((demos.obs.shape[1],))
    rms.update(demos.obs[train_idx])
    rms.frozen = True
    x = torch.as_tensor(rms.normalize(demos.obs))
    m = torch.as_tensor(demos.mask)
    y = torch.as_tensor(demos.action)
    model = ActorCritic(demos.obs.shape[1], N_ACTIONS, tuple(hidden))
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(train_idx))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            dist = masked_distribution(model.actor(x[idx]), m[idx])
            loss = -dist.log_prob(y[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def accuracy(idx) -> float:
        with torch.no_grad():
            dist = masked_distribution(model.actor(x[idx]), m[idx])
            return float((dist.probs.argmax(-1) == y[idx]).float().mean())

    stats = {
        "n_samples": n,
        "skipped": demos.skipped,
        "train_accuracy": accuracy(torch.as_tensor(train_idx)),
        "holdout_accuracy": accuracy(torch.as_tensor(hold)),
    }
    model.eval()
    return PPOAgent(model, rms, {"bc": stats, "bc_seed": seed}), stats
