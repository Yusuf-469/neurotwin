"""Warm-start pool: saved steady-state hospitals for RL training (stage2 report §3).

Why: from an empty start the hospital needs a 21-day warm-up (Welch, RESEARCH.md §6.11), so
an RL agent trained on full episodes would spend 60% of its experience in unmeasured,
unrepresentative transient states. Instead, warm states are generated once by running the
warm-up with a fixed reference policy, and training episodes start from a randomly chosen warm
state that is *branched* with the episode seed. The branch re-draws all future randomness and
the remaining durations of activities in progress (conditional on elapsed time), so two
episodes from the same warm state still have different futures.

Leakage control: pool states may only be generated from TRAINING seeds (asserted). Evaluation
does not use the pool: test episodes run the full protocol from an empty start, so reported
results do not depend on the reference warm-up policy (RESEARCH.md §6.11).

Pool size matters: near saturation the hospital state is very persistent, so a small pool
represents a few particular states rather than the stationary distribution. In a 200-run
diagnostic, a 10-state pool gave mean boarding 5.9 h vs 3.9 h for full episodes, while 200
distinct states gave 3.86 vs 3.85. Training pools therefore use at least MIN_TRAINING_POOL
distinct states.

Known limitation: the warm states reflect the reference policy's steady state (greedy
placement). An agent's own steady state may differ somewhat; the measured part of a training
episode therefore begins slightly off the agent's stationary distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..experiments.seeds import require_split
from ..simulation.config import ScenarioConfig, SimulationConfig
from ..simulation.hospital import HospitalSimulation, PlacementPolicy, run_until
from .snapshot_io import load_snapshot, save_snapshot

#: Minimum number of distinct warm states for a training pool (see pool-size note above).
MIN_TRAINING_POOL = 200


@dataclass
class WarmStartPool:
    """Saved end-of-warm-up states used to start RL training episodes."""

    states: list[HospitalSimulation]
    source_seeds: list[int]

    @classmethod
    def generate(
        cls, config: SimulationConfig, scenario: ScenarioConfig, seeds, policy: PlacementPolicy
    ) -> "WarmStartPool":
        """Run each seed from empty to the end of the warm-up and keep the state there."""
        seeds = list(seeds)
        require_split(seeds, "train")
        states = []
        for seed in seeds:
            sim = HospitalSimulation(config, scenario, seed=seed)
            run_until(sim, policy, until=config.warmup_hours)
            states.append(sim)
        return cls(states, seeds)

    def start_episode(self, episode_seed: int, rng: np.random.Generator) -> HospitalSimulation:
        """A training episode: a random warm state, branched with ``episode_seed``."""
        require_split([episode_seed], "train")
        base = self.states[int(rng.integers(len(self.states)))]
        return base.branch(episode_seed, resample_in_progress=True)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for seed, sim in zip(self.source_seeds, self.states):
            save_snapshot(sim, directory / f"warm_seed{seed}.json.gz")

    @classmethod
    def load(cls, directory: str | Path) -> "WarmStartPool":
        paths = sorted(
            Path(directory).glob("warm_seed*.json.gz"),
            key=lambda p: int(p.name[len("warm_seed") : -len(".json.gz")]),
        )
        seeds = [int(p.name[len("warm_seed") : -len(".json.gz")]) for p in paths]
        require_split(seeds, "train")
        return cls([load_snapshot(p) for p in paths], seeds)
