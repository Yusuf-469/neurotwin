"""Shared test helpers."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from neurotwin.simulation import HospitalSimulation, Unit
from neurotwin.simulation.validation.reduced import GreedyPlacement


def t_confidence_interval(samples: list[float], level: float) -> tuple[float, float]:
    """Two-sided Student-t CI for the mean of independent replications."""
    x = np.asarray(samples, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


class IntermittentPolicy:
    """Greedy, but only acts in some hours, so requests accumulate (exercises holding)."""

    def __init__(self, period: int = 3) -> None:
        self._greedy = GreedyPlacement()
        self._period = period

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        if int(sim.now) % self._period != 0:
            return None
        return self._greedy.decide(sim)
