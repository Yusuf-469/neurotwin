"""Counterfactual policy comparison from a given hospital state (RESEARCH.md v2 §6.21).

Procedure for a state S0 at time t0:
1. Copy S0 and set the measurement window to [t0, t0 + horizon].
2. For each future f = 1..N, branch the copy with seed f (and optionally new operating
   conditions): the present is kept; future arrivals, staffing and the remaining durations of
   activities in progress are re-drawn (conditional on elapsed time).
3. Run every policy on every branch. Policies share the same N futures (CRN), so differences are
   paired by future.
4. Report per-policy metrics and paired bootstrap CIs of each policy minus a reference policy.

Metric coverage: boarding metrics count requests created *inside* the window (censoring-
inclusive); J also includes patients already waiting at t0, because it integrates all waiting
time in the window. Every result is a simulation result and is labelled as such.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..experiments.metrics import episode_metrics
from ..experiments.objective import CostWeights
from ..experiments.statistics import hierarchical_paired_bootstrap
from ..simulation.config import ScenarioConfig
from ..simulation.hospital import HospitalSimulation, PlacementPolicy, run_until

LABEL = ("SIMULATION RESULTS: counterfactual outcomes of a simplified research model; "
         "not real-world or clinical evidence.")
REPORTED_METRICS = ("J_per_day", "mean_boarding_h", "p90_boarding_h", "diversion_per_100",
                    "off_service_per_100")

PolicyFactory = Callable[[], PlacementPolicy]


@dataclass(frozen=True)
class WhatIfRequest:
    horizon_hours: float = 72.0
    n_futures: int = 30
    scenario: ScenarioConfig | None = None          # None = keep current conditions
    first_future_seed: int = 0
    reference_policy: str | None = None             # default: first policy given


@dataclass
class WhatIfResult:
    label: str
    start_time: float
    horizon_hours: float
    scenario: str
    n_futures: int
    per_policy: dict[str, dict[str, dict[str, float]]]      # policy -> metric -> mean/ci
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, dict[str, list[float]]] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "start_time": self.start_time,
                "horizon_hours": self.horizon_hours, "scenario": self.scenario,
                "n_futures": self.n_futures, "per_policy": self.per_policy,
                "comparisons": self.comparisons}


def _run_future(args: tuple[HospitalSimulation, int, ScenarioConfig | None,
                            dict[str, PolicyFactory]]) -> dict[str, dict[str, float]]:
    prepared, seed, scenario, factories = args
    weights = CostWeights.load()
    out = {}
    for name, factory in factories.items():
        branch = prepared.branch(seed, scenario=scenario)
        run_until(branch, factory())
        out[name] = episode_metrics(branch, weights)
    return out


class WhatIfEngine:
    def __init__(self, policies: dict[str, PolicyFactory], workers: int = 1) -> None:
        if not policies:
            raise ValueError("need at least one policy")
        self.policies = policies
        self.workers = workers

    def evaluate(self, sim: HospitalSimulation, request: WhatIfRequest) -> WhatIfResult:
        if sim.finished:
            raise ValueError("the state must be before the end of the simulation")
        prepared = sim.clone()
        # The window is set before branching so that arrivals beyond the original horizon are
        # scheduled. If S0 is inside an open decision epoch, each policy makes that pending
        # decision itself (run_until asks the policy first), so every policy starts from S0.
        prepared.set_measurement_window(sim.now, sim.now + request.horizon_hours)
        seeds = range(request.first_future_seed, request.first_future_seed + request.n_futures)
        jobs = [(prepared, s, request.scenario, self.policies) for s in seeds]
        if self.workers > 1:
            with ProcessPoolExecutor(self.workers) as ex:
                futures = list(ex.map(_run_future, jobs))
        else:
            futures = [_run_future(j) for j in jobs]

        raw = {p: {m: [f[p][m] for f in futures] for m in REPORTED_METRICS} for p in self.policies}
        per_policy = {}
        for p, metrics in raw.items():
            per_policy[p] = {}
            for m, values in metrics.items():
                x = np.asarray(values, dtype=float)
                x = x[~np.isnan(x)]
                per_policy[p][m] = {"mean": float(x.mean()) if len(x) else math.nan,
                                    "n": int(len(x))}
        reference = request.reference_policy or next(iter(self.policies))
        comparisons = []
        for p in self.policies:
            if p == reference:
                continue
            for m in REPORTED_METRICS:
                a, b = np.asarray(raw[p][m], float), np.asarray(raw[reference][m], float)
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 2:
                    continue
                diff, lo, hi, _ = hierarchical_paired_bootstrap(a[ok][None, :], b[ok], n_boot=2000)
                comparisons.append({"policy": p, "reference": reference, "metric": m,
                                    "diff_mean": diff, "ci95": [lo, hi], "n_futures": int(ok.sum())})
        name = request.scenario.name if request.scenario else f"current ({sim.scenario.name})"
        return WhatIfResult(LABEL, sim.now, request.horizon_hours, name, request.n_futures,
                            per_policy, comparisons, raw)
