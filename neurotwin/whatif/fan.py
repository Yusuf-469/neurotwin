"""Range of possible futures from one moment ("fan chart"; dashboard Live hospital page).

From the current state S0 at time t0, each policy is run on the same N random futures (the
what-if branching of RESEARCH.md v2 §6.21: the present is kept, future arrivals, staffing and
remaining durations are re-drawn given the time already elapsed). Every ``step_hours`` the
number of patients waiting and the cost accumulated since t0 are recorded, and the futures are
summarised per time point by percentiles (10/25/50/75/90). So the chart shows the spread of
outcomes, not one lucky or unlucky path.

Because the futures are shared between policies (common random numbers), the end-of-window
costs are paired: the paired bootstrap CI of the difference and the share of futures in which
each policy was cheaper are reported. All values are simulation results.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..experiments.objective import CostWeights
from ..experiments.statistics import hierarchical_paired_bootstrap
from ..simulation.hospital import HospitalSimulation, PlacementPolicy, run_until

PERCENTILES = (10, 25, 50, 75, 90)


def _path(
    prepared: HospitalSimulation, seed: int, policy: PlacementPolicy, times: list[float],
    weights: CostWeights,
) -> tuple[list[int], list[float]]:
    sim = prepared.branch(seed)
    waiting, cost = [], []
    for t in times:
        if t > sim.now:
            run_until(sim, policy, until=t)
        waiting.append(len(sim.waiting_requests()))
        cost.append(weights.cost(sim.measured))
    return waiting, cost


def futures_fan(
    sim: HospitalSimulation,
    policies: dict[str, Callable[[], PlacementPolicy]],
    horizon_hours: float = 48.0,
    n_futures: int = 20,
    step_hours: float = 2.0,
    first_seed: int = 0,
) -> dict[str, Any]:
    """Percentile bands of waiting patients and accumulated cost for each policy."""
    if sim.finished:
        raise ValueError("the state must be before the end of the simulation")
    if not policies:
        raise ValueError("need at least one policy")
    t0 = sim.now
    end = min(t0 + horizon_hours, sim.config.horizon_hours)
    if end <= t0:
        raise ValueError("no time left in this run")
    prepared = sim.clone()
    prepared.set_measurement_window(t0, end)
    n_steps = max(1, int(round((end - t0) / step_hours)))
    times = [t0 + (end - t0) * (k / n_steps) for k in range(n_steps + 1)]
    weights = CostWeights.load()

    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, factory in policies.items():
        w_runs, c_runs = [], []
        for seed in range(first_seed, first_seed + n_futures):
            w, c = _path(prepared, seed, factory(), times, weights)
            w_runs.append(w)
            c_runs.append(c)
        raw[name] = {"waiting": np.asarray(w_runs, float), "cost": np.asarray(c_runs, float)}

    def bands(x: np.ndarray) -> dict[str, list[float]]:
        q = np.percentile(x, PERCENTILES, axis=0)
        return {f"p{p}": [round(float(v), 2) for v in row] for p, row in zip(PERCENTILES, q)}

    names = list(policies)
    out: dict[str, Any] = {
        "start_time": t0,
        "end_time": end,
        "hours": [round(t - t0, 3) for t in times],
        "n_futures": n_futures,
        "first_seed": first_seed,
        "series": {
            n: {"waiting": bands(raw[n]["waiting"]), "cost": bands(raw[n]["cost"])} for n in names
        },
        "final_cost": {
            n: {"mean": round(float(raw[n]["cost"][:, -1].mean()), 2),
                "per_future": [round(float(v), 2) for v in raw[n]["cost"][:, -1]]}
            for n in names
        },
        "comparison": None,
    }
    if len(names) >= 2:
        a, b = raw[names[0]]["cost"][:, -1], raw[names[1]]["cost"][:, -1]
        diff, lo, hi, _ = hierarchical_paired_bootstrap(a[None, :], b, n_boot=2000)
        out["comparison"] = {
            "policy": names[0],
            "reference": names[1],
            "diff_mean": round(float(diff), 2),
            "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "share_policy_cheaper": round(float(np.mean(a < b)), 3),
            "share_tied": round(float(np.mean(a == b)), 3),
        }
    return out
