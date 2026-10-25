"""Welch's graphical warm-up analysis and steady-state load summary (RESEARCH.md v2 §6.11).

Welch's method: run R independent replications, average an output series (here hourly total
occupancy) across replications at each time point, then smooth it with a centred moving
average. The warm-up period is where the smoothed curve stops trending and fluctuates around
its steady-state level.

The smoothing window must span the longest periodic component, otherwise the cycle is
mistaken for a trend. Electives arrive on weekdays only, so occupancy has a 7-day cycle and the
default window is one week (half-width 84 h). A 1-day window was tried first and suggested
11 days, which the weekly-peak pattern showed to be too short.

The reported warm-up suggestion (first day the weekly-smoothed curve reaches 98% of its mean
level over the final third of the run) is a reproducible aid to reading the curve, not a
replacement for inspecting it. The curve itself is printed so the choice can be checked.

Run: python -m neurotwin.simulation.validation.warmup [--days 42] [--reps 30] [--scenario A]
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import SCENARIOS, SimulationConfig, load_default_config
from ..entities import Unit
from ..hospital import HospitalSimulation, run_episode
from .reduced import GreedyPlacement


def occupancy_traces(config: SimulationConfig, scenario: str, seeds: range,
                     days: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hourly total occupancy (reps x hours), hourly staffed fraction, and episode metrics."""
    cfg = config.with_overrides(horizon_hours=days * 24.0, warmup_hours=min(days * 12.0, 336.0))
    totals, staffed_util = [], []
    boarding, p90, diversion, off_service = [], [], [], []
    for seed in seeds:
        sim = HospitalSimulation(cfg, SCENARIOS[scenario], seed=seed, trace_interval_hours=1.0)
        res = run_episode(sim, GreedyPlacement())
        totals.append([sum(occ.values()) for _, occ, _ in sim.trace])
        boarding.append(res.mean_boarding_hours)
        p90.append(res.p90_boarding_hours)
        diversion.append(res.diversion_rate_per_100)
        off_service.append(res.off_service_rate_per_100)
        staffed_util.append({u: sim.occupied(u) / max(1, sim.staffed_capacity(u)) for u in Unit})
    metrics = {"mean_boarding_h": np.nanmean(boarding), "p90_boarding_h": np.nanmean(p90),
               "diversion_per_100": np.nanmean(diversion),
               "off_service_per_100": np.nanmean(off_service)}
    return np.asarray(totals, dtype=float), np.asarray([[d[u] for u in Unit] for d in staffed_util]), metrics


def welch_curve(traces: np.ndarray, window: int) -> np.ndarray:
    """Cross-replication mean, then centred moving average of half-width ``window``."""
    mean = traces.mean(axis=0)
    out = np.empty_like(mean)
    for i in range(len(mean)):
        w = min(window, i, len(mean) - 1 - i)
        out[i] = mean[i - w:i + w + 1].mean()
    return out


def suggest_warmup_days(curve: np.ndarray, threshold: float = 0.98) -> int:
    daily = curve[: len(curve) // 24 * 24].reshape(-1, 24).mean(axis=1)
    level = daily[2 * len(daily) // 3:].mean()
    return int(np.argmax(daily >= threshold * level))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=42)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--scenario", default="A")
    parser.add_argument("--window", type=int, default=84, help="moving-average half-width (h)")
    args = parser.parse_args()

    traces, util, metrics = occupancy_traces(load_default_config(), args.scenario,
                                             range(args.reps), args.days)
    curve = welch_curve(traces, window=args.window)
    daily = curve[: len(curve) // 24 * 24].reshape(-1, 24).mean(axis=1)
    print(f"Scenario {args.scenario}: {args.reps} replications x {args.days} days, greedy policy")
    print("Welch curve (daily mean of smoothed total occupancy):")
    for day, value in enumerate(daily):
        print(f"  day {day:2d}: {value:7.1f}")
    print(f"Suggested warm-up (98% of final-third level): {suggest_warmup_days(curve)} days")
    print("End-of-run occupancy / staffed capacity (mean over reps): "
          + ", ".join(f"{u.value}={v:.2f}" for u, v in zip(Unit, util.mean(axis=0))))
    print("Metrics over measurement window: "
          + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
