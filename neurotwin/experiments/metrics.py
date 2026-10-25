"""Per-episode metrics over the measurement window (RESEARCH.md v2 §6.22 "Metrics").

Primary (pre-registered, all lower-is-better): mean boarding time, 90th-percentile boarding
time, diversions per 100 emergency arrivals, off-service placements per 100 general
placements. Everything else is secondary and reported descriptively.

Boarding metrics are **censoring-inclusive** (protocol amendment, 2026-10-26; see
docs/stage6_evaluation_report.md §2). Every ED admission request created in the measurement
window counts. A request still waiting at the horizon contributes (horizon - created), a lower
bound on its true wait. The pre-registered placed-only version ignored such requests, so a
policy that left hard patients waiting indefinitely looked *better*. It is undefined (NaN) when
nothing is placed and is kept only as the secondary metric ``mean_boarding_placed_only_h``.

Other request-level secondary metrics are computed from the request log (analysis only;
policies never see it).
"""

from __future__ import annotations

import math

import numpy as np

from ..simulation.entities import RequestType
from ..simulation.hospital import HospitalSimulation
from .objective import CostWeights, measured_cost

PRIMARY_METRICS = ("mean_boarding_h", "p90_boarding_h", "diversion_per_100", "off_service_per_100")
SECONDARY_METRICS = (
    "J_per_day",
    "high_priority_boarding_h",
    "elective_wait_h",
    "mean_ed_wait_h",
    "step_down_delay_h",
    "escalation_delay_h",
    "placements_per_day",
    "censored_requests",
    "censored_ed_requests",
    "mean_boarding_placed_only_h",
)
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else math.nan


def episode_metrics(sim: HospitalSimulation, weights: CostWeights) -> dict[str, float]:
    """Primary and secondary metrics of a finished episode's measurement window."""
    res = sim.result()
    warmup, horizon = sim.config.warmup_hours, sim.config.horizon_hours
    days = res.measured_hours / 24.0
    waits: dict[str, list[float]] = {"board": [], "hp": [], "sd": [], "esc": []}
    censored_ed = 0
    for r in sim.request_log():
        if r.created_at < warmup:
            continue
        placed = not math.isnan(r.placed_at)
        wait = (r.placed_at if placed else horizon) - r.created_at  # censored: lower bound
        is_ed_admission = (
            r.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU) and r.from_ed
        )
        if is_ed_admission:
            waits["board"].append(wait)
            censored_ed += not placed
            if r.priority <= 2:
                waits["hp"].append(wait)
        elif r.rtype is RequestType.STEP_DOWN:
            waits["sd"].append(wait)
        elif r.rtype is RequestType.ESCALATION:
            waits["esc"].append(wait)
    board = waits["board"]
    return {
        "mean_boarding_h": _mean(board),
        "p90_boarding_h": float(np.percentile(board, 90)) if board else math.nan,
        "diversion_per_100": res.diversion_rate_per_100,
        "off_service_per_100": res.off_service_rate_per_100,
        "J_per_day": measured_cost(sim, weights),
        "high_priority_boarding_h": _mean(waits["hp"]),
        "elective_wait_h": _mean(res.elective_waits),
        "mean_ed_wait_h": res.mean_ed_wait_hours,
        "step_down_delay_h": _mean(waits["sd"]),
        "escalation_delay_h": _mean(waits["esc"]),
        "placements_per_day": sim.measured.placements / days,
        "censored_requests": float(res.censored_requests),
        "censored_ed_requests": float(censored_ed),
        "mean_boarding_placed_only_h": res.mean_boarding_hours,
    }
