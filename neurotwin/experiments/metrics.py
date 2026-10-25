"""Per-episode metrics over the measurement window (RESEARCH.md v2 §6.22 "Metrics").

Primary (pre-registered, all lower-is-better): mean boarding time, 90th-percentile boarding
time, diversions per 100 emergency arrivals, off-service placements per 100 general
placements. Everything else is secondary and reported descriptively.

Request-level secondary metrics are computed from the request log (analysis only; policies
never see it). Only requests created inside the measurement window and placed before the
horizon are included, matching the primary boarding metrics.
"""

from __future__ import annotations

import math

import numpy as np

from ..simulation.entities import RequestType
from ..simulation.hospital import HospitalSimulation
from .objective import CostWeights, measured_cost

PRIMARY_METRICS = ("mean_boarding_h", "p90_boarding_h", "diversion_per_100", "off_service_per_100")
SECONDARY_METRICS = ("J_per_day", "high_priority_boarding_h", "elective_wait_h", "mean_ed_wait_h",
                     "step_down_delay_h", "escalation_delay_h", "placements_per_day",
                     "censored_requests")
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else math.nan


def episode_metrics(sim: HospitalSimulation, weights: CostWeights) -> dict[str, float]:
    res = sim.result()
    warmup = sim.config.warmup_hours
    days = res.measured_hours / 24.0
    waits: dict[str, list[float]] = {"hp": [], "sd": [], "esc": []}
    for r in sim.request_log():
        if r.created_at < warmup or math.isnan(r.placed_at):
            continue
        wait = r.placed_at - r.created_at
        if r.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU) and r.from_ed and r.priority <= 2:
            waits["hp"].append(wait)
        elif r.rtype is RequestType.STEP_DOWN:
            waits["sd"].append(wait)
        elif r.rtype is RequestType.ESCALATION:
            waits["esc"].append(wait)
    return {
        "mean_boarding_h": res.mean_boarding_hours,
        "p90_boarding_h": res.p90_boarding_hours,
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
    }
