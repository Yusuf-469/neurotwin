"""The 21 state features (RESEARCH.md v2 §6.5).

Normalisers are fixed constants derived from the *default* (scenario A) configuration, never
from the current scenario. If they were scenario-specific, the same physical situation would be
encoded differently in different scenarios, and the encoding itself would leak which scenario
the agent is in.

Occupancy is occupied / staffed capacity. It can exceed 1 after a shift change reduces
staffing, so it is clipped to [0, 1.5] rather than [0, 1] to keep that information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..simulation.arrivals import elective_rate, emergency_rate
from ..simulation.config import SCENARIOS, SimulationConfig
from ..simulation.entities import RequestType, Unit
from ..simulation.hospital import HospitalSimulation

N_FEATURES = 21
FEATURE_NAMES = (
    "occ_MED",
    "occ_SURG",
    "occ_ICU",
    "req_admit_general",
    "req_admit_icu",
    "req_step_down",
    "req_escalation",
    "nurse_availability",
    "doctor_availability",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "arrivals_1h",
    "arrivals_4h",
    "discharges_4h",
    "waiting_high_priority",
    "waiting_low_priority",
    "mean_elapsed_wait",
    "diversions_24h",
    "off_service_in_beds",
)
#: Wait normaliser W_max (RESEARCH.md §6.5, feature 19).
WAIT_NORMALISER_HOURS = 4.0


@dataclass(frozen=True)
class FeatureScales:
    """Scenario-independent normalisers for the state features."""

    q_max: float
    icu_beds: float
    general_beds: float
    peak_rate: float  # arrivals per hour at the weekly peak (scenario A)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "FeatureScales":
        a = SCENARIOS["A"]
        peak = float((emergency_rate(config, a).rates + elective_rate(config, a).rates).max())
        return cls(
            q_max=float(config.ed_capacity or 30),
            icu_beds=float(config.beds["ICU"]),
            general_beds=float(config.beds["MED"] + config.beds["SURG"]),
            peak_rate=peak,
        )


def features(sim: HospitalSimulation, scales: FeatureScales) -> np.ndarray:
    """The 21-feature observation vector for the current decision."""
    now = sim.now
    requests = sim.waiting_requests()
    counts = {t: 0 for t in RequestType}
    high = low = 0
    total_wait = 0.0
    for r in requests:
        counts[r.rtype] += 1
        if r.priority <= 2:
            high += 1
        else:
            low += 1
        total_wait += now - r.created_at
    occ = [min(sim.occupied(u) / max(sim.staffed_capacity(u), 1), 1.5) for u in Unit]
    nominal_nurses, nominal_doctors = sim.nominal_staff
    nurses = sum(sim.nurses_on_shift.values()) / max(sum(nominal_nurses.values()), 1)
    doctors = sim.doctors_on_shift / max(nominal_doctors, 1)
    hour = 2 * math.pi * (now % 24.0) / 24.0
    day = 2 * math.pi * (now % 168.0) / 168.0
    arrivals = sim.recent_event_times("arrivals")
    discharges = sim.recent_event_times("discharges")
    arr_1h = sum(1 for t in arrivals if t > now - 1.0)
    arr_4h = sum(1 for t in arrivals if t > now - 4.0)
    dis_4h = sum(1 for t in discharges if t > now - 4.0)
    mean_wait = total_wait / len(requests) if requests else 0.0
    return np.array(
        [
            occ[0],
            occ[1],
            occ[2],
            counts[RequestType.ADMIT_GENERAL] / scales.q_max,
            counts[RequestType.ADMIT_ICU] / scales.q_max,
            counts[RequestType.STEP_DOWN] / scales.icu_beds,
            counts[RequestType.ESCALATION] / scales.icu_beds,
            nurses,
            doctors,
            math.sin(hour),
            math.cos(hour),
            math.sin(day),
            math.cos(day),
            arr_1h / scales.peak_rate,
            arr_4h / (4.0 * scales.peak_rate),
            dis_4h / (4.0 * scales.peak_rate),
            high / scales.q_max,
            low / scales.q_max,
            mean_wait / WAIT_NORMALISER_HOURS,
            len(sim.recent_event_times("diversions")) / scales.q_max,
            sim.off_service_in_beds / scales.general_beds,
        ],
        dtype=np.float32,
    )
