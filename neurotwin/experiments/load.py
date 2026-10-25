"""Analytical offered load per scenario (policy-independent calibration, RESEARCH.md v2 §6.22).

Offered load rho_k = (expected bed-hours demanded per hour in unit group k) /
(expected staffed beds in k). It is computed from the model parameters alone, so scenario
intensities can be set by load criteria stated in advance, without looking at any policy's
results (which would let the scenario choice favour a policy).

Approximations (documented): the off-service LOS multiplier is ignored (its effect depends on
the policy); MED and SURG are pooled as "general" because off-service placement lets them
share capacity (per-ward loads are reported too, using the specialty mix).

Run: python -m neurotwin.experiments.load
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..simulation.config import SCENARIOS, ScenarioConfig, SimulationConfig, load_default_config, round_half_up
from ..simulation.distributions import sample_duration

_GRID = (np.arange(20_000) + 0.5) / 20_000           # midpoint quadrature on the uniform scale


def _mean(spec) -> float:
    return spec.mean


def _expected_staffed(physical: int, nominal_nurses: int, ratio: int, absence: float) -> float:
    n = np.arange(nominal_nurses + 1)
    pmf = stats.binom.pmf(n, nominal_nurses, 1.0 - absence)
    return float(np.sum(pmf * np.minimum(physical, n * ratio)))


def bed_demand_per_patient(config: SimulationConfig) -> dict[str, float]:
    """Expected general and ICU bed-hours per patient, and the deterioration probability."""
    h = config.deterioration_hazard_per_day / 24.0
    t = np.array([sample_duration(config.general_los, u) for u in _GRID])
    if h > 0:
        p_det = float(np.mean(1.0 - np.exp(-h * t)))
        e_min = float(np.mean((1.0 - np.exp(-h * t)) / h))      # E[min(T, D)], D ~ Exp(h)
    else:
        p_det, e_min = 0.0, float(t.mean())
    pe = np.asarray(config.priority_probs_emergency)
    pl = np.asarray(config.priority_probs_elective)
    icu = np.asarray(config.icu_first_prob)
    lam_e, lam_l = config.emergency_base_rate, config.elective_base_rate
    p_icu_first = (lam_e * pe @ icu + lam_l * pl @ icu) / (lam_e + lam_l)
    general = e_min + p_det * _mean(config.post_icu_general_los)
    icu_hours = p_icu_first * _mean(config.icu_los) + p_det * _mean(config.escalation_icu_los)
    return {"general": general, "icu": float(icu_hours), "p_deteriorate": p_det,
            "p_icu_first": float(p_icu_first)}


def offered_load(config: SimulationConfig, scenario: ScenarioConfig) -> dict[str, float]:
    demand = bed_demand_per_patient(config)
    lam = (config.emergency_base_rate + config.elective_base_rate) * scenario.arrival_multiplier
    staffed = {}
    for unit in ("MED", "SURG", "ICU"):
        physical = round_half_up(config.beds[unit] * scenario.capacity_multiplier)
        nurses = round_half_up(config.nurses[unit] * scenario.staff_multiplier)
        staffed[unit] = _expected_staffed(physical, nurses, config.patients_per_nurse[unit],
                                          config.staff_absence_prob)
    lam_e, lam_l = config.emergency_base_rate, config.elective_base_rate
    med_share = ((lam_e * config.specialty_prob_med_emergency
                  + lam_l * config.specialty_prob_med_elective) / (lam_e + lam_l))
    general_demand = lam * demand["general"]
    return {
        "rho_general": general_demand / (staffed["MED"] + staffed["SURG"]),
        "rho_MED": general_demand * med_share / staffed["MED"],
        "rho_SURG": general_demand * (1 - med_share) / staffed["SURG"],
        "rho_ICU": lam * demand["icu"] / staffed["ICU"],
        "staffed_MED": staffed["MED"], "staffed_SURG": staffed["SURG"],
        "staffed_ICU": staffed["ICU"],
    }


def main() -> None:
    cfg = load_default_config()
    d = bed_demand_per_patient(cfg)
    print(f"Per patient: general {d['general']:.1f} h, ICU {d['icu']:.2f} h, "
          f"P(deteriorate) {d['p_deteriorate']:.3f}, P(ICU first) {d['p_icu_first']:.3f}")
    print("| Scenario | rho general | rho MED | rho SURG | rho ICU |")
    print("|---|---|---|---|---|")
    for key, sc in SCENARIOS.items():
        r = offered_load(cfg, sc)
        print(f"| {key} {sc.name} | {r['rho_general']:.3f} | {r['rho_MED']:.3f} | "
              f"{r['rho_SURG']:.3f} | {r['rho_ICU']:.3f} |")


if __name__ == "__main__":
    main()
