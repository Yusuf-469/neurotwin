"""Exactly solved myopic assignment baseline (RESEARCH.md v2 §6.9).

At each decision, the waiting requests R and units K = {MED, SURG, ICU} define

    max  sum_i sum_{k in E_i} v_ik x_ik
    s.t. sum_k x_ik <= 1          for every request i
         sum_i x_ik <= cap_k      for every unit k (free staffed beds)
         x_ik in {0, 1}

with v_ik = c_i - kappa * [k is off-service for i] and
c_i = omega_p(i) * (e_i + Delta) + beta_b * [e_i + Delta > W_max].

This is a transportation problem: its constraint matrix is totally unimodular, so the LP
relaxation has an integral optimum. The production solver is scipy.optimize.milp (HiGHS); two
independent oracles (LP relaxation, Hungarian algorithm on expanded bed slots) are provided for
tests. Variables with v_ik <= 0 are dropped: placing them can never improve the objective, and
dropping them makes the policy hold instead of choosing arbitrarily between equal options.

Because every primary placement has v > 0, the MIP always fills free primary beds. It never
reserves capacity for future arrivals; it chooses off-service only when the waiting cost
exceeds kappa. This is intended: PPO vs MIP isolates the value of look-ahead (RQ1).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp

from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation
from .base import AllocationPolicy

UNITS: tuple[Unit, ...] = (Unit.MED, Unit.SURG, Unit.ICU)
Values = Sequence[dict[Unit, float]]          # per request: unit -> value (eligible units only)


def _variables(values: Values) -> list[tuple[int, Unit, float]]:
    return [(i, k, v) for i, row in enumerate(values) for k, v in row.items() if v > 0]


def _constraint_matrix(variables, n_requests: int, caps: dict[Unit, int]):
    a = np.zeros((n_requests + len(UNITS), len(variables)))
    for j, (i, k, _) in enumerate(variables):
        a[i, j] = 1.0
        a[n_requests + UNITS.index(k), j] = 1.0
    upper = np.concatenate([np.ones(n_requests), [caps.get(k, 0) for k in UNITS]])
    return a, upper


def solve_assignment(values: Values, caps: dict[Unit, int]) -> tuple[dict[int, Unit], float]:
    """Exact MILP solution: (request index -> unit, objective value)."""
    variables = _variables(values)
    if not variables:
        return {}, 0.0
    a, upper = _constraint_matrix(variables, len(values), caps)
    c = -np.array([v for _, _, v in variables])               # milp minimises
    res = milp(c, constraints=LinearConstraint(a, -np.inf, upper),
               integrality=np.ones(len(variables)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    chosen = {variables[j][0]: variables[j][1] for j in np.flatnonzero(res.x > 0.5)}
    return chosen, float(-res.fun)


def solve_lp_relaxation(values: Values, caps: dict[Unit, int]) -> tuple[np.ndarray, float]:
    """Oracle 1: LP relaxation. By total unimodularity its optimum equals the MILP optimum."""
    variables = _variables(values)
    if not variables:
        return np.zeros(0), 0.0
    a, upper = _constraint_matrix(variables, len(values), caps)
    res = linprog(-np.array([v for _, _, v in variables]), A_ub=a, b_ub=upper, bounds=(0, 1),
                  method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x, float(-res.fun)


def solve_hungarian(values: Values, caps: dict[Unit, int]) -> float:
    """Oracle 2: expand each unit into cap_k identical slots and solve the assignment problem.

    Ineligible or non-positive pairs get value 0, which is equivalent to not placing.
    """
    slots = [k for k in UNITS for _ in range(caps.get(k, 0))]
    if not values or not slots:
        return 0.0
    m = np.array([[max(row.get(k, 0.0), 0.0) for k in slots] for row in values])
    rows, cols = linear_sum_assignment(m, maximize=True)
    return float(m[rows, cols].sum())


class OptimizationAllocator(AllocationPolicy):
    """Myopic assignment, re-solved at every atomic decision.

    Re-solving after each placement is consistent: removing one placed pair from an optimal
    solution leaves an optimal solution of the reduced problem. Placements are returned in
    request-id order for determinism.
    """

    name = "mip"

    def __init__(self, review_horizon_hours: float = 4.0, breach_threshold_hours: float = 4.0,
                 breach_cost: float = 10.0, off_service_cost: float = 2.0) -> None:
        self.review_horizon_hours = review_horizon_hours
        self.breach_threshold_hours = breach_threshold_hours
        self.breach_cost = breach_cost
        self.off_service_cost = off_service_cost

    def request_values(self, sim: HospitalSimulation) -> tuple[list[int], list[dict[Unit, float]]]:
        omega = sim.config.priority_weights
        rids, values = [], []
        for r in sim.waiting_requests():
            horizon_wait = sim.now - r.created_at + self.review_horizon_hours
            c = omega[r.priority - 1] * horizon_wait
            if horizon_wait > self.breach_threshold_hours:
                c += self.breach_cost
            values.append({k: c - (self.off_service_cost if k is not r.primary_unit else 0.0)
                           for k in r.eligible_units})
            rids.append(r.rid)
        return rids, values

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        rids, values = self.request_values(sim)
        caps = {k: sim.free_staffed(k) for k in UNITS}
        chosen, _ = solve_assignment(values, caps)
        if not chosen:
            return None
        i = min(chosen, key=lambda idx: rids[idx])
        return rids[i], chosen[i]

    def params(self) -> dict[str, Any]:
        return {"review_horizon_hours": self.review_horizon_hours,
                "breach_threshold_hours": self.breach_threshold_hours,
                "breach_cost": self.breach_cost, "off_service_cost": self.off_service_cost}
