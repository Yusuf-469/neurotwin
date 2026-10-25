"""Baseline policies (RESEARCH.md v2 §6.8-§6.9): rules, MIP exactness, invariants, logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from neurotwin.experiments.objective import CostWeights, measured_cost
from neurotwin.experiments.tuning import tune
from neurotwin.policies import (DecisionLog, FCFSAllocator, LoadBalancingAllocator, LoggingPolicy,
                                OptimizationAllocator, PriorityAllocator, TrunkReservationAllocator,
                                load_baseline_params, make_policy)
from neurotwin.policies.optimization import UNITS, solve_assignment, solve_hungarian, solve_lp_relaxation
from neurotwin.simulation import (SCENARIOS, HospitalSimulation, RequestType, RequestView, Unit,
                                  load_default_config, run_episode)

CFG = load_default_config()
SHORT = CFG.with_overrides(horizon_hours=500.0, warmup_hours=200.0)


# ---------------------------------------------------------------- a fake engine for rule tests

@dataclass
class FakeSim:
    """Minimal stand-in exposing only the public queries a policy may use."""
    now: float
    requests: list[RequestView]
    free: dict[Unit, int]
    staffed: dict[Unit, int] = field(default_factory=lambda: {u: 10 for u in Unit})
    config: object = CFG

    def waiting_requests(self):
        return list(self.requests)

    def free_staffed(self, unit):
        return self.free[unit]

    def staffed_capacity(self, unit):
        return self.staffed[unit]

    def occupied(self, unit):
        return self.staffed[unit] - self.free[unit]


def req(rid, priority, created, primary=Unit.MED, rtype=RequestType.ADMIT_GENERAL):
    eligible = (Unit.ICU,) if rtype.needs_icu else (Unit.MED, Unit.SURG)
    return RequestView(rid, rtype, priority, created, primary, eligible, True)


def test_fcfs_serves_oldest_and_priority_serves_most_urgent():
    requests = [req(1, priority=4, created=1.0), req(2, priority=1, created=5.0)]
    sim = FakeSim(now=6.0, requests=requests, free={Unit.MED: 1, Unit.SURG: 0, Unit.ICU: 0})
    assert FCFSAllocator().decide(sim) == (1, Unit.MED)
    assert PriorityAllocator().decide(sim) == (2, Unit.MED)


def test_off_service_only_after_waiting_threshold():
    sim = FakeSim(now=3.0, requests=[req(1, 3, created=0.0)],
                  free={Unit.MED: 0, Unit.SURG: 2, Unit.ICU: 0})
    assert FCFSAllocator(off_service_wait_hours=4.0).decide(sim) is None
    assert FCFSAllocator(off_service_wait_hours=2.0).decide(sim) == (1, Unit.SURG)


def test_no_head_of_line_blocking_across_units():
    # The oldest request needs the ICU (full); a younger general request can still be placed.
    requests = [req(1, 1, 0.0, Unit.ICU, RequestType.ADMIT_ICU), req(2, 4, 1.0)]
    sim = FakeSim(now=2.0, requests=requests, free={Unit.MED: 1, Unit.SURG: 0, Unit.ICU: 0})
    assert FCFSAllocator().decide(sim) == (2, Unit.MED)


def test_load_balancing_overflows_when_imbalanced():
    sim = FakeSim(now=0.0, requests=[req(1, 3, 0.0)],
                  free={Unit.MED: 1, Unit.SURG: 6, Unit.ICU: 0})      # MED 90%, SURG 40%
    assert LoadBalancingAllocator(imbalance_threshold=0.2).decide(sim) == (1, Unit.SURG)
    assert LoadBalancingAllocator(imbalance_threshold=0.6).decide(sim) == (1, Unit.MED)


def test_trunk_reservation_keeps_beds_for_urgent_patients():
    policy = TrunkReservationAllocator(reserve={"MED": 2, "SURG": 2, "ICU": 1})
    low = FakeSim(now=0.0, requests=[req(1, 3, 0.0)], free={Unit.MED: 2, Unit.SURG: 0, Unit.ICU: 0})
    assert policy.decide(low) is None                                # 2 free, 2 reserved
    high = FakeSim(now=0.0, requests=[req(1, 2, 0.0)], free={Unit.MED: 1, Unit.SURG: 0, Unit.ICU: 0})
    assert policy.decide(high) == (1, Unit.MED)                      # urgent may use reserve
    ok = FakeSim(now=0.0, requests=[req(1, 5, 0.0)], free={Unit.MED: 3, Unit.SURG: 0, Unit.ICU: 0})
    assert policy.decide(ok) == (1, Unit.MED)


# ---------------------------------------------------------------- MIP exactness

def _random_instance(rng):
    n = int(rng.integers(1, 25))
    values = []
    for _ in range(n):
        if rng.random() < 0.25:
            values.append({Unit.ICU: float(rng.uniform(-2, 40))})
        else:
            primary = Unit.MED if rng.random() < 0.5 else Unit.SURG
            c = float(rng.uniform(0, 40))
            values.append({primary: c, (Unit.SURG if primary is Unit.MED else Unit.MED): c - 2.0})
    caps = {k: int(rng.integers(0, 6)) for k in UNITS}
    return values, caps


def test_milp_matches_lp_relaxation_and_hungarian_oracles():
    """Total unimodularity: MILP = LP relaxation = assignment-problem optimum, and the LP
    solution is integral (RESEARCH.md §6.9 test oracles)."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        values, caps = _random_instance(rng)
        chosen, obj = solve_assignment(values, caps)
        x, lp_obj = solve_lp_relaxation(values, caps)
        assert obj == pytest.approx(lp_obj, abs=1e-7)
        assert obj == pytest.approx(solve_hungarian(values, caps), abs=1e-7)
        assert np.allclose(x, np.round(x), atol=1e-7)
        for k in UNITS:                                              # feasibility
            assert sum(1 for u in chosen.values() if u is k) <= caps[k]
        assert all(values[i][u] > 0 for i, u in chosen.items())


def test_mip_holds_when_only_a_cheap_off_service_option_exists():
    policy = OptimizationAllocator(off_service_cost=2.0, breach_cost=0.0, review_horizon_hours=0.1)
    # Just-created P5 request: waiting cost (1.0 * 0.1) < off-service cost 2.0 -> hold.
    sim = FakeSim(now=0.0, requests=[req(1, 5, 0.0)], free={Unit.MED: 0, Unit.SURG: 3, Unit.ICU: 0})
    assert policy.decide(sim) is None
    # After a long wait the waiting cost exceeds kappa -> off-service.
    sim.now = 10.0
    assert policy.decide(sim) == (1, Unit.SURG)


def test_mip_always_fills_free_primary_beds():
    policy = OptimizationAllocator()
    sim = FakeSim(now=1.0, requests=[req(i, 5, 1.0) for i in range(3)],
                  free={Unit.MED: 2, Unit.SURG: 0, Unit.ICU: 0})
    decision = policy.decide(sim)
    assert decision is not None and decision[1] is Unit.MED


# ---------------------------------------------------------------- integration

class _Checked:
    def __init__(self, policy):
        self.policy, self.decisions = policy, 0

    def decide(self, sim):
        decision = self.policy.decide(sim)
        if decision is not None:
            rid, unit = decision
            view = next(r for r in sim.waiting_requests() if r.rid == rid)
            assert unit in view.eligible_units and sim.free_staffed(unit) > 0
        self.decisions += 1
        return decision


@pytest.mark.parametrize("name", ["fcfs", "priority", "load_balancing", "trunk_reservation", "mip"])
@pytest.mark.parametrize("scenario", ["A", "E"])
def test_every_baseline_runs_validly(name, scenario):
    checked = _Checked(make_policy(name, load_baseline_params(tuned=False)[name]))
    result = run_episode(HospitalSimulation(SHORT, SCENARIOS[scenario], seed=100_001), checked)
    assert checked.decisions > 50 and result.boarding_times


def test_trunk_reservation_never_violates_reserve_in_a_real_run():
    policy = TrunkReservationAllocator(reserve={"MED": 3, "SURG": 3, "ICU": 2})

    class Spy:
        def decide(self, sim):
            d = policy.decide(sim)
            if d is not None:
                view = next(r for r in sim.waiting_requests() if r.rid == d[0])
                if view.priority >= 3:
                    assert sim.free_staffed(d[1]) > policy.reserve[d[1]]
            return d

    run_episode(HospitalSimulation(SHORT, SCENARIOS["B"], seed=3), Spy())


def test_policies_are_deterministic():
    for name in ("priority", "mip"):
        params = load_baseline_params(tuned=False)[name]
        a = run_episode(HospitalSimulation(SHORT, seed=9), make_policy(name, params))
        b = run_episode(HospitalSimulation(SHORT, seed=9), make_policy(name, params))
        assert a.boarding_times == b.boarding_times


def test_decision_logging_is_read_only(tmp_path):
    plain = run_episode(HospitalSimulation(SHORT, seed=4), PriorityAllocator())
    logged_policy = LoggingPolicy(PriorityAllocator(), DecisionLog())
    logged = run_episode(HospitalSimulation(SHORT, seed=4), logged_policy)
    assert plain.boarding_times == logged.boarding_times
    records = logged_policy.log.records
    assert any(r["action"] == "PLACE" for r in records)
    placed = [r for r in records if r["action"] == "PLACE"]
    assert all(r["off_service"] == (r["unit"] != r["primary_unit"] and r["unit"] != "ICU")
               for r in placed)
    logged_policy.log.save_jsonl(tmp_path / "log.jsonl")
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == len(records)


# ---------------------------------------------------------------- objective and tuning

def test_measured_cost_matches_its_definition():
    sim = HospitalSimulation(SHORT, SCENARIOS["B"], seed=2)
    run_episode(sim, PriorityAllocator())
    w = CostWeights.load()
    m = sim.measured
    days = (SHORT.horizon_hours - SHORT.warmup_hours) / 24
    expected = (w.alpha * m.weighted_request_hours + w.beta * m.diversions
                + w.kappa * m.off_service_placements) / days
    assert measured_cost(sim, w) == pytest.approx(expected) and expected > 0


def test_tuning_refuses_non_validation_seeds():
    with pytest.raises(ValueError):
        tune("fcfs", [{"off_service_wait_hours": 4.0}], [200_000], workers=1)
