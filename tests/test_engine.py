"""Engine mechanics, reproducibility, CRN integrity and exact cost accounting."""

import math

import pytest

from neurotwin.simulation import (SCENARIOS, HospitalSimulation, RequestType, Unit,
                                  load_default_config, run_episode)
from neurotwin.simulation.validation.reduced import GreedyPlacement

from .helpers import IntermittentPolicy

CFG = load_default_config()


def _run(policy, scenario="A", seed=0, config=CFG):
    sim = HospitalSimulation(config, SCENARIOS[scenario], seed=seed)
    result = run_episode(sim, policy)
    return sim, result


# ---------------------------------------------------------------- reproducibility and CRN

def test_same_seed_same_result_different_seed_differs():
    _, r1 = _run(GreedyPlacement(), seed=5)
    _, r2 = _run(GreedyPlacement(), seed=5)
    _, r3 = _run(GreedyPlacement(), seed=6)
    assert r1.boarding_times == r2.boarding_times and r1.diversions == r2.diversions
    assert r1.boarding_times != r3.boarding_times


def test_crn_patients_identical_across_policies():
    """RESEARCH.md §6.17 'CRN integrity': the n-th patient is the same under every policy."""
    sim_a, _ = _run(GreedyPlacement(), seed=11)
    sim_b, _ = _run(IntermittentPolicy(period=5), seed=11)
    pa, pb = list(sim_a.patient_log()), list(sim_b.patient_log())
    n = min(len(pa), len(pb))
    assert n > 200
    attrs = ("arrival_time", "is_emergency", "priority", "specialty", "icu_first",
             "ed_treatment_time", "icu_los", "general_los", "escalation_icu_los",
             "post_icu_general_los", "deterioration_delay")
    for x, y in zip(pa[:n], pb[:n]):
        assert all(getattr(x, a) == getattr(y, a) for a in attrs)
    # Arrival processes do not depend on the policy, so both runs see every arrival.
    assert len(pa) == len(pb)
    assert sim_a.nurses_on_shift == sim_b.nurses_on_shift


def test_policies_actually_differ_in_outcomes():
    # Guards the CRN test against being vacuous (both policies doing the same thing).
    _, ra = _run(GreedyPlacement(), seed=11)
    _, rb = _run(IntermittentPolicy(period=5), seed=11)
    assert rb.mean_boarding_hours > ra.mean_boarding_hours


def test_tracing_does_not_change_the_run():
    plain = run_episode(HospitalSimulation(CFG, seed=3), GreedyPlacement())
    traced_sim = HospitalSimulation(CFG, seed=3, trace_interval_hours=1.0)
    traced = run_episode(traced_sim, GreedyPlacement())
    assert plain.boarding_times == traced.boarding_times
    assert len(traced_sim.trace) == int(CFG.horizon_hours)


# ---------------------------------------------------------------- exact accounting

@pytest.mark.parametrize("policy", [GreedyPlacement(), IntermittentPolicy(period=4)])
def test_time_integrals_equal_sum_of_individual_waits(policy):
    """Sample-path identity behind Little's law and the reward (RESEARCH.md §6.10, §6.17):
    the integral of the number waiting equals the sum of individual waiting times."""
    sim, _ = _run(policy, scenario="B", seed=2)
    horizon, warmup = CFG.horizon_hours, CFG.warmup_hours
    omega = CFG.priority_weights
    total = weighted = boarding = measured = 0.0
    for r in sim.request_log():
        end = r.placed_at if not math.isnan(r.placed_at) else horizon
        wait = end - r.created_at
        total += wait
        weighted += omega[r.priority - 1] * wait
        if r.from_ed and r.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU):
            boarding += wait
        measured += max(0.0, end - max(r.created_at, warmup))
    assert total > 100                                           # non-trivial queueing
    assert sim.acc.request_hours == pytest.approx(total, rel=1e-9)
    assert sim.acc.weighted_request_hours == pytest.approx(weighted, rel=1e-9)
    assert sim.acc.ed_boarding_hours == pytest.approx(boarding, rel=1e-9)
    assert sim.measured.request_hours == pytest.approx(measured, rel=1e-9)


def test_counters_match_logs():
    sim, result = _run(GreedyPlacement(), scenario="E", seed=4)
    placed = [r for r in sim.request_log() if r.unit is not None]
    assert sim.acc.placements == len(placed)
    off = sum(1 for r in placed if r.unit is not Unit.ICU and r.unit is not r.patient.specialty)
    assert sim.acc.off_service_placements == off
    assert sim.acc.diversions == sum(p.diverted for p in sim.patient_log())
    assert result.diversions > 0                                  # scenario E is congested


# ---------------------------------------------------------------- placement rules

class _CheckingPolicy:
    """Greedy policy that asserts engine invariants at every decision."""

    def __init__(self):
        self.greedy = GreedyPlacement()
        self.decisions = 0

    def decide(self, sim):
        for unit in Unit:
            assert sim.free_staffed(unit) == max(0, sim.staffed_capacity(unit) - sim.occupied(unit))
        if sim.config.ed_capacity is not None:
            assert sim.ed_count <= sim.config.ed_capacity
        for req in sim.waiting_requests():
            assert req.rtype.needs_icu == (req.eligible_units == (Unit.ICU,))
            assert not hasattr(req, "patient")                    # no hidden information
        self.decisions += 1
        return self.greedy.decide(sim)


@pytest.mark.parametrize("scenario", ["A", "C", "E"])
def test_invariants_hold_at_every_decision(scenario):
    policy = _CheckingPolicy()
    _run(policy, scenario=scenario, seed=8)
    assert policy.decisions > 100


def test_place_rejects_invalid_requests():
    sim = HospitalSimulation(CFG, seed=0)
    with pytest.raises(RuntimeError):
        sim.place(0, Unit.MED)                                    # no epoch open yet
    while sim.next_epoch():
        icu = [r for r in sim.waiting_requests() if r.rtype.needs_icu]
        if icu:
            with pytest.raises(ValueError):
                sim.place(icu[0].rid, Unit.MED)                   # ineligible unit
            break
    else:
        pytest.fail("no ICU request occurred")
    with pytest.raises(KeyError):
        sim.place(10 ** 9, Unit.ICU)


def test_off_service_multiplies_los_and_deterioration_leads_to_escalation():
    cfg = CFG.with_overrides(deterioration_hazard_per_day=0.2, off_service_los_multiplier=1.5)
    sim, _ = _run(GreedyPlacement(), scenario="B", seed=9, config=cfg)
    checked_off = checked_on = 0
    for r in sim.request_log():
        p = r.patient
        if (r.rtype is RequestType.ADMIT_GENERAL and r.unit is not None and not p.has_deteriorated
                and not math.isnan(p.discharged_at)):
            expected = p.general_los * (1.5 if r.unit is not p.specialty else 1.0)
            assert p.discharged_at - r.placed_at == pytest.approx(expected)
            checked_off += r.unit is not p.specialty
            checked_on += r.unit is p.specialty
    assert checked_off > 5 and checked_on > 5
    types = {r.rtype for r in sim.request_log() if r.unit is not None}
    assert {RequestType.ESCALATION, RequestType.STEP_DOWN} <= types
    for r in sim.request_log():
        if r.rtype is RequestType.ESCALATION and r.unit is not None:
            assert r.unit is Unit.ICU and r.priority == CFG.escalation_priority


def test_capacity_scenario_scales_beds():
    sim = HospitalSimulation(CFG, SCENARIOS["D"], seed=0)
    assert sim.physical_beds[Unit.MED] == round(CFG.beds["MED"] * 0.6)
    nurses, doctors = HospitalSimulation(CFG, SCENARIOS["C"], seed=0).nominal_staff
    assert doctors == 3 and nurses[Unit.ICU] == 4                 # 5*0.65=3.25, 6*0.65=3.9
