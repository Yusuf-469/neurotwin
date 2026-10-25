"""Digital twin: snapshots, clones, what-if branches, observable state and warm starts."""

import json
import math

import numpy as np
import pytest
from scipy import stats

from neurotwin.digital_twin import (
    RecordingPolicy,
    WarmStartPool,
    load_snapshot,
    observe,
    save_snapshot,
)
from neurotwin.simulation import (
    SCENARIOS,
    HospitalSimulation,
    Unit,
    load_default_config,
    run_episode,
    run_until,
)
from neurotwin.simulation.config import DistributionSpec
from neurotwin.simulation.distributions import conditional_residual, sample_duration
from neurotwin.simulation.events import EventKind
from neurotwin.simulation.validation.reduced import GreedyPlacement

from .helpers import IntermittentPolicy

CFG = load_default_config()
_PATIENT_ATTRS = (
    "arrival_time",
    "is_emergency",
    "priority",
    "specialty",
    "icu_first",
    "ed_treatment_time",
    "icu_los",
    "general_los",
    "escalation_icu_los",
    "post_icu_general_los",
    "deterioration_delay",
)


def _fingerprint(sim: HospitalSimulation) -> str:
    """Everything observable about a finished run, as a string (NaN-safe comparison)."""
    res = sim.result()
    patients = [
        [
            getattr(p, a) if not isinstance(getattr(p, a), Unit) else getattr(p, a).value
            for a in _PATIENT_ATTRS
        ]
        + [p.discharged_at, p.diverted]
        for p in sim.patient_log()
    ]
    return json.dumps(
        [
            res.boarding_times,
            res.elective_waits,
            res.ed_waits,
            res.diversions,
            vars(sim.acc),
            vars(sim.measured),
            patients,
        ]
    )


def _attrs(patients) -> str:
    return json.dumps(
        [
            [
                getattr(p, a) if not isinstance(getattr(p, a), Unit) else getattr(p, a).value
                for a in _PATIENT_ATTRS
            ]
            for p in patients
        ]
    )


# ---------------------------------------------------------------- exact snapshot / restore


@pytest.mark.parametrize("policy", [GreedyPlacement(), IntermittentPolicy(period=3)])
def test_snapshot_restore_continues_identically(policy):
    sim = HospitalSimulation(CFG, SCENARIOS["B"], seed=21)
    run_until(sim, policy, until=300.0)
    assert sim.now == 300.0 and not sim.finished
    restored = HospitalSimulation.from_snapshot(json.loads(json.dumps(sim.to_snapshot())))
    run_until(sim, policy)
    run_until(restored, policy)
    assert _fingerprint(sim) == _fingerprint(restored)


def test_snapshot_at_an_open_decision_epoch():
    sim = HospitalSimulation(CFG, SCENARIOS["C"], seed=4)
    run_until(sim, GreedyPlacement(), until=200.0)
    assert sim.next_epoch()  # now inside a decision epoch
    restored = HospitalSimulation.from_snapshot(json.loads(json.dumps(sim.to_snapshot())))
    assert restored.epoch_open
    run_until(sim, GreedyPlacement())
    run_until(restored, GreedyPlacement())
    assert _fingerprint(sim) == _fingerprint(restored)


def test_snapshot_file_roundtrip(tmp_path):
    sim = HospitalSimulation(CFG, seed=2)
    run_until(sim, GreedyPlacement(), until=150.0)
    save_snapshot(sim, tmp_path / "s.json.gz")
    restored = load_snapshot(tmp_path / "s.json.gz")
    run_until(sim, GreedyPlacement())
    run_until(restored, GreedyPlacement())
    assert _fingerprint(sim) == _fingerprint(restored)


def test_clone_is_independent_and_exact():
    sim = HospitalSimulation(CFG, seed=6)
    run_until(sim, GreedyPlacement(), until=100.0)
    twin = sim.clone()
    run_until(twin, GreedyPlacement())
    assert sim.now == 100.0 and not sim.finished  # original untouched
    run_until(sim, GreedyPlacement())
    assert _fingerprint(sim) == _fingerprint(twin)


# ---------------------------------------------------------------- what-if branches


def _warm(seed=13, until=600.0, scenario="B"):
    sim = HospitalSimulation(CFG, SCENARIOS[scenario], seed=seed)
    run_until(sim, GreedyPlacement(), until=until)
    return sim


def test_branch_preserves_the_present():
    sim = _warm()
    branch = sim.branch(7)
    a, b = observe(sim), observe(branch)
    for field in (
        "time",
        "occupied",
        "staffed_capacity",
        "waiting_requests",
        "ed_count",
        "recent_arrivals",
        "cumulative_placements",
    ):
        assert getattr(a, field) == getattr(b, field)


def test_branch_with_same_seed_gives_same_future_whatever_the_policy():
    """CRN for what-if analysis (RESEARCH.md §6.21)."""
    sim = _warm()
    n0 = len(list(sim.patient_log()))
    b1, b2 = sim.branch(7), sim.branch(7)
    assert _attrs(b1.patient_log()) == _attrs(b2.patient_log())  # redraws identical
    run_until(b1, GreedyPlacement())
    run_until(b2, IntermittentPolicy(period=4))
    new1, new2 = list(b1.patient_log())[n0:], list(b2.patient_log())[n0:]
    assert len(new1) == len(new2) > 50
    assert _attrs(new1) == _attrs(new2)
    assert b1.result().mean_boarding_hours != b2.result().mean_boarding_hours


def test_different_branch_seeds_give_different_futures():
    sim = _warm()
    n0 = len(list(sim.patient_log()))
    b1, b2 = sim.branch(7), sim.branch(8)
    run_until(b1, GreedyPlacement())
    run_until(b2, GreedyPlacement())
    assert _attrs(list(b1.patient_log())[n0:]) != _attrs(list(b2.patient_log())[n0:])


def test_branch_streams_are_independent_of_episode_streams():
    # Branch seed 7 must not replay the future of episode seed 7.
    episode = HospitalSimulation(CFG, seed=7)
    run_until(episode, GreedyPlacement(), until=48.0)
    branch = HospitalSimulation(CFG, seed=99).branch(7)
    run_until(branch, GreedyPlacement(), until=48.0)
    arrivals = lambda s: [p.arrival_time for p in s.patient_log()]
    assert arrivals(episode) != arrivals(branch)


def test_branch_never_schedules_the_past_and_respects_elapsed_time():
    sim = _warm()
    branch = sim.branch(3, resample_in_progress=True)
    m_os = CFG.off_service_los_multiplier
    checked = 0
    for t, kind, _, patient in branch._queue.entries():
        assert t >= branch.now
        if kind in (EventKind.DISCHARGE, EventKind.DETERIORATION) and not patient.has_deteriorated:
            elapsed = branch.now - patient.activity_start
            mult = m_os if patient.off_service else 1.0
            assert patient.general_los * mult >= elapsed - 1e-9
            checked += 1
    assert checked > 50


def test_branch_without_resampling_keeps_in_progress_end_times():
    sim = _warm()
    before = {
        p.pid: (t, k)
        for t, k, _, p in sim._queue.entries()
        if p is not None and k is not EventKind.ED_TREATMENT_END
    }
    branch = sim.branch(3, resample_in_progress=False)
    after = {
        p.pid: (t, k)
        for t, k, _, p in branch._queue.entries()
        if p is not None and k is not EventKind.ED_TREATMENT_END
    }
    assert before == after and len(before) > 50


@pytest.mark.parametrize(
    "spec,elapsed",
    [
        (DistributionSpec("lognormal", 96.0, 72.0), 60.0),
        (DistributionSpec("lognormal", 96.0, 72.0), 200.0),
        (DistributionSpec("lognormal", 3.0, 1.5), 4.0),
    ],
)
def test_conditional_residual_matches_rejection_sampling(spec, elapsed):
    """e + residual must follow T | T > e (two-sample KS against rejection sampling)."""
    rng = np.random.default_rng(1)
    conditional = elapsed + np.array(
        [conditional_residual(spec, elapsed, u) for u in rng.random(20_000)]
    )
    raw = np.array([sample_duration(spec, u) for u in rng.random(400_000)])
    reference = raw[raw > elapsed][:20_000]
    assert len(reference) > 2_000
    assert stats.ks_2samp(conditional, reference).pvalue > 0.001
    assert np.all(conditional >= elapsed)


def test_conditional_residual_exponential_is_memoryless():
    spec = DistributionSpec("exponential", 10.0)
    r = [conditional_residual(spec, 37.0, u) for u in np.random.default_rng(2).random(50_000)]
    assert np.mean(r) == pytest.approx(10.0, rel=0.03)


# ---------------------------------------------------------------- observable state and history


def test_observed_state_has_no_hidden_information_and_serialises():
    sim = _warm(until=300.0)
    state = observe(sim)
    hidden = ("los", "treatment_time", "deterioration")
    assert not any(h in name for name in vars(state) for h in hidden)
    text = json.dumps(state.to_dict())
    assert "general_los" not in text and "deterioration_delay" not in text
    for unit in Unit:
        assert state.free_staffed(unit.value) == sim.free_staffed(unit)
    assert all(sim.now - 24.0 <= t <= sim.now for t in state.recent_arrivals)


def test_recording_policy_does_not_change_the_run(tmp_path):
    plain = run_episode(HospitalSimulation(CFG, SCENARIOS["B"], seed=5), GreedyPlacement())
    recorder = RecordingPolicy(GreedyPlacement())
    recorded = run_episode(HospitalSimulation(CFG, SCENARIOS["B"], seed=5), recorder)
    assert plain.boarding_times == recorded.boarding_times
    assert len(recorder.history) > 100
    recorder.history.save_json(tmp_path / "history.json")
    assert len(json.loads((tmp_path / "history.json").read_text())) == len(recorder.history)


# ---------------------------------------------------------------- warm-start pool


def test_warm_start_pool_rejects_non_training_seeds():
    with pytest.raises(ValueError):
        WarmStartPool.generate(CFG, SCENARIOS["A"], [200_000], GreedyPlacement())
    pool = WarmStartPool.generate(CFG, SCENARIOS["A"], [0], GreedyPlacement())
    with pytest.raises(ValueError):
        pool.start_episode(200_001, np.random.default_rng(0))


def test_warm_start_pool_save_load_roundtrip(tmp_path):
    pool = WarmStartPool.generate(CFG, SCENARIOS["A"], [0, 1], GreedyPlacement())
    assert all(s.now == CFG.warmup_hours and not s.finished for s in pool.states)
    pool.save(tmp_path)
    loaded = WarmStartPool.load(tmp_path)
    assert loaded.source_seeds == [0, 1]
    a = pool.start_episode(500, np.random.default_rng(3))
    b = loaded.start_episode(500, np.random.default_rng(3))
    run_until(a, GreedyPlacement())
    run_until(b, GreedyPlacement())
    assert _fingerprint(a) == _fingerprint(b)


def test_warm_start_episodes_are_not_biased_relative_to_full_episodes():
    """Warm-started episodes should reproduce the measured-window statistics of full episodes
    under the same policy (checks warm-up length and conditional resampling together).

    One branch per *distinct* warm state: near saturation the state is very persistent, so
    many branches of a few states measure those states, not the stationary distribution (a
    10-state pool gave mean boarding 5.9 vs 3.9 for full episodes in a 200-run diagnostic,
    while 200 distinct states gave 3.86 vs 3.85; docs/stage4_baselines_report.md).
    """
    policy = GreedyPlacement()
    n = 60
    pool = WarmStartPool.generate(CFG, SCENARIOS["A"], range(n), policy)
    warm = [run_episode(state.branch(1000 + i), policy) for i, state in enumerate(pool.states)]
    full = [run_episode(HospitalSimulation(CFG, seed=2000 + i), policy) for i in range(n)]
    assert all(r.measured_hours == CFG.horizon_hours - CFG.warmup_hours for r in warm)
    for metric in ("time_avg_boarding", "off_service_rate_per_100"):
        x = [getattr(r, metric) for r in warm]
        y = [getattr(r, metric) for r in full]
        assert stats.ttest_ind(x, y, equal_var=False).pvalue > 0.01, metric
