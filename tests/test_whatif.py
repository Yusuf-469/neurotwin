"""What-if engine and the supporting engine features (scenario switch, measurement window)."""

import json

import numpy as np
import pytest

from neurotwin.policies import FCFSAllocator, PriorityAllocator
from neurotwin.simulation import SCENARIOS, HospitalSimulation, load_default_config, run_until
from neurotwin.simulation.config import ScenarioConfig
from neurotwin.simulation.entities import Unit
from neurotwin.whatif import WhatIfEngine, WhatIfRequest

CFG = load_default_config().with_overrides(horizon_hours=500.0, warmup_hours=100.0)


def _state(t=300.0, seed=4):
    sim = HospitalSimulation(CFG, SCENARIOS["A"], seed=seed)
    run_until(sim, PriorityAllocator(), until=t)
    return sim


def test_measurement_window_resets_metrics():
    sim = _state()
    sim.set_measurement_window(sim.now, sim.now + 48)
    assert sim.measured.placements == 0 and sim.config.horizon_hours == sim.now + 48
    run_until(sim, PriorityAllocator())
    assert sim.finished and sim.result().measured_hours == 48
    with pytest.raises(ValueError):
        sim.set_measurement_window(0.0, 10.0)                 # window in the past


def test_branch_can_switch_operating_conditions():
    sim = _state()
    sim.set_measurement_window(sim.now, sim.now + 120)
    surge = ScenarioConfig("surge", arrival_multiplier=1.6, capacity_multiplier=0.8)
    normal_counts, surge_counts = [], []
    for seed in range(10):
        for scen, counts in ((None, normal_counts), (surge, surge_counts)):
            b = sim.branch(seed, scenario=scen)
            n0 = len(list(b.patient_log()))
            run_until(b, PriorityAllocator())
            counts.append(len(list(b.patient_log())) - n0)
    assert np.mean(surge_counts) > 1.4 * np.mean(normal_counts)
    b = sim.branch(0, scenario=surge)
    assert b.physical_beds[Unit.MED] == round(CFG.beds["MED"] * 0.8)
    assert sim.physical_beds[Unit.MED] == CFG.beds["MED"]            # original untouched
    restored = HospitalSimulation.from_snapshot(json.loads(json.dumps(b.to_snapshot())))
    assert restored.physical_beds == b.physical_beds and restored.scenario == surge


def test_whatif_is_deterministic_and_uses_common_futures():
    sim = _state()
    engine = WhatIfEngine({"priority": PriorityAllocator, "fcfs": FCFSAllocator})
    req = WhatIfRequest(horizon_hours=72, n_futures=6)
    r1, r2 = engine.evaluate(sim, req), engine.evaluate(sim, req)
    assert r1.raw == r2.raw
    assert "SIMULATION" in r1.label and r1.start_time == sim.now
    assert {c["policy"] for c in r1.comparisons} == {"fcfs"}
    assert all(c["reference"] == "priority" for c in r1.comparisons)
    # The original state is not modified by the analysis.
    assert not sim.finished and sim.now == 300.0
    json.dumps(r1.as_dict())


def test_whatif_detects_a_clearly_worse_policy():
    """FCFS without overflow vs Priority under a surge: J differs in a consistent direction."""
    sim = _state(seed=7)
    engine = WhatIfEngine({"priority": PriorityAllocator,
                           "fcfs": lambda: FCFSAllocator(off_service_wait_hours=1e9)})
    res = engine.evaluate(sim, WhatIfRequest(horizon_hours=120, n_futures=12,
                                             scenario=ScenarioConfig("surge", 1.4)))
    j = next(c for c in res.comparisons if c["metric"] == "J_per_day")
    assert j["ci95"][0] > 0                                   # FCFS costlier in every resample
