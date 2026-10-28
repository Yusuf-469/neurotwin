"""'Inside the agent': decision recorder and integrated-gradients introspection."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neurotwin.api.app import V3_AGENTS, app
from neurotwin.policies import LoggingPolicy, load_baseline_params, make_policy
from neurotwin.policies.ensemble_ppo import EnsemblePPOPolicy
from neurotwin.policies.insight_recorder import InsightRecorder, compare
from neurotwin.rl.features import N_FEATURES, FeatureScales
from neurotwin.rl.introspect import (
    baseline_observation,
    explain_decision,
    input_groups,
    integrated_gradients,
)
from neurotwin.rl.actions import N_ACTIONS
from neurotwin.rl.ppo.agent import PPOAgent
from neurotwin.rl.ppo.networks import ActorCritic
from neurotwin.rl.ppo.normalize import RunningMeanStd
from neurotwin.simulation.config import SCENARIOS, load_default_config
from neurotwin.simulation.hospital import HospitalSimulation, run_until

client = TestClient(app)


def _agent(seed: int) -> PPOAgent:
    """Small random agent with a non-trivial frozen normaliser (no files needed)."""
    torch.manual_seed(seed)
    model = ActorCritic(N_FEATURES, N_ACTIONS, (16, 16))
    # orthogonal init gives a near-uniform policy; scale the output so preferences differ
    with torch.no_grad():
        model.actor[-1].weight.mul_(200.0)
    rms = RunningMeanStd((N_FEATURES,))
    rng = np.random.default_rng(seed)
    rms.update(rng.uniform(0, 1, size=(64, N_FEATURES)))
    rms.frozen = True
    return PPOAgent(model, rms)


def _rule():
    return make_policy("load_balancing", load_baseline_params()["load_balancing"])


def _run(policy, hours=72.0, seed=5):
    sim = HospitalSimulation(load_default_config(), SCENARIOS["B"], seed=seed)
    logged = LoggingPolicy(policy)
    run_until(sim, logged, until=hours)
    return sim, logged.log.records


def test_recorder_does_not_change_decisions():
    # Same agents, same seed: wrapping must leave every decision and the outcome unchanged
    # (the rule is consulted but never obeyed, and consulting it has no side effects).
    agents = [_agent(1), _agent(2)]
    sim_a, plain = _run(EnsemblePPOPolicy(agents))
    recorder = InsightRecorder(EnsemblePPOPolicy(agents), _rule())
    sim_b, wrapped = _run(recorder)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "policy_details"} for r in rs]
    assert strip(plain) == strip(wrapped)
    assert sim_a.acc == sim_b.acc
    assert len(recorder.snapshots) == len(wrapped) > 0
    assert wrapped[0]["policy"] == "ppo_ensemble"  # name passes through the wrapper


def test_snapshots_match_the_logged_decision_and_are_bounded():
    recorder = InsightRecorder(EnsemblePPOPolicy([_agent(3)]), _rule(), keep=50)
    _, records = _run(recorder)
    assert len(recorder.snapshots) == 50 and min(recorder.snapshots) == len(records) - 50
    for did, snap in recorder.snapshots.items():
        rec = records[did]
        assert snap["time"] == rec["time"]
        if rec["action"] == "HOLD":
            assert snap["agent"] is None
        else:
            assert (snap["agent"]["rid"], snap["agent"]["unit"]) == (rec["rid"], rec["unit"])
        assert snap["obs"].shape == (N_FEATURES,) and snap["mask"][0]  # HOLD always valid


def test_compare_cases():
    a = {"rid": 1, "unit": "MED"}
    assert compare(None, None) == "both_wait"
    assert compare(None, a) == "agent_waits"
    assert compare(a, None) == "rule_waits"
    assert compare(a, dict(a)) == "same"
    assert compare(a, {"rid": 1, "unit": "SURG"}) == "same_patient_other_ward"
    assert compare(a, {"rid": 2, "unit": "MED"}) == "different_patient"


def test_integrated_gradients_completeness_and_zero_at_baseline():
    agents = [_agent(4), _agent(5)]
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(0, 1.2, N_FEATURES).astype(np.float32)
        attr, fx, fb = integrated_gradients(agents, x, chosen=1, rival=0, steps=96)
        assert abs(attr.sum() - (fx - fb)) < 0.02 * max(1.0, abs(fx - fb))
    base = baseline_observation(agents).astype(np.float32)
    attr, fx, fb = integrated_gradients(agents, base, chosen=1, rival=0)
    assert np.allclose(attr, 0.0, atol=1e-6) and fx == pytest.approx(fb, abs=1e-5)


def test_attribution_finds_the_only_input_that_matters():
    # Hand-built agent whose preference for action 1 over HOLD depends on feature 3 alone.
    agent = _agent(6)
    with torch.no_grad():
        for layer in agent.model.actor:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        first, second, last = [m for m in agent.model.actor if isinstance(m, torch.nn.Linear)]
        first.weight[0, 3] = 1.0
        second.weight[0, 0] = 1.0
        last.weight[1, 0] = 5.0
    x = np.asarray(agent.obs_rms.mean, dtype=np.float32).copy()
    x[3] += 0.5
    attr, _, _ = integrated_gradients([agent], x, chosen=1, rival=0)
    assert np.argmax(np.abs(attr)) == 3 and attr[3] > 0
    assert np.allclose(np.delete(attr, 3), 0.0, atol=1e-6)


def test_input_groups_cover_every_feature_once():
    groups = input_groups(FeatureScales.from_config(load_default_config()))
    idx = sorted(i for g in groups for i in g.idx)
    assert idx == list(range(N_FEATURES))
    obs = np.zeros(N_FEATURES, dtype=np.float32)
    obs[9], obs[10] = 1.0, 0.0  # sin = 1, cos = 0 -> 06:00
    hour = next(g for g in groups if g.key == "hour")
    assert hour.show(obs) == "06:00"


def test_explain_decision_structure():
    agents = [_agent(7), _agent(8), _agent(9)]
    scales = FeatureScales.from_config(load_default_config())
    mask = np.array([True, True, True, False, False, False, False])
    out = explain_decision(agents, np.full(N_FEATURES, 0.4, np.float32), mask, scales)
    assert {a["action"] for a in out["actions"]} == {"HOLD", "ADMIT_PRIMARY_PRIORITY", "ADMIT_PRIMARY_LONGEST_WAIT"}
    assert sum(a["probability"] for a in out["actions"]) == pytest.approx(1.0, abs=1e-3)
    assert all(len(a["per_agent"]) == 3 for a in out["actions"])
    assert 1 <= out["agents_agreeing"] <= 3
    att = out["attribution"]
    assert att["runner_up"] != out["chosen_action"] and len(att["inputs"]) == 8
    # only HOLD valid: nothing to explain
    only = explain_decision(agents, np.zeros(N_FEATURES, np.float32), np.eye(7, dtype=bool)[0], scales)
    assert only["attribution"] is None and only["chosen_action"] == "HOLD"


def test_insight_endpoint_rejects_rules_and_unknown_decisions():
    sid = client.post("/api/simulation/start", json={"scenario": "A", "seed": 1, "policy": "priority"}).json()["session_id"]
    client.post(f"/api/simulation/{sid}/step", json={"hours": 24})
    assert client.get(f"/api/simulation/{sid}/insight/0").status_code == 409


@pytest.mark.skipif(not V3_AGENTS, reason="crisis-trained agents not present")
def test_insight_endpoint_for_the_latest_agent():
    sid = client.post("/api/simulation/start", json={"scenario": "B", "seed": 2, "policy": "ppo_v3"}).json()["session_id"]
    client.post(f"/api/simulation/{sid}/step", json={"hours": 48})
    decs = client.get(f"/api/simulation/{sid}/decisions?limit=20").json()["decisions"]
    r = client.get(f"/api/simulation/{sid}/insight/{decs[-1]['decision_id']}")
    assert r.status_code == 200
    j = r.json()
    assert j["n_agents"] == 5 and j["reference_rule"] == "load_balancing"
    assert j["agreement"] in {"same", "same_patient_other_ward", "different_patient", "both_wait", "agent_waits", "rule_waits"}
    assert "critic" not in j  # deliberately not reported (rl/introspect.py)
    assert client.get(f"/api/simulation/{sid}/insight/999999").status_code == 404
