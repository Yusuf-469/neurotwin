"""Range of possible futures (whatif/fan.py) and its API endpoint."""

import numpy as np
from fastapi.testclient import TestClient

from neurotwin.api.app import app
from neurotwin.policies import load_baseline_params, make_policy
from neurotwin.simulation.config import SCENARIOS, load_default_config
from neurotwin.simulation.hospital import HospitalSimulation, run_until
from neurotwin.whatif.fan import futures_fan

client = TestClient(app)


def _state(hours=60.0, seed=4):
    sim = HospitalSimulation(load_default_config(), SCENARIOS["B"], seed=seed)
    run_until(sim, make_policy("priority", load_baseline_params()["priority"]), until=hours)
    return sim


def _factory(name):
    return lambda: make_policy(name, load_baseline_params()[name])


def test_fan_is_deterministic_and_leaves_the_run_untouched():
    sim = _state()
    before = (sim.now, len(sim.waiting_requests()), vars(sim.acc).copy())
    pols = {"load_balancing": _factory("load_balancing"), "fcfs": _factory("fcfs")}
    a = futures_fan(sim, pols, horizon_hours=24, n_futures=6)
    b = futures_fan(sim, pols, horizon_hours=24, n_futures=6)
    assert a == b
    assert (sim.now, len(sim.waiting_requests()), vars(sim.acc)) == before


def test_bands_are_ordered_and_cost_never_decreases():
    out = futures_fan(_state(), {"priority": _factory("priority")}, horizon_hours=24, n_futures=8)
    assert out["hours"][0] == 0 and out["hours"][-1] == 24 and len(out["hours"]) == 13
    for key in ("waiting", "cost"):
        bands = out["series"]["priority"][key]
        rows = np.array([bands[f"p{p}"] for p in (10, 25, 50, 75, 90)])
        assert (np.diff(rows, axis=0) >= -1e-9).all()  # p10 <= p25 <= ... <= p90 at every time
    cost = np.array([out["series"]["priority"]["cost"][f"p{p}"] for p in (10, 50, 90)])
    assert (np.diff(cost, axis=1) >= -1e-9).all()  # accumulated cost only grows
    assert cost[:, 0].max() == 0  # nothing accumulated at the start
    assert out["comparison"] is None


def test_same_policy_twice_gives_identical_paired_futures():
    # Common random numbers: the same policy on the same futures must tie in every future.
    out = futures_fan(
        _state(), {"a": _factory("load_balancing"), "b": _factory("load_balancing")},
        horizon_hours=12, n_futures=5,
    )
    c = out["comparison"]
    assert c["diff_mean"] == 0 and c["share_tied"] == 1 and c["share_policy_cheaper"] == 0
    assert out["series"]["a"] == out["series"]["b"]


def test_futures_endpoint_validates_and_responds():
    sid = client.post("/api/simulation/start", json={"scenario": "A", "seed": 2, "policy": "priority"}).json()["session_id"]
    client.post(f"/api/simulation/{sid}/step", json={"hours": 24})
    r = client.post(f"/api/simulation/{sid}/futures",
                    json={"policies": ["load_balancing", "fcfs"], "horizon_hours": 12, "n_futures": 5})
    assert r.status_code == 200
    j = r.json()
    assert set(j["series"]) == {"load_balancing", "fcfs"} and j["n_futures"] == 5
    assert j["comparison"]["reference"] == "fcfs"
    assert client.post(f"/api/simulation/{sid}/futures", json={"policies": ["fcfs", "fcfs"]}).status_code == 422
    assert client.post(f"/api/simulation/{sid}/futures", json={"horizon_hours": 500}).status_code == 422
    # the session itself did not move
    assert client.get(f"/api/simulation/{sid}/state").json()["time"] == 24
