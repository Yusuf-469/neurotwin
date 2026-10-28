"""API endpoints (FastAPI TestClient; no network)."""

import pytest
from fastapi.testclient import TestClient

from neurotwin.api.app import SIMULATION_LABEL, app

client = TestClient(app)


def _start(policy="priority", scenario="B"):
    r = client.post(
        "/api/simulation/start", json={"scenario": scenario, "seed": 1, "policy": policy}
    )
    assert r.status_code == 200
    return r.json()["session_id"]


def test_health_policies_scenarios():
    assert client.get("/api/health").json()["label"] == SIMULATION_LABEL
    assert {"fcfs", "priority", "mip"} <= set(client.get("/api/policies").json()["policies"])
    assert {"A", "E", "E_severe"} <= set(client.get("/api/scenarios").json()["scenarios"])


def test_session_lifecycle_state_decisions_explanations():
    sid = _start()
    r = client.post(f"/api/simulation/{sid}/step", json={"hours": 48})
    assert r.status_code == 200 and r.json()["time"] == pytest.approx(48.0)
    state = client.get(f"/api/simulation/{sid}/state").json()
    assert state["label"] == SIMULATION_LABEL
    text = str(state["state"])
    assert "general_los" not in text and "deterioration_delay" not in text  # observable only
    hist = client.get(f"/api/simulation/{sid}/history").json()["history"]
    assert len(hist) == 13 and hist[-1]["time"] == pytest.approx(48.0)  # start + 12 x 4 h
    dec = client.get(f"/api/simulation/{sid}/decisions?limit=5").json()
    assert dec["total"] > 5 and len(dec["decisions"]) == 5
    did = dec["decisions"][0]["decision_id"]
    exp = client.get(f"/api/simulation/{sid}/explanations/{did}").json()["explanation"]
    assert exp["decision_id"] == did and "not clinical advice" in exp["limitations"]
    assert client.post(f"/api/simulation/{sid}/reset").json()["time"] == 0.0


def test_whatif_endpoint():
    sid = _start(scenario="A")
    client.post(f"/api/simulation/{sid}/step", json={"hours": 72})
    r = client.post(
        f"/api/simulation/{sid}/whatif",
        json={
            "horizon_hours": 24,
            "n_futures": 3,
            "arrival_multiplier": 1.3,
            "policies": ["priority", "fcfs"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "SIMULATION" in body["label"] and set(body["per_policy"]) == {"priority", "fcfs"}
    assert body["scenario"] == "what_if"


@pytest.mark.parametrize(
    "path,payload,code",
    [
        ("/api/simulation/start", {"scenario": "Z"}, 404),
        ("/api/simulation/start", {"policy": "nope"}, 404),
        ("/api/simulation/start", {"seed": -1}, 422),
    ],
)
def test_invalid_inputs_are_rejected(path, payload, code):
    assert client.post(path, json=payload).status_code == code


def test_bounds_on_expensive_requests():
    sid = _start()
    assert client.post(f"/api/simulation/{sid}/step", json={"hours": 10_000}).status_code == 422
    assert client.post(f"/api/simulation/{sid}/whatif", json={"n_futures": 500}).status_code == 422
    assert client.get("/api/simulation/unknown/state").status_code == 404


def test_results_endpoint_serves_stage6_data_with_label():
    body = client.get("/api/experiments/results").json()
    assert body["label"] == SIMULATION_LABEL
    assert body["summary"] is not None and len(body["comparisons"]) == 100
    assert len(body["ablations"]) == 80 and len(body["ppo_dr"]) == 25
    assert body["shield"]["collapses"]["ppo_shielded"] == 0
    assert "ppo_shielded" in client.get("/api/policies").json()["policies"]
    assert "ppo_ensemble" in client.get("/api/policies").json()["policies"]
    assert body["beat_rule"]["selected"] == "ens_bc_ppo"
    assert (
        len(body["constrained"]["comparisons"]) == 25
        and "E:lambda_mean" in body["constrained"]["budget"]
    )


def test_cors_origins_from_environment(monkeypatch):
    import importlib

    import neurotwin.api.app as app_module

    monkeypatch.setenv("NEUROTWIN_CORS_ORIGINS", "https://neurotwin.vercel.app/, https://x.dev")
    monkeypatch.setenv(
        "NEUROTWIN_CORS_ORIGIN_REGEX", r"https://neurotwin-[a-z0-9-]+-me\.vercel\.app"
    )
    try:
        reloaded = importlib.reload(app_module)
        assert "https://neurotwin.vercel.app" in reloaded.CORS_ORIGINS
        assert "https://x.dev" in reloaded.CORS_ORIGINS
        c = TestClient(reloaded.app)
        r = c.get("/api/health", headers={"Origin": "https://neurotwin.vercel.app"})
        assert r.headers.get("access-control-allow-origin") == "https://neurotwin.vercel.app"
        preview = "https://neurotwin-git-main-me.vercel.app"
        r = c.get("/api/health", headers={"Origin": preview})
        assert r.headers.get("access-control-allow-origin") == preview
        r = c.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers
    finally:
        monkeypatch.delenv("NEUROTWIN_CORS_ORIGINS")
        monkeypatch.delenv("NEUROTWIN_CORS_ORIGIN_REGEX")
        importlib.reload(app_module)


def test_custom_conditions_run_and_report_live_work():
    r = client.post(
        "/api/simulation/start",
        json={
            "scenario": "custom",
            "seed": 2,
            "policy": "priority",
            "custom": {
                "arrival_multiplier": 1.3,
                "staff_multiplier": 0.8,
                "capacity_multiplier": 1.0,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scenario"] == "custom"
    assert body["conditions"] == {
        "arrival_multiplier": 1.3,
        "staff_multiplier": 0.8,
        "capacity_multiplier": 1.0,
    }
    step = client.post(f"/api/simulation/{body['session_id']}/step", json={"hours": 48}).json()
    assert step["step"]["simulated_hours"] == 48 and step["step"]["decisions_made"] > 0
    assert step["step"]["compute_ms"] > 0
    reset = client.post(f"/api/simulation/{body['session_id']}/reset").json()
    assert reset["conditions"]["arrival_multiplier"] == 1.3  # reset keeps the custom conditions
    bad = client.post(
        "/api/simulation/start",
        json={"scenario": "custom", "custom": {"arrival_multiplier": 9}},
    )
    assert bad.status_code == 422


def test_race_runs_competitors_on_identical_futures():
    r = client.post(
        "/api/race/start",
        json={"competitors": ["priority", "priority", "fcfs"], "start": "empty", "seed": 4},
    )
    assert r.status_code == 422  # duplicates rejected
    r = client.post(
        "/api/race/start",
        json={
            "competitors": ["priority", "load_balancing", "fcfs"],
            "start": "empty",
            "seed": 4,
            "horizon_hours": 48,
        },
    )
    assert r.status_code == 200
    race = r.json()
    assert race["elapsed_hours"] == 0 and len(race["lanes"]) == 3
    last = None
    while True:
        last = client.post(f"/api/race/{race['race_id']}/step", json={"hours": 12}).json()
        if last["finished"]:
            break
    assert last["elapsed_hours"] == 48
    assert sorted(lane["rank"] for lane in last["lanes"]) == [1, 2, 3]
    costs = {lane["name"]: lane["cost"] for lane in last["lanes"]}
    assert costs["fcfs"] >= min(costs.values())
    assert all(len(lane["trace"]) == 5 for lane in last["lanes"])  # start + 4 steps
    assert client.get(f"/api/race/{race['race_id']}").json()["elapsed_hours"] == 48
    assert client.get("/api/race/doesnotexist").status_code == 404
    over = client.post(f"/api/race/{race['race_id']}/step", json={"hours": 12})
    assert over.status_code == 409


def test_race_is_reproducible_and_can_start_from_a_session():
    body = {
        "competitors": ["priority", "load_balancing"],
        "start": "empty",
        "seed": 9,
        "horizon_hours": 24,
    }
    results = []
    for _ in range(2):
        rid = client.post("/api/race/start", json=body).json()["race_id"]
        results.append(client.post(f"/api/race/{rid}/step", json={"hours": 24}).json())
    assert [l["cost"] for l in results[0]["lanes"]] == [l["cost"] for l in results[1]["lanes"]]
    sid = _start()
    client.post(f"/api/simulation/{sid}/step", json={"hours": 24})
    r = client.post(
        "/api/race/start",
        json={"competitors": ["priority", "fcfs"], "start": "session", "session_id": sid},
    )
    assert r.status_code == 200 and r.json()["start_time"] == 24


def test_state_reports_so_far_and_benchmark_reproduces_the_run():
    sid = _start(policy="priority", scenario="A")
    client.post(f"/api/simulation/{sid}/step", json={"hours": 336})
    client.post(f"/api/simulation/{sid}/step", json={"hours": 336})
    s = client.get(f"/api/simulation/{sid}/state").json()
    assert s["so_far"] is None  # still in warm-up
    while not client.post(f"/api/simulation/{sid}/step", json={"hours": 336}).json()["finished"]:
        pass
    s = client.get(f"/api/simulation/{sid}/state").json()
    assert s["so_far"]["measured_hours"] == 14 * 24
    own = client.post(f"/api/simulation/{sid}/benchmark", json={"policy": "priority"}).json()
    assert own["metrics"]["J_per_day"] == s["episode_metrics"]["J_per_day"]  # identical patients
    other = client.post(f"/api/simulation/{sid}/benchmark", json={"policy": "fcfs"}).json()
    assert other["metrics"]["J_per_day"] != own["metrics"]["J_per_day"]
    body = client.get("/api/experiments/results").json()
    assert len(body["final_comparisons"]) == 100 and body["final_robustness"] is not None


def test_state_in_the_middle_of_the_measured_period():
    """Regression: state requests between the warm-up and the horizon must not fail."""
    sid = _start(policy="priority", scenario="A")
    for _ in range(3):
        client.post(f"/api/simulation/{sid}/step", json={"hours": 288})  # 36 days
    r = client.get(f"/api/simulation/{sid}/state")
    assert r.status_code == 200
    so_far = r.json()["so_far"]
    assert so_far["measured_hours"] == 24 and so_far["turned_away_per_100"] >= 0


def test_unexpected_errors_keep_cors_headers(monkeypatch):
    import neurotwin.api.app as app_module

    def boom(_sim):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(app_module, "_so_far", boom)
    sid = _start(policy="priority", scenario="A")
    r = client.get(f"/api/simulation/{sid}/state", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 500 and "synthetic failure" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
