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
