"""REST API for the dashboard (ARCHITECTURE.md §12).

Design rules
------------
* Only *observable* state (``HospitalState``) and decision-log facts are served. Simulator
  snapshots contain hidden patient attributes and are never exposed.
* Every response carrying outcomes includes ``label`` = SIMULATION_LABEL.
* Sessions are in-memory (single-user research tool; no database needed, ARCHITECTURE.md §14).
* Inputs are bounded (step length, what-if futures and horizon) so a request cannot run for
  hours.
* The explanation endpoint uses the offline template explainer. No LLM provider is configured
  (connecting one requires the project owner's confirmation, AGENTS.md).

Run: uvicorn neurotwin.api.app:app --port 8000
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..digital_twin.state import observe
from ..explanation import TemplateExplainer
from ..experiments.metrics import episode_metrics
from ..experiments.objective import CostWeights
from ..policies import POLICY_CLASSES, DecisionLog, LoggingPolicy, load_baseline_params, make_policy
from ..policies.ppo_policy import PPOPolicy
from ..simulation.config import SCENARIOS, SEVERE_SCENARIOS, ScenarioConfig, load_default_config
from ..simulation.hospital import HospitalSimulation, run_until
from ..whatif import WhatIfEngine, WhatIfRequest

ROOT = Path(__file__).resolve().parents[2]
SIMULATION_LABEL = ("SIMULATION RESULTS from a simplified research model. Not real-world or "
                    "clinical evidence; not a medical device.")
# Agent 0 of final_A has a documented collapse mode under stress (Stage 6 report §5); the
# interactive default is agent 1. This choice affects demos only, never reported results.
AGENT_PATH = Path(os.environ.get("NEUROTWIN_PPO_AGENT",
                                 ROOT / "results" / "ppo" / "final_A" / "agent_1.pt"))
MAX_STEP_HOURS = 24 * 14
MAX_FUTURES = 50
MAX_WHATIF_HOURS = 24 * 14

ALL_SCENARIOS = {**SCENARIOS, **SEVERE_SCENARIOS}


def available_policies() -> list[str]:
    names = list(POLICY_CLASSES)
    if AGENT_PATH.exists():
        names.append("ppo")
    return names


def build_policy(name: str):
    if name == "ppo":
        if not AGENT_PATH.exists():
            raise HTTPException(404, f"no trained PPO agent at {AGENT_PATH}")
        return PPOPolicy.load(AGENT_PATH)
    if name not in POLICY_CLASSES:
        raise HTTPException(404, f"unknown policy {name!r}")
    return make_policy(name, load_baseline_params()[name])


@dataclass
class Session:
    scenario: str
    seed: int
    policy_name: str
    sim: HospitalSimulation
    policy: LoggingPolicy
    history: list[dict[str, Any]] = field(default_factory=list)


SESSIONS: dict[str, Session] = {}


def _new_session(scenario: str, seed: int, policy: str) -> Session:
    if scenario not in ALL_SCENARIOS:
        raise HTTPException(404, f"unknown scenario {scenario!r}")
    sim = HospitalSimulation(load_default_config(), ALL_SCENARIOS[scenario], seed=seed)
    session = Session(scenario, seed, policy, sim, LoggingPolicy(build_policy(policy), DecisionLog()))
    session.history.append(_history_point(sim))
    return session


def _history_point(sim: HospitalSimulation) -> dict[str, Any]:
    s = observe(sim)
    return {"time": s.time, "occupied": s.occupied, "staffed_capacity": s.staffed_capacity,
            "waiting": len(s.waiting_requests), "ed_count": s.ed_count,
            "cumulative_diversions": s.cumulative_diversions,
            "cumulative_off_service": s.cumulative_off_service}


def _get(sid: str) -> Session:
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


# ---------------------------------------------------------------------- request models

class StartRequest(BaseModel):
    scenario: str = "A"
    seed: int = Field(0, ge=0)
    policy: str = "priority"


class StepRequest(BaseModel):
    hours: float = Field(24.0, gt=0, le=MAX_STEP_HOURS)


class WhatIfBody(BaseModel):
    horizon_hours: float = Field(72.0, gt=0, le=MAX_WHATIF_HOURS)
    n_futures: int = Field(20, ge=2, le=MAX_FUTURES)
    arrival_multiplier: float = Field(1.0, gt=0, le=3)
    staff_multiplier: float = Field(1.0, gt=0, le=2)
    capacity_multiplier: float = Field(1.0, gt=0, le=2)
    policies: list[str] = Field(default_factory=lambda: ["priority", "load_balancing"])


# ---------------------------------------------------------------------- app

app = FastAPI(title="NeuroTwin API", version="0.2.0",
              description="Research simulation of hospital bed allocation. " + SIMULATION_LABEL)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "label": SIMULATION_LABEL}


@app.get("/api/policies")
def policies() -> dict[str, Any]:
    return {"policies": available_policies(), "ppo_agent": str(AGENT_PATH) if AGENT_PATH.exists() else None}


@app.get("/api/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": {k: vars(v) for k, v in ALL_SCENARIOS.items()}}


@app.post("/api/simulation/start")
def start(body: StartRequest) -> dict[str, Any]:
    session = _new_session(body.scenario, body.seed, body.policy)
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = session
    return {"session_id": sid, **_status(session)}


@app.post("/api/simulation/{sid}/step")
def step(sid: str, body: StepRequest) -> dict[str, Any]:
    session = _get(sid)
    sim = session.sim
    if sim.finished:
        raise HTTPException(409, "simulation has reached its horizon")
    target = min(sim.now + body.hours, sim.config.horizon_hours)
    # Advance in 4-hour chunks so the history has a usable resolution.
    while sim.now < target and not sim.finished:
        run_until(sim, session.policy, until=min(sim.now + 4.0, target))
        session.history.append(_history_point(sim))
    return _status(session)


@app.post("/api/simulation/{sid}/reset")
def reset(sid: str) -> dict[str, Any]:
    old = _get(sid)
    SESSIONS[sid] = _new_session(old.scenario, old.seed, old.policy_name)
    return _status(SESSIONS[sid])


def _status(session: Session) -> dict[str, Any]:
    sim = session.sim
    return {"label": SIMULATION_LABEL, "scenario": session.scenario, "seed": session.seed,
            "policy": session.policy_name, "time": sim.now, "finished": sim.finished,
            "horizon_hours": sim.config.horizon_hours, "warmup_hours": sim.config.warmup_hours}


@app.get("/api/simulation/{sid}/state")
def state(sid: str) -> dict[str, Any]:
    session = _get(sid)
    out = {"label": SIMULATION_LABEL, **_status(session), "state": observe(session.sim).to_dict()}
    if session.sim.finished:
        out["episode_metrics"] = episode_metrics(session.sim, CostWeights.load())
    return out


@app.get("/api/simulation/{sid}/history")
def history(sid: str) -> dict[str, Any]:
    return {"label": SIMULATION_LABEL, "history": _get(sid).history}


@app.get("/api/simulation/{sid}/decisions")
def decisions(sid: str, limit: int = 50) -> dict[str, Any]:
    records = _get(sid).policy.log.records
    return {"label": SIMULATION_LABEL, "total": len(records),
            "decisions": records[-max(1, min(limit, 500)):]}


@app.get("/api/simulation/{sid}/explanations/{decision_id}")
def explanation(sid: str, decision_id: int) -> dict[str, Any]:
    records = _get(sid).policy.log.records
    if not 0 <= decision_id < len(records):
        raise HTTPException(404, "unknown decision")
    return {"label": SIMULATION_LABEL,
            "explanation": TemplateExplainer().explain(records[decision_id]).as_dict()}


@app.post("/api/simulation/{sid}/whatif")
def whatif(sid: str, body: WhatIfBody) -> dict[str, Any]:
    session = _get(sid)
    if session.sim.finished:
        raise HTTPException(409, "simulation has reached its horizon")
    if not body.policies or len(body.policies) > 6:
        raise HTTPException(422, "choose 1-6 policies")
    factories = {}
    for name in body.policies:
        build_policy(name)                       # validates the name early
        factories[name] = (lambda n=name: build_policy(n))
    scenario = None
    if (body.arrival_multiplier, body.staff_multiplier, body.capacity_multiplier) != (1.0, 1.0, 1.0):
        base = session.sim.scenario
        scenario = ScenarioConfig("what_if", base.arrival_multiplier * body.arrival_multiplier,
                                  base.staff_multiplier * body.staff_multiplier,
                                  base.capacity_multiplier * body.capacity_multiplier)
    result = WhatIfEngine(factories).evaluate(
        session.sim, WhatIfRequest(body.horizon_hours, body.n_futures, scenario))
    return result.as_dict()


@app.get("/api/experiments/results")
def experiment_results() -> dict[str, Any]:
    """Pre-computed study results (Stages 6-7), served read-only with their provenance."""
    out: dict[str, Any] = {"label": SIMULATION_LABEL}
    for key, rel in {"summary": "docs/data/stage6/summary.json",
                     "comparisons": "docs/data/stage6/comparisons.json",
                     "J_comparisons": "docs/data/stage6/J_comparisons.json",
                     "robustness": "docs/data/stage7/robustness.json",
                     "crn_check": "docs/data/stage7/crn_check.json",
                     "structure": "docs/data/stage7/structure.json"}.items():
        path = ROOT / rel
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out
