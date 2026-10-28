"""REST API for the dashboard (ARCHITECTURE.md §12).

Design rules
------------
* Only *observable* state (``HospitalState``) and decision-log facts are served. Simulator
  snapshots contain hidden patient attributes and are never exposed.
* Every response carrying outcomes includes ``label`` = SIMULATION_LABEL.
* Sessions are in-memory (single-user research tool; no database needed, ARCHITECTURE.md §14).
* Inputs are bounded (step length, what-if futures and horizon) so a request cannot run for
  hours.
* Explanations: if an LLM backend is configured through hosting secrets (a free
  OpenAI-compatible provider via ``NEUROTWIN_LLM_*``, or Claude via ``ANTHROPIC_API_KEY``;
  never in code), the guarded LLM explainer is used, with a global hourly cap
  (``NEUROTWIN_LLM_MAX_PER_HOUR``, default 60) and a per-decision cache, so a public site
  cannot run up unbounded cost. Otherwise, or when the cap is reached, the offline template
  explainer is used.
* At most MAX_SESSIONS sessions are kept; the oldest is evicted first (public hosting).
* If a built dashboard exists (``dashboard/out`` or ``NEUROTWIN_STATIC_DIR``), it is served at
  ``/`` by the same process, so the API and UI share one URL.

Run: uvicorn neurotwin.api.app:app --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..digital_twin.state import observe
from ..explanation import LLMExplainer, TemplateExplainer
from ..explanation.anthropic_client import llm_configured as claude_configured
from ..explanation.openai_compatible import OpenAICompatibleClient, openai_compatible_configured


def llm_configured() -> bool:
    """True when any LLM backend is configured (free OpenAI-compatible provider or Claude)."""
    return openai_compatible_configured() or claude_configured()


def _make_llm_client():
    """The configured LLM client: the free OpenAI-compatible provider first, then Claude."""
    if openai_compatible_configured():
        return OpenAICompatibleClient.from_env()
    from ..explanation.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient()


from ..experiments.metrics import episode_metrics
from ..experiments.objective import CostWeights
from ..policies import POLICY_CLASSES, DecisionLog, LoggingPolicy, load_baseline_params, make_policy
from ..policies.ensemble_ppo import EnsemblePPOPolicy
from ..policies.ppo_policy import PPOPolicy
from ..simulation.config import SCENARIOS, SEVERE_SCENARIOS, ScenarioConfig, load_default_config
from ..simulation.hospital import HospitalSimulation, run_until
from ..whatif import WhatIfEngine, WhatIfRequest
from ..whatif.race import Race

ROOT = Path(__file__).resolve().parents[2]
SIMULATION_LABEL = (
    "SIMULATION RESULTS from a simplified research model. Not real-world or "
    "clinical evidence; not a medical device."
)
# Agent 0 of final_A has a documented collapse mode under stress (Stage 6 report §5); the
# interactive default is agent 1. This choice affects demos only, never reported results.
AGENT_PATH = Path(
    os.environ.get("NEUROTWIN_PPO_AGENT", ROOT / "results" / "ppo" / "final_A" / "agent_1.pt")
)
MAX_STEP_HOURS = 24 * 14
MAX_FUTURES = 50
MAX_WHATIF_HOURS = 24 * 14
MAX_SESSIONS = 200
# Five-agent vote of the imitation-initialised PPO agents (RESEARCH.md §6.26), if present.
ENSEMBLE_AGENTS = sorted((ROOT / "results" / "ppo" / "bc_ppo").glob("agent_*.pt"))
LLM_MAX_PER_HOUR = int(os.environ.get("NEUROTWIN_LLM_MAX_PER_HOUR", "60"))
# Browser origins allowed to call the API cross-site: the local dev server, plus any origins in
# NEUROTWIN_CORS_ORIGINS (comma-separated), e.g. a dashboard hosted separately on Vercel while
# the API runs on another host. Same-origin use (the bundled dashboard) needs none.
# NEUROTWIN_CORS_ORIGIN_REGEX optionally admits a family of origins, e.g. every deployment URL
# of one Vercel project (preview URLs change per build). The API has no auth or cookies.
CORS_ORIGIN_REGEX = os.environ.get("NEUROTWIN_CORS_ORIGIN_REGEX") or None
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"] + [
    o.strip().rstrip("/")
    for o in os.environ.get("NEUROTWIN_CORS_ORIGINS", "").split(",")
    if o.strip()
]
STATIC_DIR = Path(os.environ.get("NEUROTWIN_STATIC_DIR", ROOT / "dashboard" / "out"))

ALL_SCENARIOS = {**SCENARIOS, **SEVERE_SCENARIOS}


def available_policies() -> list[str]:
    """Baseline names, plus 'ppo' and 'ppo_shielded' (RESEARCH.md §6.25) when a trained agent
    file exists."""
    names = list(POLICY_CLASSES)
    if AGENT_PATH.exists():
        names += ["ppo", "ppo_shielded"]
    if ENSEMBLE_AGENTS:
        names.append("ppo_ensemble")
    return names


def build_policy(name: str):
    """Instantiate a policy by name (tuned baselines or the configured PPO agent)."""
    if name in ("ppo", "ppo_shielded"):
        if not AGENT_PATH.exists():
            raise HTTPException(404, f"no trained PPO agent at {AGENT_PATH}")
        return PPOPolicy.load(AGENT_PATH, shield=name == "ppo_shielded")
    if name == "ppo_ensemble":
        if not ENSEMBLE_AGENTS:
            raise HTTPException(404, "no imitation-initialised agents available")
        return EnsemblePPOPolicy.load(ENSEMBLE_AGENTS)
    if name not in POLICY_CLASSES:
        raise HTTPException(404, f"unknown policy {name!r}")
    return make_policy(name, load_baseline_params()[name])


@dataclass
class Session:
    """One interactive simulation run with its logged policy and state history."""

    scenario: str
    seed: int
    policy_name: str
    sim: HospitalSimulation
    policy: LoggingPolicy
    history: list[dict[str, Any]] = field(default_factory=list)


SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def _store(sid: str, session: Session) -> None:
    """Add or refresh a session, evicting the least recently used beyond MAX_SESSIONS."""
    SESSIONS[sid] = session
    SESSIONS.move_to_end(sid)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


# LLM explanations (only if a key is configured): per-decision cache and a global hourly cap.
_LLM_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_LLM_CALLS: deque[float] = deque()
_LLM_LOCK = threading.Lock()


def _llm_budget_available() -> bool:
    """Reserve one LLM call if fewer than LLM_MAX_PER_HOUR were made in the last hour."""
    with _LLM_LOCK:
        now = time.monotonic()
        while _LLM_CALLS and now - _LLM_CALLS[0] > 3600:
            _LLM_CALLS.popleft()
        if len(_LLM_CALLS) >= LLM_MAX_PER_HOUR:
            return False
        _LLM_CALLS.append(now)
        return True


def _conditions(scenario: str, custom: "CustomConditions | None" = None) -> ScenarioConfig:
    """Named scenario, or user-defined multipliers for scenario "custom"."""
    if scenario == "custom":
        c = custom or CustomConditions()
        return ScenarioConfig(
            "custom", c.arrival_multiplier, c.staff_multiplier, c.capacity_multiplier
        )
    if scenario in ALL_SCENARIOS:
        return ALL_SCENARIOS[scenario]
    raise HTTPException(404, f"unknown scenario {scenario!r}")


def _new_session(
    scenario: str, seed: int, policy: str, custom: CustomConditions | None = None
) -> Session:
    sim = HospitalSimulation(load_default_config(), _conditions(scenario, custom), seed=seed)
    session = Session(
        scenario, seed, policy, sim, LoggingPolicy(build_policy(policy), DecisionLog())
    )
    session.history.append(_history_point(sim))
    return session


def _history_point(sim: HospitalSimulation) -> dict[str, Any]:
    s = observe(sim)
    return {
        "time": s.time,
        "occupied": s.occupied,
        "staffed_capacity": s.staffed_capacity,
        "waiting": len(s.waiting_requests),
        "ed_count": s.ed_count,
        "cumulative_diversions": s.cumulative_diversions,
        "cumulative_off_service": s.cumulative_off_service,
    }


def _get(sid: str) -> Session:
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session (it may have expired)")
    SESSIONS.move_to_end(sid)
    return SESSIONS[sid]


# ---------------------------------------------------------------------- request models


class CustomConditions(BaseModel):
    """User-defined operating conditions for scenario "custom" (multipliers on scenario A).
    Bounds keep a single run's cost predictable on a small server."""

    arrival_multiplier: float = Field(1.0, ge=0.5, le=1.6)
    staff_multiplier: float = Field(1.0, ge=0.6, le=1.4)
    capacity_multiplier: float = Field(1.0, ge=0.6, le=1.4)


class StartRequest(BaseModel):
    """Body of POST /api/simulation/start. scenario "custom" uses ``custom``."""

    scenario: str = "A"
    seed: int = Field(0, ge=0)
    policy: str = "priority"
    custom: CustomConditions | None = None


class StepRequest(BaseModel):
    """Body of POST /api/simulation/{sid}/step (bounded step length)."""

    hours: float = Field(24.0, gt=0, le=MAX_STEP_HOURS)


class WhatIfBody(BaseModel):
    """Body of POST /api/simulation/{sid}/whatif (bounded horizon and futures)."""

    horizon_hours: float = Field(72.0, gt=0, le=MAX_WHATIF_HOURS)
    n_futures: int = Field(20, ge=2, le=MAX_FUTURES)
    arrival_multiplier: float = Field(1.0, gt=0, le=3)
    staff_multiplier: float = Field(1.0, gt=0, le=2)
    capacity_multiplier: float = Field(1.0, gt=0, le=2)
    policies: list[str] = Field(default_factory=lambda: ["priority", "load_balancing"])


# ---------------------------------------------------------------------- app

app = FastAPI(
    title="NeuroTwin API",
    version="0.2.0",
    description="Research simulation of hospital bed allocation. " + SIMULATION_LABEL,
)


@app.middleware("http")
async def _errors_as_json(request, call_next):
    """Turn unexpected server errors into JSON 500 responses *inside* the CORS layer, so the
    browser receives the error (with CORS headers) instead of an opaque network failure."""
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 - last-resort handler, the error is reported
        return JSONResponse(
            status_code=500, content={"detail": f"server error: {type(exc).__name__}: {exc}"}
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness check; also returns the simulation label."""
    return {"status": "ok", "label": SIMULATION_LABEL, "llm_explanations": llm_configured()}


@app.get("/api/policies")
def policies() -> dict[str, Any]:
    """List the policies available for sessions and what-if analysis."""
    return {
        "policies": available_policies(),
        "ppo_agent": str(AGENT_PATH) if AGENT_PATH.exists() else None,
    }


@app.get("/api/scenarios")
def scenarios() -> dict[str, Any]:
    """List the study scenarios and severe-overload scenarios."""
    return {"scenarios": {k: vars(v) for k, v in ALL_SCENARIOS.items()}}


@app.post("/api/simulation/start")
def start(body: StartRequest) -> dict[str, Any]:
    """Create a session: new simulation with the chosen scenario, seed and policy."""
    session = _new_session(body.scenario, body.seed, body.policy, body.custom)
    sid = uuid.uuid4().hex[:12]
    _store(sid, session)
    return {"session_id": sid, **_status(session)}


@app.post("/api/simulation/{sid}/step")
def step(sid: str, body: StepRequest) -> dict[str, Any]:
    """Advance the session by up to 14 days, recording history every 4 hours."""
    session = _get(sid)
    sim = session.sim
    if sim.finished:
        raise HTTPException(409, "simulation has reached its horizon")
    target = min(sim.now + body.hours, sim.config.horizon_hours)
    t0, n0, h0 = time.perf_counter(), len(session.policy.log.records), sim.now
    # Advance in 4-hour chunks so the history has a usable resolution.
    while sim.now < target and not sim.finished:
        run_until(sim, session.policy, until=min(sim.now + 4.0, target))
        session.history.append(_history_point(sim))
    # Work done by this request, so the dashboard can show that results are computed live.
    work = {
        "compute_ms": round(1000 * (time.perf_counter() - t0), 1),
        "decisions_made": len(session.policy.log.records) - n0,
        "simulated_hours": sim.now - h0,
    }
    return {**_status(session), "step": work}


@app.post("/api/simulation/{sid}/reset")
def reset(sid: str) -> dict[str, Any]:
    """Restart the session with its original scenario, seed and policy."""
    old = _get(sid)
    sc = old.sim.scenario
    custom = CustomConditions(
        arrival_multiplier=sc.arrival_multiplier,
        staff_multiplier=sc.staff_multiplier,
        capacity_multiplier=sc.capacity_multiplier,
    )
    _store(sid, _new_session(old.scenario, old.seed, old.policy_name, custom))
    return _status(SESSIONS[sid])


def _status(session: Session) -> dict[str, Any]:
    sim = session.sim
    return {
        "label": SIMULATION_LABEL,
        "scenario": session.scenario,
        "seed": session.seed,
        "policy": session.policy_name,
        "time": sim.now,
        "finished": sim.finished,
        "horizon_hours": sim.config.horizon_hours,
        "warmup_hours": sim.config.warmup_hours,
        "conditions": {
            "arrival_multiplier": sim.scenario.arrival_multiplier,
            "staff_multiplier": sim.scenario.staff_multiplier,
            "capacity_multiplier": sim.scenario.capacity_multiplier,
        },
    }


@app.get("/api/simulation/{sid}/state")
def state(sid: str) -> dict[str, Any]:
    """Observable hospital state (and episode metrics once finished)."""
    session = _get(sid)
    out = {"label": SIMULATION_LABEL, **_status(session), "state": observe(session.sim).to_dict()}
    out["so_far"] = _so_far(session.sim)
    if session.sim.finished:
        out["episode_metrics"] = episode_metrics(session.sim, CostWeights.load())
    return out


def _so_far(sim: HospitalSimulation) -> dict[str, Any] | None:
    """Running results of the measured period (after the warm-up), for live interpretation."""
    hours = min(sim.now, sim.config.horizon_hours) - sim.config.warmup_hours
    if hours <= 0:
        return None
    m = sim.measured
    arrivals = sim.measured_emergency_arrivals
    return {
        "measured_hours": hours,
        "cost_per_day": CostWeights.load().cost(m) / (hours / 24.0),
        "turned_away": m.diversions,
        "turned_away_per_100": 100.0 * m.diversions / arrivals if arrivals else 0.0,
        "wrong_ward": m.off_service_placements,
        "placements": m.placements,
        "mean_wait_h": m.request_hours / max(1, m.placements),
    }


class BenchmarkBody(BaseModel):
    policy: str


@app.post("/api/simulation/{sid}/benchmark")
def benchmark(sid: str, body: BenchmarkBody) -> dict[str, Any]:
    """Run the session's exact run (same conditions and seed, so identical patients) to the end
    with ``policy``, from an empty hospital under the full study protocol, and report its
    measured-period results. Deterministic: the session's own policy reproduces its own run."""
    session = _get(sid)
    t0 = time.perf_counter()
    sim = HospitalSimulation(load_default_config(), session.sim.scenario, seed=session.seed)
    run_until(sim, build_policy(body.policy))
    return {
        "label": SIMULATION_LABEL,
        "policy": body.policy,
        "compute_ms": round(1000 * (time.perf_counter() - t0), 1),
        "metrics": episode_metrics(sim, CostWeights.load()),
    }


@app.get("/api/simulation/{sid}/history")
def history(sid: str) -> dict[str, Any]:
    """Occupancy and queue history recorded at 4-hour resolution."""
    return {"label": SIMULATION_LABEL, "history": _get(sid).history}


@app.get("/api/simulation/{sid}/decisions")
def decisions(sid: str, limit: int = 50) -> dict[str, Any]:
    """Most recent decision-log records (at most 500)."""
    records = _get(sid).policy.log.records
    return {
        "label": SIMULATION_LABEL,
        "total": len(records),
        "decisions": records[-max(1, min(limit, 500)) :],
    }


@app.get("/api/simulation/{sid}/explanations/{decision_id}")
def explanation(sid: str, decision_id: int) -> dict[str, Any]:
    """Post-hoc explanation of one logged decision (template by default; guarded LLM if a key
    is configured and the hourly cap allows)."""
    records = _get(sid).policy.log.records
    if not 0 <= decision_id < len(records):
        raise HTTPException(404, "unknown decision")
    record = records[decision_id]
    key = (sid, decision_id)
    if llm_configured():
        if key in _LLM_CACHE:
            return {"label": SIMULATION_LABEL, "explanation": _LLM_CACHE[key]}
        if _llm_budget_available():
            try:
                result = LLMExplainer(_make_llm_client()).explain(record).as_dict()
            except Exception as exc:  # client construction failure: fall back, never 500
                result = TemplateExplainer().explain(record).as_dict()
                result["verification_issues"] = [f"LLM unavailable ({type(exc).__name__})"]
            _LLM_CACHE[key] = result
            return {"label": SIMULATION_LABEL, "explanation": result}
        result = TemplateExplainer().explain(record).as_dict()
        result["verification_issues"] = ["hourly LLM limit reached; template explanation used"]
        return {"label": SIMULATION_LABEL, "explanation": result}
    return {"label": SIMULATION_LABEL, "explanation": TemplateExplainer().explain(record).as_dict()}


# ---------------------------------------------------------------------- live races

MAX_RACES = 50
MAX_RACE_HOURS = 14 * 24
RACES: "OrderedDict[str, Race]" = OrderedDict()


class RaceStart(BaseModel):
    """Body of POST /api/race/start. start: "busy" (35-day warm-up under Load-Balancing first,
    so all competitors inherit the same realistic state), "empty", or "session" (branch the
    current state of an interactive session)."""

    competitors: list[str] = Field(..., min_length=2, max_length=6)
    scenario: str = "A"
    custom: CustomConditions | None = None
    seed: int = Field(0, ge=0)
    start: str = Field("busy", pattern="^(busy|empty|session)$")
    session_id: str | None = None
    horizon_hours: float = Field(72.0, ge=12.0, le=MAX_RACE_HOURS)


class RaceStep(BaseModel):
    hours: float = Field(6.0, gt=0, le=48.0)


@app.post("/api/race/start")
def race_start(body: RaceStart) -> dict[str, Any]:
    """Branch one hospital state into identical copies, one per competitor (CRN)."""
    if len(set(body.competitors)) != len(body.competitors):
        raise HTTPException(422, "competitors must be distinct")
    factories = {}
    for name in body.competitors:
        build_policy(name)  # validates the name early
        factories[name] = lambda n=name: build_policy(n)
    t0 = time.perf_counter()
    if body.start == "session":
        if not body.session_id:
            raise HTTPException(422, "start=session needs session_id")
        base = _get(body.session_id).sim
        label = "your live run"
    else:
        conditions = _conditions(body.scenario, body.custom)
        base = HospitalSimulation(load_default_config(), conditions, seed=body.seed)
        if body.start == "busy":
            warm = make_policy("load_balancing", load_baseline_params()["load_balancing"])
            run_until(base, warm, until=base.config.warmup_hours)
        label = conditions.name
    if base.finished:
        raise HTTPException(409, "that run has already finished")
    try:
        race = Race(base, factories, body.horizon_hours, seed=body.seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    rid = uuid.uuid4().hex[:12]
    RACES[rid] = race
    RACES.move_to_end(rid)
    while len(RACES) > MAX_RACES:
        RACES.popitem(last=False)
    return {
        "label": SIMULATION_LABEL,
        "race_id": rid,
        "start_label": label,
        "start_time": race.start,
        "setup_ms": round(1000 * (time.perf_counter() - t0), 1),
        **race.standings(),
    }


@app.get("/api/race/{rid}")
def race_state(rid: str) -> dict[str, Any]:
    """Current standings of a race (read-only)."""
    if rid not in RACES:
        raise HTTPException(404, "unknown race (it may have expired)")
    RACES.move_to_end(rid)
    return RACES[rid].standings()


@app.post("/api/race/{rid}/step")
def race_step(rid: str, body: RaceStep) -> dict[str, Any]:
    """Advance every competitor by the same simulated time; report live standings."""
    if rid not in RACES:
        raise HTTPException(404, "unknown race (it may have expired)")
    race = RACES[rid]
    RACES.move_to_end(rid)
    if race.finished:
        raise HTTPException(409, "the race is over")
    t0 = time.perf_counter()
    out = race.step(body.hours)
    return {"compute_ms": round(1000 * (time.perf_counter() - t0), 1), **out}


@app.post("/api/simulation/{sid}/whatif")
def whatif(sid: str, body: WhatIfBody) -> dict[str, Any]:
    """Counterfactual comparison of policies from the session's current state."""
    session = _get(sid)
    if session.sim.finished:
        raise HTTPException(409, "simulation has reached its horizon")
    if not body.policies or len(body.policies) > 6:
        raise HTTPException(422, "choose 1-6 policies")
    factories = {}
    for name in body.policies:
        build_policy(name)  # validates the name early
        factories[name] = lambda n=name: build_policy(n)
    scenario = None
    if (body.arrival_multiplier, body.staff_multiplier, body.capacity_multiplier) != (
        1.0,
        1.0,
        1.0,
    ):
        base = session.sim.scenario
        scenario = ScenarioConfig(
            "what_if",
            base.arrival_multiplier * body.arrival_multiplier,
            base.staff_multiplier * body.staff_multiplier,
            base.capacity_multiplier * body.capacity_multiplier,
        )
    result = WhatIfEngine(factories).evaluate(
        session.sim, WhatIfRequest(body.horizon_hours, body.n_futures, scenario)
    )
    return result.as_dict()


@app.get("/api/experiments/results")
def experiment_results() -> dict[str, Any]:
    """Pre-computed study results (Stages 6-7), served read-only with their provenance."""
    out: dict[str, Any] = {"label": SIMULATION_LABEL}
    for key, rel in {
        "summary": "docs/data/stage6/summary.json",
        "comparisons": "docs/data/stage6/comparisons.json",
        "J_comparisons": "docs/data/stage6/J_comparisons.json",
        "final_summary": "docs/data/final/summary.json",
        "final_comparisons": "docs/data/final/comparisons.json",
        "final_J_comparisons": "docs/data/final/J_comparisons.json",
        "final_robustness": "docs/data/final/robustness.json",
        "robustness": "docs/data/stage7/robustness.json",
        "crn_check": "docs/data/stage7/crn_check.json",
        "structure": "docs/data/stage7/structure.json",
        "ablations": "docs/data/stage7/ablation_comparisons.json",
        "ppo_dr": "docs/data/stage7/ppo_dr_comparisons.json",
        "constrained": "docs/data/stage7/constrained_comparisons.json",
        "shield": "docs/data/stage7/shield_comparisons.json",
        "beat_rule": "docs/data/stage7/bc_ppo_test.json",
        "beat_rule_validation": "docs/data/stage7/bc_ppo_validation.json",
    }.items():
        path = ROOT / rel
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


# Serve the built dashboard (static export) from the same process, if present. Mounted last so
# that every /api route takes precedence.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
