# Stages 8–11 Report: What-If Engine, Explanation Layer, API, Dashboard and Integration

**Date:** 2026-10-26
**Code:** `neurotwin/whatif/`, `neurotwin/explanation/`, `neurotwin/api/`, `dashboard/`
**Tests:** `tests/test_whatif.py` (4), `tests/test_explanation.py` (5), `tests/test_api.py` (8); full suite 140 passing.

> Everything here is tooling around the research results. No numbers in this report are study findings except the performance benchmark (§5).

## 1. What-if engine (Stage 8)
`WhatIfEngine.evaluate(state, request)` compares policies **from the same hospital state on the same random futures**:
1. Copy the state and set a measurement window [now, now + horizon].
2. For each future seed, branch: keep the present, re-draw future arrivals and staffing and the remaining time of activities in progress (conditional on elapsed time, Stage 3). Operating conditions can also be changed, e.g. "+30% demand from now", via the new `branch(seed, scenario=...)`. Changing conditions mid-run also enables a future PPO-DR with warm starts.
3. Run every policy on every branch, and report per-policy means and paired bootstrap CIs against a reference policy.

**Tests:**
- the analysis is deterministic and doesn't modify the original state;
- the comparison table is built correctly;
- switching to a surge scenario raises arrivals by over 40% on average (bed counts change, and a snapshot round trip preserves them);
- a clearly worse policy is detected with a CI excluding 0.

**Coverage note:** boarding metrics count requests created inside the window. J also includes patients already waiting at the branch point, because it integrates all waiting time.

## 2. Explanation layer (Stage 10)
**Flow (AGENTS.md rule 15):** Policy → Decision → Simulation → **Decision log** → Explanation. The explanation package has **no import path to the simulation or policy code**, which a structural test enforces.

- **Decision log (extended):** for every decision, the logger records:
  - decision id and time;
  - free staffed beds per unit;
  - the waiting queue by priority, and the longest wait;
  - **the feasible actions**;
  - the chosen placement;
  - for PPO, **the action probabilities**.

  Logging is read-only (tested).
- **Template explainer (default, offline):**
  - Describes what the policy's rule does and the logged situation.
  - For PPO it reports the learned action probabilities and explicitly says they are "not a confidence in the decision's quality".
  - Every number in its text is checked against the logged facts (tested over 300 decisions).
  - Reliability indicator: "high (generated only from logged facts)".
- **LLM explainer (interface ready, not connected):**
  - A grounded prompt that says to use only the facts, not judge quality, and give no clinical advice.
  - A **fact-check guard**: any number in the generated text that isn't in the logged facts sets the reliability indicator to "low" and is listed.
  - Provider failure falls back to the template.
  - Tested with fake clients.

  **No provider is configured.** Connecting one sends decision data to an external service and needs the project owner's API key and confirmation (AGENTS.md).

## 3. API (Stage 11 integration)
FastAPI app (`uvicorn neurotwin.api.app:app --port 8000`). Endpoints are listed in ARCHITECTURE.md §12.1. Rules:
- Only observable state is served; simulator snapshots (which contain hidden data) are never exposed.
- Every result-bearing response carries a SIMULATION label.
- Inputs are bounded: step ≤ 14 days, what-if ≤ 50 futures and ≤ 14 days.
- Interactive PPO sessions use **agent 1** by default (configurable with `NEUROTWIN_PPO_AGENT`), because agent 0 has the documented collapse mode (Stage 6 §5). This affects demos only, never reported results.

## 4. Dashboard (Stage 9)
Next.js 15 + React 19 + TypeScript + Recharts 3, with typed interfaces for every API payload.

**Pages:**
- **Overview.**
- **Simulation:** controls; unit occupancy, queue and ED panels; occupancy and queue chart with the warm-up period marked; decision log with an "explain" button; what-if panel with paired CIs.
- **Study results:** means ± 95% CI for every metric, scenario and policy; robustness index; CRN narrowing; learned-reservation table.

A permanent banner states that it is a simulation and not a clinical tool.

**Checks:**
- `tsc --noEmit` clean; production build succeeds.
- Smoke test against the running API:
  - health OK;
  - a PPO session runs 48 h;
  - `/`, `/simulation` and `/experiments` return HTTP 200.
- An interactive check in a browser is still recommended. The automated checks cover building, serving and the API calls, but not clicking through the UI.

## 5. Performance (Stage 11)
From the recorded runtimes of the 2,500 Stage 6 test episodes (49 simulated days each, single core):

| Policy | Median s/episode (A) | Median s/episode (stress) | Slowest episode | Minimum speed vs real time |
|---|---|---|---|---|
| FCFS | 0.14 | 0.24 | 0.44 s | 9.6 million × |
| Priority | 0.15 | 0.27 | 0.43 s | 9.9 million × |
| Load-Balancing | 0.14 | 0.30 | 0.46 s | 9.2 million × |
| Trunk-Reservation | 0.15 | 0.27 | 0.45 s | 9.4 million × |
| MIP | 4.04 | 5.88 | 7.62 s | 555,000 × |
| PPO | 1.05 | 2.12 | 3.87 s | 1.09 million × |

The PROJECT_SPEC target of 10× real time is exceeded by more than four orders of magnitude.

**Load testing** was not performed. The API is a single-user, local research tool, and its expensive operations (what-if, long steps) are bounded instead.

## 6. Known issues
- **The experiment runner hit Windows' memory commit limit.** Running 16 training processes at once ("paging file is too small") killed workers. Training now uses at most 8 parallel processes; evaluation, which uses less memory per process, ran fine with 18.
- **Deprecation warning:** Starlette's TestClient warns that `httpx` support is deprecated (a warning only; tests pass).
- **Sessions are in memory** and are lost when the API restarts. This is acceptable for a research tool.
