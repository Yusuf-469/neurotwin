---
title: NeuroTwin
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Hospital digital twin comparing RL, optimisation and rules
---

# NeuroTwin V2

**A hospital digital twin for evaluating reinforcement-learning bed allocation under operational stress.**

NeuroTwin simulates patient flow through an emergency department, medical and surgical wards and an ICU. Capacity is limited by nurse staffing, and the model includes off-service placement, ICU step-down and deterioration. It compares a reinforcement-learning policy (PPO, **implemented from scratch** in PyTorch) with rule-based policies and an exactly solved optimisation model. Comparisons use normal conditions and stress conditions the agent never saw in training, with pre-registered statistics.

> **Simulation only.** NeuroTwin is a final-year research project. The model is not calibrated to any real hospital, all results are simulation results, and it is not a clinical or decision-support tool.

## What is in it

| Component | Highlights | Report |
|---|---|---|
| Discrete-event simulator | Written from scratch; Common Random Numbers by design; validated against M/M/c and M/M/c/K (15/15 inside 95% CI; detects 5% input errors); Welch warm-up | [Stage 2](docs/stage2_validation_report.md) |
| Digital twin | Exact snapshot/restore; what-if branching with remaining stays re-drawn given elapsed time (inspection-paradox safe); warm-start pool | [Stage 3](docs/stage3_digital_twin_report.md) |
| Baselines | FCFS, Priority, Load-Balancing, Trunk-Reservation, and an exact myopic MIP checked against LP and Hungarian oracles; tuned on validation seeds with the RL cost | [Stage 4](docs/stage4_baselines_report.md) |
| RL | Gymnasium env (7 masked actions, semi-Markov reward); in-house PPO with γ^Δt GAE, checked against stable-baselines3 and MaskablePPO | [Stage 5](docs/stage5_rl_report.md) |
| Evaluation | 2,500 test episodes; hierarchical paired bootstrap; Holm over 100 comparisons; protocol committed before the test run | [Stage 6](docs/stage6_evaluation_report.md) |
| Robustness | Degradation index, CRN variance reduction, learned bed-reservation structure, ablations | Stage 7 (in progress) |
| Platform | What-if engine, post-hoc explanation layer with fact-check guard, FastAPI backend, Next.js dashboard | [Stages 8–11](docs/stage8_11_platform_report.md) |

## Main findings (simulation, test seeds)

- **PPO matches the best tuned rule, Load-Balancing:** no significant difference in overall cost in any scenario. It **beats** Priority, Trunk-Reservation, FCFS and the exact one-step optimiser in most scenarios, including stress it never trained on.
- The apparent validation advantage over Load-Balancing **disappeared on fresh test seeds**, a case of model-selection optimism.
- **Learned-policy failure mode:** one of five PPO agents collapsed (holding patients while beds emptied) in 2% of its stress episodes. No rule-based policy ever collapsed.
- **Under normal load, PPO learned a soft bed-reservation policy.** It holds more often as free beds run out, the threshold shape queueing theory predicts. Under unseen stress that structure disappears.
- **Common random numbers narrowed confidence intervals up to 12.5×.** Without them, PPO's 9% advantage over Priority would have been undetectable.

Full numbers, CIs and caveats are in the stage reports. One protocol amendment (censoring-inclusive boarding time) is disclosed in Stage 6 §2.

## Repository layout

```
neurotwin/
  simulation/     discrete-event engine, config, RNG streams, analytical validation
  digital_twin/   observable state, history, snapshots, warm-start pool
  policies/       baselines, exact MIP, PPO policy wrapper, decision log
  rl/             Gymnasium env, actions, features, in-house PPO (rl/ppo/)
  experiments/    seeds, objective, tuning, training, evaluation, statistics, robustness
  whatif/         counterfactual engine
  explanation/    post-hoc explanations (template + guarded LLM interface)
  api/            FastAPI backend
dashboard/        Next.js + TypeScript dashboard
config/           all model, policy, reward and PPO parameters (JSON)
docs/             stage reports and committed result data (docs/data/)
tests/            pytest suite (140 tests)
```

## Setup

Python 3.10+ (developed on 3.11.9, Windows 11), Node.js 20+ for the dashboard.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest                 # full suite, ~3-4 min
.\.venv\Scripts\python.exe -m pytest -m "not slow"   # fast subset
```

## Reproducing the study

All runs are deterministic given their seeds. Times are for a 20-core laptop CPU.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m neurotwin.simulation.validation.report --heavy           # analytical validation (~1 min)
& $py -m neurotwin.simulation.validation.warmup --days 84         # Welch warm-up (~1 min)
& $py -m neurotwin.experiments.load                               # scenario offered loads
& $py -m neurotwin.experiments.tuning                             # baseline tuning, validation seeds (~5 min)
& $py -m neurotwin.rl.ppo.validate cartpole                       # PPO gate 2 (~1 min)
& $py -m neurotwin.rl.ppo.validate hospital                       # PPO gate 3 (~4 min)
& $py -m neurotwin.experiments.train_ppo --name final_A --agents 5 --workers 5   # ~30 min
& $py -m neurotwin.experiments.evaluate --split test --out docs/data/stage6       # ~4 min
& $py -m neurotwin.experiments.robustness
& $py -m neurotwin.experiments.crn_check
& $py -m neurotwin.experiments.structure
& $py -m neurotwin.experiments.ablations train --workers 8      # ~80 min
& $py -m neurotwin.experiments.ablations evaluate
```
Seed ranges (training, validation, test) are disjoint and asserted in code (`neurotwin/experiments/seeds.py`). On Windows, keep parallel training at 8 workers or fewer: PyTorch's DLLs can exhaust the memory commit limit.

## Running the platform

**Docker (same as the hosted version, one URL):**
```powershell
docker build -t neurotwin .
docker run -p 7860:7860 neurotwin            # open http://localhost:7860  (demo: /simulation/?demo=1)
```

**Development (hot reload):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn neurotwin.api.app:app --port 8000   # API
cd dashboard; npm install; npm run dev                                     # http://localhost:3000
```
The five final PPO agents (`results/ppo/final_A/agent_*.pt`, 52 KB each) are committed. The
dashboard's PPO policy uses agent 1; agent 0 has the documented collapse mode.

## Deploying free on Hugging Face Spaces
1. Create a free account at huggingface.co and click **New Space**: SDK **Docker**, hardware **CPU basic (free)**, visibility **Public**.
2. Push this repository to the Space (Hugging Face shows the exact `git remote add` command; use an access token with *write* permission as the password).
3. The Space builds the `Dockerfile` and serves the app on its URL.

**Optional LLM explanations at zero cost:** the offline template explainer is the default and needs nothing. To use a free-tier LLM instead, add three **secrets** under the Space's *Settings → Variables and secrets*:

| Secret | Example (Google Gemini free tier) | Example (Groq free tier) |
|---|---|---|
| `NEUROTWIN_LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://api.groq.com/openai/v1` |
| `NEUROTWIN_LLM_MODEL` | a Gemini model name listed in AI Studio | a model name listed in the Groq console |
| `NEUROTWIN_LLM_API_KEY` | your key from aistudio.google.com | your key from console.groq.com |

Never put keys in code or commit them. Explanations stay fact-checked against the decision log, capped at `NEUROTWIN_LLM_MAX_PER_HOUR` (default 60) and cached. If the provider fails or the cap is reached, the template is used.

### Optional: dashboard on Vercel (API stays on the Space)
The Python backend cannot run on Vercel, for three reasons:
- PyTorch exceeds Vercel's function size limit.
- Simulation sessions live in server memory.
- What-if runs exceed its free time limits.

Only the static dashboard goes there:
1. On vercel.com, choose **Add New → Project**, import the GitHub repository and set **Root Directory** to `dashboard`.
2. Add the environment variable `NEXT_PUBLIC_API_URL` = your Space URL, e.g. `https://<user>-neurotwin.hf.space` (no trailing slash), then deploy.
3. In the Space's *Settings → Variables and secrets*, add the variable `NEUROTWIN_CORS_ORIGINS` = your Vercel URL, e.g. `https://neurotwin.vercel.app`, so the browser may call the API cross-site. To also admit the project's preview and Git-branch URLs, set `NEUROTWIN_CORS_ORIGIN_REGEX`, e.g. `https://neurotwin(-[a-z0-9-]+-<team>)?\.vercel\.app`.

A free Space sleeps after inactivity. The first request then takes a minute while it wakes, and the dashboard shows the error until the API responds.

## Documents

- [PROJECT_SPEC.md](PROJECT_SPEC.md): goals and scope
- [RESEARCH.md](RESEARCH.md): research questions, verified literature, the locked experimental design, and a changelog of every design change with its reason
- [ARCHITECTURE.md](ARCHITECTURE.md): components and interfaces
- [ROADMAP.md](ROADMAP.md): stages and completion criteria
- [AGENTS.md](AGENTS.md): rules for AI coding agents working on this repository
- `docs/`: one report per stage, with the evidence behind every claim

## Limitations

- Model parameters are plausible assumptions, not estimates from real data.
- The learned look-ahead is about half a day (γ = 0.95 per hour).
- Results hold only for the evaluated scenarios and parameter ranges.
- The LLM explanation backend is implemented and tested, but no provider is connected.
