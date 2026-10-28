# NeuroTwin V2

**A hospital digital twin for evaluating reinforcement-learning bed allocation under operational stress.**

**Live demo:** https://neurotwin-swart.vercel.app (the API sleeps when idle; the first load can take about a minute)

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
| Robustness | Degradation index, CRN variance reduction, learned bed-reservation structure, ablations, domain randomisation, constrained PPO | [Stage 7](docs/stage7_analysis_report.md) |
| Platform | What-if engine and live decision-maker race (identical patients for every competitor), post-hoc explanation layer with fact-check guard, FastAPI backend, Next.js dashboard with an animated live hospital | [Stages 8–11](docs/stage8_11_platform_report.md) |
| Inside the agent | For any live RL decision: each voting network's action probabilities, what Load-Balancing would have done in the same state (consulted, never obeyed), and integrated-gradients attributions of the chosen-vs-runner-up logit margin with a completeness check. The critic's value is deliberately not shown (not calibrated in absolute terms) | `neurotwin/rl/introspect.py` |
| Possible futures and reproduction | From any live moment, 20 shared random futures per decision-maker as percentile bands with a paired comparison; and any of the 1,500 runs of the final test re-simulated on the server and compared digit by digit with the published file | `neurotwin/whatif/fan.py`, `POST /api/reproduce` |

## Main findings (simulation, test seeds)

- **PPO matches the best tuned rule, Load-Balancing:** no significant difference in overall cost in any scenario. It **beats** Priority, Trunk-Reservation, FCFS and the exact one-step optimiser in most scenarios, including stress it never trained on.
- The apparent validation advantage over Load-Balancing **disappeared on fresh test seeds**, a case of model-selection optimism.
- **RL agent v2 beats Load-Balancing on normal days.** PPO initialised by imitating Load-Balancing, fine-tuned with RL, five agents voting: **8.5% lower cost** in the training scenario (Holm p = 0.0005) on a fresh, never-used seed block; ties in demand surge, staff shortage and bed loss; 3.5% higher under combined stress (not significant). Pure imitation alone was 58% worse than the rule, so the RL fine-tuning is what makes it beat its teacher. Pre-registered (RESEARCH.md v2.8); selection fixed before the one-time test.
- **RL agent v3 (crisis-trained) beats Load-Balancing in two situations and loses in none:** −5.1% cost on normal days (Holm p = 0.024) and −5.3% under staff shortage (p = 0.0005) on a second fresh seed block; ties under demand surge, bed loss and combined stress. Crisis conditions are part of v3's training (disclosed). Pre-registered as RESEARCH.md v2.10.
- **Learned-policy failure mode:** one of five PPO agents collapsed (holding patients while beds emptied) in 2% of its stress episodes. No rule-based policy ever collapsed.
- **A parameter-free safety guard fixes it:** forbidding HOLD while the ED is full (pre-registered, no retraining) cut collapses from 5 to 0 of 1,250 test episodes. The cost: +4.4% in the staff-shortage scenario; elsewhere no significant change, and the guarded agent still ties Load-Balancing.
- **Under normal load, PPO learned a soft bed-reservation policy.** It holds more often as free beds run out, the threshold shape queueing theory predicts. Under unseen stress that structure disappears.
- **Common random numbers narrowed confidence intervals up to 12.5×.** Without them, PPO's 9% advantage over Priority would have been undetectable.
- **Ablations:** the priority-weighted reward is the component that matters most (uniform weights give +11% cost). No single observation group is necessary.
- **The fixed diversion penalty has no measurable effect,** because diversions are too rare in training to give a learning signal. A pre-registered λ-conditioned constrained PPO with online dual ascent adapts its multiplier as designed, but does **not** reduce diversions under stress. This is a negative result: a constraint cannot steer an outcome the policy never learned to influence.
- **Domain-randomised PPO** has nominally 3–5% lower cost under every stress scenario at no cost in A, but none of these differences is significant after correction.

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

## Deploying for free

The live demo runs on two free tiers:

| Part | Host | What it runs |
|---|---|---|
| API and simulator | Render (free web service, Docker) | this repository's `Dockerfile`; the port comes from `PORT` |
| Dashboard | Vercel (root directory `dashboard`) | the static Next.js export |

1. **Render:** create a *Web Service* from this repository, runtime **Docker**, plan **Free**, health check `/api/health`.
2. **Vercel:** import the repository, set **Root Directory** to `dashboard` and the environment variable
   `NEXT_PUBLIC_API_URL` to the Render URL (no trailing slash).
3. **CORS:** on Render, set `NEUROTWIN_CORS_ORIGINS` to the Vercel URL. To also admit the project's preview
   URLs, set `NEUROTWIN_CORS_ORIGIN_REGEX`, e.g. `https://neurotwin(-[a-z0-9-]+-<team>)?\.vercel\.app`.

The Python backend cannot run on Vercel itself, for three reasons:
- PyTorch exceeds Vercel's function size limit.
- Simulation sessions live in server memory.
- What-if runs exceed its free time limits.

A free Render service sleeps after 15 idle minutes; the first request then takes about a minute.

**Optional LLM explanations at zero cost:** the offline template explainer is the default and needs nothing. To use a free-tier LLM instead, set three **secret environment variables** on the API host:

| Variable | Example (Google Gemini free tier) | Example (Groq free tier) |
|---|---|---|
| `NEUROTWIN_LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://api.groq.com/openai/v1` |
| `NEUROTWIN_LLM_MODEL` | a Gemini model name listed in AI Studio | a model name listed in the Groq console |
| `NEUROTWIN_LLM_API_KEY` | your key from aistudio.google.com | your key from console.groq.com |

Never put keys in code or commit them. Explanations stay fact-checked against the decision log, capped at `NEUROTWIN_LLM_MAX_PER_HOUR` (default 60) and cached. If the provider fails or the cap is reached, the template is used.

## Documents

- [PROJECT_SPEC.md](PROJECT_SPEC.md): goals and scope
- [RESEARCH.md](RESEARCH.md): research questions, verified literature, the locked experimental design, and a changelog of every design change with its reason
- [ARCHITECTURE.md](ARCHITECTURE.md): components and interfaces
- [ROADMAP.md](ROADMAP.md): stages and completion criteria
- `docs/`: one report per stage, with the evidence behind every claim

## Limitations

- Model parameters are plausible assumptions, not estimates from real data.
- The learned look-ahead is about half a day (γ = 0.95 per hour).
- Results hold only for the evaluated scenarios and parameter ranges.
- The LLM explanation backend is implemented and tested, but no provider is connected.
