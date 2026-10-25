# NeuroTwin V2

A research simulation of hospital bed allocation. It compares a reinforcement-learning policy (PPO, implemented in-house) against heuristic, bed-reservation and exactly solved optimisation baselines, in a discrete-event hospital model validated against queueing theory.

> **Simulation only.** NeuroTwin is a final-year research project. It is not a clinical system, is not calibrated to any real hospital, and makes no claim about real-world effectiveness. All results are simulation results.

## Status

| Stage | Status |
|---|---|
| 1. Research design | Complete: [RESEARCH.md](RESEARCH.md) v2.1, [design review](docs/stage1_design_review.md) |
| 2. Discrete-event simulator | Complete: [validation report](docs/stage2_validation_report.md) |
| 3. Digital twin state | Complete: [report](docs/stage3_digital_twin_report.md) |
| 4. Baseline policies | Complete: [report](docs/stage4_baselines_report.md) |
| 5. RL environment and in-house PPO | Complete: [report](docs/stage5_rl_report.md) |
| 6. Experimental evaluation (test seeds) | Complete: [report](docs/stage6_evaluation_report.md) |
| 7. Robustness and generalisation analysis | Next |
| 8–12 | Planned: [ROADMAP.md](ROADMAP.md) |

## Setup (Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest                     # full suite (about 1 minute)
.\.venv\Scripts\python.exe -m pytest -m "not slow"       # fast subset
```

Reproduce the validation evidence:

```powershell
.\.venv\Scripts\python.exe -m neurotwin.simulation.validation.report --heavy
.\.venv\Scripts\python.exe -m neurotwin.simulation.validation.warmup --days 84 --reps 30
.\.venv\Scripts\python.exe -m neurotwin.experiments.load        # offered load per scenario
.\.venv\Scripts\python.exe -m neurotwin.experiments.tuning      # baseline tuning (validation seeds)
```

## Documents

- [PROJECT_SPEC.md](PROJECT_SPEC.md): goals and scope
- [RESEARCH.md](RESEARCH.md): research questions, literature, and the locked experimental design
- [ARCHITECTURE.md](ARCHITECTURE.md): components and interfaces
- [ROADMAP.md](ROADMAP.md): stages and completion criteria
- [AGENTS.md](AGENTS.md): rules for AI coding agents working on this repository
