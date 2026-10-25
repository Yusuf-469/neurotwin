# Stage 5 Report: RL Environment and In-House PPO

**Date:** 2026-10-26
**Code:** `neurotwin/rl/` (environment, actions, features, `ppo/`), `neurotwin/policies/ppo_policy.py`, `neurotwin/experiments/train_ppo.py`
**Config:** `config/ppo_default.json`
**Tests:** 113 passing in total (`tests/test_rl.py` adds 19, including a slow CartPole learning test)

> All performance numbers here are **validation-seed** results from scenario A, used for model selection, and are therefore optimistically biased. They are not the study's results. Test seeds (200000–200049) have not been used. All numbers are simulation results.

## 1. Environment

- **Decision steps:** one step is one atomic decision. A placement keeps the epoch open (Δt = 0) while further placements are feasible. HOLD, or running out of feasible placements, advances time. `info["dt_hours"]` makes the process semi-Markov.
- **Actions:** 7 masked actions (RESEARCH.md §6.3). The mask equals exactly the set of feasible placements (tested at every decision of a stressed episode). Choosing a masked action raises an error instead of silently falling back.
- **Observation:** 21 features (§6.5), with normalisers fixed from scenario A so the encoding cannot reveal the scenario. The space is declared as [−1, ∞) because counts and waits are unbounded.
- **Reward:** −(α ΔW + β ΔD + κ ΔO) × 0.1. The sum of rewards over an episode equals −0.1 × J **exactly** (tested to 1e-9).
- **Equivalence:** replaying an environment episode's decisions through `run_episode` reproduces it identically, so the environment adds no hidden mechanics.
- **Training starts:** a warm-start pool of 200 distinct states from training seeds. Evaluation always uses the full protocol from an empty start.

## 2. In-house PPO

Written from scratch in PyTorch (`neurotwin/rl/ppo/`):
- separate tanh actor and critic networks with orthogonal initialisation;
- logit-level masking (−1e9);
- **GAE with per-step discount γ^Δt** (semi-Markov), with time-limit truncation bootstrapped from V(s_T);
- clipped surrogate and clipped value loss;
- entropy bonus, per-minibatch advantage normalisation, global gradient clipping;
- Adam with ε = 1e-5, linear learning-rate decay, and running observation normalisation (frozen at evaluation).

Each implementation detail is commented with its rationale and tagged "[37 details]" after Huang et al. (2022).

## 3. Validation gates (RESEARCH.md §6.18)

| Gate | Result |
|---|---|
| 1. Unit tests | **Pass.** GAE vs hand computation with Δt ∈ {0, 1, 2}; no leakage across episode boundaries; clipped loss equals the plain surrogate inside the trust region and has zero gradient outside it; masked actions have probability and gradient exactly 0; the trainer passes γ^Δt to GAE, which a spy test confirms fails if Δt is ignored |
| 2. CartPole-v1 vs stable-baselines3 (200k steps × 5 seeds, matched hyperparameters) | **Pass.** Final return 446.5 [341.9, 551.1] vs 369.5 [237.4, 501.6]; p = 0.24; learning curves track each other |
| 3. Hospital env vs sb3-contrib MaskablePPO (300k steps × 5 seeds, per-step discounting) | **Pass.** −425.6 [−491.1, −360.1] vs −441.1 [−516.0, −366.3]; p = 0.68; curves overlap |

Gate 3 also shows that the reference implementation learns as slowly as ours at those settings. The difficulty described next is a property of the problem, not of our code.

## 4. Making PPO learn: a low signal-to-noise problem

**First attempt (v2.0 settings: γ = 0.99/h, λ = 0.95, entropy 0.01, 300k steps):** J = 329.8, against 168.7 for Priority. The value function learned (explained variance 0.01 → 0.77), but the policy barely moved (KL → 0, clip fraction 0, entropy ≈ 0.95). In deterministic evaluation it chose HOLD 42% of the time, which is very costly in scenario A.

**Diagnosis:** a single placement changes total waiting cost very little compared with the randomness of future arrivals. Long-horizon advantage estimates are therefore dominated by noise, the policy gradients cancel out, and the entropy bonus keeps the policy near uniform. This is a known difficulty in queueing control with RL, which is why Sun, Shi and Dai use queueing-informed value approximations.

**Development sweep (600k steps, 10 validation seeds):**

| Setting | J/day | HOLD rate |
|---|---|---|
| γ 0.99, λ 0.95, entropy 0.01 | 312.8 | 42% |
| entropy 0 | 226.0 | 34% |
| γ 0.95, λ 0.9, entropy 0.001 | 193.7 | 16% |
| γ 0.90, λ 0.9, entropy 0.001 | 184.7 | 11% |
| larger batch (16 × 1024) | 511.8 | 48% |

Shorter horizons learn; larger batches (fewer updates) do not.

**Selection run (3M steps, 2 agents per γ, 20 validation seeds):**

| γ per hour | Agent mean J |
|---|---|
| 0.80 | 148.9, 152.4 |
| 0.90 | 151.4, 153.0 |
| **0.95 (selected: lowest mean, rule fixed in advance)** | 147.9, 151.0 |

The differences between γ values are within noise. γ = 0.95/h has a half-life of about 13.5 h, so the agent's look-ahead spans hours, not days. RQ1 is interpreted with this in mind.

## 5. Final configuration: 5 independent agents

`config/ppo_default.json`: γ = 0.95/h (semi-Markov), λ = 0.9, entropy 0.001, 3M steps, 8 envs × 256 steps, 10 epochs, 8 minibatches, lr 3e-4 annealed, clip 0.2, value clip 0.2, observation normalisation on, 64-64 tanh. Each agent trains in about 26 minutes on one CPU core; the five ran in parallel.

**Validation results (scenario A, 20 seeds, full protocol, deterministic policy):**

| Agent | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| J/day | 147.9 | 151.0 | 150.0 | 154.0 | 153.2 | **151.2** |
| Paired Δ vs Priority (168.7) | −20.9 | −17.7 | −18.8 | −14.7 | −15.5 | −17.5 |

For reference, tuned Load-Balancing scores 164.4 and the MIP 179.8. In the selection runs, paired 95% CIs against Priority excluded zero for 5 of 6 agents; against Load-Balancing, for 3 of 6.

**Reproducibility check:** agents 0 and 1 of the final run have the same seeds and configuration as the two γ = 0.95 selection agents, and they reproduced the same validation J (147.9, 151.0). Training is deterministic given the seed.

**Honest reading:** on validation data, PPO beats every tuned baseline in mean cost, consistently against Priority and probably against Load-Balancing. Whether this holds on fresh test seeds, and under the unseen stress scenarios, is the Stage 6–7 question. Model selection on 20 validation seeds introduces some optimistic bias, and the test protocol exists to remove it.

## 6. Limitations and open items

- The look-ahead horizon is limited by the γ needed for learning (discussed above).
- The warm-start pool reflects the greedy policy's steady state (Stage 3 report).
- Training uses scenario A only. PPO-DR and PPO-Lagrangian are optional extensions, and warm starts combined with scenario sampling are not implemented yet.
- `PPOPolicy` evaluates the deterministic argmax policy. Stochastic evaluation is not used for the headline results.
