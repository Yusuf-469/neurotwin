# NeuroTwin V2 - Research Specification

**Version:** 2.3 (2026-10-26). See [Changelog](#changelog) at the end for every change from v1 and its justification.

## 1. Research Problem

Hospital resource allocation is a critical operational challenge that affects patient wait times, resource utilization, and overall healthcare delivery efficiency. A central and well-studied instance is **inpatient bed allocation**: when an admitted patient's primary (specialty) ward is full, the hospital must choose between making the patient wait (boarding in the emergency department) and placing them in a non-primary ward ("off-service placement" or "overflow"), which is associated with worse care coordination and longer stays. A second, less visible decision is **bed reservation**: whether to keep a bed free for a likely higher-priority arrival instead of filling it immediately.

Traditional allocation methods rely on static rules or one-step (myopic) optimization, which cannot reason about the future consequences of filling or reserving a bed. Whether an adaptive reinforcement-learning policy that reasons about the future can outperform these methods, and whether it stays robust when operating conditions change, is the research question.

## 2. Research Question

**Primary Research Question:**
In a validated discrete-event hospital simulation, can a reinforcement-learning (PPO) bed-allocation policy achieve lower patient boarding time and diversion than heuristic, bed-reservation, and myopic-optimal baselines, and does that advantage persist under operating conditions not seen during training?

**Secondary Research Questions:**
1. **Look-ahead value:** Does learned look-ahead (PPO) outperform optimal *myopic* allocation (the MIP baseline), and under which conditions does the gap grow or shrink?
2. **Robustness:** How does each policy's performance degrade as operational stress increases (demand surge, staff shortage, capacity loss, combined)?
3. **Generalization:** How well does a policy trained only under normal conditions perform on unseen stress scenarios?
4. **Policy structure:** Does the learned policy exhibit threshold (bed-reservation) structure similar to the classical trunk-reservation policies that are optimal in simpler queueing models (Miller, 1969)? How does it compare to a tuned trunk-reservation baseline?
5. **State features:** Which state features contribute most to learned performance (ablations)?
6. **Constraint handling (extension):** Does treating diversion as a constraint (PPO-Lagrangian) maintain a diversion budget under stress better than a fixed reward penalty?
7. **Trade-offs:** What trade-offs exist between boarding time, tail boarding time, diversion, and off-service placement?
8. **Training diversity:** Does training on randomised *mild* variations of normal conditions (domain randomisation) make PPO more robust to *unseen, more severe* stress than training on Normal only? (§6.23)

## 3. Proposed Hypothesis

**Primary Hypothesis (to be tested experimentally):**
A PPO policy trained in the simulated environment will reduce mean boarding time compared with heuristic and myopic-optimal baselines under the training (Normal) conditions, and will retain a statistically significant, though smaller, advantage under at least some unseen stress scenarios.

**Null Hypothesis:**
There is no statistically significant difference in the primary metrics between the PPO policy and the best baseline in any evaluated scenario.

**Structure Hypothesis (RQ4):**
The learned policy will hold low-priority patients when free staffed beds fall below a small threshold, i.e. it will approximate a trunk-reservation policy.

*Note added in v2.3, without changing the hypothesis:* the tuned trunk-reservation baseline chose zero reserve in scenario A. This is consistent with the cμ rule for delay systems (§6.8), so theory predicts holding behaviour mainly where diversion is frequent. The hypothesis is kept as pre-registered and will be tested as stated. Reporting will distinguish holding in A from holding under stress.

**Important:**
- All hypotheses must be tested through controlled experiments
- No results may be fabricated
- All findings must be reported honestly, including negative results (e.g. PPO not beating the tuned trunk-reservation baseline is a valid and reportable outcome)
- Claims must be limited to the simulated environment

## 4. Research Gap

**What has been done (see §5 for citations):**
- PPO has been applied to inpatient overflow decisions in a time-periodic queueing model and matched or beat approximate dynamic programming (Sun, Shi & Dai, MSOM) — **the closest prior work to NeuroTwin**
- Approximate dynamic programming for inpatient overflow (Dai & Shi, 2019)
- Hospital-wide bed assignment with adaptive robust optimization and ML forecasts (Bertsimas & Pauphilet, 2024)
- DDQN for admission-time bed allocation on a single-site retrospective simulation (Kim et al., 2026)
- DES + actor-critic RL for ED patient prioritization (Zattoni, 2026)
- Deep Q-learning for fair ventilator allocation (Li et al., 2025)
- DES validated as a digital twin for bed management (Oakley, Onggo & Worthington, 2020)
- Digital twins with formal models and SMT solvers for bed-bay planning (Sieve et al., 2025)
- Trunk reservation is optimal for multi-class loss systems with ordered rewards (Miller, 1969)

**What is less explored:**
- **Generalization of learned bed-allocation policies to unseen stress conditions.** Prior RL work typically trains and evaluates under the same (or calibrated) conditions.
- **Comparison against a tuned bed-reservation (trunk-reservation) baseline.** RL is usually compared to FCFS-like rules, which are weak; a tuned threshold policy is the classical strong baseline.
- **Separating the value of look-ahead from the value of optimization** by comparing against an exactly solved myopic assignment problem.
- **Inspection of learned policy structure** against queueing theory.
- **Simulator correctness evidence.** DES studies often report validation weakly (Mohiuddin et al., 2017); analytical validation in reduced configurations is rarely reported alongside RL results.

**NeuroTwin's Specific Contribution:**
A controlled, reproducible experimental framework that:
1. Implements a DES hospital simulation (two specialty general wards + ICU, off-service placement, step-down and escalation) **validated against closed-form queueing results** in reduced configurations
2. Implements PPO **in-house** (validated against a reference implementation), with action masking and semi-Markov discounting
3. Compares PPO against FCFS, Priority, Load-Balancing, **tuned Trunk-Reservation**, and an **exactly solved myopic assignment** baseline under Common Random Numbers
4. Tests generalization: train under Normal conditions only, evaluate under unseen stress
5. Measures robustness via a signed degradation metric with paired bootstrap CIs
6. Analyses learned policy structure against trunk-reservation theory
7. Supports counterfactual what-if evaluation from cloned states

**Claims we should NOT make:**
- "First work to apply RL / PPO to hospital bed allocation" (false — Sun, Shi & Dai; Kim et al.)
- "First work to combine digital twins and RL" (not defensible)
- "Our simulation accurately represents real hospital operations"
- Any claim of clinical or real-world effectiveness

**Claims we CAN make (if supported by results):**
- "We evaluate generalization of a PPO bed-allocation policy to unseen stress conditions"
- "We compare against tuned threshold and exactly solved myopic baselines"
- "Our simulator reproduces Erlang B/C results in reduced configurations"
- "All comparisons use Common Random Numbers and paired statistics"

## 5. Literature Review

### 5.0 Citation verification status

Every source below was checked on 2026-10-25 against a publisher, arXiv, or repository record. Figures are quoted only where they appear in the verified abstract. Five v1 citations could not be verified and were removed (see §5.8).

| # | Source | Status |
|---|---|---|
| 1 | Li et al. (2025), IJCAI / arXiv 2309.08560 | Verified |
| 2 | Kim et al. (2026), BMC Med. Inform. Decis. Mak. | Verified (abstract only) |
| 4 | Zattoni (2026), PhD thesis, Univ. Bologna | Verified (front matter / abstract) |
| 5 | Bertsimas & Pauphilet (2024), Management Science | Verified |
| 6 | Ajmi et al. (2023), IEEE SMC | Verified (abstract) |
| 7 | Ordu et al. (2023), Healthcare Analytics | Verified (abstract) |
| 8* | Mohiuddin et al. (2017), BMJ Open | Verified (replaces unverifiable v1 Paper 8) |
| 9 | Oakley, Onggo & Worthington (2020), HCMS | Verified |
| 10 | Sieve et al. (2025), EPTCS 418 / arXiv 2505.06287 | Verified |
| 12 | Fujimoto et al. (2024), arXiv 2402.03590 | Verified |
| 13 | Kirk et al. (2023), JAIR | Verified |
| 14 | Schäfer et al. (2023), HCMS | Verified |
| 17 | Sun, Shi & Dai, MSOM / arXiv 2410.13767 | Verified (new) |
| 18 | Dai & Shi (2019), MSOM | Verified (new) |
| 19 | Miller (1969), Management Science | Verified (new) |

### 5.1 RL for Hospital Resource Allocation

**Paper 17: Inpatient Overflow Management with Proximal Policy Optimization** *(closest prior work)*
- **Authors**: J. Sun, P. Shi, J. G. Dai
- **Venue**: Manufacturing & Service Operations Management (DOI 10.1287/msom.2025.0215); arXiv 2410.13767
- **Problem**: Whether arriving patients wait for their primary unit or overflow to an alternative unit
- **Method**: PPO in a time-periodic, long-run average-cost setting; "atomic actions" that decompose multi-patient routing into sequential assignments; partially shared policy network; queueing-informed value function approximation
- **Findings (abstract)**: Matches or outperforms benchmarks including ADP, on cases with up to 20 patient classes and 20 wards
- **Relevance**: Establishes PPO as credible for overflow. NeuroTwin adopts the sequential (atomic) decision idea (§6.3) and differs in its focus: generalization under unseen stress, trunk-reservation and exact-myopic baselines, and validated simulation

**Paper 1: Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources**
- **Authors**: Y. Li, C. Mao, K. Huang, H. Wang, Z. Yu, M. Wang, Y. Luo
- **Venue**: IJCAI 2025 (AI and Social Good track); arXiv 2309.08560
- **Problem**: Ventilator allocation during shortage
- **Method**: Transformer-based deep Q-network modelling patients' disease courses and interactions
- **Findings (abstract)**: Reduced excess deaths and more equitable allocation than severity- and comorbidity-based protocols
- **Relevance**: Multi-objective reward design; fairness considerations

**Paper 2: Reinforcement learning-based decision support for admission-time inpatient bed allocation**
- **Authors**: T. Kim, M. Son, F. Rismanchian, et al.
- **Venue**: BMC Medical Informatics and Decision Making (2026), DOI 10.1186/s12911-026-03758-w
- **Method**: Double DQN over a retrospective DES built from 40,302 admissions across 21 departments at one tertiary hospital in South Korea
- **Findings (abstract)**: DDQN improved throughput, overall waiting time and short-stay queue control vs FCFS under the primary condition; **no policy dominated every outcome**
- **Limitations (stated by authors)**: Single-site retrospective simulation; prospective and external validation required
- **Relevance**: Bed-allocation framing; motivates multi-metric reporting rather than a single score

**Paper 4: A discrete event simulation and deep reinforcement learning framework for queue management and patient prioritization in an emergency department**
- **Author**: L. Zattoni (PhD thesis, University of Bologna, 2026; amsdottorato eprint 12809)
- **Method**: DES validated as a digital twin of a real ED; prioritization as an MDP; two actor-critic agents; a DL model estimating time to end of pathway
- **Relevance**: DES + RL integration pattern; DES as digital twin

**Paper 6: Self-adaptive agents based on reinforcement learning to optimize patient scheduling in emergency department**
- **Authors**: F. Ajmi, S. Ben Othman, F. Ajmi, H. Zgaya-Biau, J.-M. Renard, G. Smith, S. Hammadi
- **Venue**: IEEE International Conference on Systems, Man, and Cybernetics (SMC), 2023
- **Method**: Multi-agent framework (ABOS) with metaheuristics; agents adapt via RL; real data from Lille University Hospital ED
- **Relevance**: Alternative (multi-agent) decomposition; not adopted in V1

### 5.2 Queueing and Operations Research Foundations

**Paper 18: Inpatient Overflow: An Approximate Dynamic Programming Approach**
- **Authors**: J. G. Dai, P. Shi
- **Venue**: MSOM 21(4):894–911 (2019), DOI 10.1287/msom.2018.0730
- **Method**: Multiclass, multipool queueing model; overflow as an average-cost MDP; ADP with fluid control and single-pool approximations
- **Relevance**: Defines the overflow problem NeuroTwin models; motivates off-service placement as an explicit decision with a cost

**Paper 5: Hospital-Wide Inpatient Flow Optimization**
- **Authors**: D. Bertsimas, J. Pauphilet
- **Venue**: Management Science 70(7):4893–4911 (2024), DOI 10.1287/mnsc.2023.4933
- **Method**: Multistage adaptive robust optimization with ML predictions, unifying bed assignment across the hospital
- **Findings (published abstract)**: In simulations calibrated to a 600-bed academic medical center, off-service placement reduced by 24% on average; ED and post-anaesthesia boarding delays reduced by 35% and 18%; adaptive linear decision rules outperform static assignment
- **Relevance**: Off-service placement and boarding as the key metrics; optimization-based baseline perspective

**Paper 19: A Queueing Reward System with Several Customer Classes**
- **Author**: B. L. Miller
- **Venue**: Management Science 16(3):234–245 (1969), DOI 10.1287/mnsc.16.3.234
- **Result**: For a multi-server loss system with ordered class rewards and class-independent service rates, a trunk-reservation (threshold) admission policy is optimal
- **Relevance**: Theoretical basis for the Trunk-Reservation baseline and for the policy-structure analysis (RQ4)

**Paper 14: Combining machine learning and optimization for the operational patient-bed assignment problem**
- **Authors**: F. Schäfer, M. Walther, D. G. Grimm, A. Hübner
- **Venue**: Health Care Management Science 26(4):785–806 (2023), DOI 10.1007/s10729-023-09652-5
- **Method**: ML forecasting of emergency admissions + hyper-heuristic (pilot method, greedy look-ahead)
- **Findings (abstract)**: Up to 17% lower RMSE for arrival forecasts vs historical averages; up to 5.3% objective improvement over a literature benchmark
- **Relevance**: Optimization baseline perspective; forecasting extension (PROJECT_SPEC §5.7)

### 5.3 Discrete-Event Simulation in Healthcare

**Paper 7: A comprehensive and integrated hospital decision support system for efficient and effective healthcare services delivery using discrete event simulation**
- **Authors**: M. Ordu, E. Demir, C. Tofallis, M. M. Gunal
- **Venue**: Healthcare Analytics 4 (2023) 100248
- **Method**: Whole-hospital DES with specialty-level forecasting, using Hospital Episode Statistics
- **Relevance**: Whole-hospital DES scope; scenario analysis methodology

**Paper 8\*: Patient flow within UK emergency departments: a systematic review of the use of computer simulation modelling methods**
- **Authors**: S. Mohiuddin et al.
- **Venue**: BMJ Open 7(5):e015007 (2017)
- **Findings (abstract)**: 21 studies (19 DES); evidence base small; **reporting of modelling assumptions was poor**
- **Relevance**: Motivates NeuroTwin's explicit assumption register and analytical validation (§6.17)

**Paper 9: Symbiotic simulation for the operational management of inpatient beds: model development and validation using Δ-method**
- **Authors**: D. Oakley, B. S. Onggo, D. Worthington
- **Venue**: Health Care Management Science 23(1):153–169 (2020), DOI 10.1007/s10729-019-09485-1
- **Method**: Simulation loading current hospital state at run time; conditional LOS distributions; Δ-Method validation for stochastic symbiotic models
- **Relevance**: Digital twin pattern (state loading = NeuroTwin's what-if cloning); validation methodology

### 5.4 Digital Twins in Healthcare

**Paper 10: BedreFlyt: Improving Patient Flows through Hospital Wards with Digital Twins**
- **Authors**: Sieve, Kobialka, Slaughter, Schlatte, Johnsen, Tapia Tarifa (University of Oslo)
- **Venue**: ASQAP 2025, EPTCS 418, pp. 1–15; arXiv 2505.06287
- **Method**: Executable formal models (ABS), knowledge graphs, Z3 SMT solver for bed-bay allocation
- **Relevance**: What-if scenarios in digital twins; constraint-based optimization

### 5.5 RL Generalization and Robustness

**Paper 12: Assessing the Impact of Distribution Shift on Reinforcement Learning Performance**
- **Authors**: T. Fujimoto, J. Suetterlein, S. Chatterjee, A. Ganguly
- **Venue**: arXiv 2402.03590 (NeurIPS 2023 RegML workshop)
- **Key Contribution**: Point estimates and convergence plots can hide overfitting; proposes robustness-oriented evaluation under distribution shift
- **Relevance**: Justifies reporting full distributions and degradation, not single scores

**Paper 13: A Survey of Zero-shot Generalisation in Deep Reinforcement Learning**
- **Authors**: R. Kirk, A. Zhang, E. Grefenstette, T. Rocktäschel
- **Venue**: JAIR 76:201–264 (2023), DOI 10.1613/jair.1.14174
- **Relevance**: Formalism for train/test context distributions; NeuroTwin's scenarios A–E are controllable factors of variation in this sense

### 5.6 RL Methodology (implementation references)

- Schulman et al. (2017), Proximal Policy Optimization Algorithms, arXiv 1707.06347
- Schulman et al. (2016), High-Dimensional Continuous Control Using Generalized Advantage Estimation, ICLR
- Huang et al. (2022), The 37 Implementation Details of Proximal Policy Optimization, ICLR Blog Track
- Henderson et al. (2018), Deep Reinforcement Learning That Matters, AAAI
- Ng, Harada & Russell (1999), Policy Invariance under Reward Transformations, ICML
- Tessler, Mankowitz & Mannor (2019), Reward Constrained Policy Optimization, ICLR
- Ray, Achiam & Amodei (2019), Benchmarking Safe Exploration in Deep Reinforcement Learning, OpenAI technical report

### 5.7 What-If / Counterfactual Analysis

NeuroTwin's what-if analysis is **simulation-based counterfactual evaluation**: the same cloned state and the same random-number streams are run under different policies. It is not causal inference from observational data, and no causal-inference literature is cited in support of it. The closest verified methodological precedent is the run-time state loading of Oakley, Onggo & Worthington (2020).

### 5.8 Citations removed in v2 (could not be verified)

| v1 entry | Reason |
|---|---|
| Paper 3: "Adaptive Hospital Bed Allocation Using Deep RL: A DDPG-Based Framework" (Scitepress, 2025) | No record found |
| Paper 8: "DES for ED modelling: a systematic review of validation methods" (2022) | No record found; replaced by Mohiuddin et al. (2017) |
| Paper 11: "GE Healthcare Digital Twin" (20–40% ED wait reduction) | Not a publication; the 20–40% figure traces to a simulation vendor's marketing material, not GE or peer review |
| Paper 15: "From prediction to intervention: causal digital twins..." (J. Transl. Med., 2026) | No record found |
| Paper 16: "Quantum-Causal Clinical Operations Engine" (NES, 2026) | No record found |

---

## 6. Experimental Design (v2, locked)

## 6.1 Hospital Model

### Units
- **ED**: arrival, triage, ED treatment (requires a doctor), then a **bed request** if admitted
- **Medical ward (MED)** and **Surgical ward (SURG)**: general beds; each patient has a primary specialty
- **ICU**: critical-care beds

### Staffed capacity
A bed can be filled only if it is **staffed**: `staffed_k(t) = min(physical_beds_k, nurses_k(t) × patients_per_nurse_k)`. Default ratios are 1:5 for general wards and 1:2 for ICU (**assumption**, configurable). Staff shortage scenarios reduce `nurses_k(t)` and therefore effective capacity.

### Patient pathways
1. **General pathway**: ED → bed request → MED or SURG (primary or off-service) → discharge
2. **ICU pathway**: ED → bed request → ICU → *step-down ready* → general bed (primary or off-service) → discharge
3. **Escalation**: a general-ward patient may *deteriorate* and then requires an ICU bed

### Bed requests
Every patient waiting for a bed is a **request** with a priority and a wait clock:

| Request type | Origin | Eligible units | Priority | Wait clock starts |
|---|---|---|---|---|
| ED general | ED (or elective arrival) | Primary ward, or other ward (off-service) | Triage priority 1–5 | Bed request time |
| ED ICU | ED | ICU | Triage priority 1–5 | Bed request time |
| Step-down | ICU patient ready to leave ICU | Primary ward, or other ward (off-service) | 3 | Step-down-ready time |
| Escalation | Deteriorated general-ward patient | ICU | 1 | Deterioration time |

Priority 1 is most urgent everywhere (consistent with ESI-style triage).

### Diversion
An arriving emergency patient is **diverted** (turned away) if the number of patients in the ED (in treatment + boarding) is at the ED capacity `Q_max`. Diversion is the "overflow" outcome of v1, renamed to avoid confusion with off-service placement.

### Off-service placement
Placing a general-pathway patient in the non-primary ward is allowed and recorded. It multiplies that patient's remaining general LOS by `m_os` (default 1.1, **assumption**; sensitivity analysis at 1.0, 1.1, 1.25).

### Discharge
Discharge is **automatic** when a patient's LOS ends. It is not a decision: in this model delaying a discharge never helps any objective and only blocks beds (removed from the v1 action space; see changelog).

## 6.2 Decision Problem

### Decision points (epochs)
A decision epoch occurs when at least one **non-HOLD action is feasible** at the current simulation time, triggered by:
1. A new bed request (arrival after ED treatment, step-down ready, deterioration)
2. A staffed bed becoming free (discharge, transfer, shift change increasing staffed capacity)
3. The periodic review timer (every 4 simulated hours) if feasible actions exist

If no non-HOLD action is feasible, no decision is requested and the simulation advances.

### Sequential (atomic) decisions within an epoch
Following the atomic-action idea of Sun, Shi & Dai, multiple placements at the same instant are made **one at a time**. After each non-HOLD action, the policy is queried again at the same timestamp (Δt = 0) with the updated state. The epoch ends when the policy chooses HOLD or no non-HOLD action remains feasible. This keeps the action space small and avoids patient × bed combinatorics.

### Information available to every policy at decision time
- Current state features (§6.5) and the full list of waiting requests (type, priority, elapsed wait, primary specialty)
- Staffed free capacity per unit
- **Not available**: future arrivals, realised LOS, future deterioration times, future discharges

## 6.3 Action Space (7 actions)

| Action | Name | Effect | Masked when |
|---|---|---|---|
| 0 | HOLD | End the epoch; leave all requests waiting | Never |
| 1 | ADMIT_PRIMARY_PRIORITY | Highest-priority (ties: longest wait) ED general request whose primary ward has a free staffed bed → primary ward | No such request |
| 2 | ADMIT_PRIMARY_LONGEST_WAIT | Longest-waiting ED general request whose primary ward has a free staffed bed → primary ward | No such request |
| 3 | ADMIT_OFF_SERVICE | Highest-priority ED general request whose primary ward is full → other ward, if it has a free staffed bed | No such request |
| 4 | ADMIT_ICU | Highest-priority ED ICU request → ICU | No ICU request or no free staffed ICU bed |
| 5 | STEP_DOWN | Longest-waiting step-down request → primary ward if free, else other ward (off-service) | No step-down request or no free staffed general bed |
| 6 | ESCALATE | Longest-waiting escalation request → ICU | No escalation request or no free staffed ICU bed |

**Why this design:**
1. **Every action is a real trade-off.** Actions 1–3 and 5 compete for general beds; 4 and 6 compete for ICU beds; HOLD expresses bed reservation; 3 expresses overflow.
2. **Ordering choice is learnable.** Actions 1 vs 2 let the agent choose between priority and wait-time ordering instead of having one hard-coded.
3. **Compact and maskable.** 7 discrete actions with exact feasibility masks; no combinatorial explosion.
4. **Comparable.** Every baseline acts through the same request/placement primitives.

Masking is applied at the logit level (invalid logits set to −∞ before softmax); see §6.18.

## 6.4 Decision Timing

Hybrid: event-driven epochs (§6.2) plus a 4-hour periodic review. Because epochs are irregular, the decision process is a **semi-Markov decision process**: each step has a duration `Δt_t = τ_{t+1} − τ_t ≥ 0`, which enters the reward (§6.8) and the discount (§6.18).

All policies are queried at exactly the same epochs under CRN, which preserves fair comparison.

## 6.5 State Representation (21 features)

| # | Feature | Normalization |
|---|---|---|
| 1 | MED staffed occupancy | [0, 1] |
| 2 | SURG staffed occupancy | [0, 1] |
| 3 | ICU staffed occupancy | [0, 1] |
| 4 | ED general requests waiting | / Q_max |
| 5 | ED ICU requests waiting | / Q_max |
| 6 | Step-down requests waiting | / ICU beds |
| 7 | Escalation requests waiting | / ICU beds |
| 8 | Nurse availability ratio (on shift / nominal) | [0, 1] |
| 9 | Doctor availability ratio (on shift / nominal) | [0, 1] |
| 10 | Hour of day (sin) | [−1, 1] |
| 11 | Hour of day (cos) | [−1, 1] |
| 12 | Day of week (sin) | [−1, 1] |
| 13 | Day of week (cos) | [−1, 1] |
| 14 | Arrival rate, 1 h window | / nominal peak rate |
| 15 | Arrival rate, 4 h window | / nominal peak rate |
| 16 | Discharge rate, 4 h window | / nominal peak rate |
| 17 | High-priority (P1–2) requests waiting | / Q_max |
| 18 | Low-priority (P3–5) requests waiting | / Q_max |
| 19 | Mean elapsed wait of waiting requests | / W_max |
| 20 | Diversions in the last 24 h | / Q_max |
| 21 | Off-service patients currently in beds | / general beds |

v1 listed "Average wait time in ED" (feature 14) and "Average ED waiting time" (feature 19) as two features with the same meaning; v2 removes the duplicate.

**Optional experimental features** (extension, PROJECT_SPEC §5.7): arrival forecast for the next 4 h; per-ward expected discharges in the next 24 h (from LOS *distributions*, never realised LOS).

## 6.6 Patient Generation and Stochastic Model

### Arrival processes
- **Emergency**: non-homogeneous Poisson, `λ_E(t) = base_E × f_hour(t) × f_day(t)`, generated by thinning
- **Elective**: non-homogeneous Poisson restricted to weekday daytime, `λ_L(t) = base_L × g(t)`
- Default mix: 70% emergency / 30% elective by expected daily volume (**assumption**)

### Attributes drawn at arrival (all from dedicated RNG streams, §6.7)
- Triage priority: P1–2: 15%, P3: 50%, P4–5: 35% (emergency); electives P4–5
- Primary specialty: MED/SURG (emergency 60/40; elective 30/70; **assumption**)
- Pathway: ICU-first probability by priority (P1: 0.6, P2: 0.3, P3: 0.05, P4–5: 0; **assumption**)
- ED treatment time (emergency only): lognormal
- ICU LOS and general LOS: lognormal (means within the ranges in §6.20)
- Deterioration time: exponential with hazard `h_det` per general-ward day; the patient deteriorates if this time falls before the end of their general LOS

All attributes are drawn **at arrival** and stored hidden from policies. This is what makes CRN hold even though policies change *when* things happen (§6.7).

### Assumptions register
Every parameter marked **assumption** above is a modelling choice, not an empirical estimate. Each is listed with its default and range in §6.20, and the two most influential (`m_os`, `h_det`) receive sensitivity analysis.

## 6.7 Baseline Fairness and Common Random Numbers

### Information equivalence
All policies (FCFS, Priority, Load-Balancing, Trunk-Reservation, MIP, PPO) observe the same information (§6.2) and act through the same placement primitives at the same epochs.

### Common Random Numbers (CRN)
All policies experience identical stochastic scenarios for a given seed.

**Implementation requirement (critical for CRN to actually hold):** each stochastic source uses its **own RNG stream**, derived from the scenario seed with `numpy.random.SeedSequence.spawn`:

| Stream | Draws |
|---|---|
| `arrivals_emergency` | Emergency arrival times |
| `arrivals_elective` | Elective arrival times |
| `patient_attributes` | Priority, specialty, pathway, ED treatment time, LOS, deterioration time (drawn at arrival) |
| `staffing` | Unplanned staff absences (staff-shortage scenarios) |

Because patient attributes are drawn at arrival, the *n*-th patient has identical attributes under every policy. A policy that places a patient earlier or later does not shift any other random draw. A single shared RNG would silently break CRN, because different policies would consume draws in different orders.

### Tuning budget fairness
Every baseline with a tunable parameter (Trunk-Reservation thresholds, off-service wait threshold `W_os`, load-balancing threshold `θ_lb`) is tuned on **scenario A, validation seeds only**, which is the same data PPO's hyperparameters are tuned on.

## 6.8 Baseline Policies

All baselines process requests through the same primitives as the PPO actions. Escalation requests (priority 1) are included in the priority ordering.

| Policy | Rule |
|---|---|
| **FCFS** | Serve requests in order of elapsed wait. Primary placement when a staffed bed is free; off-service once the request has waited ≥ `W_os` (default 4 h) and the other ward has a free bed |
| **Priority** | As FCFS, but order by priority (1 first), ties by elapsed wait |
| **Load-Balancing** | Priority order; place in primary ward unless its occupancy exceeds the other ward's by more than `θ_lb`, then off-service |
| **Trunk-Reservation** | Priority order; a P3–5 request is placed in unit *k* only if free staffed beds in *k* exceed the reservation level `r_k`; P1–2 always placed when a bed is free. `r_MED, r_SURG, r_ICU` tuned by grid search (§6.7). Grounded in Miller (1969) |
| **MIP (myopic)** | Exact solution of the one-epoch assignment problem in §6.9 |
| **PPO** | In-house PPO (§6.18) |
| **PPO-Lagrangian** | Extension (§6.19) |

**Tuning results (scenario A, validation seeds 100000–100019, objective J per measured day; `config/baselines_tuned.json`, `docs/stage4_baselines_report.md`):**

| Policy | Tuned parameters | J (95% CI over seeds) |
|---|---|---|
| FCFS | W_os = 4 h | 276.8 [164.8, 388.9] |
| Priority | W_os = 12 h | 168.7 [116.7, 220.8] |
| Load-Balancing | θ_lb = 0.10 | 164.4 [109.4, 219.3] |
| Trunk-Reservation | r = (0, 0, 0), W_os = 12 h | 168.7 (identical to Priority) |
| MIP | κ_MIP = 20, Δ = 4 h, β_b = 10 | 179.8 [114.6, 244.9] |

**Trunk reservation tunes to zero reserve in scenario A.** This is consistent with queueing theory rather than a failure of the baseline. Miller (1969) shows that reservation is optimal in *loss* systems, where turned-away customers are gone. In a *delay* system with class-dependent waiting costs and equal service rates, the cμ rule (serve the highest weighted class first, never idle) is optimal, and that is the Priority policy. Diversion is rare in A (0.1 per 100), so A behaves as a delay system. Reservation can only pay where losses (diversions) are common. Per protocol, the baseline keeps its A-tuned parameters on the stress scenarios, and the PPO policy-structure analysis (§6.20) tests whether learned holding behaviour appears where theory predicts it can help.

## 6.9 Myopic Assignment Formulation (MIP baseline)

At each decision epoch, the MIP decides all placements for the current epoch at once (it does not use atomic steps).

### Sets and data
- `R`: waiting requests; `K = {MED, SURG, ICU}`; `E_i ⊆ K` eligible units of request *i*
- `cap_k`: free staffed beds in unit *k*
- `e_i`: elapsed wait of request *i* (data); `p(i)`: priority; `ω_p`: priority weight
- `Δ`: expected time to next review (4 h)
- `W_max`: wait-threshold for a breach; `β_b`: breach cost; `κ`: off-service cost
- `os_{ik} = 1` if unit *k* is off-service for request *i*

### Decision variables
- `x_{ik} ∈ {0, 1}`: request *i* placed in unit *k* (for *k* ∈ `E_i`)

### Objective
Maximise the waiting cost avoided now, minus off-service cost:

```
max  Σ_i Σ_{k∈E_i} ( c_i − κ · os_{ik} ) · x_{ik}

c_i = ω_{p(i)} · (e_i + Δ) + β_b · 1[ e_i + Δ > W_max ]
```

`c_i` is the cost that request *i* incurs if it is left waiting until the next review. Because `e_i` is data at decision time, the breach indicator is a constant and no big-M constraints are needed.

### Constraints
```
1. Σ_{k∈E_i} x_{ik} ≤ 1            ∀ i ∈ R      (each request placed at most once)
2. Σ_{i: k∈E_i} x_{ik} ≤ cap_k     ∀ k ∈ K      (staffed capacity)
3. x_{ik} = 0                       k ∉ E_i      (eligibility)
```

### Structure and solution method
- Constraints 1–2 define a **transportation problem** (bipartite requests → units with integer capacities). Its constraint matrix is **totally unimodular**, so the LP relaxation has an integral optimum.
- Solver: `scipy.optimize.milp` (HiGHS) with integrality constraints. SciPy is already a dependency of scikit-learn, so no new dependency is added.
- **Test oracles:** (a) the LP relaxation (`scipy.optimize.linprog`) must return the same objective value with integral `x`; (b) expanding each unit into `cap_k` identical slots gives an assignment problem whose `scipy.optimize.linear_sum_assignment` optimum must match.
- Staffing is folded into `cap_k` because nurses are ward-specific in V1. If a float-nurse pool is added later, it introduces a coupling side constraint that breaks total unimodularity, and the MILP solver is then required. The code keeps the MILP path for that reason.

### Tuned off-service cost κ_MIP (v2.3)
`κ` in the MIP objective is a **tuning parameter** (κ_MIP ∈ {2, 5, 10, 20, 40, 80}), not fixed to the reward's κ = 2. Reason: the myopic model cannot see that off-service stays are longer (`m_os`) and consume future capacity, so with κ_MIP = 2 it overflowed almost every waiting patient, which scored J = 213. A tuned surrogate cost gives the MIP the same tuning budget as the heuristics, and it tuned to κ_MIP = 20 (J = 180). It still trails Priority and Load-Balancing (J ≈ 164–169): **an exactly solved one-step optimiser, even with a tuned surrogate, does not beat simple rules here**. This is the gap RQ1 asks whether learned look-ahead can close.

### Why myopic, and what the comparison means
The MIP is the **best possible one-epoch decision** given current information. Since every `c_i > 0`, it always fills a free primary bed when there is a request for it. It therefore **never reserves beds**, and it chooses off-service only when the waiting cost exceeds `κ`. PPO, by contrast, optimises discounted future cost. **PPO vs MIP therefore isolates the value of look-ahead**, which is RQ1. (v1 justified myopia by claiming "RL is also myopic", which is incorrect; see changelog.)

## 6.10 Reward Function

### Formulation: time-integrated cost per decision step
For decision step *t* spanning `[τ_t, τ_{t+1})`:

```
R_t = − [ α · ∫_{τ_t}^{τ_{t+1}} Σ_{i ∈ R(τ)} ω_{p(i)} dτ      (priority-weighted request-hours of waiting)
        + β · D_t                                             (diversions in the interval)
        + κ · O_t ]                                           (off-service placements made at step t)
```

### Why this form
1. **Aligned with the primary metric.** With uniform `ω`, the integral term summed over an episode equals the total request waiting time, so the return measures exactly what is evaluated. This is the same accounting identity behind Little's law.
2. **Penalises sustained congestion.** A queue that stays long keeps costing reward every hour it persists.
3. **Handles irregular epochs.** Cost accrues in proportion to elapsed time, and atomic steps with Δt = 0 accrue no waiting cost.
4. **Few, interpretable weights.** Three cost terms, each tied to a reported metric.

### Terms removed from v1, with justification
| v1 term | Reason removed |
|---|---|
| Difference form `C(s_t) − C(s_{t+1})` | Level terms (queue length, average wait) only reward *changes*; summed over an episode the return telescopes to the terminal state's cost, plus a (1−γ)-weighted integral. It is not aligned with the time-averaged metrics |
| Throughput reward (ε) | In a stable system throughput = arrivals − diversions, so it double-counts the diversion penalty and is exploitable |
| Transfer penalty (δ) | In v2 every transfer (step-down, escalation) is clinically indicated; penalising it would penalise correct care |
| Utilisation variance (ζ) | Ward imbalance is now an explicit decision (off-service) with its own cost κ |

### Reward-hacking review
| Risk | Mitigation |
|---|---|
| Holding everyone to "save" off-service cost | Waiting cost grows every hour; diversions rise as the ED fills |
| Filling beds with low-priority patients | Priority weights ω; diversion penalty when high-priority arrivals are later blocked |
| Ignoring step-down requests | They are requests with waiting cost (ω_3); they also block ICU beds for escalations |
| Off-service everything | κ per placement; LOS multiplier `m_os` lengthens bed occupancy |

### Weights (tuned on scenario A validation seeds only)
- α = 1.0 (per request-hour), ω = (P1: 4, P2: 3, P3: 2, P4: 1, P5: 1)
- β = 20 (per diversion; ≈ 20 request-hours)
- κ = 2 (per off-service placement)

These are initial values. Tuning ranges are in §6.20; the final values and the tuning log are reported.

## 6.11 Episode Design

- **Horizon**: 49 simulated days (1,176 h)
- **Start**: empty hospital, nominal staffing (the first shift's absences are drawn at t = 0)
- **Warm-up**: **35 days**, excluded from metrics. Rule: Welch's method with a **one-week** smoothing window, 30 replications, suggestion = first day reaching 98% of the final-third level, **rounded up to whole weeks**. At the v2.3 calibrated load (ρ = 0.93) the suggestion was 29 days, giving 35. (At the v2.1 load of 0.88 it was 21 days; heavier load relaxes more slowly.) The v2.0 default of 1 day was far too short. See `docs/stage2_validation_report.md` §3 and `docs/stage4_baselines_report.md` §2.
- **Measurement**: days 36–49 (14 days, starting on the same weekday every episode)
- **Termination**: truncation at 49 days. For PPO, the final step bootstraps from the value function (truncation, not a true terminal state).
- **RL training episodes start from a warm-start pool** (Stage 3) of **at least 200 distinct states** (near saturation the state is very persistent: a 10-state pool misrepresented the stationary distribution, with mean boarding 5.9 h vs 3.9 h). The states come from the end of the warm-up of training-seed runs under a reference greedy policy, and each is branched with the episode's own training seed. In-progress stays are re-drawn conditional on elapsed time. Warm-started episodes reproduce full-episode statistics (Welch t-test p ≈ 0.85), whereas a naive restart of in-progress stays nearly triples boarding, an inspection-paradox bias (`docs/stage3_digital_twin_report.md`). **Validation and test episodes do not use the pool**: they run the full 35-day protocol from an empty start.

**Why empty start + warm-up:** an empty start is reproducible and identical for all policies, but it introduces **initialisation bias** (an empty hospital is atypical). The warm-up removes that bias from metrics.

## 6.12 Train / Validation / Test Protocol

### Seed ranges (disjoint, fixed in configuration)
| Purpose | Environment seeds | Scenarios |
|---|---|---|
| Training | 0 – 99,999 (fresh seed per episode) | A only |
| Validation (tuning, early stopping) | 100,000 – 100,019 | A only |
| Test | 200,000 – 200,049 | A, B, C, D, E |

### PPO training seeds
K = 5 independent PPO agents per configuration (network initialisation and minibatch order), training seeds 0–4. Results are reported per agent and aggregated (Henderson et al., 2018).

### Rules
1. All tuning (PPO hyperparameters, reward weights, baseline thresholds) uses **validation seeds on scenario A only**.
2. Hyperparameters are locked before any test run.
3. Each policy is evaluated on the test set once; the run logs are kept.
4. Scenario A test results use held-out test seeds. This is an in-distribution test, not training on test data, and it is required for the degradation metric (§6.15).

v1 tuned on "A + mild B/C/D perturbations". That leaks the test distribution into tuning and is removed.

## 6.13 Multi-Seed Protocol

- **Development**: 5–10 seeds; not reported as results
- **Final evaluation**: 50 test seeds per (policy, scenario); K = 5 agents for PPO
- **Paired design**: for each seed, every policy faces the same patients and staffing (CRN)

### Approximate power
For a paired comparison with n = 50, a two-sided test at α = 0.05 has about 80% power to detect a standardised paired effect `d_z ≈ 0.4`. After Holm correction over 100 comparisons, the per-test threshold is stricter, so the detectable effect is larger. **The power analysis is repeated with pilot variance estimates** before final evaluation, and the seed count is increased if needed.

## 6.14 Statistical Methodology

### Primary: paired bootstrap
1. For each test seed *s*, compute `d_s = M(PPO, s) − M(baseline, s)`. For PPO, `M(PPO, s)` is the mean over the K agents (**hierarchical bootstrap**: resample agents, then seeds).
2. Resample B = 10,000 times; report the mean difference and its 95% percentile CI.
3. Two-sided bootstrap p-value for H0: μ_d = 0.

### Sensitivity analysis
The Wilcoxon signed-rank test is reported alongside for every primary comparison. It is **not** chosen conditionally on a normality pre-test, since choosing the test from the same data inflates error rates.

### Multiple comparisons
Primary family: PPO vs 5 baselines × 5 scenarios × 4 primary metrics = **100 comparisons**, Holm–Bonferroni corrected. Secondary metrics are reported descriptively.

### Effect sizes
`d_z` (mean paired difference / SD of differences) and Cliff's delta, always reported next to p-values.

### Reporting
Actual p-values, CIs, effect sizes, per-seed results, and per-agent results for PPO.

## 6.15 Robustness Metric

For metric *m*, policy *p*, stress scenario *S*:

```
Degradation_m(p, S) = s_m · ( M(p, S) − M(p, A) ) / M(p, A) × 100%
```

with `s_m = +1` for lower-is-better metrics and `−1` for higher-is-better metrics, so **positive always means worse**. CRN pairs the seeds across scenarios, so degradation is computed per seed and reported with a paired bootstrap CI.

**Robustness Index** of policy *p*: the mean degradation over stress scenarios B–E and the primary metrics (lower = more robust), with a bootstrap CI.

## 6.16 Ablation Experiments (PPO only, retrained per ablation, K = 5)

| ID | Change | Question |
|---|---|---|
| A | Full state, full reward | Reference |
| B | Remove demand history (14–16) | Do demand trends enable anticipation? |
| C | Remove temporal features (10–13) | Do daily/weekly cycles matter? |
| D | Remove staffing features (8–9) | Does explicit staffing state help under staff shortage? |
| E | Remove priority aggregates (17–19) | Does priority mix information enable reservation behaviour? |
| F1 | β = 0 | Is diversion avoided without an explicit penalty? |
| F2 | κ = 0 | How much off-service placement does the agent use when it is free? |
| F3 | Uniform ω | Effect of priority weighting on tail and high-priority waits |
| G | Remove step-down/escalation state (6–7) | Does the agent use ICU-flow information? |

## 6.17 Simulator Validation (analytical)

Before any policy result is reported, the DES engine is checked against closed-form results in reduced configurations. The checks are kept as permanent regression tests.

| Check | Reduced configuration | Analytical result |
|---|---|---|
| M/M/c delay | Homogeneous Poisson λ, exponential LOS (mean 1/μ), one unit with c beds, unlimited FIFO queue, no staffing limit, no diversion | Erlang C probability of waiting; `W_q = C(c, λ/μ) / (cμ − λ)` |
| M/M/c/K finite waiting room | As above, with ED capacity `Q_max`; arrivals finding the ED full are diverted, so K = c + Q_max | M/M/c/K blocking probability, W_q and P(wait). Erlang B (K = c) cannot be configured because the ED always has a waiting room; the formulas are tested to reduce exactly to Erlang B at K = c and Erlang C as K → ∞ |
| Little's law | Full model, any policy | `L = λ_eff · W` for each queue over the measurement window |
| NHPP arrivals | Arrival generator alone | Counts per hour-of-week bin match `∫ λ(t) dt` (chi-square goodness of fit) |
| CRN integrity | Two different policies, same seed | Identical arrival times and patient attributes for every patient index |

**Acceptance:** for each check, the analytical value lies within the 95% CI of the simulated estimate over ≥ 30 seeds, at loads ρ = 0.5, 0.8, 0.95.

**Result (2026-10-25):** 15/15 analytical values inside the 95% CI. A mutation test shows that a 5% error in the arrival rate or a 10% error in mean LOS is detected (`docs/stage2_validation_report.md`). The automated regression tests use 99% CIs to keep the family-wise false-alarm rate low with fixed seeds.

**What this does and does not show:** it shows the engine implements its stochastic model correctly. It does **not** show the model matches any real hospital.

## 6.18 PPO Implementation and Validation

PPO is implemented in-house in PyTorch (ARCHITECTURE.md §8.5).

**Algorithmic details:**
- Actor-critic MLP, orthogonal initialisation; logit-level action masking
- **Semi-Markov discounting:** step discount `γ_t = γ^{Δt_t / 1h}`. GAE uses per-step `γ_t`: `δ_t = R_t + γ_t V(s_{t+1}) − V(s_t)`, `Â_t = δ_t + γ_t λ Â_{t+1}`
- Clipped surrogate objective, clipped value loss, entropy bonus, advantage normalisation, gradient-norm clipping, learning-rate annealing, running observation normalisation (frozen at evaluation)
- Truncation at the episode horizon bootstraps from `V(s_T)`

**Validation gates (all must pass before hospital results are reported):**
1. Unit tests: GAE matches a hand-computed example (including Δt ≠ 1h); clipped loss equals the unclipped loss inside [1−ε, 1+ε]; masked actions have probability exactly 0
2. CartPole-v1: learning curves over 5 seeds comparable to stable-baselines3 PPO with matched hyperparameters
3. Hospital environment: in-house vs reference PPO (sb3-contrib MaskablePPO, test-only dependency) learning curves with overlapping CIs, using Δt-independent discounting for the comparison (the reference does not support SMDP discounting)

## 6.19 Extension: Constrained PPO (PPO-Lagrangian)

Constrained MDP: minimise expected waiting and off-service cost subject to `E[diversions per day] ≤ c_div`. The β term is replaced by a Lagrange multiplier λ ≥ 0 updated by dual ascent on the measured constraint violation (Tessler et al., 2019; Ray et al., 2019).

**Experiment:** PPO with fixed β vs PPO-Lagrangian with a budget `c_div` equal to fixed-β PPO's diversion rate on scenario A validation seeds. Compare constraint satisfaction and waiting cost on stress scenarios B–E.

**Hypothesis:** a fixed β tuned on Normal conditions under-penalises diversion under stress, whereas the constraint formulation keeps diversions closer to budget.

Run only after the core experiment matrix is complete.

## 6.20 Policy-Structure Analysis (RQ4)

1. On test-seed trajectories, record PPO's action probabilities as a function of (free staffed beds in unit *k*, priority mix of waiting requests).
2. For low-priority-only queues, estimate P(HOLD | free beds = b) for b = 0, 1, 2, … and test for a monotone threshold shape (isotonic regression fit vs unconstrained fit).
3. Compare the estimated thresholds with the tuned Trunk-Reservation levels `r_k`.
4. Report honestly whether threshold structure is present, absent, or state-dependent.

This is a direct, non-LLM interpretability analysis grounded in queueing theory.

## 6.21 What-If Experiment Design

For a state S₀, each future seed *f* defines a branch `S₀.branch(f)`: the present is kept, and all future randomness comes from independent branch-namespace streams. Not-yet-started durations are re-drawn, and durations already in progress are re-drawn **conditional on elapsed time** (because the true remaining time is unknown at S₀). Every policy runs on the same branches (CRN), and outcomes are compared over 50 future seeds per S₀ with paired CIs. All results come from actual simulation runs and are labelled as simulation results.

## 6.23 Extension: Domain-Randomised Training (RQ8)

**Policy variant PPO-DR:** identical to PPO (same architecture, hyperparameters, K = 5 agents), except that each training episode samples scenario multipliers uniformly from a **mild range strictly inside the Normal neighbourhood**:

| Multiplier | PPO-DR training range | Nearest test scenario value |
|---|---|---|
| Arrivals | 0.95 – 1.04 | E: 1.08, B: 1.10 |
| Staff | 0.97 – 1.0 | E: 0.90, C: 0.88 |
| Physical capacity | 0.97 – 1.0 | E: 0.90, D: 0.88 |

**Leakage control (two conditions, both checked):**
1. Every test scenario B–E lies strictly outside the training range on every perturbed axis.
2. The **worst corner** of the training range (arrivals 1.04, staff and capacity 0.97) has offered load 0.981 (general) and 0.854 (ICU), **below the mildest test scenario** (D: 1.020 / 0.878).

Condition 2 is needed because axis-wise milder ranges can still combine into a test-level load. An earlier candidate range (arrivals up to 1.05, staff and capacity down to 0.95) passed condition 1 but had a corner load of 1.046, heavier than B and D. RQ8 therefore measures **extrapolation** to more severe stress, not interpolation. Tuning still uses scenario A validation seeds only. This is the "experiment that explicitly studies that setting" exception in AGENTS.md rule 12, and it is documented as such.

**Comparison:** PPO vs PPO-DR on A–E using the same paired test seeds, with degradation (§6.15) and the Robustness Index. A possible outcome is that PPO-DR is more robust but slightly worse on A. That trade-off is reported either way.

## 6.24 Variance-Reduction Check for CRN

CRN is claimed to tighten policy comparisons; this is measured, not assumed.

1. For each baseline vs PPO comparison on scenario A, compute the 95% CI of the mean difference two ways:
   - **paired (CRN):** both policies on the same 50 test seeds
   - **independent:** each policy on its own disjoint set of 50 seeds (an extra 50-seed block per policy: 300,000 – 300,049, used only here)
2. Report the CI width ratio and the variance reduction `1 − Var(paired diff) / (Var(M_1) + Var(M_2))`, which equals the induced positive correlation between the two policies' metrics.
3. Report it honestly even if the reduction is small for some metrics.

## 6.22 Experiment Matrix

### Primary
| Policy | A (held-out seeds) | B | C | D | E |
|---|---|---|---|---|---|
| FCFS | ✓ | ✓ | ✓ | ✓ | ✓ |
| Priority | ✓ | ✓ | ✓ | ✓ | ✓ |
| Load-Balancing | ✓ | ✓ | ✓ | ✓ | ✓ |
| Trunk-Reservation | ✓ | ✓ | ✓ | ✓ | ✓ |
| MIP (myopic) | ✓ | ✓ | ✓ | ✓ | ✓ |
| PPO (K = 5) | ✓ | ✓ | ✓ | ✓ | ✓ |

PPO is **trained** on scenario A training seeds only.

### Extended
| Experiment | Policies |
|---|---|
| Stress sweep (arrival ×1.1 … ×1.6; staff ×0.9 … ×0.6; capacity ×0.9 … ×0.5) | All |
| Severe overload: v2.0 scenario values (ρ 1.19–1.77) | All |
| Ablations A–G | PPO |
| Sensitivity: `m_os` ∈ {1.0, 1.1, 1.25}; `h_det` ×{0.5, 1, 2} | All |
| PPO-Lagrangian | PPO, PPO-Lagrangian |
| Domain-randomised training (§6.23) | PPO, PPO-DR |
| CRN variance reduction (§6.24) | All baselines vs PPO, scenario A |
| Policy structure | PPO vs Trunk-Reservation |
| What-if | All |

### Scenarios (v2.3: calibrated on analytical offered load)

**Calibration criteria (stated before any policy result was examined):** analytical offered load ρ on the general wards (`neurotwin.experiments.load`, policy-independent): A ≈ 0.93 (heavily loaded but stable); single-factor stress B, C, D ≈ 1.02 (just past saturation, where allocation decisions matter most); combined stress E ≈ 1.10.

| Scenario | Arrivals | Staff | Physical capacity | ρ general | ρ ICU | Use |
|---|---|---|---|---|---|---|
| A – Normal | 1.0× | 1.0× | 1.0× | 0.930 | 0.821 | Train, validate, test |
| B – High demand | 1.10× | 1.0× | 1.0× | 1.023 | 0.903 | Test |
| C – Staff shortage | 1.0× | 0.88× | 1.0× | 1.068 | 0.985 | Test |
| D – Capacity loss | 1.0× | 1.0× | 0.88× | 1.020 | 0.878 | Test |
| E – Combined | 1.08× | 0.90× | 0.90× | 1.098 | 1.064 | Test |

C's load is the closest achievable to 1.02: nurse counts are integers, so 0.90× leaves the hospital below saturation (0.989) and anything from 0.84× to 0.89× removes the same nurse (1.068).

**Why v2.0's values were replaced:** v2.0 used B 1.35×, C 0.65×, D 0.6×, E (1.25, 0.7, 0.6), chosen without calibration. Their offered loads are 1.19–1.77. A pilot showed every policy diverting 42–59% of arrivals in E, so differences between policies were swamped and robustness comparisons would have been uninformative. These values are **retained as a "severe overload" extension** (`SEVERE_SCENARIOS`) and reported separately.

### Metrics
| Metric | Direction | Role |
|---|---|---|
| Mean boarding time (bed request → placement, ED requests) | Lower | **Primary** |
| 90th-percentile boarding time | Lower | **Primary** |
| Diversion rate (per 100 arrivals) | Lower | **Primary** |
| Off-service placement rate (per 100 general placements) | Lower | **Primary** |
| Wait-threshold breach rate (`e > W_max`) | Lower | Secondary |
| Mean ED wait (arrival → ED treatment start) | Lower | Secondary (not directly policy-controlled) |
| High-priority (P1–2) mean boarding time | Lower | Secondary |
| ICU step-down delay; escalation delay | Lower | Secondary |
| Throughput (placements per day) | Context | Secondary |
| Staffed occupancy per unit | Context | Secondary |
| Mean LOS | Context | Secondary |
| Decision latency | Lower | Secondary |
| Episode return | Diagnostic | PPO only |

---

## Locked Experimental Design Summary (v2)

1. **Model**: ED + MED + SURG + ICU; staffed capacity; off-service placement; step-down; escalation; automatic discharge
2. **Decision timing**: event-driven epochs + 4-hour review; atomic sequential decisions; semi-Markov
3. **Action space**: 7 masked actions (§6.3)
4. **State**: 21 features (§6.5)
5. **Reward**: time-integrated priority-weighted waiting + diversion + off-service cost (§6.10)
6. **Episodes**: 49 days = 35-day warm-up (Welch, 1-week window, rounded up to whole weeks) + 14 measured days, empty start, truncation bootstrapping; RL training starts from a ≥200-state warm-start pool
7. **Separation**: train and tune on A only; test on A (held-out seeds) and B–E
8. **CRN**: independent RNG streams per source; attributes drawn at arrival
9. **Seeds**: 50 test seeds; K = 5 PPO agents; disjoint seed ranges
10. **Statistics**: hierarchical paired bootstrap; Holm over 100 comparisons; d_z and Cliff's delta; Wilcoxon sensitivity
11. **MIP**: myopic transportation problem, exact, with LP and Hungarian test oracles
12. **Baselines**: FCFS, Priority, Load-Balancing, tuned Trunk-Reservation, MIP
13. **Validation**: analytical simulator checks (§6.17) and PPO validation gates (§6.18) before any results
14. **Ablations**: A–G (§6.16)

## Remaining Experimental Hyperparameters (not locked; tuned on scenario A validation seeds)

### PPO
| Parameter | Range |
|---|---|
| Learning rate | 1e-4, 3e-4, 1e-3 (annealed) |
| Rollout length | 2048, 4096 steps |
| Minibatch size | 64, 128, 256 |
| Update epochs | 4, 10 |
| Clip range ε | 0.1, 0.2, 0.3 |
| Entropy coefficient | 0.0, 0.01, 0.05 |
| γ (per hour) | 0.98, 0.99, 0.995 |
| GAE λ | 0.9, 0.95 |
| Hidden layers | [64, 64], [128, 128], [256, 256] |
| Training steps | 1M, 3M, 5M |

### Reward weights
| Weight | Range |
|---|---|
| α | fixed 1.0 (scale) |
| ω | default (4, 3, 2, 1, 1); ablation uniform |
| β | 5 – 50 |
| κ | 0.5 – 5 |

### Baselines
| Parameter | Range |
|---|---|
| `W_os` (FCFS/Priority off-service threshold) | 2, 4, 8 h |
| `θ_lb` | 0.1, 0.2, 0.3 |
| `r_MED, r_SURG, r_ICU` | 0 – 5 each (grid) |
| MIP `β_b`, `W_max` | `W_max` = 4 h (as metric); `β_b` 5 – 20 |

### Lagrangian (extension)
| Parameter | Range |
|---|---|
| λ learning rate | 1e-3, 1e-2 |
| Budget `c_div` | from fixed-β PPO on A validation |

### Simulation parameters (assumptions; calibrated for plausibility, not to a specific hospital)
| Parameter | Default | Range |
|---|---|---|
| Beds MED / SURG / ICU | 72 / 70 / 12 | — |
| Nurses on shift MED / SURG / ICU (nominal) | 15 / 14 / 6 | — |
| ED doctors on shift (nominal) | 5 | — |
| Staff absence probability per shift | 0.05 | — |
| Emergency base rate (patients needing a bed) | 0.9258 / h (22.2 / day) | — |
| Elective base rate (weekly mean) | 0.3968 / h (9.5 / day, weekdays 08–16) | — |
| General LOS mean | 4 days | 3 – 5 days |
| ICU LOS mean | 3 days | 2 – 4 days |
| ED treatment time mean | 3 h | 2 – 4 h |
| Patients per nurse (general / ICU) | 5 / 2 | 4–6 / 1–2 |
| ED capacity `Q_max` | 30 | — |
| Deterioration hazard `h_det` | 0.01 / day | ×0.5 – ×2 |
| Off-service LOS multiplier `m_os` | 1.1 | 1.0 – 1.25 |

**v2.3 calibration:** base rates are set so that scenario A has analytical offered load **ρ = 0.93** on the general wards (policy-independent; see §6.22).

**Pilot (30 × 84 days, greedy reference policy):** staffed occupancy MED 0.94, SURG 0.88, ICU 0.80; mean boarding 3.99 h; 90th percentile 9.15 h; diversion 0.12 per 100 arrivals; off-service 29.4 per 100 general placements (`docs/stage4_baselines_report.md` §2).

History of this table:
- v2.1 calibrated on occupancy under the greedy policy (ρ = 0.88). Under sensible policies that left almost no congestion (mean boarding 0.03–0.19 h), so allocation decisions barely mattered.
- v2.0 listed 40/30/12 beds at 2.5 arrivals/h, which would have run far above 100% load.

Defaults live in `config/simulation_default.json`.

### Implementation
| Item | Choice |
|---|---|
| RNG | `numpy.random.default_rng` with `SeedSequence.spawn` streams |
| Time unit | hours (float), event-driven (no fixed timestep) |
| Queue structures | Priority queue (heap) for the event list; per-type request lists |
| Distributions | Lognormal LOS / treatment; exponential deterioration; NHPP by thinning |

## Claims We MUST NOT Make

1. That NeuroTwin improves, or would improve, real hospital operations or patient outcomes
2. That the simulation represents any specific real hospital, or is calibrated to real data
3. That any policy is ready for, or validated for, clinical use
4. That simulation results are real-world evidence
5. That we are the first to apply RL or PPO to bed allocation
6. That LLM explanations are accurate descriptions of the policy's reasoning (they are post-hoc text)
7. That results generalise beyond the evaluated scenarios and parameter ranges
8. That the analytical validation shows realism (it shows implementation correctness only)

## Claims We CAN Make (if the results support them)

- PPO is compared systematically against heuristic, tuned threshold, and exactly solved myopic baselines in a unified simulation
- Generalization is tested: training under Normal conditions, evaluation under unseen stress
- Robustness is quantified with a signed degradation metric and paired CIs
- The simulator reproduces Erlang B/C and Little's law in reduced configurations
- The in-house PPO matches a reference implementation on standard and project tasks
- Comparisons use CRN with paired, multiplicity-corrected statistics over 50 seeds and 5 independent PPO agents
- All code, configurations and seeds are released for reproducibility

---

## Changelog

### v2.3 — 2026-10-26 (Stage 4; `docs/stage4_baselines_report.md`)
- **Scenario recalibration (§6.22)** on analytical, policy-independent offered load, with criteria stated before examining policy results. A: ρ 0.88 → 0.93 (base rates raised 5.8%). B–E: from ρ 1.19–1.77 to 1.02–1.10. The old values are kept as a severe-overload extension
- Warm-up 21 → 35 days and horizon 35 → 49 days (Welch re-run at the new load, rounded up to whole weeks)
- PPO-DR range shrunk so its worst corner is below every test scenario's load (§6.23)
- Warm-start pool: minimum 200 distinct states (§6.11)
- MIP off-service cost is a tuned surrogate (§6.9); tuning results recorded (§6.8)
- RQ4 note: trunk reservation tunes to zero in A, consistent with the cμ rule (hypothesis unchanged)

### v2.2 — 2026-10-25 (Stage 3; `docs/stage3_digital_twin_report.md`)
- §6.11: RL training episodes start from a validated warm-start pool (training seeds only); evaluation unchanged
- §6.21: what-if branches defined precisely (branch-namespace streams, conditional resampling of in-progress durations)

### v2.1 — 2026-10-25 (Stage 2 pilot evidence; `docs/stage2_validation_report.md`)
- Warm-up 1 day → 21 days and horizon 14 → 35 days, from the Welch analysis (§6.11)
- §6.17: M/M/c/c row replaced by M/M/c/K (the ED always has a waiting room); the formulas are tested against Erlang B and C as limits. Validation results recorded
- §6.20: placeholder hospital size replaced by the calibrated defaults and pilot load figures
- Model detail made explicit: arrivals are patients who will need a bed; triage is instantaneous at arrival; at most one deterioration per patient (see `neurotwin/simulation/hospital.py` docstring)

### v2.0 — 2026-10-25 (approved by project owner; rationale in `docs/stage1_design_review.md`)

**Defect fixes**
- **A1** Reward: difference-based → time-integrated cost (§6.10); throughput, transfer and utilisation-variance terms removed with justification
- **A2** MIP waiting constraint used absolute arrival timestamps → elapsed wait as data (§6.9)
- **A3** MIP overflow indicator was non-linear → constant breach indicator (§6.9)
- **A4** "RL is also myopic" was incorrect → PPO vs myopic MIP framed as the value of look-ahead (§6.9, RQ1)
- **A5** MIP identified as a totally unimodular transportation problem; exact solve with LP and Hungarian oracles (§6.9)
- **A6** Priority scale contradiction → priority 1 most urgent everywhere
- **A7** PPO now evaluated on scenario A with held-out test seeds (§6.12, §6.22)
- **A8** Degradation metric took an absolute value then claimed a sign → signed formula (§6.15)
- **A9** K = 5 independent PPO agents; hierarchical bootstrap (§6.12, §6.14)
- **A10** Action PRIORITY_SWAP duplicated ASSIGN_GENERAL; DISCHARGE_READY was a non-decision → action space redesigned (§6.3)
- **A11** TRANSFER_TO_ICU had no deterioration process in the model → escalation added (§6.1, §6.6)
- **A12** Validation used mild B/C/D perturbations (test-distribution leakage) → validation on A only (§6.12)
- **A13** Duplicate state feature (average ED wait listed twice) removed (§6.5)
- **A14** "Overflow" meant both diversion and long waits → split into diversion, off-service placement, and wait-threshold breach (§6.1, §6.22)
- **A15** CRN implementation requirements made explicit (independent streams, attributes drawn at arrival) (§6.7)
- **A16** Statistical test chosen after a normality pre-test → Wilcoxon reported as a fixed sensitivity analysis (§6.14)
- Broken cross-references to non-existent §7.2, §8.1 and §10.1 removed

**Enhancements**
- Specialty wards + off-service placement, aligned with the inpatient-overflow literature (Dai & Shi; Sun, Shi & Dai)
- Trunk-Reservation baseline and policy-structure analysis (RQ4), grounded in Miller (1969)
- Analytical simulator validation (§6.17)
- In-house PPO with semi-Markov GAE and validation gates (§6.18)
- PPO-Lagrangian extension (§6.19)
- Welch warm-up check; power analysis; explicit disjoint seed ranges
- Domain-randomised PPO variant with a strictly milder training range (RQ8, §6.23)
- Measured CRN variance reduction (§6.24)

**Literature**
- All citations verified; authors, venues and DOIs corrected (Papers 1, 2, 5, 6, 7, 9, 10, 12, 13, 14)
- Five unverifiable entries removed (§5.8)
- Added Sun, Shi & Dai (closest prior work), Dai & Shi (2019), Miller (1969), Mohiuddin et al. (2017)

### v1.0
Initial specification.
