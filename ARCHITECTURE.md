# NeuroTwin V2 - Architecture Specification

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NeuroTwin V2 System                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          SCENARIO ENGINE                              │    │
│  │              (Configuration, Parameters, Random Seeds)                 │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DISCRETE-EVENT HOSPITAL SIMULATION                │    │
│  │        (Entities: Patients, Staff, Beds, ICU, ED, Wards)            │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DIGITAL TWIN STATE                           │    │
│  │  (Current State │ History │ Snapshots │ Serialization │ What-If)     │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          DECISION LAYER                              │    │
│  │         (Receives state, returns allocation decisions)                │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│          ┌─────────────────────────┼─────────────────────────┐               │
│          │                         │                         │               │
│          ▼                         ▼                         ▼               │
│  ┌───────────────┐     ┌───────────────────────┐     ┌───────────────┐    │
│  │   HEURISTIC   │     │    OPTIMIZATION       │     │      RL       │    │
│  │   POLICIES    │     │    (Myopic MIP)       │     │  (own PPO)    │    │
│  │  FCFS/Priority│     │                       │     │               │    │
│  │  LoadBalance  │     │                       │     │               │    │
│  └───────────────┘     └───────────────────────┘     └───────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        EXPERIMENT ENGINE                            │    │
│  │    (Multi-seed runner │ Metrics collection │ Statistics)             │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          WHAT-IF ENGINE                             │    │
│  │       (Counterfactual: compare policies from comparable states)      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────┐     ┌───────────────────────┐     ┌───────────────┐    │
│  │   DASHBOARD   │     │    EXPLANATION        │     │      API      │    │
│  │   (Next.js)   │     │    LAYER (LLM)        │     │   (FastAPI)   │    │
│  │               │     │  Post-hoc only │ Facts │     │               │    │
│  └───────────────┘     └───────────────────────┘     └───────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DATA STORAGE (Minimal)                            │    │
│  │          Local files / SQLite (escalate to PostgreSQL if needed)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Core Architecture Principles

1. **Research-First**: The research engine is the core product; dashboard is a visualization layer
2. **LLM Outside Decision Loop**: LLM explains decisions post-hoc; never controls or overrides
3. **Minimal Infrastructure**: Use simplest appropriate storage; add complexity only when needed
4. **Reproducibility**: All experiments use seeded randomness and documented configurations
5. **Statistical Rigor**: Multi-seed evaluation with appropriate statistical tests

## 3. Major Components

### 3.1 Scenario Engine (`scenario/`)
Manages experiment configuration and parameters.
- **Configuration Management**: Scenario parameters, random seeds, policy configs
- **Parameter Validation**: Ensures valid parameter combinations
- **Seed Management**: Centralized random seed handling

### 3.2 Simulation Engine (`simulation/`)
Core discrete-event simulation modeling hospital operations.
- **Entities**: Patient (emergency, elective), Doctor, Nurse, Bed, ICUBed, EmergencyDepartment,
  Ward (MED, SURG specialty), ICU, BedRequest
- **Events**: ArrivalEvent, TriageEvent, EDTreatmentStart/End, BedRequestEvent, AdmissionEvent,
  StepDownReadyEvent, DeteriorationEvent, TransferEvent, DischargeEvent, ShiftChangeEvent,
  PeriodicReviewEvent
- **Scheduler**: Event queue management, time advancement
- **State Manager**: Maintains current simulation state
- **Analytical Validation** (`simulation/validation/`): Reduced configurations checked against
  closed-form queueing results (Erlang C, Erlang B, Little's law, arrival-rate goodness of fit).
  Kept as permanent regression tests so engine changes cannot silently break correctness.
  See `docs/stage1_design_review.md` B1.

**Note**: Operating rooms are out of scope for V1.

### 3.3 Digital Twin Layer (`digital_twin/`)
Maintains computational representation of simulated hospital state.
- **Observable State** (`state.py`): `HospitalState` built by `observe(sim)` from public engine
  queries only; this is the only state policies, the API and the dashboard may see
- **State History** (`history.py`): `StateHistory`, `RecordingPolicy` (read-only recording)
- **Snapshots** (`snapshot_io.py` + engine `to_snapshot/from_snapshot`): complete state incl.
  hidden attributes and RNG states; exact restore (byte-identical continuation, tested)
- **What-If Branching** (engine `branch(seed)`): same present, new future randomness from
  branch-namespace streams; in-progress durations re-drawn conditional on elapsed time
- **Warm-Start Pool** (`warm_start.py`): steady-state training starts from training seeds only
- **Counterfactual Engine**: policy comparison over many branches is Stage 8 (`whatif/`)

Snapshots contain hidden information and are never passed to policies.

### 3.4 Decision Layer (`policies/`)
Contains all allocation policies evaluated in experiments.
- **Heuristic Policies**:
  - `FCFSAllocator`: First-come, first-served
  - `PriorityAllocator`: Severity/urgency-based allocation
  - `LoadBalancingAllocator`: Distribute across resources
  - `TrunkReservationAllocator`: Priority order with tuned bed-reservation thresholds
    (Miller, 1969; `RESEARCH.md` §6.8)
- **Optimization Policy**:
  - `OptimizationAllocator`: Myopic assignment (`RESEARCH.md` §6.9), solved exactly with
    `scipy.optimize.milp`
- **RL Policy**:
  - `PPOPolicy`: PPO-based allocation policy (in-house PPO, see 3.5)

### 3.5 Reinforcement Learning Layer (`rl/`)
RL agent for learning adaptive allocation policies.
- **Gymnasium Environment**: `HospitalResourceEnv` compatible with Gymnasium
- **Observation Space**: 21 operational state features (`RESEARCH.md` §6.5)
- **Action Space**: 7 masked, sequential (atomic) placement actions (`RESEARCH.md` §6.3)
- **Reward Function**: Time-integrated priority-weighted waiting + diversion + off-service cost
  (`RESEARCH.md` §6.10); semi-Markov (variable step duration)
- **Algorithm**: PPO implemented in-house in PyTorch (`rl/ppo/`). `stable-baselines3` is kept
  only as a reference implementation to validate against (CartPole and the hospital env), not as
  the training path. Rationale: the algorithm is the core technical contribution and must be
  fully understood and modifiable (per-step semi-Markov discounting, logit-level action masking,
  Lagrangian extension), which off-the-shelf SB3 PPO does not support without patching.
- **Training**: Train under normal conditions; evaluate under stress
- **Optional extension**: PPO-Lagrangian (overflow as a constraint instead of a fixed reward
  weight), see `docs/stage1_design_review.md` B3

### 3.6 Experiment Engine (`experiments/`)
Automated experiment execution and evaluation.
- **Scenario Runner**: Executes simulations with given config
- **Multi-Seed Runner**: Runs experiments across multiple seeds
- **Metrics Collector**: Collects all metrics per timestep and aggregate
- **Statistical Analyzer**: Computes means, std, CIs, significance tests
- **Robustness Analyzer**: Measures performance degradation under stress

### 3.7 What-If Engine (`whatif/`)
Counterfactual analysis capability.
- **State Clone**: Create comparable initial states
- **Policy Evaluator**: Run multiple policies from same initial state
- **Comparison Generator**: Generate comparative results
- **Dashboard Integration**: Display what-if analysis results

### 3.8 Explanation Layer (`explanation/`)
LLM-based explanations for decisions (post-hoc only).
- **Decision Logger**: Records all decisions with state context
- **Context Retriever**: Retrieves relevant context for explanation
- **Prompt Constructor**: Builds prompts from actual logged data
- **Explanation Generator**: Generates natural language via LLM
- **Reliability Indicator**: Clearly labeled, not statistical confidence unless actual

**Critical**: LLM never controls simulation; never overrides decisions; never fabricates data.

### 3.9 API Layer (`api/`)
FastAPI backend exposing system functionality.
- **Simulation Endpoints**: Start, pause, step, reset simulation
- **State Endpoints**: Get current state, state history
- **Policy Endpoints**: Get/evaluate policies
- **Experiment Endpoints**: Run experiments, get results
- **What-If Endpoints**: Trigger counterfactual analysis
- **Explanation Endpoints**: Get explanations for decisions

### 3.10 Dashboard Layer (`dashboard/`)
Next.js web interface for visualization and interaction.
- **Simulation Control Panel**: Start/pause/step controls, speed
- **Real-time Visualization**: Hospital state, queues, utilization
- **Metrics Dashboard**: Charts with statistical summaries
- **What-If Interface**: Scenario comparison visualization
- **Decision Log**: Recent decisions with explanations

## 4. Data Flow

```
[Scenario Config + Seed]
        │
        ▼
┌───────────────────┐
│  Scenario Engine  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Discrete-Event   │
│  Simulation       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Digital Twin     │
│  State Update    │
└─────────┬─────────┘
          │
          ├──────────────────────┐
          │                      │
          ▼                      ▼
┌───────────────────┐   ┌───────────────────┐
│  Decision Layer   │   │  History/Snapshot  │
│  (Policy Loop)    │   │  Storage          │
└─────────┬─────────┘   └───────────────────┘
          │
          ▼
┌───────────────────┐
│  Experiment      │
│  Engine          │
└─────────┬─────────┘
          │
          ├──────────────────────┐
          │                      │
          ▼                      ▼
┌───────────────────┐   ┌───────────────────┐
│  What-If Engine  │   │  Metrics/Stats    │
│  (Counterfactual)│   │  Collection       │
└─────────┬─────────┘   └───────────────────┘
          │                      │
          └──────────┬───────────┘
                     │
                     ▼
          ┌───────────────────┐
          │  Dashboard + API  │
          └───────────────────┘

Decision Flow (per timestep):
1. Simulation advances → produces state
2. State goes to Decision Layer
3. Policy selects action
4. Action logged → LLM Explanation (post-hoc)
5. Action applied to simulation
```

## 5. Simulation Layer (`simulation/`)

### 5.1 Entity Model (V1 Scope, RESEARCH.md v2 §6.1)
```
Entity (base)
├── Patient (id, arrival_time, is_emergency, priority [1 = most urgent], specialty [MED|SURG],
│            pathway [GENERAL|ICU_FIRST], location, is_off_service)
│     # Hidden attributes, drawn at arrival for CRN and never visible to policies:
│     # ed_treatment_time, icu_los, general_los, deterioration_time
├── Staff (id, type, shift_schedule, is_available)
│   ├── Doctor                      # ED treatment
│   └── Nurse (assigned_unit)       # determines staffed capacity
├── Bed (id, unit, is_occupied, patient_id)
├── BedRequest (patient_id, type [ED_GENERAL|ED_ICU|STEP_DOWN|ESCALATION], priority,
│               created_at, eligible_units)
└── Unit (id, physical_beds, nurses_on_shift, patients_per_nurse)
    ├── EmergencyDepartment (treatment_queue, capacity Q_max)
    ├── Ward (specialty MED | SURG)
    └── ICU
```

### 5.2 Event Model (as implemented in `neurotwin/simulation/events.py`)
Scheduled events, processed in this order when they occur at the same time:
```
EventKind
├── SHIFT_CHANGE        # draw staffing for the new shift (fixed draw order, CRN-safe)
├── DISCHARGE           # automatic at end of LOS; frees a bed            -> trigger
├── ICU_PHASE_END       # ICU stay over -> STEP_DOWN request             -> trigger
├── DETERIORATION       # general-ward patient -> ESCALATION request     -> trigger
├── ED_TREATMENT_END    # -> admission request (ED boarding begins)      -> trigger
├── EMERGENCY_ARRIVAL   # triage (instant), diversion check, ED queue
├── ELECTIVE_ARRIVAL    # -> admission request                           -> trigger
├── PERIODIC_REVIEW     # every 4 h                                      -> trigger
└── TRACE               # optional occupancy snapshot; never a trigger, no RNG
```
Triage, diversion, ED treatment start, admission and transfers are **instantaneous state
changes** inside event handlers or `place()`, not separate scheduled events. Admissions and
transfers happen only through `HospitalSimulation.place()` during an open decision epoch.

### 5.3 Simulation Parameters
```python
SimulationConfig:
    horizon_hours: float              # 14 days = 336 h
    warmup_hours: float               # 24 h (Welch-checked)
    seed: int                         # root seed -> SeedSequence.spawn streams
    emergency_base_rate: float        # per hour, modulated by f_hour(t) * f_day(t)
    elective_base_rate: float         # per hour, weekday daytime only
    priority_probs: Dict[int, float]  # P1..P5
    icu_first_prob: Dict[int, float]  # by priority
    specialty_probs: Dict[str, Dict[str, float]]  # emergency / elective -> MED/SURG
    beds: Dict[str, int]              # MED, SURG, ICU physical beds
    nurses: Dict[str, int]            # nominal nurses on shift per unit
    patients_per_nurse: Dict[str, int]
    doctors: int                      # nominal ED doctors on shift
    ed_capacity: int                  # Q_max (diversion threshold)
    los_params: Dict[str, Tuple[float, float]]    # lognormal mean/std: general, icu, ed_treatment
    deterioration_hazard_per_day: float
    off_service_los_multiplier: float # m_os
    review_interval_hours: float      # 4 h
```
All parameters marked as assumptions in `RESEARCH.md` §6.6 / §6.20 live in config files, not code.

### 5.4 Patient Flow (V1)
```
Emergency arrival ──(ED full)──► Diversion
       │
       ▼
Triage → ED treatment (doctor) → Bed request ◄── Elective arrival
                                     │
                    ┌────────────────┴───────────────┐
                    ▼                                ▼
         General ward (primary or            ICU
          off-service)                        │
              │   ▲                          ▼
              │   └──── Step-down request ◄── step-down ready
              │
              ├── deterioration → Escalation request → ICU
              ▼
          Discharge (automatic at end of LOS)
```

### 5.5 Random Number Streams (CRN)
Each stochastic source has its own stream spawned from the root seed
(`arrivals_emergency`, `arrivals_elective`, `patient_attributes`, `staffing`). Patient attributes
are drawn at arrival, so policy decisions never change other draws. A shared RNG would silently
break Common Random Numbers (`RESEARCH.md` §6.7).

## 6. Digital Twin Layer (`digital_twin/`)

### 6.1 State Schema
```python
HospitalState:
    timestamp: float
    simulation_time: float

    # Resources (per unit: MED, SURG, ICU)
    beds: Dict[UnitID, List[BedState]]
    staffed_capacity: Dict[UnitID, int]
    doctors_on_shift: int
    nurses_on_shift: Dict[UnitID, int]

    # Queues
    ed_treatment_queue: List[Patient]     # Waiting for a doctor
    bed_requests: List[BedRequest]        # ED_GENERAL, ED_ICU, STEP_DOWN, ESCALATION

    # Utilization
    staffed_occupancy: Dict[UnitID, float]
    off_service_in_beds: int

    # Counts (cumulative over measurement window)
    patients_in_system: int
    placements: int
    off_service_placements: int
    diversions: int

    # History (for RL state)
    arrival_history: List[float]          # Arrival timestamps in the last 24 h
    discharge_history: List[float]        # Discharge timestamps in the last 24 h
    diversion_history: List[float]

    # Hidden patient attributes are NOT part of HospitalState exposed to policies.
    # They are part of the full simulation snapshot used for what-if cloning.
```

### 6.2 What-If/Counterfactual Capability
```python
class WhatIfEngine:
    def clone_state(state: HospitalState) -> HospitalState:
        """Create deep copy for counterfactual simulation"""

    def evaluate_policy_from_state(
        state: HospitalState,
        policy: AllocationPolicy,
        horizon: float
    ) -> SimulationResult:
        """Run policy from cloned state for specified horizon"""

    def compare_policies(
        state: HospitalState,
        policies: List[AllocationPolicy],
        horizon: float
    ) -> Dict[str, SimulationResult]:
        """Compare multiple policies from same initial state"""
```

## 7. Policy Layer (`policies/`)

### 7.1 Policy Interface
```python
class AllocationPolicy(ABC):
    @abstractmethod
    def select_action(
        state: HospitalState,
        waiting_patients: List[Patient],
        available_beds: Dict[str, List[Bed]]
    ) -> AllocationDecision:
        """Select allocation action given current state"""
```

### 7.2 Heuristic Policies
```python
class FCFSAllocator(AllocationPolicy):
    """Assign earliest patient to first available appropriate bed"""

class PriorityAllocator(AllocationPolicy):
    """Assign highest-priority patient to most appropriate available bed"""

class LoadBalancingAllocator(AllocationPolicy):
    """Primary ward unless its occupancy exceeds the other ward's by more than theta_lb"""

class TrunkReservationAllocator(AllocationPolicy):
    """Priority order; P3-5 placed in unit k only if free staffed beds > r_k (tuned on A)"""
```
Off-service rule for FCFS/Priority: place off-service once a request has waited >= `W_os`
(`RESEARCH.md` §6.8).

### 7.3 Optimization Policy (Myopic Assignment)
```python
class OptimizationAllocator(AllocationPolicy):
    """Solve the myopic transportation problem in RESEARCH.md §6.9 at each epoch"""
    # Solver: scipy.optimize.milp (HiGHS). SciPy is already required by scikit-learn,
    # so no new dependency is added.
    # The problem (requests -> units with integer staffed capacities) is a transportation
    # problem, so its constraint matrix is totally unimodular and the LP relaxation is integral.
    # Test oracles: linprog LP relaxation and linear_sum_assignment over expanded bed slots
    # must give the same objective value.
    # The MILP path is kept for a future float-nurse coupling constraint, which would break TU.
```

## 8. Reinforcement Learning Layer (`rl/`)

### 8.1 Gymnasium Environment
```python
class HospitalResourceEnv(gymnasium.Env):
    observation_space: Box(low=-1, high=1, shape=(21,), dtype=np.float32)
    # 21 features, exact list and normalisation in RESEARCH.md §6.5

    action_space: Discrete(7)
    # 0 HOLD, 1 ADMIT_PRIMARY_PRIORITY, 2 ADMIT_PRIMARY_LONGEST_WAIT, 3 ADMIT_OFF_SERVICE,
    # 4 ADMIT_ICU, 5 STEP_DOWN, 6 ESCALATE  (RESEARCH.md §6.3)

    step(action) -> (obs, reward, terminated, truncated, info)
    #   info["action_mask"]: np.ndarray[bool] of shape (7,) for the NEXT decision
    #   info["dt_hours"]: duration of this step (semi-Markov discount)
    reset(seed) -> (obs, info)       # info["action_mask"] for the first decision
```

### 8.2 Action Space Design Principles
- Compact, meaningful decisions (not assign_patient_X_to_bed_Y)
- Sequential (atomic) placements within an epoch; HOLD ends the epoch
- Every action is a real trade-off (competition for general beds or for ICU beds)
- Exact feasibility masks exposed via `info`; invalid actions are impossible, not penalised

### 8.3 Reward Function
```python
def compute_reward(step: StepRecord) -> float:
    # Priority-weighted request-hours of waiting accrued over [tau_t, tau_{t+1}).
    # Accumulated exactly by the simulator: between events the request set is constant,
    # so the integral is a sum of (sum of weights) * (event gap).
    waiting_cost = alpha * step.weighted_request_hours
    diversion_cost = beta * step.diversions
    off_service_cost = kappa * step.off_service_placements
    return -(waiting_cost + diversion_cost + off_service_cost)
```

**Why this form** (`RESEARCH.md` §6.10): summed over an episode with uniform weights, the
waiting term equals total request waiting time, so the return is aligned with the primary
metric. The v1 difference-based reward telescoped to the terminal state and is not used.

**Important**: Reward weights are tuned on scenario A validation seeds only and reported.

### 8.4 Training Configuration
```python
PPOConfig:
    total_timesteps: int          # Training duration
    learning_rate: float          # PPO learning rate
    n_steps: int                  # Steps per update
    batch_size: int               # Minibatch size
    n_epochs: int                # PPO epochs per update
    gamma: float                  # Discount factor
    clip_range: float            # PPO clipping
    ent_coef: float              # Entropy coefficient (encourages exploration)
    vf_coef: float               # Value function coefficient
    max_grad_norm: float          # Gradient clipping
    gae_lambda: float
    gamma_per_hour: float        # step discount = gamma_per_hour ** dt_hours

    # Training scenario and seeds (RESEARCH.md §6.12)
    train_scenario: ScenarioConfig  # Normal operations (A) only
    train_env_seed_range: Tuple[int, int]   # (0, 99_999), fresh seed per episode
    agent_seeds: List[int]          # K = 5 independent agents
```

### 8.5 In-House PPO Implementation (`rl/ppo/`)
```
rl/ppo/
├── networks.py      # Actor-critic MLP, orthogonal init
├── masking.py       # Logit-level action masking (invalid actions -> -inf)
├── buffer.py        # Rollout buffer with per-step discount gamma ** dt (semi-Markov)
├── gae.py           # Generalised Advantage Estimation
├── losses.py        # Clipped surrogate, clipped value loss, entropy bonus
├── normalize.py     # Running observation normalisation (frozen at evaluation)
└── trainer.py       # Rollout/update loop, LR annealing, grad clipping, checkpoints
```

Validation before any hospital result is reported (`docs/stage1_design_review.md` B2):
1. CartPole-v1 learning curves comparable to stable-baselines3 under matched hyperparameters.
2. Hospital env: in-house vs reference PPO learning curves with overlapping CIs.
3. Unit tests: GAE against a hand-computed example, clipped loss identity inside the trust
   region, masked actions have zero probability.

Multiple training seeds: K >= 5 independent agents per configuration (Henderson et al., 2018).

## 9. Experiment Layer (`experiments/`)

### 9.1 Scenario Configurations
```python
class ScenarioConfig:
    name: str
    arrival_rate_multiplier: float = 1.0    # 1.0 = baseline
    staff_multiplier: float = 1.0          # 1.0 = full staffing
    capacity_multiplier: float = 1.0       # 1.0 = full capacity
    # Emergency share is a SimulationConfig parameter (default 70%), not a scenario knob.
```

### 9.2 Experiment Configuration
```python
class ExperimentConfig:
    scenarios: List[ScenarioConfig]
    policies: List[str]                    # Policy names to evaluate
    seeds_per_condition: int              # 30-50 for final eval
    simulation_time_limit: float
    metrics: List[str]                   # Metrics to collect
    save_snapshots: bool                  # Save state histories
```

### 9.3 Results Structure
```python
class ExperimentResult:
    experiment_id: str
    scenario: str
    policy: str
    seed: int
    runtime: float

    # Metrics
    metrics: Dict[str, float]

    # Per-timestep data (if saved)
    time_series: Optional[Dict[str, List[float]]]
```

## 10. What-If Engine (`whatif/`)

### 10.1 Capability
Given a current simulation state, allow users to:
1. Modify parameters (e.g., increase demand by 30%)
2. Evaluate all policies from that point
3. Compare outcomes across policies

### 10.2 Example Usage
```
Current State:
  ICU occupancy = 85%
  General bed occupancy = 70%
  ED queue = 12 patients
  Available staff = 80%

User Action: "Increase emergency arrivals by 30%"

System Response (simulation results):
  Policy A (FCFS): [results]
  Policy B (Priority): [results]
  Policy C (Trunk-Reservation): [results]
  Policy D (Optimization): [results]
  Policy E (RL): [results]
```

### 10.3 Display Requirements
- Clearly label as simulation results
- Show uncertainty (variance across seeds)
- Allow comparison of multiple metrics

## 11. Explanation Layer (`explanation/`)

### 11.1 Components
```python
class ExplanationGenerator:
    def generate_explanation(
        decision: AllocationDecision,
        state: HospitalState,
        policy_name: str
    ) -> Explanation:
        """Generate post-hoc explanation for a decision"""

class Explanation:
    decision_id: str
    timestamp: float
    policy: str

    summary: str              # One-sentence summary
    reasoning: str           # Detailed explanation

    # Clearly labeled, NOT statistical confidence unless calculated as such
    reliability_indicator: str  # e.g., "Low/Medium/High based on..."

    facts_used: List[str]    # Actual state data referenced
    alternatives_considered: List[str]
    limitations: str         # Known limitations of explanation
```

### 11.2 Safety Constraints
- **Never** control simulation decisions
- **Never** override RL or policy decisions
- **Never** fabricate simulation data
- **Always** distinguish facts from generated text
- **Always** include uncertainty/limitations

## 12. API Layer (`api/`)

### 12.1 Endpoints
```
# Simulation
POST   /api/simulation/start
POST   /api/simulation/pause
POST   /api/simulation/step
POST   /api/simulation/reset
GET    /api/simulation/status

# State
GET    /api/state/current
GET    /api/state/history
GET    /api/state/snapshot/{id}

# Policies
GET    /api/policies
POST   /api/policies/{name}/evaluate

# Experiments
POST   /api/experiments/run
GET    /api/experiments/{id}/status
GET    /api/experiments/{id}/results
GET    /api/experiments/{id}/statistics

# What-If
POST   /api/whatif/evaluate
GET    /api/whatif/results/{id}

# Explanations
GET    /api/explanations/{decision_id}
GET    /api/explanations/recent
```

## 13. Dashboard Layer (`dashboard/`)

### 13.1 Pages
- `/` - Overview with key metrics
- `/simulation` - Real-time state and controls
- `/experiments` - Experiment configuration and results
- `/whatif` - Counterfactual analysis interface
- `/decisions` - Decision log with explanations

### 13.2 Key Components
- `SimulationControl`: Start/pause/step, speed slider
- `HospitalVisualization`: Bed states, queue lengths, staff
- `MetricChart`: Time-series with CI bands
- `PolicyComparison`: Side-by-side policy results
- `WhatIfPanel`: Scenario modification and comparison
- `DecisionCard`: Individual decision with explanation

## 14. Data Storage

### 14.1 Initial Approach (Minimal)
- **Local Files/JSON**: Configuration, results
- **SQLite**: Experiment metadata, state snapshots, metrics

### 14.2 Escalation Criteria
Introduce PostgreSQL when:
- Multiple concurrent experiments require shared state
- Complex querying of historical results needed
- Team collaboration requires shared database

### 14.3 NOT Mandatory for Core
- Redis: Not needed for core functionality
- MLflow: Can use simple file-based logging initially
- Kubernetes/AWS: Out of scope for V1

## 15. Technology Stack

### 15.1 Backend/Research
- Python 3.10+
- NumPy, Pandas
- scikit-learn (for baselines/optional ML)
- SciPy (already required by scikit-learn): `milp` for the MIP baseline, statistics
- Gymnasium
- PyTorch (in-house PPO)
- stable-baselines3 + sb3-contrib (test/dev only: reference PPO and MaskablePPO for validating
  the in-house PPO; never on the training path)
- FastAPI
- Pydantic

### 15.2 Frontend
- Next.js
- React
- TypeScript
- Chart.js or Recharts

### 15.3 Storage
- SQLite (initial)
- PostgreSQL (if needed)

### 15.4 Infrastructure
- Docker (for development reproducibility)
- Local execution for experiments

## 16. Experiment Tracking

### 16.1 Required Tracking
```python
ExperimentRecord:
    experiment_id: str
    timestamp: datetime
    config: ExperimentConfig
    seeds: List[int]
    policies: List[str]
    scenarios: List[str]
    results: Dict[str, List[ExperimentResult]]
    statistics: Dict[str, StatisticalSummary]
```

### 16.2 Metrics Logged
- All simulation events
- All policy decisions with state context
- Per-timestep metrics
- Aggregate results
- Random seeds for reproducibility
