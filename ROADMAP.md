# NeuroTwin V2 - Implementation Roadmap

## Overview

This roadmap organizes implementation into 12 stages, prioritized by research importance. The research engine (simulation, baselines, RL, experiments) comes before UI/infrastructure components.

**Estimated Total Duration**: ~21.5 weeks
**Key Principle**: Research-first development; dashboard and infrastructure are secondary.

---

## Stage 1 — Research Foundation

**Duration**: 2 weeks

### Objective
Establish research formulation and validate simulation scope.

### Components
- Systematic literature review (RL in healthcare, hospital resource allocation, digital twins)
- Research gap identification and refinement
- Simulation scope definition and justification
- Baseline policy definitions (FCFS, Priority, Load Balance, Optimization)
- Optimization formulation research and selection
- RL action/observation space preliminary design
- Critical review of the locked design (`docs/stage1_design_review.md` Part A)

### Expected Output
- Refined research questions and hypothesis
- Documented literature review findings
- Justified simulation scope
- Selected optimization formulation
- Preliminary RL design document

### Testing Requirements
- Research methodology review
- Peer validation of scope decisions

### Completion Criteria
- [x] Literature review completed and documented (all citations verified 2026-10-25; RESEARCH.md §5)
- [x] Research gap clearly articulated (RESEARCH.md §4, positioned against Sun, Shi & Dai)
- [x] Simulation scope justified against research questions (RESEARCH.md §6.1)
- [x] Optimization baseline selected with justification (RESEARCH.md §6.9)
- [x] Preliminary RL design reviewed (`docs/stage1_design_review.md`)
- [x] Design review items approved and written into RESEARCH.md v2 (2026-10-25)

**Status: complete (2026-10-25).**

---

## Stage 2 — Discrete-Event Hospital Simulation

**Duration**: 3 weeks

### Objective
Implement the core simulation engine with all hospital entities and patient flow.

### Components
- Project structure and configuration management
- Entity classes: Patient, Doctor, Nurse, Bed, BedRequest
- Unit classes: EmergencyDepartment, Ward (MED, SURG), ICU, with staffed capacity
- Event classes per ARCHITECTURE.md §5.2 (incl. StepDownReady, Deterioration, Diversion,
  PeriodicReview)
- Event queue and scheduler
- Patient flow logic (primary/off-service placement, step-down, escalation)
- Random seed management: independent CRN streams, attributes drawn at arrival
- Exact accumulation of weighted request-hours between events (reward and metrics)

### Expected Output
- Complete simulation engine with all V1 entity types
- Patient flow: Arrival → Triage → Admission → Discharge/Transfer
- Working scheduler with reproducible execution
- All entities configurable via SimulationConfig

### Testing Requirements
- Unit tests for all entity classes
- Integration test for patient flow
- Reproducibility test with fixed seeds
- Analytical validation (`docs/stage1_design_review.md` B1), kept as regression tests:
  - M/M/c reduced config vs Erlang C waiting time and P(wait)
  - M/M/c/c reduced config vs Erlang B blocking (overflow) probability
  - Little's law on every queue in the full model
  - Non-homogeneous arrival counts vs integrated λ(t) (chi-square)
  - Each at loads ρ = 0.5, 0.8, 0.95 over >= 30 seeds
- CRN integrity test: two different policies, same seed → identical arrivals and attributes
- Welch warm-up check on pilot runs (scenario A)

### Completion Criteria
- [x] Analytical values fall inside the simulated 95% CIs for all checks (15/15; `docs/stage2_validation_report.md`)
- [x] All entity classes functional
- [x] Events execute in correct temporal order
- [x] Patient flows through complete journey (incl. step-down, escalation, off-service)
- [x] Simulation runs to completion with known results
- [x] Random seed produces identical results; CRN integrity verified
- [x] Warm-up determined by Welch's method (21 days)
- [x] Operating rooms NOT implemented (V1 scope)

**Status: complete (2026-10-25).** 50 tests passing.

---

## Stage 3 — Digital Twin State Management

**Duration**: 1.5 weeks

### Objective
Implement state management with history, serialization, and what-if capability.

### Components
- HospitalState data class
- State manager service
- State snapshot storage
- State serialization (JSON export/import)
- State history preservation
- What-if state cloning

### Expected Output
- Complete state representation
- JSON serialization/deserialization
- Snapshot mechanism for reproducibility
- State history for analysis

### Testing Requirements
- Unit tests for state serialization
- Test state updates reflect events
- Test history preservation
- Test state cloning for what-if

### Completion Criteria
- [x] State reflects simulation after every event (observable state built from live engine queries)
- [x] State serializes to JSON correctly
- [x] State restores from JSON (byte-identical continuation)
- [x] State history maintained
- [x] What-if state cloning functional (`clone`, `branch` with conditional resampling)
- [x] Warm-start pool validated against full episodes (`docs/stage3_digital_twin_report.md`)

**Status: complete (2026-10-25).**

---

## Stage 4 — Baseline Allocation Policies

**Duration**: 2 weeks

### Objective
Implement all baseline policies for comparison.

### Components
- AllocationPolicy base class
- FCFSAllocator
- PriorityAllocator
- LoadBalancingAllocator
- TrunkReservationAllocator (thresholds tuned by grid search on scenario A validation seeds)
- OptimizationAllocator (myopic transportation problem, `scipy.optimize.milp`; LP-relaxation
  and Hungarian-algorithm oracle tests)
- Baseline tuning harness (W_os, θ_lb, r_k) restricted to validation seeds on scenario A
- Policy evaluation framework
- Decision logging

### Expected Output
- All baseline policies functional
- Common policy interface
- Decision logs for analysis
- Policy validation tests

### Testing Requirements
- Unit tests for each allocator
- Integration test with simulation
- Test policy produces valid allocations
- Test decision logging

### Completion Criteria
- [x] All allocator classes functional
- [x] FCFS assigns oldest patient first
- [x] Priority assigns by severity
- [x] Load balancing respects the θ_lb rule
- [x] Trunk reservation never places P3-5 when free beds <= r_k
- [x] MIP objective equals LP-relaxation and Hungarian oracle objectives on random instances
- [x] Optimization policy implements selected formulation
- [x] Policies swappable in simulation
- [x] Baselines tuned on scenario A validation seeds (`config/baselines_tuned.json`)
- [x] Scenarios recalibrated on analytical offered load (`docs/stage4_baselines_report.md`)

**Status: complete (2026-10-26).** 94 tests passing.

---

## Stage 5 — RL Environment and Agent

**Duration**: 4 weeks (was 3; +1 week for the in-house PPO and its validation)

### Objective
Create Gymnasium-compatible RL environment, implement PPO in-house, validate it, and train it.

### Components
- HospitalResourceEnv (Gymnasium.Env)
- Observation space design and implementation
- Action space design and implementation (with action masks exposed to the agent)
- Reward function implementation
- Environment validation
- In-house PPO in PyTorch (`rl/ppo/`, see ARCHITECTURE.md §8.5): masking, GAE with
  per-step discount, clipped losses, observation normalisation
- PPO validation against stable-baselines3 (CartPole-v1, then hospital env)
- Training loop with K >= 5 independent training seeds
- Checkpointing

### Expected Output
- Gymnasium-compatible environment
- Documented observation/action space
- Justified reward formulation
- Trained PPO agent
- Saved checkpoints

### Testing Requirements
- Environment passes Gymnasium checks
- Test action/observation spaces
- Test reward calculation
- Test episode termination
- Test training convergence (indicative)
- GAE vs hand-computed example; clipped-loss identity inside [1-ε, 1+ε]; masked actions
  have zero probability
- CartPole-v1 and hospital-env learning curves vs stable-baselines3 (matched hyperparameters)

### Completion Criteria
- [x] Environment instantiates and passes Gymnasium passive checks
- [x] `reset()` returns valid observation
- [x] `step()` returns valid (obs, reward, terminated, truncated, info)
- [x] Actions produce valid state transitions (masks = feasible placements; env ≡ direct simulation)
- [x] In-house PPO passes all unit tests and matches the reference on CartPole-v1 and the hospital env
- [x] PPO agent trains without errors on the hospital env (K = 5 training seeds)
- [x] Checkpoints save/load correctly

**Status: complete (2026-10-26).** See `docs/stage5_rl_report.md`.

---

## Stage 6 — Experimental Evaluation Framework

**Duration**: 2 weeks

### Objective
Build automated multi-seed experiment execution and statistical evaluation.

### Components
- Scenario configuration system (A-E + stress sweeps)
- Experiment runner
- Multi-seed execution (30-50 seeds per condition)
- Metric collection
- Results aggregation
- Statistical significance testing
- Effect size calculation

### Expected Output
- All scenario configurations
- Automated experiment execution
- Statistical results with CIs and significance
- Results exportable to CSV/JSON

### Testing Requirements
- Test scenario configs load correctly
- Test experiment runs to completion
- Test results are persisted
- Test statistical functions

### Completion Criteria
- [ ] All 5+ scenario families configurable
- [ ] Multi-seed execution works
- [ ] Results stored correctly
- [ ] Statistical tests implemented
- [ ] Results exportable

---

## Stage 7 — Robustness and Generalization Analysis

**Duration**: 1.5 weeks

### Objective
Evaluate policy generalization and robustness under stress.

### Components
- Train/test scenario separation validation
- Generalization testing (normal train → stress eval)
- Robustness degradation metrics
- Ablation studies (where scientifically justified)
- Performance comparison tables
- Policy-structure analysis: PPO hold probabilities vs free beds, compared with tuned
  trunk-reservation thresholds (RESEARCH.md §6.20)
- Sensitivity analysis: off-service LOS multiplier and deterioration hazard
- PPO-DR (domain-randomised training on a strictly milder range) vs PPO on A–E
  (RESEARCH.md §6.23)
- CRN variance-reduction measurement: paired vs independent-seed CI widths (RESEARCH.md §6.24)
- Optional: PPO-Lagrangian (diversion as a constraint) vs fixed-β PPO under stress
  (RESEARCH.md §6.19; only after core results are complete)

### Expected Output
- Generalization results
- Robustness analysis
- Ablation findings (if conducted)
- Comparative performance summary

### Testing Requirements
- Verify train/test separation
- Verify no retraining on test scenarios
- Validate robustness metrics

### Completion Criteria
- [ ] Policy-structure analysis reported (threshold present, absent, or state-dependent)
- [ ] RL trained only on normal scenario
- [ ] Evaluation covers all stress scenarios
- [ ] No retraining per scenario
- [ ] Robustness degradation measured
- [ ] Ablations documented (if conducted)

---

## Stage 8 — What-If Engine

**Duration**: 1.5 weeks

### Objective
Implement counterfactual analysis capability.

### Components
- State cloning for counterfactual
- Policy evaluation from cloned states
- Multi-policy comparison
- What-if API endpoints
- What-if dashboard interface

### Expected Output
- Counterfactual analysis from any state
- Policy comparison from same initial conditions
- What-if dashboard panel

### Testing Requirements
- Test state cloning produces independent copy
- Test policy evaluation from cloned state
- Test multi-policy comparison

### Completion Criteria
- [ ] States clone correctly
- [ ] Counterfactual simulation works
- [ ] Policies compare from same initial state
- [ ] What-if interface functional

---

## Stage 9 — Dashboard

**Duration**: 2 weeks

### Objective
Create interactive web dashboard for visualization and control.

### Components
- Next.js project setup
- Simulation control panel
- Real-time state visualization
- Metrics charts with statistical summaries
- What-if comparison display
- Decision log display

### Expected Output
- Running dashboard in development
- Real-time hospital visualization
- Interactive simulation controls
- Metrics with confidence intervals
- What-if results display

### Testing Requirements
- Dashboard builds without errors
- Test API connections
- Test real-time updates
- Test simulation controls

### Completion Criteria
- [ ] Dashboard runs in development mode
- [ ] Simulation controllable through UI
- [ ] State visualizes correctly
- [ ] Charts show metrics with CIs
- [ ] What-if results display

---

## Stage 10 — Explanation Layer

**Duration**: 1 week

### Objective
Implement post-hoc LLM explanation for decisions.

### Components
- Decision logger with context
- Context retriever
- Prompt constructor
- LLM integration
- Explanation formatter
- Reliability indicator (clearly labeled, not statistical confidence)

### Expected Output
- Natural language explanations
- Decision context retrieval
- Clear limitation disclaimers
- Explanation API endpoints

### Testing Requirements
- Test explanation generation
- Test context retrieval
- Test explanation formatting
- Verify LLM cannot control simulation

### Completion Criteria
- [ ] Explanations generate for sample decisions
- [ ] Explanations reference correct context
- [ ] Limitations clearly stated
- [ ] LLM outside decision loop verified

---

## Stage 11 — Integration and Performance

**Duration**: 1 week

### Objective
Integrate components and validate end-to-end functionality.

### Components
- Full system integration
- End-to-end tests
- Performance optimization
- Bug fixes
- Full pipeline validation

### Expected Output
- Complete integrated system
- All components working together
- Performance validated (10x real-time target)
- Known issues documented

### Testing Requirements
- Full system integration tests
- End-to-end scenario tests
- Performance benchmarks
- Load testing

### Completion Criteria
- [ ] All components integrated
- [ ] Full simulation runs end-to-end
- [ ] Performance acceptable
- [ ] No critical bugs

---

## Stage 12 — Documentation and Presentation

**Duration**: 1.5 weeks

### Objective
Finalize documentation and prepare for submission.

### Components
- README update with setup instructions
- API documentation
- Architecture documentation final review
- Code comments and docstrings
- Reproducibility package
- Project presentation materials

### Expected Output
- Complete documentation
- Clean code base
- Presentation-ready materials

### Testing Requirements
- Documentation review
- Code review
- Reproducibility check

### Completion Criteria
- [ ] README complete
- [ ] API documented
- [ ] All docstrings present
- [ ] Code follows style guide
- [ ] Presentation ready

---

## Stage Summary

| Stage | Duration | Focus |
|-------|----------|-------|
| 1 | 2 weeks | Research Foundation |
| 2 | 3 weeks | Simulation |
| 3 | 1.5 weeks | Digital Twin |
| 4 | 2 weeks | Baselines |
| 5 | 4 weeks | RL Environment & In-house PPO |
| 6 | 2 weeks | Experiments |
| 7 | 1.5 weeks | Robustness Analysis |
| 8 | 1.5 weeks | What-If Engine |
| 9 | 2 weeks | Dashboard |
| 10 | 1 week | Explanation Layer |
| 11 | 1 week | Integration |
| 12 | 1.5 weeks | Documentation |
| **Total** | **~21.5 weeks** | |

## Stage Priority Order

If time is constrained, prioritize in this order:
1. Stage 1 (Research) - essential for valid research
2. Stage 2 (Simulation) - foundation of everything
3. Stage 4 (Baselines) - required for comparison
4. Stage 5 (RL) - core research contribution
5. Stage 6 (Experiments) - required for evaluation
6. Stage 7 (Robustness) - key research question
7. Stage 3 (Digital Twin) - needed for what-if
8. Stage 8 (What-If) - differentiating feature
9. Stage 10 (Explanation) - post-hoc only
10. Stage 9 (Dashboard) - visualization
11. Stage 11 (Integration) - validation
12. Stage 12 (Documentation) - final polish
