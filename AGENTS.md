# NeuroTwin V2 - AI Coding Agent Instructions

This file contains strict instructions for AI coding agents working on the NeuroTwin V2 project. All agents must read and follow these guidelines.

## General Principles

### 1. Incremental Implementation
- **NEVER** implement the entire project at once
- Work on one stage at a time as defined in ROADMAP.md
- Complete each stage fully before moving to the next
- Request clarification if requirements conflict

### 2. Code Inspection Before Modification
- **ALWAYS** read existing code before modifying it
- Use the `read` tool to inspect files before editing
- Understand the existing patterns and conventions
- Check related files to understand context
- Do not modify code you have not read

### 3. Scope Control
- **DO NOT** modify unrelated files
- When implementing a feature, only change files relevant to that feature
- If a change requires modifying unrelated code, ask for confirmation first
- Be aware of file interdependencies

### 4. Architectural Decisions
- **ALWAYS** explain architectural decisions in code comments
- Document why a particular approach was chosen
- If a significant architectural change is needed, ask for confirmation before implementing
- Reference ARCHITECTURE.md when making architecture-related decisions

### 5. Dependency Management
- **AVOID** unnecessary dependencies
- Before adding a new dependency:
  1. Verify it solves a problem that cannot be solved with existing dependencies
  2. Check if the dependency is already used in the project
  3. Ensure the dependency is actively maintained
  4. Consider the added maintenance burden
- Prefer standard library when possible
- Document the purpose of each dependency added
- **Do NOT add infrastructure (Redis, MLflow, Kubernetes, AWS) unless there is a concrete requirement**

### 6. Testing Requirements
- **ALWAYS** write tests for new code
- Unit tests are required for all new classes and functions
- Integration tests required for new components
- Tests must pass before marking a stage complete
- Do not write tests that cannot fail (e.g., `assert True`)

### 7. Secret Management
- **NEVER** hardcode secrets in source code
- Use environment variables or configuration files for secrets
- Never commit secrets to version control
- If a secret is needed, use `os.environ.get()` or similar patterns
- Never log secrets

## Research Integrity Rules

### 8. Experimental Results Integrity
- **NEVER** fabricate experimental results
- **NEVER** modify results to make them look better
- **NEVER** claim real-world medical effectiveness
- **NEVER** claim clinical validity
- All metrics must come from actual simulation runs
- All claims must be supported by reproducible experiments
- Include confidence intervals and statistical significance where applicable
- Report negative results honestly

### 9. Real-World Claims
- **NEVER** claim that NeuroTwin improves real hospital operations
- **NEVER** suggest the system is ready for clinical deployment
- **NEVER** present simulation results as validated real-world outcomes
- **NEVER** treat simulation outcomes as real-world evidence
- Always qualify results as "in simulation" or "in the simulated environment"
- The system is a research project, not a medical device

### 10. Reproducibility
- **ALWAYS** use seeded random number generation
- Document all random seeds used in experiments
- Make all configuration parameters explicit
- Ensure experiments can be reproduced with the same seed
- Log all parameters that affect simulation behavior
- **Always preserve train/test scenario separation**

### 11. Reward/Metric Manipulation
- **NEVER** modify reward functions or metrics to artificially improve results
- Reward formulations must be experimentally justified
- Do not optimize solely for utilization
- Document any changes to reward/metrics and justify them

### 12. Training/Evaluation Separation
- **NEVER** retrain RL model specifically on a test scenario unless the experiment explicitly studies that setting
- Always maintain separation between training and evaluation conditions
- Document any deviation from this principle

## Code Organization

### 13. Code Organization
- **KEEP** research code separate from production code
- Use `simulation/` for simulation-specific code
- Use `digital_twin/` for state management
- Use `policies/` for allocation policies (heuristic, optimization, RL)
- Use `rl/` for reinforcement learning
- Use `experiments/` for experiment infrastructure
- Use `whatif/` for what-if engine
- Use `explanation/` for LLM explanation layer
- Use `api/` for FastAPI backend
- Use `dashboard/` for Next.js frontend
- Do not mix concerns across directories

### 14. Documentation
- **DOCUMENT** all assumptions in code comments
- Document why certain design decisions were made
- Document limitations of the simulation
- Document when code makes simplifying assumptions
- Update docstrings when changing code behavior

## Pre-Implementation Checklist

Before implementing any feature, verify:
- [ ] Read and understood ARCHITECTURE.md
- [ ] Read and understood relevant sections of ROADMAP.md
- [ ] Read existing similar code to understand patterns
- [ ] Understood what the feature should and should not do
- [ ] Identified all files that need to be created or modified
- [ ] Identified all dependencies required (if any)
- [ ] Planned for tests

## Pre-Modification Checklist

Before modifying existing code, verify:
- [ ] Read the entire file you are modifying
- [ ] Read related files to understand context
- [ ] Understood how the existing code works
- [ ] Understood what your change will do
- [ ] Identified all places that use the code you are changing
- [ ] Have a plan for testing the change

## Major Architectural Changes

For any of the following, **ask for confirmation before proceeding**:
- Adding new major components not in ARCHITECTURE.md
- Changing the data flow between components
- Modifying entity or event models
- Changing the RL environment interface
- Adding new external service integrations
- Modifying the database schema
- Changing API endpoints
- Adding new dependencies
- Operating room modeling (out of V1 scope)

## LLM Explanation Layer Rules

### 15. LLM Safety Constraints
- **The LLM explanation layer must NEVER control or override simulation decisions**
- Correct flow: RL/Policy → Decision → Simulation → Decision Log → LLM Explanation
- Incorrect flow: RL → LLM → Decision
- Never allow LLM to modify simulation state
- Never allow LLM to override policy decisions
- Never allow LLM to fabricate simulation data
- Always distinguish facts from generated explanations
- Do not call an LLM-generated score "confidence" unless it is actually a statistical confidence measure; use "reliability indicator" instead

## Stage Completion

When completing a stage from ROADMAP.md:
1. All tests pass
2. Documentation is updated
3. No new dependencies added without justification
4. Code follows existing patterns
5. All completion criteria from ROADMAP.md are met

## Directory Structure

```
neurotwin/
├── simulation/          # Discrete-event simulation
├── digital_twin/        # State management and what-if
├── policies/            # Allocation policies (FCFS, Priority, Optimization, RL)
├── rl/                  # Reinforcement learning
├── experiments/         # Experiment infrastructure
├── whatif/              # What-if/counterfactual engine
├── explanation/         # LLM explanation layer
├── api/                 # FastAPI backend
├── dashboard/           # Next.js frontend
├── tests/              # Test suite
├── docs/               # Documentation
└── config/             # Configuration files
```

## Technology Guidelines

### Approved Technologies
- **Simulation**: Python standard library + custom DES framework
- **ML/Statistics**: scikit-learn, PyTorch, numpy, pandas
- **RL**: Gymnasium, stable-baselines3 (PPO primary)
- **API**: FastAPI, Pydantic
- **Database**: SQLite (initial), PostgreSQL (if needed)
- **Dashboard**: Next.js, React, TypeScript, Chart.js/Recharts
- **Container**: Docker (for development reproducibility)

### Technology Decision Process
1. State the problem you are trying to solve
2. Evaluate if existing approved technologies can solve it
3. If no, research alternatives with clear justification
4. Propose the new technology with rationale
5. Wait for confirmation before adding

### NOT Mandatory (Add Only If Needed)
- Redis
- MLflow
- Kubernetes
- AWS
- SAC, DQN (PPO is primary RL algorithm)

## File Naming Conventions

- Python modules: `snake_case.py`
- Python classes: `PascalCase`
- Python functions/methods: `snake_case`
- TypeScript files: `camelCase.tsx` or `camelCase.ts`
- Configuration: `snake_case.yaml` or `snake_case.json`
- Tests: `test_*.py` or `*_test.py`

## Code Style

- Follow PEP 8 for Python code
- Follow TypeScript/React conventions for frontend
- Use type hints in Python
- Use interfaces for TypeScript
- Maximum line length: 100 characters
- Use meaningful variable names

## Commit Messages

- Use clear, descriptive commit messages
- Start with verb: "Add", "Fix", "Update", "Remove", "Refactor"
- Reference issue numbers when applicable
- Keep commits focused on a single change

## Priority Principles

When in doubt, prioritize:
1. **Research validity** over dashboard polish
2. **Reproducibility** over convenience
3. **Statistical rigor** over speed of results
4. **Honest reporting** over positive results
5. **Core implementation** over infrastructure

## When Uncertain

If you are uncertain about:
- Whether a change is within scope
- How to implement something
- Whether a dependency is needed
- Whether a test is sufficient
- Whether a claim is justified

**Ask for clarification before proceeding.**

## Important Reminders

> **This is a research project for a final-year CSE-AIML project.**
> The goal is to demonstrate the potential of AI-driven resource allocation in simulation, NOT to build a production hospital system.
>
> **Do not claim real-world effectiveness.**
> Do not suggest deployment readiness.
> Do not fabricate results.
> Do not treat simulation as reality.
>
> **The simulation is a simplified model.**
> Real hospital operations are far more complex and involve clinical, legal, and ethical factors not captured in this simulation.
>
> **Research results have priority over dashboard polish.**
> The research engine is the core product.
