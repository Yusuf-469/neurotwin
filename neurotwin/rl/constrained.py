"""Constants of the λ-conditioned constrained PPO (RESEARCH.md v2.6 §6.19), fixed before any run.

Training draws λ ~ U[0, LAMBDA_MAX] per episode and appends λ / LAMBDA_MAX to the observation.
Deployment runs online dual ascent every UPDATE_HOURS:
    λ <- clip(λ + ETA * (diversions per 100 arrivals in the last 24 h - BUDGET), 0, LAMBDA_MAX)
starting from LAMBDA0 (the fixed-β value, config/reward_default.json).
"""

LAMBDA_MAX = 100.0
LAMBDA0 = 20.0
BUDGET_PER_100 = 3.0
ETA = 5.0
UPDATE_HOURS = 24.0


def dual_ascent_step(lam: float, diversions_per_100: float) -> float:
    """One projected dual-ascent update of the diversion multiplier."""
    return min(LAMBDA_MAX, max(0.0, lam + ETA * (diversions_per_100 - BUDGET_PER_100)))
