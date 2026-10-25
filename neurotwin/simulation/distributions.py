"""Duration sampling by inverse CDF from a single uniform draw.

Why inverse CDF: every patient consumes a *fixed* number of uniforms from the
``patient_attributes`` stream, whatever the distribution type or the patient's pathway. This
keeps the stream aligned patient-by-patient (RESEARCH.md v2 §6.7) and makes it possible to
switch a distribution (e.g. to exponential for analytical validation) without disturbing any
other draw.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from .config import DistributionSpec

_STANDARD_NORMAL = NormalDist()
# Uniforms come from Generator.random(), which is in [0, 1). Clamp away from the endpoints so
# the inverse CDFs stay finite. The clamp is reached with probability ~1e-15 per draw.
_EPS = 1e-15


def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """(mu, sigma) of the underlying normal for a lognormal with the given mean and sd."""
    sigma2 = math.log1p((sd / mean) ** 2)
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


def sample_duration(spec: DistributionSpec, u: float) -> float:
    """Map one uniform u in [0, 1) to a duration with distribution ``spec``."""
    u = min(max(u, _EPS), 1.0 - _EPS)
    if spec.dist == "fixed":
        return spec.mean
    if spec.dist == "exponential":
        return -spec.mean * math.log1p(-u)
    mu, sigma = lognormal_params(spec.mean, spec.sd)
    return math.exp(mu + sigma * _STANDARD_NORMAL.inv_cdf(u))


def conditional_residual(spec: DistributionSpec, elapsed: float, u: float) -> float:
    """Remaining duration given that ``elapsed`` hours have already passed, from one uniform.

    Samples T - e from the distribution of T conditional on T > e by inverse CDF:
    p = F(e) + u * (1 - F(e)), T = F^-1(p). This is how a what-if branch re-randomises an
    activity already in progress without assuming it just started (the conditional-LOS idea
    used in symbiotic hospital simulation, Oakley, Onggo & Worthington, 2020). For the
    exponential distribution this reduces to memorylessness.
    """
    if elapsed < 0:
        raise ValueError("elapsed must be non-negative")
    u = min(max(u, _EPS), 1.0 - _EPS)
    if spec.dist == "fixed":
        return max(spec.mean - elapsed, 0.0)
    if spec.dist == "exponential":
        return -spec.mean * math.log1p(-u)
    mu, sigma = lognormal_params(spec.mean, spec.sd)
    f_e = _STANDARD_NORMAL.cdf((math.log(elapsed) - mu) / sigma) if elapsed > 0 else 0.0
    p = min(f_e + u * (1.0 - f_e), 1.0 - _EPS)
    # If elapsed is far in the tail (F(e) within ~1e-15 of 1), p is clamped and the quantile
    # can fall below e; the residual is then ~0 and the activity ends immediately.
    return max(math.exp(mu + sigma * _STANDARD_NORMAL.inv_cdf(p)) - elapsed, 0.0)


def exponential_from_uniform(rate: float, u: float) -> float:
    """Exponential(rate) by inverse CDF; rate 0 gives infinity (event never happens)."""
    if rate <= 0:
        return math.inf
    u = min(max(u, _EPS), 1.0 - _EPS)
    return -math.log1p(-u) / rate
