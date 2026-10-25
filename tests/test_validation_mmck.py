"""Statistical validation of the engine against M/M/c and M/M/c/K (RESEARCH.md v2 §6.17).

Each check runs 30 independent replications (fixed seeds 0..29) of the production engine in a
reduced configuration and requires the analytical value to lie inside the 99% t-interval of
the replication means.

Why 99% here when the report uses 95%: this file contains ~15 interval checks. At 95% the
chance that at least one fails purely by chance is about 50%, and seeds are fixed, so a chance
failure would be permanent and tempt seed shopping. 99% keeps the family-wise false-alarm rate
near 14% while still detecting real modelling errors, which shift estimates far outside the
interval. The 95% intervals are reported by the validation report script.
"""

import pytest

from neurotwin.simulation.validation.queueing import (erlang_c, mmc_mean_wait, mmck_blocking,
                                                      mmck_mean_wait, mmck_wait_probability)
from neurotwin.simulation.validation.reduced import estimate_replication, mmck_config

from .helpers import t_confidence_interval

SEEDS = range(30)
C, MEAN_SERVICE = 10, 10.0
MU = 1.0 / MEAN_SERVICE


def _replicate(config):
    return [estimate_replication(config, seed) for seed in SEEDS]


def _assert_in_ci(samples, analytic, label):
    lo, hi = t_confidence_interval(samples, 0.99)
    assert lo <= analytic <= hi, f"{label}: analytic {analytic:.4f} outside 99% CI [{lo:.4f}, {hi:.4f}]"


def _check_mmc(rho, horizon, warmup):
    lam = rho * C * MU
    estimates = _replicate(mmck_config(servers=C, arrival_rate=lam, mean_service=MEAN_SERVICE,
                                       ed_capacity=None, horizon=horizon, warmup=warmup))
    _assert_in_ci([e.mean_wait for e in estimates], mmc_mean_wait(C, lam, MU), "W_q")
    _assert_in_ci([e.wait_probability for e in estimates], erlang_c(C, lam / MU), "P(wait)")
    assert all(e.blocking == 0 for e in estimates)


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_mmc_matches_erlang_c(rho):
    _check_mmc(rho, horizon=4000.0, warmup=400.0)


@pytest.mark.slow
def test_mmc_matches_erlang_c_heavy_load():
    # Relaxation time grows like 1 / (1 - sqrt(rho))^2, so rho = 0.95 needs long runs.
    _check_mmc(0.95, horizon=30000.0, warmup=3000.0)


@pytest.mark.parametrize("rho_offered,ed_capacity", [(0.9, 5), (1.2, 5), (1.0, 2)])
def test_mmck_matches_finite_waiting_room(rho_offered, ed_capacity):
    lam = rho_offered * C * MU
    k = C + ed_capacity
    estimates = _replicate(mmck_config(servers=C, arrival_rate=lam, mean_service=MEAN_SERVICE,
                                       ed_capacity=ed_capacity, horizon=4000.0, warmup=400.0))
    _assert_in_ci([e.blocking for e in estimates], mmck_blocking(C, k, lam, MU), "blocking")
    _assert_in_ci([e.mean_wait for e in estimates], mmck_mean_wait(C, k, lam, MU), "W_q")
    _assert_in_ci([e.wait_probability for e in estimates],
                  mmck_wait_probability(C, k, lam, MU), "P(wait)")
