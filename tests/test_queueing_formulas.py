"""The analytical formulas must be right before the simulator is judged against them."""

import math

import pytest

from neurotwin.simulation.validation.queueing import (erlang_b, erlang_c, mmc_mean_wait,
                                                      mmck_blocking, mmck_distribution,
                                                      mmck_mean_wait, mmck_wait_probability)


def _erlang_b_direct(c: int, a: float) -> float:
    terms = [a ** k / math.factorial(k) for k in range(c + 1)]
    return terms[-1] / sum(terms)


@pytest.mark.parametrize("c,a", [(1, 1.0), (5, 3.0), (10, 8.0), (20, 19.5)])
def test_erlang_b_matches_direct_formula(c, a):
    assert erlang_b(c, a) == pytest.approx(_erlang_b_direct(c, a), rel=1e-12)


def test_known_small_cases():
    assert erlang_b(1, 1.0) == pytest.approx(0.5)
    assert erlang_c(1, 0.7) == pytest.approx(0.7)                 # M/M/1: P(wait) = rho
    lam, mu = 0.7, 1.0
    assert mmc_mean_wait(1, lam, mu) == pytest.approx(lam / (mu * (mu - lam)))


def test_erlang_c_rejects_unstable_load():
    with pytest.raises(ValueError):
        erlang_c(10, 10.0)


@pytest.mark.parametrize("c,lam,mu", [(10, 8.0, 1.0), (3, 2.0, 1.0)])
def test_mmck_limits(c, lam, mu):
    # K = c is the loss system: blocking equals Erlang B.
    assert mmck_blocking(c, c, lam, mu) == pytest.approx(erlang_b(c, lam / mu), rel=1e-12)
    # K -> infinity recovers M/M/c (Erlang C).
    k = c + 3000
    assert mmck_wait_probability(c, k, lam, mu) == pytest.approx(erlang_c(c, lam / mu), rel=1e-9)
    assert mmck_mean_wait(c, k, lam, mu) == pytest.approx(mmc_mean_wait(c, lam, mu), rel=1e-9)


def test_mmck_distribution_sums_to_one_even_when_overloaded():
    p = mmck_distribution(10, 40, 30.0, 1.0)
    assert sum(p) == pytest.approx(1.0)
    assert p[-1] > 0.5                                              # heavily overloaded
