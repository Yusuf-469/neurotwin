import numpy as np
import pytest
from scipy import stats

from neurotwin.simulation import SCENARIOS, load_default_config
from neurotwin.simulation.arrivals import (HOURS_PER_WEEK, PiecewiseConstantRate,
                                           ThinningArrivalProcess, elective_rate, emergency_rate)


def test_integral_of_piecewise_rate():
    rate = PiecewiseConstantRate(np.arange(HOURS_PER_WEEK, dtype=float))
    assert rate.integral(0.5, 2.25) == pytest.approx(0 * 0.5 + 1 * 1 + 2 * 0.25)
    assert rate.integral(0, 2 * HOURS_PER_WEEK) == pytest.approx(2 * rate.rates.sum())


def test_base_rates_are_weekly_mean_rates():
    cfg = load_default_config()
    for fn, base in ((emergency_rate, cfg.emergency_base_rate), (elective_rate, cfg.elective_base_rate)):
        assert fn(cfg, SCENARIOS["A"]).rates.mean() == pytest.approx(base)
        assert fn(cfg, SCENARIOS["B"]).rates.mean() == pytest.approx(1.35 * base)


def test_electives_only_in_window():
    cfg = load_default_config()
    rates = elective_rate(cfg, SCENARIOS["A"]).rates.reshape(7, 24)
    assert np.all(rates[5:] == 0)                  # weekend
    assert np.all(rates[:5, :8] == 0) and np.all(rates[:5, 16:] == 0)
    assert np.all(rates[:5, 8:16] > 0)


def test_thinning_reproduces_hour_of_week_profile():
    """RESEARCH.md §6.17 'NHPP arrivals': chi-square goodness of fit per hour-of-week bin."""
    cfg = load_default_config()
    rate = emergency_rate(cfg, SCENARIOS["A"])
    process = ThinningArrivalProcess(rate, np.random.default_rng(12345))
    weeks = 60
    counts = np.zeros(HOURS_PER_WEEK)
    t = process.next_arrival(0.0)
    while t < weeks * HOURS_PER_WEEK:
        counts[int(t) % HOURS_PER_WEEK] += 1
        t = process.next_arrival(t)
    expected = rate.rates * weeks
    # Total count is Poisson(sum expected); compare shape conditional on the total.
    _, p_value = stats.chisquare(counts, expected * counts.sum() / expected.sum())
    assert p_value > 0.001
    assert counts.sum() == pytest.approx(expected.sum(), rel=0.03)
