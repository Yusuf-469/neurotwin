import math

import numpy as np
import pytest

from neurotwin.simulation.config import DistributionSpec
from neurotwin.simulation.distributions import exponential_from_uniform, sample_duration
from neurotwin.simulation.rng import STREAM_NAMES, RandomStreams


def test_streams_are_reproducible_and_distinct():
    a, b = RandomStreams(7), RandomStreams(7)
    for name in STREAM_NAMES:
        assert np.array_equal(a[name].random(5), b[name].random(5))
    c = RandomStreams(7)
    draws = [tuple(c[name].random(3)) for name in STREAM_NAMES]
    assert len(set(draws)) == len(STREAM_NAMES)


def test_streams_are_independent_of_each_other():
    # Consuming one stream must not change another (the basis of CRN).
    a, b = RandomStreams(3), RandomStreams(3)
    a["arrivals_emergency"].random(1000)
    assert np.array_equal(a["patient_attributes"].random(10), b["patient_attributes"].random(10))


@pytest.mark.parametrize(
    "spec",
    [
        DistributionSpec("lognormal", 96.0, 72.0),
        DistributionSpec("lognormal", 3.0, 1.5),
        DistributionSpec("exponential", 10.0),
    ],
)
def test_inverse_cdf_sampling_matches_mean_and_sd(spec):
    u = np.random.default_rng(0).random(200_000)
    x = np.array([sample_duration(spec, ui) for ui in u])
    assert x.mean() == pytest.approx(spec.mean, rel=0.02)
    expected_sd = spec.mean if spec.dist == "exponential" else spec.sd
    assert x.std() == pytest.approx(expected_sd, rel=0.05)
    assert np.all(x > 0)


def test_fixed_and_extreme_uniforms():
    assert sample_duration(DistributionSpec("fixed", 2.0), 0.3) == 2.0
    spec = DistributionSpec("lognormal", 5.0, 2.0)
    assert math.isfinite(sample_duration(spec, 0.0)) and math.isfinite(sample_duration(spec, 1.0))
    assert exponential_from_uniform(0.0, 0.5) == math.inf
