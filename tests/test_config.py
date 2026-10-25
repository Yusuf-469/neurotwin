import json

import pytest

from neurotwin.simulation import SCENARIOS, SimulationConfig, load_default_config
from neurotwin.simulation.config import DEFAULT_CONFIG_PATH, round_half_up


def _raw_default() -> dict:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_default_config_loads_and_profiles_are_normalised():
    cfg = load_default_config()
    assert sum(cfg.hourly_profile) / 24 == pytest.approx(1.0)
    assert sum(cfg.weekly_profile) / 7 == pytest.approx(1.0)


def test_unknown_key_is_rejected():
    raw = _raw_default()
    raw["arival_rate"] = 3.0          # typo must not be silently ignored
    with pytest.raises(ValueError, match="unknown"):
        SimulationConfig.from_dict(raw)


def test_missing_key_is_rejected():
    raw = _raw_default()
    del raw["doctors"]
    with pytest.raises(ValueError, match="missing"):
        SimulationConfig.from_dict(raw)


def test_comment_keys_are_ignored():
    raw = _raw_default()
    raw["_note"] = "anything"
    SimulationConfig.from_dict(raw)


@pytest.mark.parametrize("override", [
    {"priority_probs_emergency": [0.5, 0.5, 0.5, 0.0, 0.0]},
    {"warmup_hours": 10_000.0},                       # warm-up longer than the horizon
    {"beds": {"MED": 1, "SURG": 1}},
    {"ed_capacity": 0},
    {"general_los": {"dist": "lognormal", "mean": 10.0, "sd": 0.0}},
    {"escalation_priority": 6},
])
def test_invalid_values_are_rejected(override):
    with pytest.raises(ValueError):
        load_default_config().with_overrides(**override)


def test_scenarios_match_research_spec():
    # RESEARCH.md v2 §6.22 scenario table.
    assert SCENARIOS["B"].arrival_multiplier == 1.35
    assert SCENARIOS["C"].staff_multiplier == 0.65
    assert SCENARIOS["D"].capacity_multiplier == 0.6
    e = SCENARIOS["E"]
    assert (e.arrival_multiplier, e.staff_multiplier, e.capacity_multiplier) == (1.25, 0.7, 0.6)


def test_round_half_up():
    assert round_half_up(2.5) == 3 and round_half_up(3.5) == 4 and round_half_up(2.4) == 2
