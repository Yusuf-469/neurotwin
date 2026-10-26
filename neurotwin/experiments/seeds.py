"""Disjoint seed ranges (RESEARCH.md v2 §6.12, §6.24).

Defined in one place so that train/validation/test separation can be asserted in code rather
than trusted. Ranges are half-open [start, stop).
"""

from __future__ import annotations

TRAIN = range(0, 100_000)  # environment seeds for training (scenario A only)
VALIDATION = range(100_000, 100_020)  # tuning and early stopping (scenario A only)
TEST = range(200_000, 200_050)  # final evaluation, A-E
CRN_INDEPENDENT = range(300_000, 300_050)  # §6.24 variance-reduction check only
FRESH_TEST = range(400_000, 400_050)  # §6.26 one-time final test of the selected candidate

_ALL = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
    "crn_independent": CRN_INDEPENDENT,
    "fresh_test": FRESH_TEST,
}


def split_of(seed: int) -> str:
    """Name of the registered seed range containing ``seed``."""
    for name, rng in _ALL.items():
        if seed in rng:
            return name
    raise ValueError(f"seed {seed} is outside every registered range")


def require_split(seeds, split: str) -> None:
    """Raise if any seed is not in ``split`` (guards against train/test leakage)."""
    bad = [s for s in seeds if s not in _ALL[split]]
    if bad:
        raise ValueError(f"seeds {bad[:5]} are not in the {split!r} range")
