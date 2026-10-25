"""Independent random-number streams for Common Random Numbers (RESEARCH.md v2 §6.7).

Each stochastic source draws only from its own stream, so a policy that changes *when* things
happen cannot shift any other random draw. A single shared generator would silently break CRN,
because different policies would consume draws in a different order.

The streams are spawned from one root seed with ``numpy.random.SeedSequence.spawn``, which
gives statistically independent child streams. The spawn order defines each stream's identity:
**new streams must only ever be appended to STREAM_NAMES**, or every existing seed would change.
"""

from __future__ import annotations

import numpy as np

STREAM_NAMES: tuple[str, ...] = (
    "arrivals_emergency",
    "arrivals_elective",
    "patient_attributes",
    "staffing",
)


class RandomStreams:
    """Named, independent ``numpy.random.Generator`` streams derived from one root seed."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed
        children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
        self._generators = {name: np.random.default_rng(child)
                            for name, child in zip(STREAM_NAMES, children)}

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._generators[name]
