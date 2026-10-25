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


#: Namespace for what-if branches. Branch streams for seed s are statistically independent of the
#: episode streams for the same integer s, so branching can never replay an episode's future.
BRANCH_NAMESPACE = 1


class RandomStreams:
    """Named, independent ``numpy.random.Generator`` streams derived from one root seed.

    ``namespace`` 0 is used for episodes (and must stay the default so existing seeds keep
    their streams); other namespaces give independent families of streams for the same seed.
    """

    def __init__(self, seed: int, namespace: int = 0) -> None:
        if seed < 0 or namespace < 0:
            raise ValueError("seed and namespace must be non-negative")
        self.seed = seed
        self.namespace = namespace
        spawn_key = () if namespace == 0 else (namespace,)
        children = np.random.SeedSequence(seed, spawn_key=spawn_key).spawn(len(STREAM_NAMES))
        self._generators = {
            name: np.random.default_rng(child) for name, child in zip(STREAM_NAMES, children)
        }

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._generators[name]
