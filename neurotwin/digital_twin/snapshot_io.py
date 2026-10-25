"""Snapshot files (complete simulator state, ARCHITECTURE.md §3.3).

Format: JSON as written by Python's ``json`` module with ``allow_nan=True`` (non-finite floats
such as "never deteriorates" are written as Infinity / NaN). gzip is used when the path ends
in ``.gz``. Snapshots contain hidden information and are for reproducibility and what-if
analysis only; they must not be served to policies or shown as observations.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ..simulation.hospital import HospitalSimulation


def save_snapshot(sim: HospitalSimulation, path: str | Path) -> None:
    """Write the complete simulator state to JSON (gzip if the path ends in .gz)."""
    path = Path(path)
    data = json.dumps(sim.to_snapshot())
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(data)
    else:
        path.write_text(data, encoding="utf-8")


def load_snapshot_dict(path: str | Path) -> dict[str, Any]:
    """Read a snapshot file into a plain dict."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(path: str | Path) -> HospitalSimulation:
    """Restore a simulation from a snapshot file."""
    return HospitalSimulation.from_snapshot(load_snapshot_dict(path))
