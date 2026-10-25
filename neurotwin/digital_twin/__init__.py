"""Digital twin: observable state, history, snapshots and warm starts (ARCHITECTURE.md §3.3).

Two kinds of state are kept strictly apart:

* ``HospitalState`` is what a policy, the API or the dashboard may see: only information
  available at decision time (RESEARCH.md §6.2).
* A *snapshot* (``HospitalSimulation.to_snapshot``) is the complete simulator state, including
  hidden patient attributes and RNG states. It exists for reproducibility, what-if branching
  and warm starts, and must never be passed to a policy.
"""

from .history import RecordingPolicy, StateHistory
from .snapshot_io import load_snapshot, save_snapshot
from .state import HospitalState, observe
from .warm_start import WarmStartPool

__all__ = ["HospitalState", "observe", "StateHistory", "RecordingPolicy", "save_snapshot",
           "load_snapshot", "WarmStartPool"]
