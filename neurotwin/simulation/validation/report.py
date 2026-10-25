"""Print the analytical validation table with 95% CIs (RESEARCH.md v2 §6.17).

Run: python -m neurotwin.simulation.validation.report [--heavy]

Uses the same reduced configurations and seeds (0..29) as tests/test_validation_mmck.py, so the
printed numbers are the evidence behind the tests. ``--heavy`` adds the rho = 0.95 M/M/c case.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import stats

from .queueing import (erlang_c, mmc_mean_wait, mmck_blocking, mmck_mean_wait,
                       mmck_wait_probability)
from .reduced import estimate_replication, mmck_config

C, MEAN_SERVICE, SEEDS = 10, 10.0, range(30)
MU = 1.0 / MEAN_SERVICE


def _ci95(samples: list[float]) -> tuple[float, float, float]:
    x = np.asarray(samples, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args()

    cases = [("M/M/10", 0.5, None, 4000.0, 400.0), ("M/M/10", 0.8, None, 4000.0, 400.0)]
    if args.heavy:
        cases.append(("M/M/10", 0.95, None, 30000.0, 3000.0))
    cases += [("M/M/10/15", 0.9, 5, 4000.0, 400.0), ("M/M/10/15", 1.2, 5, 4000.0, 400.0),
              ("M/M/10/12", 1.0, 2, 4000.0, 400.0)]

    print("| System | rho (offered) | Quantity | Analytical | Simulated mean | 95% CI | In CI |")
    print("|---|---|---|---|---|---|---|")
    for name, rho, q_max, horizon, warmup in cases:
        lam = rho * C * MU
        cfg = mmck_config(servers=C, arrival_rate=lam, mean_service=MEAN_SERVICE,
                          ed_capacity=q_max, horizon=horizon, warmup=warmup)
        est = [estimate_replication(cfg, s) for s in SEEDS]
        if q_max is None:
            rows = [("W_q (h)", mmc_mean_wait(C, lam, MU), [e.mean_wait for e in est]),
                    ("P(wait)", erlang_c(C, lam / MU), [e.wait_probability for e in est])]
        else:
            k = C + q_max
            rows = [("Blocking", mmck_blocking(C, k, lam, MU), [e.blocking for e in est]),
                    ("W_q (h)", mmck_mean_wait(C, k, lam, MU), [e.mean_wait for e in est]),
                    ("P(wait)", mmck_wait_probability(C, k, lam, MU),
                     [e.wait_probability for e in est])]
        for quantity, analytic, samples in rows:
            mean, lo, hi = _ci95(samples)
            ok = "yes" if lo <= analytic <= hi else "**no**"
            print(f"| {name} | {rho} | {quantity} | {analytic:.4f} | {mean:.4f} | "
                  f"[{lo:.4f}, {hi:.4f}] | {ok} |")


if __name__ == "__main__":
    main()
