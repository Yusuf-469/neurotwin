"""Running observation normalisation ([37 details]).

Mean and variance are tracked with the parallel (Chan et al.) update over each batch of raw
observations. They are frozen at evaluation so a deployed policy is a fixed function of the
observation.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Running mean/variance normaliser with clipping; can be frozen."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.clip = clip
        self.frozen = False

    def update(self, batch: np.ndarray) -> None:
        if self.frozen:
            return
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "RunningMeanStd":
        rms = cls(np.asarray(d["mean"]).shape, clip=d["clip"])
        rms.mean, rms.var, rms.count = np.asarray(d["mean"]), np.asarray(d["var"]), d["count"]
        return rms


class IdentityNormalizer(RunningMeanStd):
    """No-op normaliser (used for matched comparisons with the reference implementation)."""

    def update(self, batch: np.ndarray) -> None:
        return

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)
