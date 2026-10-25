"""Trained agent: network + observation normaliser, with save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .masking import masked_distribution
from .networks import ActorCritic
from .normalize import IdentityNormalizer, RunningMeanStd


class PPOAgent:
    def __init__(self, model: ActorCritic, obs_rms: RunningMeanStd,
                 meta: dict[str, Any] | None = None) -> None:
        self.model = model
        self.obs_rms = obs_rms
        self.meta = meta or {}

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray | None = None,
            deterministic: bool = True) -> int:
        """Action for one raw observation. Deterministic = argmax of the masked policy."""
        x = torch.as_tensor(self.obs_rms.normalize(obs)).unsqueeze(0)
        logits, _ = self.model(x)
        m = None if mask is None else torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        dist = masked_distribution(logits, m)
        if deterministic:
            return int(dist.probs.argmax(dim=-1).item())
        return int(dist.sample().item())

    @torch.no_grad()
    def action_probabilities(self, obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        x = torch.as_tensor(self.obs_rms.normalize(obs)).unsqueeze(0)
        logits, _ = self.model(x)
        m = None if mask is None else torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        return masked_distribution(logits, m).probs[0].numpy()

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(),
                    "obs_rms": None if isinstance(self.obs_rms, IdentityNormalizer)
                    else self.obs_rms.state_dict(),
                    "obs_dim": self.model.actor[0].in_features,
                    "n_actions": self.model.actor[-1].out_features,
                    "hidden": [m.out_features for m in self.model.actor[:-1]
                               if isinstance(m, torch.nn.Linear)],
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        d = torch.load(path, weights_only=False)
        model = ActorCritic(d["obs_dim"], d["n_actions"], tuple(d["hidden"]))
        model.load_state_dict(d["model"])
        model.eval()
        rms = (IdentityNormalizer((d["obs_dim"],)) if d["obs_rms"] is None
               else RunningMeanStd.from_state_dict(d["obs_rms"]))
        rms.frozen = True
        return cls(model, rms, d["meta"])
