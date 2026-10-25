"""Actor-critic MLP."""

from __future__ import annotations

import math

import torch
from torch import nn


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    # [37 details] Orthogonal initialisation with gain sqrt(2) for hidden layers, 0.01 for the
    # policy output (near-uniform initial policy) and 1.0 for the value output; zero biases.
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(n_in: int, hidden: tuple[int, ...], n_out: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [_layer(n_in, h, math.sqrt(2)), nn.Tanh()]
        n_in = h
    layers.append(_layer(n_in, n_out, out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic networks ([37 details]: no shared trunk, the default for small
    MLP problems; avoids the policy and value losses competing for the same features)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, n_actions, out_gain=0.01)
        self.critic = _mlp(obs_dim, hidden, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
