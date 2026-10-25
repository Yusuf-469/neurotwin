"""PPO losses (Schulman et al., 2017)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossInfo:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def ppo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
             new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
             entropy: torch.Tensor, clip: float, vf_clip: float | None, ent_coef: float,
             vf_coef: float) -> tuple[torch.Tensor, LossInfo]:
    """Total loss = clipped surrogate + vf_coef * value loss - ent_coef * entropy.

    Clipped surrogate: L = E[max(-A * r, -A * clip(r, 1 - ε, 1 + ε))], r = π_new / π_old.
    Inside the trust region (|r - 1| <= ε) this equals the plain policy-gradient surrogate
    -A * r (tested); outside, the gradient is cut off in the direction that would move the
    policy further.
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    pg_loss = torch.max(-advantages * ratio,
                        -advantages * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
    if vf_clip is None:
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
    else:
        # [37 details] Value-loss clipping, as in the original PPO implementation.
        clipped = old_values + torch.clamp(new_values - old_values, -vf_clip, vf_clip)
        v_loss = 0.5 * torch.max((new_values - returns) ** 2, (clipped - returns) ** 2).mean()
    ent = entropy.mean()
    total = pg_loss + vf_coef * v_loss - ent_coef * ent
    with torch.no_grad():
        # [37 details] Low-variance KL estimator (k3, Schulman 2020) for monitoring.
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1).abs() > clip).float().mean().item()
    return total, LossInfo(pg_loss.item(), v_loss.item(), ent.item(), approx_kl, clip_frac)
