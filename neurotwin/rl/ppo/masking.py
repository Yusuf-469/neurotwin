"""Logit-level action masking (Huang & Ontañón, 2022; RESEARCH.md §6.3).

Invalid actions get a logit of -1e9 before the softmax, so their probability underflows to
exactly 0 in float32. Their log-probabilities, entropy contributions and gradients are then
zero. A finite large negative number is used instead of -inf so that 0 * logit never produces
NaN in the entropy.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

MASK_VALUE = -1e9


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor | None) -> Categorical:
    if mask is not None:
        logits = torch.where(mask, logits, torch.full_like(logits, MASK_VALUE))
    return Categorical(logits=logits)
