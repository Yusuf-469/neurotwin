"""In-house PPO (Schulman et al., 2017) with GAE (Schulman et al., 2016), logit-level action
masking and semi-Markov discounting (RESEARCH.md v2 §6.18; ARCHITECTURE.md §8.5).

Implementation choices follow Huang et al., "The 37 Implementation Details of Proximal Policy
Optimization" (ICLR Blog Track, 2022); each detail used is marked "[37 details]" with its
name where it is implemented.
stable-baselines3 is used only as a reference to validate this implementation.
"""

from .agent import PPOAgent
from .trainer import PPOConfig, train

__all__ = ["PPOAgent", "PPOConfig", "train"]
