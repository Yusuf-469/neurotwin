"""PPO training loop over a list of environments (synchronous vectorisation)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

import gymnasium as gym
import numpy as np
import torch

from .agent import PPOAgent
from .gae import compute_gae
from .losses import ppo_loss
from .masking import masked_distribution
from .networks import ActorCritic
from .normalize import IdentityNormalizer, RunningMeanStd


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 256                 # rollout length per env per update
    learning_rate: float = 3e-4
    anneal_lr: bool = True             # [37 details] linear learning-rate annealing
    n_epochs: int = 10
    n_minibatches: int = 8
    gamma: float = 0.99                # per unit of dt (per hour for the hospital env)
    semi_markov: bool = True           # discount gamma ** dt; False = gamma per step (used
                                       # only for matched comparisons with the reference PPO)
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_clip: float | None = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5         # [37 details] global gradient clipping
    norm_adv: bool = True              # [37 details] per-minibatch advantage normalisation
    normalize_obs: bool = True
    hidden: tuple[int, ...] = (64, 64)
    seed: int = 0                      # network init, sampling and minibatch order
    torch_threads: int = 1

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps


@dataclass
class TrainLog:
    updates: list[dict] = field(default_factory=list)
    episode_returns: list[tuple[int, float]] = field(default_factory=list)   # (global step, R)


def train(env_fns: list[Callable[[], gym.Env]], config: PPOConfig,
          episode_seeds: Iterator[int], log_every: int = 0,
          callback: Callable[[int, PPOAgent, TrainLog], None] | None = None
          ) -> tuple[PPOAgent, TrainLog]:
    """Train PPO. ``episode_seeds`` yields a seed for every episode reset (e.g. TRAIN seeds).

    Each env may provide ``info["action_mask"]`` (default: all valid) and ``info["dt_hours"]``
    (default: 1, i.e. ordinary MDP discounting).
    """
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    rng = np.random.default_rng(config.seed)
    envs = [fn() for fn in env_fns]
    n_envs, n_steps = len(envs), config.n_steps
    obs_dim = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n
    model = ActorCritic(obs_dim, n_actions, config.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)  # [37 details] Adam epsilon 1e-5
    obs_rms = RunningMeanStd((obs_dim,)) if config.normalize_obs else IdentityNormalizer((obs_dim,))
    agent = PPOAgent(model, obs_rms, {"config": asdict(config)})
    log = TrainLog()

    def reset(i: int):
        o, info = envs[i].reset(seed=next(episode_seeds))
        return o, info.get("action_mask", np.ones(n_actions, dtype=bool))

    raw_obs, masks = map(list, zip(*(reset(i) for i in range(n_envs))))
    ep_returns = np.zeros(n_envs)
    buf_obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    buf_mask = np.zeros((n_steps, n_envs, n_actions), dtype=bool)
    buf_act = np.zeros((n_steps, n_envs), dtype=np.int64)
    buf_logp = np.zeros((n_steps, n_envs))
    buf_val = np.zeros((n_steps, n_envs))
    buf_rew = np.zeros((n_steps, n_envs))
    buf_gam = np.zeros((n_steps, n_envs))
    buf_done = np.zeros((n_steps, n_envs))

    n_updates = max(1, config.total_steps // config.batch_size)
    global_step, t_start = 0, time.perf_counter()
    for update in range(1, n_updates + 1):
        if config.anneal_lr:
            optimizer.param_groups[0]["lr"] = config.learning_rate * (1 - (update - 1) / n_updates)
        for t in range(n_steps):
            obs_rms.update(np.asarray(raw_obs))
            norm = obs_rms.normalize(np.asarray(raw_obs))
            mask = np.asarray(masks)
            with torch.no_grad():
                logits, values = model(torch.as_tensor(norm))
                dist = masked_distribution(logits, torch.as_tensor(mask))
                actions = dist.sample()
                logps = dist.log_prob(actions)
            buf_obs[t], buf_mask[t] = norm, mask
            buf_act[t], buf_logp[t], buf_val[t] = actions.numpy(), logps.numpy(), values.numpy()
            for i, env in enumerate(envs):
                o, r, term, trunc, info = env.step(int(actions[i]))
                dt = info.get("dt_hours", 1.0) if config.semi_markov else 1.0
                gamma_t = config.gamma ** dt
                ep_returns[i] += r
                if trunc and not term:
                    # Time-limit truncation: bootstrap from the value of the final state.
                    with torch.no_grad():
                        v_final = model.value(torch.as_tensor(obs_rms.normalize(o)).unsqueeze(0))
                    r = r + gamma_t * float(v_final)
                buf_rew[t, i], buf_gam[t, i] = r, gamma_t
                buf_done[t, i] = float(term or trunc)
                if term or trunc:
                    log.episode_returns.append((global_step + (t + 1) * n_envs, ep_returns[i]))
                    ep_returns[i] = 0.0
                    o, m = reset(i)
                else:
                    m = info.get("action_mask", np.ones(n_actions, dtype=bool))
                raw_obs[i], masks[i] = o, m
        global_step += n_steps * n_envs

        with torch.no_grad():
            last_values = model.value(torch.as_tensor(obs_rms.normalize(np.asarray(raw_obs)))).numpy()
        adv, ret = compute_gae(buf_rew, buf_val, buf_gam, buf_done, last_values, config.gae_lambda)

        b_obs = torch.as_tensor(buf_obs.reshape(-1, obs_dim))
        b_mask = torch.as_tensor(buf_mask.reshape(-1, n_actions))
        b_act = torch.as_tensor(buf_act.reshape(-1))
        b_logp = torch.as_tensor(buf_logp.reshape(-1), dtype=torch.float32)
        b_val = torch.as_tensor(buf_val.reshape(-1), dtype=torch.float32)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        batch = config.batch_size
        mb_size = batch // config.n_minibatches
        infos = []
        for _ in range(config.n_epochs):
            perm = rng.permutation(batch)
            for start in range(0, batch, mb_size):
                idx = torch.as_tensor(perm[start:start + mb_size])
                logits, new_values = model(b_obs[idx])
                dist = masked_distribution(logits, b_mask[idx])
                mb_adv = b_adv[idx]
                if config.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                loss, info = ppo_loss(dist.log_prob(b_act[idx]), b_logp[idx], mb_adv, new_values,
                                      b_val[idx], b_ret[idx], dist.entropy(), config.clip,
                                      config.vf_clip, config.ent_coef, config.vf_coef)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                infos.append(info)

        explained = 1 - np.var(ret - buf_val) / (np.var(ret) + 1e-8)          # [37 details] explained variance (diagnostic)
        recent = [r for _, r in log.episode_returns[-20:]]
        entry = {"update": update, "step": global_step,
                 "policy_loss": float(np.mean([i.policy_loss for i in infos])),
                 "value_loss": float(np.mean([i.value_loss for i in infos])),
                 "entropy": float(np.mean([i.entropy for i in infos])),
                 "approx_kl": float(np.mean([i.approx_kl for i in infos])),
                 "clip_fraction": float(np.mean([i.clip_fraction for i in infos])),
                 "explained_variance": float(explained),
                 "mean_return_last20": float(np.mean(recent)) if recent else float("nan"),
                 "sps": global_step / (time.perf_counter() - t_start)}
        log.updates.append(entry)
        if log_every and update % log_every == 0:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()},
                  flush=True)
        if callback is not None:
            callback(update, agent, log)
    obs_rms.frozen = True
    return agent, log
