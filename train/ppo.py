"""A small PPO loop over `lockstep_train`'s vector environment.

Every step is a turn of a real match: the environment's own engine, under
the fuel and the rules Lockstep plays it with, on practice data only. The
loop is plain PyTorch, kept short so you can read and change it; nothing
in it is tuned for one environment. Swap it for any RL library you like:
`lockstep_train.make_vec` is a standard Gymnasium vector env.

What it does, per update:

1. **Collect** `rollout` turns from each of `num_envs` matches (each
   match runs on its own thread). A match that ends starts the next one
   at once (Gymnasium's "same step" autoreset).
2. **Score** each turn with GAE (generalised advantage estimation). The
   reward is the engine's own per-turn training reward (`reward="auto"`,
   the "Reward" line of the environment's brief), scaled by a running
   estimate of the return's spread so every environment trains at a
   similar scale.
3. **Update** the policy and the value network for a few epochs of
   clipped PPO on minibatches; the learning rate and the entropy bonus
   fall linearly to zero by the end of the run.
4. **Update the input normalisers** with the turns just seen.

A checkpoint is written after every update (`checkpoint.pt`): `--resume`
continues from it. One CSV row per update goes to `metrics.csv`.

Seeds: training matches start from training seeds 1000 + 1000 × `seed`
(each env `i` adds `i`), and every later match draws its seed from the
environment's own generator. Practice seeds 0 to 19 (`lockstep run --seed
n`) are left for evaluation: a drawn seed lands on one about once in 200
million matches.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .policy import Policy

FIRST_TRAINING_SEED = 1000

METRICS = ["update", "steps", "seconds", "steps_per_second", "matches", "mean_score", "mean_return",
           "policy_loss", "value_loss", "entropy", "clip_fraction"]  # fmt: skip


@dataclass
class Config:
    steps: int = 100_000
    minutes: float | None = None  # stop after this long, whatever `steps` says
    num_envs: int = 8
    rollout: int = 128
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    seed: int = 0


def default_num_envs() -> int:
    """One match per core, leaving two for the learner and the machine, at most 8."""
    return min(8, max(1, (os.cpu_count() or 4) - 2))


class ReturnScaler:
    """Divides rewards by a running standard deviation of the discounted
    return, so the value network sees numbers of about 1 whatever the
    environment's score range."""

    def __init__(self, num_envs: int, gamma: float):
        self.gamma, self.ret = gamma, np.zeros(num_envs)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        b_mean, b_var, n = self.ret.mean(), self.ret.var(), len(self.ret)
        delta, total = b_mean - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / total) / total
        self.count = total
        self.ret[done] = 0.0
        return (reward / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s: dict) -> None:
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


def gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages and value targets for a `[T, N]` rollout; a finished match
    stops the sum (a match always ends by itself, so there is nothing to
    bootstrap past it)."""
    adv = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values


def finished_scores(infos: dict) -> list[float]:
    """The score of every match that ended this step (none for a match an
    engine fault voided)."""
    if "final_info" not in infos:
        return []
    scores = infos["final_info"].get("score")
    mask = infos.get("_final_info")
    if scores is None or mask is None:
        return []
    return [float(s) for s, m in zip(scores, mask, strict=False) if m and s is not None]


def train(vec, net: Policy, cfg: Config, out_dir: Path, resume: bool = False, log=print) -> int:
    """Train `net` in place on the vector env `vec`; returns the steps done."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    n = vec.num_envs
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    scaler = ReturnScaler(n, cfg.gamma)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"

    steps, update = 0, 0
    obs, _ = vec.reset(seed=FIRST_TRAINING_SEED + 1000 * cfg.seed)
    if resume:
        blob = torch.load(checkpoint, weights_only=False, map_location="cpu")
        net.load_state_dict(blob["state_dict"])
        opt.load_state_dict(blob["optimizer"])
        scaler.load(blob["scaler"])
        steps, update = blob["steps"], blob["update"]
        log(f"  resumed at {steps:,} steps from {checkpoint}")
    else:
        # Warm-up: measure the inputs for one rollout before learning, so
        # the first update already sees normalised numbers.
        seen = []
        for _ in range(cfg.rollout):
            seen.append(obs)
            with torch.no_grad():
                actions, _, _ = net.act(*net.tensors(obs))
            obs, *_ = vec.step(net.to_env(actions))
        net.update_normalizers(net.tensors({k: np.concatenate([o[k] for o in seen]) for k in obs}))

    metrics_file = open(out_dir / "metrics.csv", "a" if resume else "w", newline="")  # noqa: SIM115
    metrics = csv.DictWriter(metrics_file, fieldnames=METRICS)
    if not resume:
        metrics.writeheader()

    started, start_steps, last_log = time.monotonic(), steps, -10.0
    scores: list[float] = []
    returns: list[float] = []
    episode_return = np.zeros(n)
    batch = cfg.rollout * n

    def progress() -> float:
        frac = steps / cfg.steps
        if cfg.minutes:
            frac = max(frac, (time.monotonic() - started) / (60 * cfg.minutes))
        return frac

    while progress() < 1.0:
        left = max(0.0, 1.0 - progress())
        for group in opt.param_groups:
            group["lr"] = cfg.lr * left
        entropy_coef = cfg.entropy * left

        # 1. Collect.
        buf_obs, buf_act = [], []
        buf_logp = torch.zeros(cfg.rollout, n)
        buf_val = np.zeros((cfg.rollout, n), dtype=np.float32)
        buf_rew = np.zeros((cfg.rollout, n), dtype=np.float32)
        buf_done = np.zeros((cfg.rollout, n), dtype=np.float32)
        for t in range(cfg.rollout):
            x = net.tensors(obs)
            with torch.no_grad():
                actions, logp, value = net.act(*x)
            obs, reward, terminated, truncated, infos = vec.step(net.to_env(actions))
            done = np.logical_or(terminated, truncated)
            buf_obs.append(x)
            buf_act.append(actions)
            buf_logp[t], buf_val[t], buf_done[t] = logp, value.numpy(), done
            buf_rew[t] = scaler(reward, done)
            episode_return += reward
            for i in np.flatnonzero(done):
                returns.append(float(episode_return[i]))
                episode_return[i] = 0.0
            scores += finished_scores(infos)
        steps += batch

        # 2. Score.
        with torch.no_grad():
            last_value = net.value(net.features(*net.tensors(obs))).squeeze(-1).numpy()
        adv, ret = gae(buf_rew, buf_val, buf_done, last_value, cfg.gamma, cfg.lam)

        # 3. Update. [T, N, ...] → [T*N, ...], every buffer in the same order.
        xs = tuple(torch.cat([step[k] for step in buf_obs]) for k in range(len(net.inputs)))
        acts = [torch.cat([step[h] for step in buf_act]) for h in range(len(net.heads))]
        old_logp = buf_logp.reshape(-1)
        adv_t = torch.from_numpy(adv.reshape(-1))
        ret_t = torch.from_numpy(ret.reshape(-1))
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_fraction": []}
        for _ in range(cfg.epochs):
            for idx in torch.randperm(batch, generator=gen).split(cfg.minibatch):
                logp, entropy, value = net.evaluate(tuple(x[idx] for x in xs), [a[idx] for a in acts])
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old_logp[idx]).exp()
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                value_loss = (value - ret_t[idx]).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["clip_fraction"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())

        # 4. The input normalisers learn from what this rollout saw.
        net.update_normalizers(xs)
        update += 1

        torch.save(
            {
                "state_dict": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state(),
                "steps": steps,
                "update": update,
            },  # fmt: skip
            checkpoint,
        )
        seconds = time.monotonic() - started
        recent_scores, recent_returns = scores[-20:], returns[-20:]
        row = {
            "update": update,
            "steps": steps,
            "seconds": round(seconds, 2),
            "steps_per_second": round((steps - start_steps) / max(seconds, 1e-9)),
            "matches": len(scores),
            "mean_score": round(float(np.mean(recent_scores)), 3) if recent_scores else "",
            "mean_return": round(float(np.mean(recent_returns)), 3) if recent_returns else "",
            **{k: round(float(np.mean(v)), 5) for k, v in stats.items()},
        }
        metrics.writerow(row)
        metrics_file.flush()
        if seconds - last_log < 5 and progress() < 1.0:
            continue
        last_log = seconds
        log(
            f"  {steps:>9,} steps  {seconds:6.0f} s  {row['steps_per_second']:>6} steps/s  "
            f"matches {len(scores):<5} score (last 20) {row['mean_score'] or '-':>7}"
        )
    metrics_file.close()
    return steps
