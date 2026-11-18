"""The supervised recipe: learn from the truth a data environment reveals.

Many data environments tell you, some turns later, what was really true
about a decision: a **feedback window**, an input whose rows are the
decisions that resolved this turn (card-fraud's `feedback` has the columns
`age`, `decision`, `outcome`, `amount`, `valid`). On practice seeds you can
collect as many `(what you saw, what was true)` pairs as you like.
`lockstep_train.LabelledStream` joins each row back to the observation the
decision was made on.

    python -m train.main --folder <invite> --recipe supervised [--matches 16] [--epochs 20]

1. **Collect**: play `matches` training matches with the declared
   no-action (what is true doesn't depend on what you decide) and keep the
   pairs.
2. **Fit**: train one discrete head (`--head`, default the first) to
   predict the truth column with cross-entropy: truth value `v` is the
   head's code `v`. That fits environments whose output codes line up with
   the truth (card-fraud: outcome 1 "fraud" and decision 1 "decline").
   When yours don't, change `label_to_index` below. Matches are split 80/20
   by seed and the held-out accuracy is printed every epoch.
3. Every other head keeps its starting weights (a unit head plays about
   0, the middle of its range: card-fraud's `confidence` plays 0.5).
   `--recipe ppo --init training/<name>/weights.pt` continues from these
   weights with PPO, but measure before you keep the result: on card-fraud
   a minute of PPO from here fell from about 56 to 15 on practice seeds
   (PPO samples its decisions, and a few random declines end a match).
   Training the other heads on what you know of the task is usually better.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .policy import Policy

FIRST_COLLECT_SEED = 500_000


def label_to_index(truth: float, low: int) -> int:
    """The head's index for a truth value: the code with the same value."""
    return int(round(truth)) - low


def collect(env, matches: int, seed: int, stream_kwargs: dict, log=print):
    """`(observations by name, truths, match number per pair)` from `matches`
    no-action matches."""
    from lockstep_train import LabelledStream

    stream = LabelledStream(env, **stream_kwargs)
    obs_list, truths, owners = [], [], []
    started = time.monotonic()
    for m in range(matches):
        obs, info = env.reset(seed=FIRST_COLLECT_SEED + 1000 * seed + m)
        stream.reset()
        stream.feed(info["turn"], obs)
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(None)
            for obs_then, label in stream.feed(info["turn"], obs, reward):
                obs_list.append(obs_then)
                truths.append(label.truth)
                owners.append(m)
            done = done or truncated
    log(f"  collected {len(truths):,} labelled turns from {matches} matches in {time.monotonic() - started:.1f} s")
    if not truths:
        raise SystemExit("no labels arrived: is the feedback input named right? (--label-input, --truth-column)")
    stacked = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    return stacked, np.asarray(truths), np.asarray(owners)


def fit(net: Policy, head_name: str | None, obs, truths, owners, matches: int, epochs: int, seed: int, log=print):
    """Train the trunk and one discrete head on the pairs."""
    torch.manual_seed(seed)
    k = next(
        (i for i, h in enumerate(net.heads) if h.kind == "discrete" and (head_name is None or h.name == head_name)),
        None,
    )
    if k is None:
        raise SystemExit(f"no discrete head named {head_name!r}: the heads are {[h.name for h in net.heads]}")
    head = net.heads[k]
    if head.shape != ():
        raise SystemExit(f"head {head.name!r} has shape {head.shape}: this recipe trains a single-code head")
    y = np.array([label_to_index(t, head.low) for t in truths])
    keep = (y >= 0) & (y < head.n)
    xs = net.tensors(obs)
    net.update_normalizers(tuple(x[torch.from_numpy(keep)] for x in xs))
    held_out = owners >= int(matches * 0.8)
    train_rows = np.flatnonzero(keep & ~held_out)
    test_rows = np.flatnonzero(keep & held_out)
    counts = np.bincount(y[keep], minlength=head.n)
    log(f"  head {head.name!r}: {len(train_rows):,} turns to learn from, {len(test_rows):,} held out")
    log(f"  how often each code is the truth: {counts.tolist()}")
    targets = torch.from_numpy(y)
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith("value")], lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        for idx in torch.from_numpy(train_rows)[torch.randperm(len(train_rows), generator=gen)].split(256):
            logits = net.head_outputs(net.features(*(x[idx] for x in xs)))[k]
            loss = torch.nn.functional.cross_entropy(logits, targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(test_rows) and (epoch == epochs or epoch % 5 == 0):
            with torch.no_grad():
                idx = torch.from_numpy(test_rows)
                pred = net.head_outputs(net.features(*(x[idx] for x in xs)))[k].argmax(-1).numpy()
            truth = y[test_rows]
            recall = [f"{c}: {np.mean(pred[truth == c] == c):.2f}" for c in range(head.n) if np.any(truth == c)]
            accuracy = np.mean(pred == truth)
            log(f"  epoch {epoch:>3}  held-out accuracy {accuracy:.3f}  recall by code {', '.join(recall)}")
