"""The policy network, built from the environment's own spaces.

`lockstep_train` gives every environment two `Dict` spaces whose entries
are exactly the tensors Lockstep's ONNX shell feeds and reads
(`.lockstep/sdk/docs/onnx-shell.md` in a starter, or LOCKSTEP-ENVIRONMENTS
docs/onnx-shell.md):

- **observations**: one entry per declared input, named after it, float32
  (an f32 input as it is, a u8 input already divided by 255) or int32;
- **actions**: one entry per head. A *unit* head (an f32 output) is a
  `Box(-1, 1)`: the shell maps -1 to the output's low bound and 1 to its
  high bound. A *discrete* head (an i32 or u8 output, or one slice or
  column of it) is `Discrete(n)` or `MultiDiscrete`: the model gives n
  logits per element and the shell plays the largest.

So the network needs no knowledge of any environment: one input per
observation entry, one output per head, and the export names them the
same way. Nothing here is tuned for a particular task. Make it yours:
change `hidden`, the trunk, or how an input is read (for example a small
convolution for an image), as long as `forward` keeps its outputs.

What the network does:

1. **Normalise.** Each input is scaled by a running mean and standard
   deviation measured while training (`Normalizer`), then clipped to
   [-10, 10]. The numbers are saved in the model, so the exported file
   does the same scaling. An int32 input is turned into floats first.
2. **Trunk.** All inputs are concatenated and go through two small
   `tanh` layers.
3. **Heads.** A unit head is `tanh` of a linear layer (training samples
   around it with a learned spread). A discrete head is a linear layer
   giving `[*shape, n]` logits.

A separate value network (for PPO) reads the same normalised inputs. It
is never exported: the shell refuses a model with outputs it doesn't use.

Only operators the Lockstep preflight allows are used (Reshape, Sub, Mul,
Clip, Gemm/MatMul, Tanh, Concat, Cast). `nn.Flatten`, `nn.GELU`,
`nn.Embedding` on an input and similar are refused there: check with
`python -m train.selfcheck` after a change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

#: Bounds on the learned log standard deviation of a unit head's samples.
LOG_STD_MIN, LOG_STD_MAX = -4.0, 0.5

#: Normalised inputs are clipped to this range.
CLIP = 10.0

# The distributions below are built thousands of times a second; checking
# their arguments each time costs more than the network.
torch.distributions.Distribution.set_default_validate_args(False)


@dataclass(frozen=True)
class Input:
    name: str
    shape: tuple[int, ...]
    is_int: bool

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Head:
    name: str
    kind: str  # "unit" or "discrete"
    shape: tuple[int, ...]
    n: int = 0  # discrete: how many codes
    low: int = 0  # discrete: the code of index 0

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def width(self) -> int:
        """Numbers the network computes for this head."""
        return self.size * (self.n if self.kind == "discrete" else 1)


def describe(spec) -> dict:
    """The network's inputs and heads, from a folder's `lockstep_train` spec
    (`env.lockstep`), as plain data: saved with the weights so a policy can
    be rebuilt (for the export or the self-check) without an environment."""
    return {
        "inputs": [{"name": o.name, "shape": list(o.shape), "int": o.dtype == "i32"} for o in spec.obs],
        "heads": [{"name": h.name, "kind": h.kind, "shape": list(h.shape), "n": h.n, "low": h.low} for h in spec.heads],
    }


class Normalizer(nn.Module):
    """A running mean and variance of one input, kept in the model.

    `update` is called on each batch of observations while training; the
    exported graph only applies `(x - mean) * inv_std`, clipped."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.register_buffer("inv_std", torch.ones(size))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.double().reshape(-1, self.mean.numel())
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean.double()
        total = self.count + n
        mean = self.mean.double() + delta * n / total
        m2 = self.var.double() * self.count + batch_var * n + delta.pow(2) * self.count * n / total
        self.mean.copy_(mean.float())
        self.var.copy_((m2 / total).float())
        self.count.copy_(total)
        self.inv_std.copy_(1.0 / torch.sqrt(self.var + 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) * self.inv_std).clamp(-CLIP, CLIP)


def mlp(width_in: int, hidden: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, width_out)
    )


class Policy(nn.Module):
    """One input per observation entry, one output per head."""

    def __init__(self, layout: dict, hidden: int = 64):
        super().__init__()
        self.layout = layout
        self.hidden = hidden
        self.inputs = [Input(i["name"], tuple(i["shape"]), i["int"]) for i in layout["inputs"]]
        self.heads = [Head(h["name"], h["kind"], tuple(h["shape"]), h["n"], h["low"]) for h in layout["heads"]]
        width = sum(i.size for i in self.inputs)
        self.norms = nn.ModuleList(Normalizer(i.size) for i in self.inputs)
        self.trunk = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.out = nn.ModuleList(nn.Linear(hidden, h.width) for h in self.heads)
        units = sum(h.size for h in self.heads if h.kind == "unit")
        self.log_std = nn.Parameter(torch.full((units,), -0.5))
        self.value = mlp(width, hidden, 1)
        # Small last layers: the first policy is close to uniform logits
        # and unit outputs near 0, so early training explores evenly.
        for layer in self.out:
            nn.init.orthogonal_(layer.weight, 0.01)
            nn.init.zeros_(layer.bias)

    @classmethod
    def from_spec(cls, spec, hidden: int = 64) -> Policy:
        """The network for a folder: `Policy.from_spec(env.lockstep)`."""
        return cls(describe(spec), hidden)

    # -- the network ---------------------------------------------------------

    def features(self, *xs: torch.Tensor) -> torch.Tensor:
        """Normalised inputs, side by side: `[batch, total size]`."""
        parts = []
        for spec, norm, x in zip(self.inputs, self.norms, xs, strict=True):
            parts.append(norm(x.float().reshape(-1, spec.size)))
        return torch.cat(parts, 1)

    def head_outputs(self, feats: torch.Tensor) -> list[torch.Tensor]:
        """Per head: a unit head's mean before `tanh` `[batch, *shape]`, or a
        discrete head's logits `[batch, *shape, n]`."""
        z = self.trunk(feats)
        outs = []
        for head, layer in zip(self.heads, self.out, strict=True):
            y = layer(z)
            outs.append(y.reshape(-1, *head.shape) if head.kind == "unit" else y.reshape(-1, *head.shape, head.n))
        return outs

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """What the exported model computes: per head, a unit head's value in
        [-1, 1] or a discrete head's logits. The shell takes it from there."""
        outs = self.head_outputs(self.features(*xs))
        return tuple(torch.tanh(y) if h.kind == "unit" else y for h, y in zip(self.heads, outs, strict=True))

    # -- training ------------------------------------------------------------

    def _dists(self, feats: torch.Tensor):
        """One distribution per head; a unit head samples before `tanh`."""
        dists, k = [], 0
        for head, y in zip(self.heads, self.head_outputs(feats), strict=True):
            if head.kind == "unit":
                std = self.log_std[k : k + head.size].clamp(LOG_STD_MIN, LOG_STD_MAX).exp().reshape(head.shape)
                dists.append(torch.distributions.Normal(y, std))
                k += head.size
            else:
                dists.append(torch.distributions.Categorical(logits=y))
        return dists

    def act(self, *xs: torch.Tensor):
        """Sample an action for every row: `(actions, log_prob, value)`.
        `actions` holds per head the pre-`tanh` sample (unit) or the index
        (discrete), which is what `evaluate` re-scores."""
        feats = self.features(*xs)
        actions, log_prob = [], 0
        for d in self._dists(feats):
            a = d.sample()
            actions.append(a)
            log_prob = log_prob + d.log_prob(a).reshape(a.shape[0], -1).sum(1)
        return actions, log_prob, self.value(feats).squeeze(-1)

    def evaluate(self, xs, actions):
        """`(log_prob, entropy, value)` of stored actions under the current
        weights."""
        feats = self.features(*xs)
        log_prob, entropy = 0, 0
        for d, a in zip(self._dists(feats), actions, strict=True):
            log_prob = log_prob + d.log_prob(a).reshape(a.shape[0], -1).sum(1)
            entropy = entropy + d.entropy().reshape(a.shape[0], -1).sum(1)
        return log_prob, entropy, self.value(feats).squeeze(-1)

    def update_normalizers(self, xs) -> None:
        for spec, norm, x in zip(self.inputs, self.norms, xs, strict=True):
            norm.update(x.float().reshape(-1, spec.size))

    # -- between numpy and torch ---------------------------------------------

    def tensors(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
        """A batch of observations `{name: [batch, *shape]}` → the inputs."""
        return tuple(
            torch.from_numpy(np.ascontiguousarray(obs[i.name], dtype=np.int32 if i.is_int else np.float32))
            for i in self.inputs
        )

    def to_env(self, actions: list[torch.Tensor]) -> dict[str, np.ndarray]:
        """Sampled actions → what the vector env steps on."""
        out = {}
        for head, a in zip(self.heads, actions, strict=True):
            out[head.name] = torch.tanh(a).numpy().astype(np.float32) if head.kind == "unit" else a.numpy()
        return out

    @torch.no_grad()
    def play(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The deterministic action for a batch, exactly as the shell plays
        the exported model: unit heads as they are, discrete heads by the
        largest logit with NaN never winning."""
        return outputs_to_actions(self.heads, [y.numpy() for y in self(*self.tensors(obs))])


def argmax(logits: np.ndarray) -> np.ndarray:
    """The shell's argmax: ties to the lowest index, a NaN never wins (plain
    `np.argmax` would pick a NaN), all NaN gives 0."""
    return np.argmax(np.where(np.isnan(logits), -np.inf, logits), axis=-1)


def outputs_to_actions(heads: list[Head], outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Model outputs (each with the batch axis) → env actions."""
    return {
        h.name: (np.asarray(y, dtype=np.float32) if h.kind == "unit" else argmax(y))
        for h, y in zip(heads, outputs, strict=True)
    }


def save(path, net: Policy, **extra) -> None:
    torch.save({"layout": net.layout, "hidden": net.hidden, "state_dict": net.state_dict(), **extra}, path)


def load(path) -> tuple[Policy, dict]:
    blob = torch.load(path, weights_only=True, map_location="cpu")
    net = Policy(blob["layout"], blob["hidden"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob
