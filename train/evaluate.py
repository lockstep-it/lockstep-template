"""Score a policy on practice seeds it never trained on, next to doing nothing.

The policy played is the **exported model** (`models/policy.onnx`, run by
ONNX Runtime and turned into actions by the shell's rules), so the number
is what Lockstep's shell would play on the same seeds. Practice seed `n` is
the match `lockstep run --seed n` plays, so you can watch any of them.

    python -m train.evaluate --folder <invite folder> [--name my-policy] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .policy import Head, outputs_to_actions


class OnnxPlayer:
    """Plays an exported model the way the shell does: feeds the inputs it
    has, reads one output per head, argmax for logits (NaN never wins)."""

    def __init__(self, path: Path, spec):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = [(i.name, i.type) for i in self.session.get_inputs()]
        self.heads = [Head(h.name, h.kind, h.shape, h.n, h.low) for h in spec.heads]

    def outputs(self, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
        feed = {
            name: np.asarray(obs[name], dtype=np.int32 if kind == "tensor(int32)" else np.float32)[None]
            for name, kind in self.inputs
        }
        return self.session.run([h.name for h in self.heads], feed)

    def __call__(self, obs: dict[str, np.ndarray]) -> dict:
        actions = outputs_to_actions(self.heads, self.outputs(obs))
        return {k: (v[0] if v.ndim else v) for k, v in actions.items()}


def play(env, policy, seed: int, record: list | None = None, every: int = 1) -> dict:
    """One whole match on practice seed `seed`; `policy(obs)` or None for the
    declared no-action. Returns how it ended."""
    obs, info = env.reset(seed=seed)
    turn, total = 0, 0.0
    while True:
        if record is not None and turn % every == 0:
            record.append(obs)
        obs, reward, terminated, truncated, info = env.step(None if policy is None else policy(obs))
        total += reward
        turn += 1
        if terminated or truncated:
            return {
                "seed": seed,
                "score": info.get("score"),
                "ending": info.get("ending"),
                "turns": info.get("turns", turn),
                "return": round(total, 4),
                "fault": info.get("fault"),
            }


def mean(results: list[dict]) -> float | None:
    s = [r["score"] for r in results if r["score"] is not None]
    return round(float(np.mean(s)), 3) if s else None


def evaluate(folder: Path, model: Path, seeds: int, env_kwargs: dict, record_to: Path | None = None) -> dict:
    """The policy and the no-action on practice seeds 0 to `seeds` - 1."""
    import lockstep_train

    env = lockstep_train.make(folder, **env_kwargs)
    player = OnnxPlayer(model, env.lockstep)
    recorded: list = []
    every = max(1, env.lockstep.max_turns // 100)
    policy = [play(env, player, s, recorded if record_to else None, every) for s in range(seeds)]
    if record_to is not None and recorded:
        np.savez_compressed(record_to, **{k: np.stack([o[k] for o in recorded]) for k in recorded[0]})
    nothing = [play(env, None, s) for s in range(seeds)]
    return {
        "seeds": list(range(seeds)),
        "policy": policy,
        "do_nothing": nothing,
        "policy_mean": mean(policy),
        "do_nothing_mean": mean(nothing),
    }


def report(result: dict) -> str:
    lines = ["  seed   policy        do-nothing"]
    for p, d in zip(result["policy"], result["do_nothing"], strict=True):
        ps = "fault" if p["score"] is None else f"{p['score']:7.2f} {p['ending'] or ''}"
        ds = "fault" if d["score"] is None else f"{d['score']:7.2f} {d['ending'] or ''}"
        lines.append(f"  {p['seed']:>4}   {ps:<13} {ds}")
    lines.append(f"  mean   {result['policy_mean']!s:<13} {result['do_nothing_mean']!s}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=".", help="the invite's folder (default: here)")
    p.add_argument("--name", default="my-policy", help="the policy bundle's folder name")
    p.add_argument("--seeds", type=int, default=10, help="practice seeds 0 to N-1 (default 10)")
    p.add_argument("--opponents", nargs="*", default=None, help="who plays the other seats (as in training)")
    args = p.parse_args()
    folder = Path(args.folder).expanduser().resolve()
    model = folder / args.name / "models" / "policy.onnx"
    if not model.is_file():
        print(f"no exported policy at {model}: train one first (python -m train.main --folder {args.folder})")
        return 2
    kwargs = {"opponents": args.opponents} if args.opponents else {}
    result = evaluate(folder, model, args.seeds, kwargs)
    print(report(result))
    out = folder / "training" / args.name / "evaluation.json"
    if out.parent.is_dir():
        out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
