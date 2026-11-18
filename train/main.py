"""Train a policy for an invite, export it, and check it: one command.

    python -m train.main --folder ~/acme-card-fraud              # PPO, 100,000 turns
    python -m train.main --folder ~/acme-card-fraud --minutes 10
    python -m train.main --folder ~/acme-card-fraud --recipe supervised

`--folder` is the folder `lockstep invite get` made. What it writes there:

    my-policy/                     the policy bundle you check and submit
      lockstep.toml
      models/policy.onnx
    training/my-policy/            yours to keep: never submitted
      weights.pt                   the torch network (with its input normalisers)
      checkpoint.pt  metrics.csv   PPO's resume point and one row per update
      evaluation.json              practice seeds 0-9: the policy and the no-action
      observations.npz             what the self-check compares torch and ONNX on

Then, in that folder: `lockstep check --agent my-policy`, `lockstep run
--agent my-policy --seed 3` to watch a match, and `lockstep submit --agent
my-policy` when you are happy (TUTORIAL.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from . import ppo, supervised
from .evaluate import evaluate, report
from .export import export, folder_identity, write_bundle
from .policy import Policy, load, save
from .selfcheck import selfcheck


def find_folder(start: Path) -> Path:
    """The invite's folder: `start` or the nearest folder above holding
    `.lockstep/declaration.json`."""
    for d in [start, *start.parents]:
        if (d / ".lockstep" / "declaration.json").is_file():
            return d
    raise SystemExit(
        f"{start} isn't an invite's folder (no .lockstep/declaration.json in it or above it).\n"
        "Make one with `lockstep invite get <invite id>`, then pass it: --folder <that folder>."
    )


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=".", help="the invite's folder (default: here)")
    p.add_argument("--name", default="my-policy", help="the bundle's folder name (default my-policy)")
    p.add_argument("--recipe", choices=["ppo", "supervised"], default="ppo")
    p.add_argument("--steps", type=int, default=100_000, help="ppo: turns to train, over all matches (default 100,000)")
    p.add_argument("--minutes", type=float, default=None, help="ppo: stop after this many minutes")
    p.add_argument(
        "--num-envs", type=int, default=None, help=f"ppo: matches at once (default {ppo.default_num_envs()} here)"
    )
    p.add_argument("--rollout", type=int, default=128, help="ppo: turns per match between updates")
    p.add_argument("--lr", type=float, default=3e-4, help="ppo: learning rate")
    p.add_argument("--entropy", type=float, default=0.01, help="ppo: entropy bonus")
    p.add_argument("--hidden", type=int, default=64, help="width of the network's layers")
    p.add_argument("--seed", type=int, default=0, help="the run's seed: same seed, same folder, same result")
    p.add_argument("--seat", type=int, default=0, help="the seat you play (several-seat environments)")
    p.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help='who plays the other seats: "random" (the default in an invite folder), "do-nothing", '
        "or a policy bundle folder such as an earlier my-policy-v1",
    )
    p.add_argument(
        "--reward",
        default="auto",
        help="auto (default): the engine's per-turn training reward, as the brief's \"Reward\" line says; "
        "score: the score at the end of the match only; series:<key>: a series the environment declares",
    )
    p.add_argument("--init", type=Path, default=None, help="start from these weights (a weights.pt)")
    p.add_argument("--resume", action="store_true", help="ppo: continue from training/<name>/checkpoint.pt")
    p.add_argument("--matches", type=int, default=16, help="supervised: matches to collect labels from")
    p.add_argument("--epochs", type=int, default=20, help="supervised: passes over the labelled turns")
    p.add_argument("--head", default=None, help="supervised: the discrete head to train (default the first)")
    p.add_argument("--label-input", default="feedback", help="supervised: the feedback input's name")
    p.add_argument("--truth-column", default="outcome", help="supervised: the column holding the truth")
    p.add_argument("--age-column", default="age", help="supervised: the column saying how many turns ago")
    p.add_argument("--decision-column", default="decision", help="supervised: the column holding your decision")
    p.add_argument("--valid-column", default="valid", help="supervised: the column marking real rows")
    p.add_argument("--eval-seeds", type=int, default=10, help="practice seeds to score on afterwards (0 to skip)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    # One thread: the networks are small, and a fixed thread count keeps a
    # seeded run the same from one run to the next.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    import lockstep_train

    folder = find_folder(Path(args.folder).expanduser().resolve())
    run_dir = folder / "training" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {"seat": args.seat, "reward": args.reward}
    if args.opponents:
        env_kwargs["opponents"] = args.opponents
    started = time.monotonic()

    if args.recipe == "ppo":
        num_envs = args.num_envs or ppo.default_num_envs()
        vec = lockstep_train.make_vec(folder, num_envs, **env_kwargs)
        spec = vec.lockstep
    else:
        env = lockstep_train.make(folder, **env_kwargs)
        spec = env.lockstep
    environment, _ = folder_identity(folder, spec.declaration)
    print(f"{environment} ({spec.declaration['title']}), seat {args.seat} of {spec.seats}")
    print(f"  inputs:  {', '.join(f'{o.name} {list(o.shape)}' for o in spec.obs)}")
    heads = (f"{h.name} " + (f"{h.n} codes" if h.kind == "discrete" else "in [-1, 1]") for h in spec.heads)
    print(f"  outputs: {', '.join(heads)}")

    if args.init:
        net, _ = load(args.init)
        net.train()
        print(f"  starting from {args.init}")
    else:
        net = Policy.from_spec(spec, args.hidden)

    if args.recipe == "ppo":
        cfg = ppo.Config(
            steps=args.steps, minutes=args.minutes, num_envs=num_envs, rollout=args.rollout,
            lr=args.lr, entropy=args.entropy, seed=args.seed,
        )  # fmt: skip
        limit = f"{args.steps:,} turns" + (f" or {args.minutes:g} minutes" if args.minutes else "")
        print(f"Training with PPO: {num_envs} matches at once, up to {limit}.")
        steps = ppo.train(vec, net, cfg, run_dir, resume=args.resume)
        vec.close()
    else:
        print(f"Training on revealed truth: {args.matches} matches, {args.epochs} epochs.")
        stream = {
            "value": args.label_input, "truth_col": args.truth_column, "age_col": args.age_column,
            "decision_col": args.decision_column, "valid_col": args.valid_column,
        }  # fmt: skip
        obs, truths, owners = supervised.collect(env, args.matches, args.seed, stream)
        supervised.fit(net, args.head, obs, truths, owners, args.matches, args.epochs, args.seed)
        steps = len(truths)
    trained = time.monotonic() - started
    print(f"Trained in {trained:.0f} s.")

    weights = run_dir / "weights.pt"
    save(weights, net, recipe=args.recipe, steps=steps, seed=args.seed)
    onnx_file = export(net, run_dir / "policy.onnx")
    _, interface_version = folder_identity(folder, spec.declaration)
    bundle = write_bundle(folder / args.name, onnx_file, environment, interface_version, steps)
    print(f"Exported {bundle.relative_to(folder)}/ (models/policy.onnx, {onnx_file.stat().st_size:,} bytes).")

    if args.eval_seeds > 0:
        t = time.monotonic()
        result = evaluate(
            folder, bundle / "models" / "policy.onnx", args.eval_seeds, env_kwargs, run_dir / "observations.npz"
        )
        (run_dir / "evaluation.json").write_text(json.dumps(result, indent=1))
        took = time.monotonic() - t
        print(f"Practice seeds 0-{args.eval_seeds - 1} (never trained on), the exported model ({took:.0f} s):")
        print(report(result))

    print("Self-check:")
    ok = selfcheck(folder, args.name, env_kwargs)
    print(f"Done in {time.monotonic() - started:.0f} s.")
    if not ok:
        print("Failed: fix what is FAILED above, then export again.")
        return 1
    print(
        f"Next, in {folder}:\n"
        f"  lockstep check --agent {args.name}            what Lockstep checks, and one practice match\n"
        f"  lockstep run --agent {args.name} --seed 3     a match to watch on the replay page\n"
        f"  lockstep submit --agent {args.name}           when you are happy: final"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
