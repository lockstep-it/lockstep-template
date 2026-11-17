"""Check an exported policy before `lockstep check`: the bundle, the shell's
rules, and that the exported model plays what the trained network plays.

    python -m train.selfcheck --folder <invite folder> [--name my-policy]

1. **The bundle**: `<name>/lockstep.toml` is a version 3 `policy` bundle
   for this invite's environment and interface version, with one model,
   `policy`, at `models/policy.onnx`.
2. **The rules** (`train/shellrules.py`): inputs and outputs as the ONNX
   shell reads them for this environment, and the host's preflight rules.
3. **Same actions**: the torch network (`training/<name>/weights.pt`) and
   the exported model (run by ONNX Runtime) on observations recorded from
   real practice matches: every unit head within 1e-4, every discrete head
   the same choice (rows where two logits are within 1e-4 are counted, not
   failed: a tie is a coin either way).

Exit 0 when everything passes, 1 when something fails.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

from . import shellrules
from .evaluate import OnnxPlayer, play
from .export import folder_identity
from .policy import argmax, load

UNIT_TOLERANCE = 1e-4
TIE = 1e-4


def bundle_problems(bundle: Path, folder: Path, declaration: dict) -> list[str]:
    manifest = bundle / "lockstep.toml"
    if not manifest.is_file():
        return [f"no {manifest}"]
    m = tomllib.loads(manifest.read_text(encoding="utf-8"))
    environment, interface_version = folder_identity(folder, declaration)
    out = []
    if m.get("schema_version") != 3:
        out.append(f"schema_version is {m.get('schema_version')}, not 3")
    if m.get("kind") != "policy":
        out.append(f'kind is {m.get("kind")!r}, not "policy"')
    if m.get("environment") != environment:
        out.append(f"environment is {m.get('environment')!r}; this folder is {environment!r}")
    if m.get("interface_version") != interface_version:
        out.append(f"interface_version is {m.get('interface_version')}; this folder's is {interface_version}")
    models = m.get("models", {})
    if list(models) != ["policy"]:
        out.append(f"a policy bundle has one model named `policy`; this one has {list(models)}")
    elif models["policy"].get("path") != "models/policy.onnx" or not (bundle / "models/policy.onnx").is_file():
        out.append("[models.policy] path must be models/policy.onnx, and the file must exist")
    return out


def compare(net, player: OnnxPlayer, obs: dict[str, np.ndarray]) -> tuple[float, int, int, int]:
    """`(largest unit difference, rows compared, discrete mismatches, ties)`."""
    rows = len(next(iter(obs.values())))
    with torch.no_grad():
        torch_out = [y.numpy() for y in net(*net.tensors(obs))]
    worst, mismatches, ties = 0.0, 0, 0
    for r in range(rows):
        onnx_out = player.outputs({k: v[r] for k, v in obs.items()})
        for h, t, o in zip(net.heads, torch_out, onnx_out, strict=True):
            t, o = t[r], o[0]
            if h.kind == "unit":
                worst = max(worst, float(np.max(np.abs(t - o))) if t.size else 0.0)
                continue
            if np.array_equal(argmax(t), argmax(o)):
                continue
            top2 = np.sort(np.where(np.isnan(t), -np.inf, t), axis=-1)[..., -2:]
            if np.all((top2[..., 1] - top2[..., 0] < TIE) | (argmax(t) == argmax(o))):
                ties += 1
            else:
                mismatches += 1
    return worst, rows, mismatches, ties


def selfcheck(folder: Path, name: str, env_kwargs: dict | None = None, log=print) -> bool:
    import lockstep_train

    bundle = folder / name
    model = bundle / "models" / "policy.onnx"
    weights = folder / "training" / name / "weights.pt"
    env = lockstep_train.make(folder, **(env_kwargs or {}))
    spec = env.lockstep
    ok = True

    bad = bundle_problems(bundle, folder, spec.declaration)
    for b in bad:
        log(f"  FAILED  bundle: {b}")
    if not bad:
        log(f"  ok      bundle {bundle.name}/: lockstep.toml v3, kind policy, {spec.declaration['name']}")
    ok &= not bad
    if not model.is_file():
        return False

    bad = shellrules.problems(model, spec)
    for b in bad:
        log(f"  FAILED  model: {b}")
    if not bad:
        ins = ", ".join(o.name for o in spec.obs)
        outs = ", ".join(f"{h.name} {'logits' if h.kind == 'discrete' else '[-1, 1]'}" for h in spec.heads)
        log(f"  ok      model {model.stat().st_size:,} bytes; inputs {ins}; outputs {outs}")
        log("  ok      opset, operators, fixed shapes and sizes within Lockstep's preflight rules")
    ok &= not bad

    if not weights.is_file():
        log(f"  note    no {weights}: the torch comparison is skipped")
        return ok
    net, _ = load(weights)
    player = OnnxPlayer(model, spec)
    recorded = folder / "training" / name / "observations.npz"
    if recorded.is_file():
        with np.load(recorded) as z:
            obs = {k: z[k] for k in z.files}
    else:
        seen: list = []
        play(env, player, 0, seen)
        obs = {k: np.stack([o[k] for o in seen[:500]]) for k in seen[0]}
    worst, rows, mismatches, ties = compare(net, player, obs)
    same = worst <= UNIT_TOLERANCE and mismatches == 0
    ok &= same
    tie_note = f", {ties} near-ties" if ties else ""
    log(
        f"  {'ok    ' if same else 'FAILED'}  torch and the export agree on {rows} recorded observations: "
        f"unit heads within {worst:.1e}, {mismatches} different choices{tie_note}"
    )
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=".", help="the invite's folder (default: here)")
    p.add_argument("--name", default="my-policy", help="the policy bundle's folder name")
    args = p.parse_args()
    folder = Path(args.folder).expanduser().resolve()
    print(f"Self-check of {args.name} in {folder}:")
    if selfcheck(folder, args.name):
        print(f"Passed. Next, in {folder}: lockstep check --agent {args.name}")
        return 0
    print("Failed: fix what is FAILED above, then export again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
