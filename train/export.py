"""Export a trained policy to ONNX, and write the policy bundle Lockstep takes.

The model, as the ONNX shell requires it (onnx-shell.md, "The model"):

- named inputs, one per observation entry, `[1, *shape]`, float32 (int32
  for an i32 input);
- named outputs, one per action head: a unit head's values `[1, *shape]`,
  a discrete head's logits `[1, *shape, n]`; nothing else (the value
  network is left out);
- opset 17, batch 1, fixed shapes, every weight inside the one file.

The bundle, in the invite's folder (`lockstep check --agent <name>` reads it):

    <name>/
      lockstep.toml        schema_version 3, kind = "policy"
      models/policy.onnx
"""

from __future__ import annotations

import contextlib
import io
import logging
import subprocess
import tomllib
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .policy import Policy

OPSET = 17


def example_inputs(net: Policy) -> tuple[torch.Tensor, ...]:
    """One observation of zeros per input, batch 1: what the export traces."""
    return tuple(torch.zeros((1, *i.shape), dtype=torch.int32 if i.is_int else torch.float32) for i in net.inputs)


def export(net: Policy, path: Path) -> Path:
    """Write `path`: one input per observation entry, one output per head."""
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    # The exporter talks a lot (it traces at opset 18 and converts down to
    # 17); its messages say nothing about this model, so they are muted.
    # The checks after the export are what count.
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            example_inputs(net),
            str(path),
            input_names=[i.name for i in net.inputs],
            output_names=[h.name for h in net.heads],
            opset_version=OPSET,
            dynamo=True,
            # Weights inside the file: a bundle carries one model file.
            external_data=False,
            verbose=False,
        )
    # Keep only what the graph uses (the exporter may leave unused
    # weights behind, which only make the file bigger).
    model = onnx.load(str(path))
    used = {name for node in model.graph.node for name in node.input}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.save(model, str(path))
    return path


def folder_identity(folder: Path, declaration: dict) -> tuple[str, int]:
    """`(environment, interface_version)` for the bundle: the invite's own
    (`invite.toml`, written by `lockstep invite get`); in a folder with no
    invite, `local/<name>` and the declaration's version, as the CLI does."""
    invite = folder / "invite.toml"
    if invite.is_file():
        t = tomllib.loads(invite.read_text(encoding="utf-8"))
        return t["environment"], int(t["interface_version"])
    return f"local/{declaration['name']}", int(declaration["interface_version"])


def template_version() -> str:
    """This checkout's commit, for the bundle's provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unversioned"
    except (OSError, subprocess.SubprocessError):
        return "unversioned"


def write_bundle(bundle: Path, onnx_file: Path, environment: str, interface_version: int, steps: int) -> Path:
    """Write the `policy` bundle: `lockstep.toml` and `models/policy.onnx`."""
    (bundle / "models").mkdir(parents=True, exist_ok=True)
    target = bundle / "models" / "policy.onnx"
    if onnx_file.resolve() != target.resolve():
        target.write_bytes(onnx_file.read_bytes())
    built = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    (bundle / "lockstep.toml").write_text(
        "# A trained policy, written by lockstep-template (train/export.py).\n"
        "# Lockstep plays it with its own ONNX shell. Check it with\n"
        f"# `lockstep check --agent {bundle.name}` in the invite's folder.\n"
        "schema_version = 3\n"
        f'environment = "{environment}"\n'
        f"interface_version = {interface_version}\n"
        'kind = "policy"\n'
        "\n"
        "[models.policy]\n"
        'path = "models/policy.onnx"\n'
        "\n"
        "# Self-reported, shown to the company as written.\n"
        "[provenance]\n"
        'tool = "lockstep-template"\n'
        f'tool_version = "{template_version()}"\n'
        'language = "python"\n'
        f"steps = {int(steps)}\n"
        f'built_at = "{built}"\n',
        encoding="utf-8",
    )
    return bundle


def run_onnx(path: Path, net: Policy, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run the exported model on a batch of observations, one row at a time
    (the model's batch is 1). Returns one array per head, `[rows, ...]`."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    rows = len(next(iter(obs.values())))
    outs: list[list[np.ndarray]] = [[] for _ in net.heads]
    for r in range(rows):
        feed = {
            i.name: np.ascontiguousarray(obs[i.name][r : r + 1], dtype=np.int32 if i.is_int else np.float32)
            for i in net.inputs
            if i.name in names
        }
        for k, y in enumerate(session.run([h.name for h in net.heads], feed)):
            outs[k].append(y)
    return [np.concatenate(o) for o in outs]
