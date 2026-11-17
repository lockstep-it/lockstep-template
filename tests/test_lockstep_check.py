"""The real check: `lockstep check --agent my-policy` in the invite's folder
runs Lockstep's own preflight, the ONNX shell's checks and a practice match.

Needs the lockstep CLI with the candidate commands (`LOCKSTEP_BIN`, else
`lockstep` on PATH) and the runtime's ONNX shell (`LOCKSTEP_RUNTIME_DIR`, or
the CLI's cache after one `lockstep run`); skipped otherwise. CI has
neither until the CLI is released.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from train.main import main


def cli() -> str | None:
    exe = os.environ.get("LOCKSTEP_BIN") or shutil.which("lockstep")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30).stdout
    except OSError:
        return None
    return exe if "\n  check " in out else None


@pytest.mark.parametrize("name", ["card-fraud", "jetpack-joust", "ticket-desk", "panda-pick"])
def test_lockstep_check_accepts_the_export(invite, name):
    exe = cli()
    if exe is None:
        pytest.skip("no lockstep CLI with `check` (LOCKSTEP_BIN or PATH)")
    folder = invite(name)
    assert main(["--folder", str(folder), "--steps", "256", "--num-envs", "2", "--eval-seeds", "1"]) == 0
    out = subprocess.run(
        [exe, "--env-file", "none", "--json", "check", "--agent", "my-policy"],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads(out.stdout)
    assert report["exit_code"] == 0, report
