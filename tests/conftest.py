"""Invite folders for the tests: the committed fixtures (tests/fixtures/),
each copied into a temporary folder so training can write into it.

panda-pick is laid out from a sibling LOCKSTEP-ENVIRONMENTS checkout
(`LOCKSTEP_ENVIRONMENTS_PATH` names another) when its engine is built there
and `LOCKSTEP_RUNTIME_DIR` holds the physics builds; otherwise its tests
are skipped.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CATALOG = (
    Path(os.environ.get("LOCKSTEP_ENVIRONMENTS_PATH") or ROOT.parent / "LOCKSTEP-ENVIRONMENTS") / "packages" / "catalog"
)


def _panda_pick(dest: Path) -> Path | None:
    project = CATALOG / "panda-pick"
    runtime = os.environ.get("LOCKSTEP_RUNTIME_DIR")
    if (
        not (project / "build" / "engine.wasm").is_file()
        or not runtime
        or not (Path(runtime) / "mujoco.wasm").is_file()
    ):
        return None
    lock = dest / ".lockstep"
    lock.mkdir(parents=True)
    shutil.copy(project / "build" / "engine.wasm", lock / "engine.wasm")
    shutil.copy(project / "declaration.json", lock / "declaration.json")
    shutil.copytree(project / "model", lock / "model")
    mjcf = re.findall(
        r'"([^"]+)"', re.search(r"mjcf\s*=\s*\[([^\]]*)\]", (project / "environment.toml").read_text())[1]
    )
    (dest / "runtime.lock").write_text("runtime = 1\n")
    (dest / "invite.toml").write_text(
        'format = 1\ninvite = "inv_fixture"\nenvironment = "lockstep/panda-pick"\nversion = "1.0.0"\n'
        "interface_version = 1\nseats_min = 1\nseats_max = 1\n"
        f"mjcf = [{', '.join(f'{m!r}'.replace(chr(39), chr(34)) for m in mjcf)}]\n"
    )
    return dest


@pytest.fixture
def invite(tmp_path):
    """`invite(name)` → a fresh copy of that invite folder."""

    def make(name: str) -> Path:
        dest = tmp_path / name
        if name == "panda-pick":
            if _panda_pick(dest) is None:
                pytest.skip("panda-pick: no LOCKSTEP-ENVIRONMENTS build beside this repo, or no LOCKSTEP_RUNTIME_DIR")
            return dest
        shutil.copytree(FIXTURES / name, dest)
        return dest

    return make
