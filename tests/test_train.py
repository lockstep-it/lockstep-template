"""`python -m train.main` end to end on invite folders: train, export, bundle,
score on practice seeds, self-check."""

from __future__ import annotations

import json
import os
import tomllib

import pytest
import torch

from train.main import main
from train.policy import load


@pytest.mark.parametrize("name", ["card-fraud", "jetpack-joust", "ticket-desk", "panda-pick"])
def test_256_steps_train_export_and_check(invite, name):
    folder = invite(name)
    assert main(["--folder", str(folder), "--steps", "256", "--num-envs", "2", "--eval-seeds", "2"]) == 0
    manifest = tomllib.loads((folder / "my-policy" / "lockstep.toml").read_text())
    assert manifest["schema_version"] == 3 and manifest["kind"] == "policy"
    assert manifest["environment"] == f"lockstep/{name}"
    assert manifest["models"] == {"policy": {"path": "models/policy.onnx"}}
    assert manifest["provenance"]["tool"] == "lockstep-template" and manifest["provenance"]["steps"] >= 256
    assert sorted(p.name for p in (folder / "my-policy").rglob("*") if p.is_file()) == ["lockstep.toml", "policy.onnx"]
    result = json.loads((folder / "training" / "my-policy" / "evaluation.json").read_text())
    assert [r["seed"] for r in result["policy"]] == [0, 1]


def test_same_seed_same_policy(invite):
    folder = invite("ticket-desk")
    weights = []
    for name in ("a", "b"):
        args = ["--folder", str(folder), "--name", name, "--steps", "512", "--num-envs", "2", "--eval-seeds", "0"]
        assert main(args) == 0
        weights.append(load(folder / "training" / name / "weights.pt")[0].state_dict())
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


def test_supervised_beats_doing_nothing_on_card_fraud(invite):
    """The smoke bar: a few seconds of the supervised recipe beat the
    no-action on practice seeds 0-9 (never trained on)."""
    folder = invite("card-fraud")
    assert main(["--folder", str(folder), "--recipe", "supervised"]) == 0
    result = json.loads((folder / "training" / "my-policy" / "evaluation.json").read_text())
    assert result["do_nothing_mean"] == pytest.approx(20.0)
    assert result["policy_mean"] > result["do_nothing_mean"] + 10, result["policy_mean"]


def test_train_against_an_earlier_policy(invite):
    """Self-play by generations: the other seat plays an earlier export
    (through the runtime's ONNX shell, so LOCKSTEP_RUNTIME_DIR or the CLI's
    cache must hold it)."""
    if not os.environ.get("LOCKSTEP_RUNTIME_DIR"):
        pytest.skip("an exported opponent plays through the runtime's ONNX shell: set LOCKSTEP_RUNTIME_DIR")
    folder = invite("jetpack-joust")
    base = ["--folder", str(folder), "--steps", "256", "--num-envs", "2", "--eval-seeds", "1"]
    assert main([*base, "--name", "gen-1"]) == 0
    assert main([*base, "--name", "gen-2", "--opponents", "gen-1"]) == 0
    assert (folder / "gen-2" / "models" / "policy.onnx").is_file()
