"""The export against the ONNX shell's rules (train/shellrules.py), and the
self-check catching an export that plays differently from the network."""

from __future__ import annotations

import lockstep_train
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from train import shellrules
from train.export import export, write_bundle
from train.policy import Policy, save
from train.selfcheck import selfcheck


def exported(folder, name="my-policy"):
    env = lockstep_train.make(folder)
    net = Policy.from_spec(env.lockstep)
    run = folder / "training" / name
    run.mkdir(parents=True)
    save(run / "weights.pt", net)
    onnx_file = export(net, run / "policy.onnx")
    write_bundle(folder / name, onnx_file, f"lockstep/{folder.name}", 1, 0)
    return env.lockstep, net, folder / name / "models" / "policy.onnx"


def test_one_output_per_head_as_the_shell_reads_them(invite):
    folder = invite("jetpack-joust")
    spec, _, path = exported(folder)
    assert shellrules.problems(path, spec) == []
    m = onnx.load(str(path))
    outs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
    assert outs == {"flight": [1, 4], "strike": [1, 2]}  # unit [1, *S]; logits [1, *S, n]
    assert [o.version for o in m.opset_import] == [17]
    assert {n.op_type for n in m.graph.node} <= shellrules.ALLOWED_OPS


def test_the_rules_name_what_is_wrong(invite, tmp_path):
    folder = invite("card-fraud")
    spec, _, _ = exported(folder)
    # A model with a misnamed input, a flat `action` output, an extra
    # output and an operator the preflight refuses.
    x = helper.make_tensor_value_info("pay", TensorProto.FLOAT, [1, 10])
    y = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 4])
    v = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 10])
    graph = helper.make_graph(
        [helper.make_node("Gelu", ["pay"], ["value"]), helper.make_node("Slice", ["value"], ["action"])],
        "bad",
        [x],
        [y, v],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    bad = tmp_path / "bad.onnx"
    onnx.save(model, str(bad))
    text = "\n".join(shellrules.problems(bad, spec))
    for expected in (
        "`pay` isn't a declared input",
        "no output `decision` (float32 [1, 3]",
        "no output `confidence` (float32 [1]",
        "`action` isn't one the shell reads",
        "`value` isn't one the shell reads",
        "uses `Gelu`",
        "uses `Slice`",
    ):
        assert expected in text, (expected, text)


def test_selfcheck_passes_and_catches_a_stale_export(invite, capsys):
    folder = invite("card-fraud")
    _, net, _ = exported(folder)
    assert selfcheck(folder, "my-policy")
    # Change the network after exporting: the export no longer plays it.
    with torch.no_grad():
        for layer in net.out:
            layer.weight.add_(torch.from_numpy(np.random.default_rng(0).normal(0, 1, layer.weight.shape)).float())
    save(folder / "training" / "my-policy" / "weights.pt", net)
    lines = []
    assert not selfcheck(folder, "my-policy", log=lines.append)
    assert any(line.startswith("  FAILED  torch and the export") for line in lines), lines


def test_bundle_for_another_invite_fails(invite):
    folder = invite("ticket-desk")
    exported(folder)
    toml = folder / "my-policy" / "lockstep.toml"
    toml.write_text(toml.read_text().replace("lockstep/ticket-desk", "acme/ticket-desk"))
    lines = []
    assert not selfcheck(folder, "my-policy", log=lines.append)
    assert any("environment is 'acme/ticket-desk'" in line for line in lines), lines
