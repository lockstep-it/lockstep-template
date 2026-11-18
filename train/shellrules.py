"""The rules an exported model must meet before Lockstep plays it, checked
here without Lockstep.

Two sets of rules, both written down by Lockstep:

- **The ONNX shell's** (onnx-shell.md): the model's inputs and outputs, by
  name, type and shape, against the environment's declaration. The names
  and shapes come from `lockstep_train`'s spec of the folder, which is the
  shell's own mapping.
- **The host's preflight** (LOCKSTEP-PLATFORM docs/studio-host.md,
  "Inference"): opset 13 to 21 of the default domain only, IR 7 to 10,
  weights inside the file, the operator allowlist, fixed shapes and the
  size limits.

This is a copy of the documented rules, for a quick answer and for CI.
The real check is `lockstep check --agent <bundle>` in the invite's
folder: it runs the host's own preflight and plays a practice match
through the shell, and it also reports what the model costs a turn.
"""

from __future__ import annotations

import math
from pathlib import Path

#: The operators the host's preflight allows (studio-host `onnx/preflight.rs`).
ALLOWED_OPS = frozenset(
    {
        *("Identity", "Constant", "Reshape", "Squeeze", "Unsqueeze"),
        *("Cast", "Concat", "Transpose", "Gather", "GatherND", "Tile"),
        *("Add", "Sub", "Mul", "Div", "Neg", "Abs", "Relu", "LeakyRelu", "PRelu", "Clip", "Min", "Max"),
        *("Sign", "Floor", "Ceil", "Round", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"),
        *("Not", "And", "Or", "Xor", "Where", "HardSigmoid", "HardSwish", "Sum", "Mean", "Reciprocal"),
        *("IsNaN", "IsInf", "Tanh", "Sigmoid", "Exp", "Log", "Sqrt"),
        *("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "GlobalAveragePool"),
        *("Softmax", "LogSoftmax", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceProd", "ArgMax"),
        *("LayerNormalization", "ReduceLogSumExp", "MaxPool", "AveragePool", "Gemm", "MatMul", "Conv"),
    }
)

OPSET_MIN, OPSET_MAX = 13, 21
IR_MIN, IR_MAX = 7, 10
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_NODES = 1024
MAX_PARAMS = 4 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 1 << 20
MAX_LOGITS = 1_000_000
MAX_STATE_ELEMENTS = 65_536


def _shape(value_info) -> list[int | str]:
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims


def _dtype(value_info) -> str:
    import onnx

    return onnx.helper.tensor_dtype_to_np_dtype(value_info.type.tensor_type.elem_type).name


def problems(path: Path, spec) -> list[str]:
    """Everything wrong with the model at `path` for the folder whose
    `lockstep_train` spec is `spec`, one sentence each (empty: it passes)."""
    import onnx

    out: list[str] = []
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return [f"the model is {size:,} bytes; Lockstep takes at most {MAX_FILE_BYTES:,}."]
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    # -- the file -------------------------------------------------------------
    if not IR_MIN <= model.ir_version <= IR_MAX:
        out.append(f"IR version {model.ir_version}; Lockstep takes {IR_MIN} to {IR_MAX}.")
    for op in model.opset_import:
        if op.domain not in ("", "ai.onnx"):
            out.append(f"the operator set `{op.domain}` isn't allowed: only ONNX's own operators are.")
        elif not OPSET_MIN <= op.version <= OPSET_MAX:
            out.append(f"opset {op.version}; Lockstep takes {OPSET_MIN} to {OPSET_MAX} (export with opset 17).")
    if model.functions:
        out.append("the model defines functions; export them inlined.")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in g.initializer):
        out.append("some weights live in a separate file; export with every weight inside the model.")
    if g.sparse_initializer:
        out.append("the model has sparse weights; Lockstep takes dense ones only.")
    if len(g.node) > MAX_NODES:
        out.append(f"{len(g.node)} nodes; Lockstep takes at most {MAX_NODES}.")
    params = sum(math.prod(t.dims) for t in g.initializer)
    if params > MAX_PARAMS:
        out.append(f"{params:,} weights; Lockstep takes at most {MAX_PARAMS:,}.")
    for n in g.node:
        if n.op_type not in ALLOWED_OPS:
            out.append(f"node `{n.name or n.op_type}` uses `{n.op_type}`, which Lockstep's preflight refuses.")
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for a in n.attribute):
            out.append(f"node `{n.name or n.op_type}` holds a subgraph; Lockstep takes none.")

    # -- inputs: one per declared input the model uses ------------------------
    weights = {t.name for t in g.initializer}
    obs = {o.name: o for o in spec.obs}
    inputs = [i for i in g.input if i.name not in weights]
    state_in = None
    for i in inputs:
        shape, dtype = _shape(i), _dtype(i)
        if i.name == "state_in":
            state_in = (shape, dtype)
            continue
        o = obs.get(i.name)
        if o is None:
            out.append(f"the model input `{i.name}` isn't a declared input (they are {', '.join(obs)}).")
            continue
        want_dtype = "int32" if o.dtype == "i32" else "float32"
        want = [1, *o.shape]
        if shape != want or dtype != want_dtype:
            out.append(f"the model input `{i.name}` is {dtype} {shape}; the shell feeds {want_dtype} {want}.")

    # -- outputs: exactly one per head ----------------------------------------
    outputs = {o.name: o for o in g.output}
    logits = 0
    for h in spec.heads:
        o = outputs.pop(h.name, None)
        want = [1, *h.shape] if h.kind == "unit" else [1, *h.shape, h.n]
        what = "values in [-1, 1]" if h.kind == "unit" else f"logits, {h.n} codes from {h.low}"
        if o is None:
            out.append(f"the model has no output `{h.name}` (float32 {want}: {what}).")
            continue
        shape, dtype = _shape(o), _dtype(o)
        if shape != want or dtype != "float32":
            out.append(f"the model output `{h.name}` is {dtype} {shape}; the shell reads float32 {want} ({what}).")
        if h.kind == "discrete":
            logits += h.elements * h.n
    state_out = outputs.pop("state_out", None)
    for name in outputs:
        out.append(f"the model output `{name}` isn't one the shell reads: remove it (the shell refuses extra outputs).")
    if logits > MAX_LOGITS:
        out.append(f"{logits:,} logits in all; the shell takes at most {MAX_LOGITS:,}.")

    # -- memory ---------------------------------------------------------------
    if (state_in is None) != (state_out is None):
        out.append("a model with memory has both `state_in` and `state_out`, or neither.")
    elif state_in is not None and state_out is not None:
        so = (_shape(state_out), _dtype(state_out))
        n = math.prod(d if isinstance(d, int) else 0 for d in state_in[0])
        if state_in != so or state_in[1] != "float32" or not 1 <= n <= MAX_STATE_ELEMENTS:
            out.append(
                f"`state_in` {state_in} and `state_out` {so} must be float32 of one shape, "
                f"1 to {MAX_STATE_ELEMENTS:,} elements."
            )

    # -- fixed shapes everywhere ----------------------------------------------
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as e:  # noqa: BLE001 - any failure is the model's
        out.append(f"ONNX's shape inference fails on the model: {e}")
        return out
    for vi in [*inferred.graph.value_info, *inferred.graph.output]:
        shape = _shape(vi)
        if not shape and not vi.type.tensor_type.HasField("shape"):
            out.append(f"`{vi.name}` has no known shape; every shape must be fixed.")
        elif any(not isinstance(d, int) for d in shape):
            out.append(f"`{vi.name}` has shape {shape}; every shape must be fixed.")
        elif math.prod(shape) > MAX_TENSOR_ELEMENTS:
            out.append(
                f"`{vi.name}` has {math.prod(shape):,} elements; Lockstep takes at most {MAX_TENSOR_ELEMENTS:,}."
            )
    return out
