"""Training's action map against the golden the generic ONNX shell is tested
against (`agent-onnx`'s `denormalize`). The fixture is a byte-identical copy
of LOCKSTEP-INTERFACE's `tests/fixtures/policy/action_mapping.json`: when the
two disagree, a policy plays a different action sealed than the one that
earned its training return.

Bounds: the golden states them as declared (±inf for an open bound).
`lockstep_train` clamps an infinite bound to the dtype's range so the space
stays a valid Box, so each case also runs with that clamped equivalent; the
answer must be the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from train.core.policy import actions_to_env, obs_to_tensors

GOLDEN = json.loads((Path(__file__).parent / "fixtures/policy/action_mapping.json").read_text())
F32 = np.finfo(np.float32)


def num(v) -> np.float32:
    return np.float32({"nan": np.nan, "inf": np.inf, "-inf": -np.inf}.get(v, v))


def clamped(bound: np.float32) -> np.float32:
    return np.float32(np.clip(bound, F32.min, F32.max))


@pytest.mark.parametrize("case", GOLDEN["actions"], ids=lambda c: c["case"])
@pytest.mark.parametrize("bounds", ["declared", "clamped like lockstep_train"])
def test_actions_match_the_shell(case, bounds):
    low, high = num(case["low"]), num(case["high"])
    if bounds != "declared":
        low, high = clamped(low), clamped(high)
    box = spaces.Box(low=np.array([low]), high=np.array([high]), dtype=np.float32)
    out = actions_to_env(np.array([num(case["output"])], dtype=np.float32), box)
    assert out.dtype == np.float32
    # Exact: both sides compute in f32, in the same order.
    assert out[0] == num(case["expected"]), case["case"]


class _Net:
    input_names = ["pixels"]
    input_dtypes = {"pixels": "uint8"}


def test_u8_observations_are_fed_as_f32_over_255():
    cases = GOLDEN["observations_u8"]
    bytes_ = np.array([[c["byte"] for c in cases]], dtype=np.uint8)
    (fed,) = obs_to_tensors({"pixels": bytes_}, _Net())
    got = fed.numpy()[0]
    want = np.array([c["expected"] for c in cases], dtype=np.float32)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)
