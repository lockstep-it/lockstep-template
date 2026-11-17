# Lockstep training template

Train a policy for a [Lockstep](https://lockstep.it) assessment, export it
as an ONNX model, and check it the way Lockstep will, all on your own
machine. Lockstep plays the model you submit with its own ONNX shell, on
the same engine you train against here.

This repo is for **trained policies**. To write your agent as plain code
(Rust, C or Python) you don't need it: `lockstep agent new --lang rust|c|python`
in your invite's folder makes a starter for you.

**New here? Read [TUTORIAL.md](TUTORIAL.md):** from `lockstep invite get`
to a submitted policy, step by step, with how long each step takes.

## In five commands

```sh
lockstep invite get inv_…                 # your invite's folder, e.g. ./acme-card-fraud
git clone https://github.com/lockstep-it/lockstep-template && cd lockstep-template
task setup                                # .venv with PyTorch, ONNX and lockstep-train
task train FOLDER=../acme-card-fraud      # train, export, score on practice seeds, self-check
cd ../acme-card-fraud && lockstep check --agent my-policy && lockstep submit --agent my-policy
```

No `task`? Every task is one line of Python: `python -m train.main --folder
../acme-card-fraud` (see `Taskfile.yml`).

## What you need

- Python 3.11 or later, and about 1 GB of disk for PyTorch.
- The `lockstep` CLI ([lockstep.it](https://lockstep.it)): `invite get`,
  `check`, `run` and `submit`.
- [Task](https://taskfile.dev) (optional).
- No GPU: the networks here are small and train on the CPU.

## What is in it

```
train/
  main.py        the one command: train → export → score → self-check
  policy.py      the network, built from the environment's spaces: one input
                 per declared input, one output per action head
  ppo.py         a small PPO loop over lockstep_train.make_vec (the default recipe)
  supervised.py  learn from the truth a data environment reveals (LabelledStream)
  export.py      the ONNX export (one output per head) and the policy bundle
  evaluate.py    score the exported model and the no-action on practice seeds
  selfcheck.py   the bundle, the shell's rules, torch and ONNX playing the same
  shellrules.py  the ONNX shell's and the host's documented rules, as code
tests/           the template's own tests, on committed invite folders (tests/fixtures)
```

Training reads your invite's folder and writes two things into it:
`my-policy/` (the bundle you check and submit: `lockstep.toml` and
`models/policy.onnx`) and `training/my-policy/` (weights, checkpoint,
metrics, scores: yours, never submitted).

`lockstep-train` is Lockstep's Python package that steps the environment's
real engine as a Gymnasium environment, with spaces that match the ONNX
shell exactly. Use it directly to write your own training loop:
`lockstep_train.make(folder)`, `lockstep_train.make_vec(folder, 8)`.

## Commands

| | |
|---|---|
| `task setup` | `.venv` with the training stack and lockstep-train |
| `task train FOLDER=<invite folder>` | train (PPO, 100,000 turns), export to `<folder>/my-policy`, score, self-check. `RECIPE=supervised`, `STEPS=`, `MINUTES=`, `NAME=`, more flags after `--` (`python -m train.main --help`) |
| `task evaluate FOLDER=…` | the exported model and the no-action on practice seeds 0 to 9 |
| `task selfcheck FOLDER=…` | the bundle, the shell's rules, torch against ONNX |
| `task test` | the template's tests |
| `task lint` | ruff |

## Testing the template

`task test` trains, exports and checks on three committed invite folders
(card-fraud, jetpack-joust, ticket-desk: `tests/fixtures/SOURCE.txt`), with
no server. With a `lockstep` CLI that has `check` (`LOCKSTEP_BIN=` or on
PATH) and the runtime's files (`LOCKSTEP_RUNTIME_DIR=`), it also runs
`lockstep check` on each export, trains against an earlier export, and
adds panda-pick from a sibling LOCKSTEP-ENVIRONMENTS checkout. CI runs the
rest on Linux, macOS and Windows (`.github/workflows/ci.yml`).

## License

MIT or Apache-2.0, at your option.
