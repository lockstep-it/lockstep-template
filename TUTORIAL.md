# Tutorial: from your invite to a submitted trained policy

A company has invited you to a Lockstep assessment. You can answer it with
a program you write yourself (`lockstep agent new` makes a starter in Rust,
C or Python), or with a **trained policy**: a neural network you train on
your own machine and send as one ONNX file. This tutorial is the second
way. It takes about 15 minutes the first time, most of it installing.

The times below were measured on a 2021 MacBook Pro (M1 Max) while it was
busy with other work; yours may be faster.

| Step | What you do | How long |
|---|---|---|
| 1 | Get your invite's folder | under a minute |
| 2 | Read what the environment asks of you | 5 to 10 minutes |
| 3 | Set up this repo | a few minutes (PyTorch is a large download) |
| 4 | Train, export and check, in one command | seconds to minutes: you choose |
| 5 | Read the scores | a minute |
| 6 | Check it as Lockstep will | seconds |
| 7 | Watch a match | a minute |
| 8 | Make it better | as long as you like |
| 9 | Submit | a minute, then Lockstep runs it (up to an hour) |

The examples use card-fraud, a fraud-detection assessment. Every command
works the same for any environment.

## 1. Get your invite's folder

Install the `lockstep` CLI (see [lockstep.it](https://lockstep.it)), then
use the invite id from your invite's page:

```sh
lockstep invite get inv_…
```

This makes a folder named after the environment, for example
`acme-card-fraud/`:

```
invite.toml          which invite and environment this folder is for
runtime.lock         the version of Lockstep's runtime your policy will run on
environment.txt      what the environment is: the goal, the scoring, every input and output
environment.html     the same, for a browser
.lockstep/           the environment's engine and practice data (don't edit)
```

Everything you need to train is in that folder. Nothing private is in it:
the practice data is for you, and Lockstep scores your submission on
other, private scenarios of the same kind.

## 2. Read what the environment asks of you

Open `environment.html` (or `environment.txt`). Four things matter for
training:

- **The goal and the scoring**: what a good match looks like, out of 100.
- **The inputs**: what your policy sees each turn. Each has a name, a
  shape and units.
- **The outputs**: what your policy decides each turn. A number output
  (f32) has a low and a high bound; a coded output (i32 or u8) is one of a
  few codes, such as approve, decline or step-up.
- **The reward**: a number the engine gives you each turn for training.
  For card-fraud the rewards of a match add up to its score.

Also note the **no-action**: what happens on a turn your policy sends
nothing. For card-fraud it is "approve, with confidence 0.5", which scores
about 20. A trained policy should beat that.

## 3. Set up this repo

You need Python 3.11 or later. Clone this repo next to your invite's
folder and install:

```sh
git clone https://github.com/lockstep-it/lockstep-template
cd lockstep-template
task setup
```

`task setup` makes `.venv/` here with PyTorch, ONNX, ONNX Runtime and
`lockstep-train`, Lockstep's package that runs the environment's real
engine in Python. No GPU is needed.

Without [Task](https://taskfile.dev):

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt "lockstep-train>=1.0,<2"
```

(On Windows, `.venv\Scripts\pip`.) Then use `.venv/bin/python -m train.main
…` wherever this tutorial says `task train …`.

## 4. Train, export and check, in one command

```sh
task train FOLDER=../acme-card-fraud
```

That does five things and tells you about each:

1. **Trains** with PPO for 100,000 turns, running 8 matches at once. Each
   turn is a turn of a real match: the same engine, rules and fuel
   Lockstep uses.
2. **Exports** the network to `../acme-card-fraud/my-policy/`:
   `lockstep.toml` and `models/policy.onnx`. That folder is what you
   submit.
3. **Scores** the exported model on practice seeds 0 to 9, which training
   never used, next to the no-action.
4. **Self-checks** the export: the bundle is right for your invite, the
   model's inputs and outputs are what Lockstep's ONNX shell reads, it uses
   only the operators Lockstep allows, and it chooses exactly what the
   trained network chooses.
5. Prints the next commands.

PPO's 100,000 turns took 20 to 45 seconds on card-fraud. `MINUTES=10`
trains for ten minutes instead; `STEPS=1000000` for a million turns.
Training writes a checkpoint after every update, so you can stop it with
Ctrl-C and continue later with `task train FOLDER=… -- --resume`.

### Two recipes

- **PPO** (the default) learns from the reward, by trial and error. It
  suits control tasks (robots, games) where the reward comes often.
- **Supervised** (`RECIPE=supervised`) suits data tasks where the
  environment later tells you what was true (card-fraud's `feedback`
  input: whether each payment was fraud). It plays 16 practice matches,
  collects every `(what you saw, what was true)` pair, and trains the
  first coded output to predict the truth.

For card-fraud, **use the supervised recipe**:

```sh
task train FOLDER=../acme-card-fraud RECIPE=supervised
```

It trains in about 5 seconds (under 20 with the export, the scoring and
the checks) and scored **56.6** on practice seeds 0 to 9, against **20.0** for the no-action. PPO
from scratch did worse than the no-action after a minute (11.8): its first
policy declines payments at random, the match ends early in a "decline
storm", and the score gives it almost nothing to follow out of there. That
is normal for some environments: the brief's tags (`method:…`) hint at
what works.

What one minute of PPO did on three environments (the mean score on
practice seeds 0 to 9, the exported model against the no-action):

| | turns trained | policy | no-action |
|---|---|---|---|
| card-fraud | 140,000 | 11.8 | 20.0 |
| ticket-desk | 744,000 | 24.4 | 24.8 |
| jetpack-joust (against a random opponent) | 628,000 | 50.0 | 50.0 |

So the PPO here is a working start, not a finished answer: it learns
slowly, and the work of the assessment is making it learn what matters
(step 8).

## 5. Read the scores

The end of a run looks like this:

```
Practice seeds 0-9 (never trained on), the exported model (2 s):
  seed   policy        do-nothing
     0     55.70 completed   20.00 completed
     1     77.25 completed   20.00 loss-cap
  …
  mean   56.553        20.0
Self-check:
  ok      bundle my-policy/: lockstep.toml v3, kind policy, card-fraud
  ok      model 73,612 bytes; inputs payment, card, …; outputs decision logits, confidence [-1, 1]
  ok      opset, operators, fixed shapes and sizes within Lockstep's preflight rules
  ok      torch and the export agree on 1000 recorded observations: …
```

Practice seed `n` is the same match `lockstep run --seed n` plays, so you
can watch any of them. The private scenarios Lockstep scores you on are
different seeds of the same kind, so expect a similar but not identical
mean.

`training/my-policy/` in your invite's folder keeps the run: `weights.pt`
(the network), `metrics.csv` (one row per PPO update: the score of recent
matches, the losses), `evaluation.json` (the table above).

## 6. Check it as Lockstep will

In your invite's folder:

```sh
cd ../acme-card-fraud
lockstep check --agent my-policy
```

This runs Lockstep's own checks on your bundle, the same ones it runs when
you submit, and plays one practice match through Lockstep's ONNX shell:

```
What Lockstep checks when you submit:
  ok      made for interface_version 1, the environment's (from its lockstep.toml)
  ok      the bundle is 66 KB (at most 95 MiB)
  ok      a trained policy the ONNX shell can play
A practice match on seed 0 (1 seat, your agent in it):
  ok      finished at turn 2000: completed, score 55.7
  ok      no bad turns: 2,000 turns played, none crashed, ran out of fuel or sent an invalid output
  ok      fuel: the busiest turn used 215,937 of the 1,000,000 a turn may use (22%); …
```

The score is the one step 5 showed for seed 0: training, the self-check and
Lockstep all play your model the same way. **Fuel** is Lockstep's measure
of work: your model's turn must fit the environment's budget. A bigger
network costs more; `check` tells you how much of the budget you use.

## 7. Watch a match

```sh
lockstep run --agent my-policy --seed 3
```

It plays practice seed 3, prints the result and saves the match in
`.lockstep/runs/<n>/match.archive`. Open
<https://app.lockstep.it/replay> and drop that file on the page: the
replay plays in your browser, and the file stays on your machine.

## 8. Make it better

- **Train longer**: `MINUTES=30`. Watch `training/my-policy/metrics.csv`:
  when `mean_score` stops rising, more time won't help much.
- **Change the network** in `train/policy.py`: wider layers (`-- --hidden
  128`), a different trunk, a convolution for an image input. Keep one
  output per head. Lockstep allows a fixed set of ONNX operators (no
  `Flatten`, `GELU`, `Slice` or embedding lookups on inputs, for
  example); the self-check names any operator it refuses, and `lockstep
  check` reports the fuel.
- **Change the reward**: `-- --reward score` trains on the final score
  only; `-- --reward series:<key>` on a series the environment declares.
- **Change the supervised labels**: `train/supervised.py`'s
  `label_to_index` decides which code each truth trains. Card-fraud's
  `confidence` output stays at 0.5 with this recipe, and its score gives
  10 points for a calibrated confidence: training it is a good next step.
- **Two-seat environments**: by default the other seat plays at random.
  `-- --opponents my-policy-v1` makes it play an earlier export of yours
  (any policy bundle folder in the invite's folder), so you can train
  generation by generation:

  ```sh
  task train FOLDER=../acme-joust NAME=my-policy-v1
  task train FOLDER=../acme-joust NAME=my-policy-v2 -- --opponents my-policy-v1
  ```

  `lockstep-train` 1.0 drives one seat at a time: both seats learning at
  once (self-play) isn't in it yet.
- **Write your own loop.** `lockstep_train.make(folder)` is a Gymnasium
  environment and `lockstep_train.make_vec(folder, 8)` a vector
  environment, with observation and action spaces that are exactly the
  ONNX shell's. Any RL library works; export with `train/export.py` and
  check with `task selfcheck FOLDER=…`.

Each run writes `my-policy/` again. Use `NAME=` to keep several (`NAME=v2`
writes `v2/` and `training/v2/`) and `lockstep check --agent v2` to check
one.

## 9. Submit

When you are happy with a policy:

```sh
cd ../acme-card-fraud
lockstep submit --agent my-policy --source ../lockstep-template/train
```

`submit` runs `lockstep check` first, then asks once: **a candidate's
submission is final.** If Lockstep doesn't accept it (the same checks as
step 6), it doesn't count and you can submit again. `--source` sends your
training code too, so the company can read how you made the policy; it
is optional. Lockstep then plays your policy on 20 private scenarios (up
to an hour), and your invite's page shows what happens next.

## How Lockstep plays your model

You don't need this to train, but it explains the export. Lockstep's ONNX
shell runs your `policy.onnx` once per turn:

- It feeds one model input per declared input, by name, shape `[1, …]`:
  float32 as the environment sends it (a u8 input divided by 255), or
  int32. Your model may leave inputs out.
- It reads one model output per action head. A number output is one tensor
  of values in −1 to 1: −1 becomes the output's low bound, 1 its high
  bound. A coded output gives one score (logit) per code, and the shell
  plays the code with the highest score. An output with slices or columns
  has one head per slice or column, named `<output>.<part>`.
- The model has no other outputs (training's value network is left out),
  fixed shapes, batch 1, opset 17, and every weight inside the file.

`lockstep_train`'s spaces follow these rules exactly, which is why the
policy plays the same in training and once submitted. The full rules are
in `.lockstep/sdk/docs/onnx-shell.md` of any starter `lockstep agent new`
makes.

## If something goes wrong

- **`isn't an invite's folder`**: pass the folder `lockstep invite get`
  made (it has `.lockstep/declaration.json`).
- **The physics files or the ONNX shell are missing** (robot
  environments, or training against an earlier policy): run `lockstep run`
  once in the invite's folder; it fetches them.
- **`lockstep check` says the model was made for another version**: the
  company updated the environment. Run `lockstep invite get` again, then
  train again.
- **The self-check FAILED on "torch and the export agree"**: something in
  your network exports differently (a Python `if` on values, an operator
  ONNX does differently). The line says which head.
