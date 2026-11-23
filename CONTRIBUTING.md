# Contributing

Thanks for helping. This is a research harness, so the most valuable contributions are careful
experiments and honest results, including negative ones.

## Setup

```sh
uv sync
uv run pytest -q          # must pass before you open a pull request
uv run prepare.py         # only needed to train or evaluate
```

You need an Apple Silicon Mac to train. The tests run on the CPU, so they also run in CI.

## Kinds of contribution

### 1. Experiments (changes to `train.py`)

This is the normal path, and it is what the research agents do too.

- One idea per pull request. Run it through `uv run swarm/run.py --lane <lane> --desc ... --hypothesis ...`
  so it is measured exactly like everything else.
- In the pull request, include the ledger row, the `runs/<stamp>-<lane>/result.json`, your machine
  (chip and memory), and the base checkpoint's commit (from `~/.cache/inkling-autoresearch/base.json`).
- Claims of improvement need `SEED = 1` and `SEED = 2` re-runs, and the mean must beat the baseline by
  more than twice the seed standard deviation (see `program.md`).
- New modules must start as a no-op where possible (zero-initialised output projections, closed
  gates), so warm-starting from the base does not damage it. Add your switch to `CASES` in
  `tests/test_model.py`, and to `NO_OP_AT_INIT` if it is meant to be a no-op at init.
- Negative results are welcome. Put them in `results/notes-<lane>.md`.

### 2. Harness changes (`prepare.py`, `pretrain_base.py`, `evaluate()`)

These change what every result means, so open an issue first. A harness change invalidates all
earlier ledger rows, and it must say so in the pull request. Bug fixes in measurement code are the
exception: fix them, and say which past results they affect.

### 3. Tooling (`swarm/`, tests, docs)

Ordinary pull requests are fine. Keep the swarm's two invariants: training is serialized through the
GPU lock, and the ledger is written only by `swarm/run.py`.

## Rules that keep results comparable

- Do not raise `TIME_BUDGET`, `MAX_PARAMS` or the data in a results pull request.
- Never edit `results/ledger.tsv` by hand, and never cherry-pick seeds.
- Report `val_bpb` and reasoning accuracy together. A reasoning gain bought with a knowledge loss is
  a trade-off, not a win.
- Compare at matched `mean_loops_per_token` when halting is involved, or say that you did not.

## Style

Plain Python and MLX, no new dependencies in the training path. Short commit messages that say what
changed and why, for example `depth-attn: K=3 over pre-core states (val_bpb 1.712, reason_ood 0.21)`.
