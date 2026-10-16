# Contributing

## Setup

```bash
uv sync --locked --extra dev
uv run pre-commit install
```

ANARCI (bioconda) is needed only for `rbdaim predict` on raw sequences; see
`environment.yml`. Everything else, including the whole test suite, runs from
the pip environment.

## Before opening a PR

```bash
make lint typecheck test
```

These are exactly what CI runs. If `make test` passes locally and CI fails, that
is a bug in the setup worth reporting.

## Conventions

- **Every config field is read by code.** A setting nothing consumes is a claim
  about behaviour that does not exist; delete it instead.
- **Tests must run without the dataset or the network.** AntiBERTy's weights
  ship inside its wheel, so building a real model in a test is fine. Mark
  `@pytest.mark.slow` only for things a plain `pip install` cannot provide
  (today: ANARCI), and those are excluded from CI.
- **A test must be able to fail.** Do not recompute the logic under test in the
  test body — exercise the real function and assert on what it did. If a fix is
  worth a test, reintroduce the bug once and watch the test go red.
- **Evaluation never trains.** `rbdaim score` consumes checkpoints that
  already exist.
- **Don't report a metric computed on training data.** If a code path has no
  held-out split, it returns `None` rather than a flattering number. See
  `train.full_model`.
- **Don't change behaviour the published checkpoints depend on** without saying
  so loudly. Several details of the dataset and training loop are chosen to match
  the original research code, not because they are what you would pick today.
- The coverage floor only moves up. Lowering it needs a reason in the PR body.
