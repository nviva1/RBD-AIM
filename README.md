# RBD-AIM classifier

Antibody → SARS-CoV-2 RBD epitope classification with a paired-chain AntiBERTy
model, refactored from the research code accompanying Terekhov SS, Ivanisenko
NV *et al.*, "Mining antibody functionality via AI-guided structural landscape
profiling", *Nature Communications* 2026 —
[10.1038/s41467-026-70553-6](https://www.nature.com/articles/s41467-026-70553-6).

Give it a heavy and a light chain sequence; it returns one of 12 RBD epitope
classes.

```console
$ rbdaim predict --heavy EVQLVESGG... --light DIQMTQSPS... --checkpoints weights/
predicted_epitope=D1
# per-class sigmoid scores (independent, not a distribution)
    D1: 0.8412
    D2: 0.2231
     C: 0.1104
   ...
```

## Scope

This repository is **one model**: the paired-chain AntiBERTy CLS+CDR classifier.
The full RBD-AIM pipeline also has homology and structure-based components,
which live elsewhere and are not reimplemented here.

What this repo does: train the classifier, score it, run inference with it. It
is a refactor of the original research code into an installable package — see
the paper for the method and its results.

## Install

```bash
git clone https://github.com/nviva1/RBD-AIM
cd rbd-aim-classifier
uv sync --locked --extra dev          # or: pip install -e ".[dev]"
```

Python ≥ 3.10. AntiBERTy ships its own weights, so the first import pulls them
in automatically; everything else is pinned in `uv.lock`.

**One dependency is not on PyPI.** `rbdaim predict` on *raw* sequences needs
ANARCI, which `abnumber` uses for antibody numbering and which is published
through bioconda:

```bash
conda install -c bioconda anarci     # or: conda env create -f environment.yml
```

Training, `rbdaim score`, `predict_numbered` and the entire test suite work
without it — the dataset ships pre-numbered sequences. If it is missing, you get
an error that says so and names this command, rather than a `ModuleNotFoundError`
from somewhere inside abnumber.

## Getting the data and weights

Neither is redistributable with the source, and neither has a baked-in default
path — a default that resolves only on the author's laptop is worse than an
error message.

**Dataset.** The labeled Fab dataframe (`dataset_production.pkl`) accompanies
the paper — see its Data Availability section. 1,810 rows, of which 1,788 carry
a class label and 1,311 are curated; [`docs/DATA.md`](docs/DATA.md) documents the
column contract the loader validates on read, and why those three counts are not
interchangeable.

**Weights.** The 20 published checkpoints are at
[huggingface.co/nviva1/rbd-aim-epitope](https://huggingface.co/nviva1/rbd-aim-epitope):

```bash
huggingface-cli download nviva1/rbd-aim-epitope --local-dir weights/
```

Point the commands below at whatever directory you put them in.

The release contains, per pooling: five numbered replicates that held out
**fold 0** (`NN_cv_0_{CLS,CDR}_{0..4}.pth`), and five `full` models that held
out nothing (`NN_cv_{0..4}_{CLS,CDR}_full.pth`). So **fold 0 is the only fold
these weights can honestly score** — `rbdaim score --fold 1` is an error naming
the folds that are available, not a number.

## Usage

```bash
# Cross-validation training (the scorable run)
rbdaim train data.path=data/dataset_production.pkl

# CDR pooling instead of CLS, two folds, on a GPU
rbdaim train data.path=data/dataset_production.pkl \
    model=antiberty_cdr 'train.folds=[0,1]' train.device=cuda

# The shipped inference model: trains on everything, reports no validation score
rbdaim train data.path=data/dataset_production.pkl train.full_model=true

# One replicate of the ensemble (ensemble_id 0..4 for both poolings).
rbdaim train data.path=data/dataset_production.pkl train.ensemble_id=3

# Score checkpoints over a held-out fold. Selection is fold-aware: only
# checkpoints that held this fold out are used.
rbdaim score --checkpoints weights/ --fold 0 data.path=data/dataset_production.pkl

# Or download the released weights and score those
rbdaim score --from-hub --fold 0 data.path=data/dataset_production.pkl

# Inspect the composed configuration
rbdaim config data.path=data/dataset_production.pkl
```

Configuration is Hydra; `rbdaim config` prints what any override composes to.
An unrecognised key is an error rather than a silent default, so
`train.epocs=10` fails instead of silently doing nothing.

## Scoring the released weights

`rbdaim score --from-hub` downloads the published checkpoints from the Hugging
Face Hub and runs them over held-out rows of the dataset, reporting metrics from
real inference rather than from any stored numbers:

```bash
rbdaim score --from-hub --fold 0 data.path=data/dataset_production.pkl
```

Checkpoint selection is fold-aware: only checkpoints that held out the fold being
scored are used, so the reported numbers are not measured on training data. The
release holds out fold 0 only, so fold 0 is the only fold it can honestly score;
asking for another is an error naming the ones that work.

`--allow-full-models` and `--allow-other-folds` disable that guard. They exist so
that including training data in a score has to be a deliberate act; the numbers
they produce are inflated and are not held-out numbers.

Pin `--hub-revision <sha>` when a number needs to be reproducible — an unpinned
download tracks the repo's default branch and can change between runs.

## Development

```bash
make install     # uv sync --locked --extra dev
make lint        # ruff check + format --check
make typecheck   # mypy --strict
make test        # pytest, excluding -m slow
make test-all    # everything, including tests that need AntiBERTy weights
make smoke DATA=data/dataset_production.pkl   # real end-to-end run
make score-hub DATA=data/dataset_production.pkl  # released weights, fold 0
```

CI runs lint, typecheck, the fast suite on Python 3.10/3.11/3.12 and a packaging
check on every push and PR. AntiBERTy's weights ship inside its wheel, so tests
that build a real model run in CI too; the `slow` marker is only for things a
plain `pip install` cannot provide, which today means ANARCI.

## Citation

If you use this code or the weights, please cite the paper:

```bibtex
@article{terekhov2026rbdaim,
  title   = {Mining antibody functionality via AI-guided structural landscape profiling},
  author  = {Terekhov, Stanislav S. and Ivanisenko, Nikita V. and others},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-026-70553-6}
}
```

See `CITATION.cff` for the machine-readable form.

## License

Apache-2.0 — see [LICENSE](LICENSE).

This is a derivative refactor of [tanishAI/rbd-aim](https://github.com/tanishAI/rbd-aim)
(MIT) and depends on [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy)
(MIT). Both upstream notices are preserved in [NOTICE](NOTICE), as MIT requires.
