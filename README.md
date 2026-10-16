# RBD-AIM

Epitope classification for anti-SARS-CoV-2-RBD antibodies. A paired-chain
AntiBERTy model maps a heavy/light Fab sequence pair to one of 12 RBD epitope
classes.

Refactor of the research code accompanying Terekhov SS, Ivanisenko NV *et al.*,
"Mining antibody functionality via AI-guided structural landscape profiling",
*Nature Communications* 2026,
[10.1038/s41467-026-70553-6](https://www.nature.com/articles/s41467-026-70553-6).

```console
$ rbdaim predict --heavy EVQLVESGG... --light DIQMTQSPS... --checkpoints weights/
predicted_epitope=A
# per-class sigmoid scores (independent, not a distribution)
     A: 0.9180
     C: 0.0750
     B: 0.0460
   ...
```

## Scope

This repository implements a single component of RBD-AIM: the AntiBERTy CLS+CDR
classifier. The homology and structure-based components of the published
pipeline are not included.

Supported workflows are training, scoring against held-out folds, and inference.

## Install

```bash
git clone https://github.com/nviva1/RBD-AIM
cd RBD-AIM
uv sync --locked --extra dev          # or: pip install -e ".[dev]"
```

Requires Python ≥ 3.10. AntiBERTy's weights ship inside its wheel; all other
dependencies are pinned in `uv.lock`.

### Optional: raw-sequence inference

`rbdaim predict` numbers raw sequences with ANARCI. Training, `rbdaim score`,
`predict_numbered` and the test suite do not need it, since the dataset ships
pre-numbered sequences.

ANARCI is a Python wrapper around `hmmscan` from HMMER, which is a C program and
therefore not installable from PyPI:

```bash
pip install -e ".[predict]"           # ANARCI; requires HMMER on PATH
conda env create -f environment.yml   # ANARCI + HMMER together
```

Alternatively, `[predict-nohmmer]` installs ANARCII, a pure-Python
reimplementation requiring no external binary.

## Data and weights

Neither is redistributable with the source, and neither has a default path.

**Dataset.** The labeled Fab dataframe (`dataset_production.pkl`) accompanies
the paper; see its Data Availability section. It holds 1,810 rows, of which
1,788 carry a class label and 1,311 are curated. [`docs/DATA.md`](docs/DATA.md)
specifies the column contract validated on load, and why those counts are not
interchangeable.

**Weights.** 20 checkpoints are hosted at
[huggingface.co/nviva1/rbd-aim-epitope](https://huggingface.co/nviva1/rbd-aim-epitope).
`rbdaim score --from-hub` fetches them automatically, or download them manually:

```bash
hf download nviva1/rbd-aim-epitope --local-dir weights/
```

Per pooling, the release contains five replicates that held out **fold 0**
(`NN_cv_0_{CLS,CDR}_{0..4}.pth`) and five models trained on all data
(`NN_cv_{0..4}_{CLS,CDR}_full.pth`). Fold 0 is therefore the only fold these
weights can score without evaluating on their own training data.

## Usage

```bash
# Cross-validation training
rbdaim train data.path=data/dataset_production.pkl

# CDR pooling, selected folds, GPU
rbdaim train data.path=data/dataset_production.pkl \
    model=antiberty_cdr 'train.folds=[0,1]' train.device=cuda

# Inference model: trains on all non-holdout rows, reports no validation score
rbdaim train data.path=data/dataset_production.pkl train.full_model=true

# Score the published checkpoints, downloading them on demand
rbdaim score --from-hub --fold 0 data.path=data/dataset_production.pkl

# Score a local checkpoint directory
rbdaim score --checkpoints weights/ --fold 0 data.path=data/dataset_production.pkl

# Print the composed configuration
rbdaim config data.path=data/dataset_production.pkl
```

Configuration uses Hydra. Unrecognised keys raise rather than falling back to a
default, so `train.epocs=10` fails instead of training with the default epoch
count.

### Scoring semantics

Checkpoint selection is fold-aware: only checkpoints that held out the fold
being scored are used, so reported metrics are never measured on training data.
Requesting a fold with no eligible checkpoint raises an error listing the folds
that have one.

`--allow-full-models` and `--allow-other-folds` lift that restriction. Metrics
produced with either flag include training data and are not held-out metrics.

Pass `--hub-revision <sha>` to pin the download. Unpinned downloads track the
Hub repository's default branch and may change between runs.

Inference reports independent per-class sigmoid scores, matching the per-class
BCE training objective. The scores are not a probability distribution over
classes and are not calibrated; treat them as a ranking.

## Development

```bash
make install     # uv sync --locked --extra dev
make lint        # ruff check + format --check
make typecheck   # mypy --strict
make test        # pytest, excluding -m slow
make test-all    # including tests that require ANARCI
make smoke DATA=data/dataset_production.pkl      # end-to-end run on real data
make score-hub DATA=data/dataset_production.pkl  # published weights, fold 0
```

CI runs lint, type checking, the test suite on Python 3.10/3.11/3.12, and a
packaging check on every push and pull request. AntiBERTy's weights ship in its
wheel, so tests that construct a real model run in CI; the `slow` marker covers
only tests requiring ANARCI.

## Citation

```bibtex
@article{terekhov2026rbdaim,
  title   = {Mining antibody functionality via AI-guided structural landscape profiling},
  author  = {Terekhov, Stanislav S. and Ivanisenko, Nikita V. and others},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-026-70553-6}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.

## License

Apache-2.0, see [LICENSE](LICENSE).

Derived from [tanishAI/rbd-aim](https://github.com/tanishAI/rbd-aim) (MIT) and
depends on [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy) (MIT). Both
upstream notices are preserved in [NOTICE](NOTICE).
