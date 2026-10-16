# RBD-AIM

A paired-chain AntiBERTy classifier that assigns anti-SARS-CoV-2-RBD antibodies
to one of twelve receptor-binding-domain epitope classes from heavy- and
light-chain variable sequences.

This repository is a refactor of the research code accompanying Terekhov SS,
Ivanisenko NV *et al.*, "Mining antibody functionality via AI-guided structural
landscape profiling", *Nature Communications* 2026,
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

The repository implements a single component of the published RBD-AIM pipeline:
the AntiBERTy CLS+CDR epitope classifier. The homology-based and
structure-based components are not included.

Three workflows are supported: cross-validation training, scoring of
checkpoints against held-out folds, and inference on individual antibodies.

## Requirements

Python 3.10 or later. The AntiBERTy weights are distributed within the
`antiberty` wheel; all remaining dependencies are pinned in `uv.lock`.

## Installation

```bash
git clone https://github.com/nviva1/RBD-AIM
cd RBD-AIM
uv sync --locked --extra dev          # alternatively: pip install -e ".[dev]"
```

### Optional components

Inference on unnumbered sequences requires ANARCI, which performs the antibody
numbering. Training, `rbdaim score`, `predict_numbered` and the test suite do
not require it, as the distributed dataset contains pre-numbered sequences.

ANARCI is a Python wrapper around the `hmmscan` executable provided by HMMER.
HMMER is implemented in C and cannot be installed from the Python Package
Index, so one of the following is required:

```bash
pip install -e ".[predict]"           # ANARCI; HMMER must be present on PATH
conda env create -f environment.yml   # installs ANARCI and HMMER together
```

The `[predict-nohmmer]` extra installs ANARCII, a pure-Python reimplementation
that requires no external executable.

## Data and model weights

Neither the dataset nor the weights are redistributed with the source, and
neither has a default path.

### Dataset

The labelled Fab dataframe (`dataset_production.pkl`) accompanies the
publication; refer to its Data Availability statement. The frame contains 1,810
records, of which 1,788 carry a class label and 1,311 are curated. The column
contract validated on load, together with the distinction between these three
subsets, is specified in [`docs/DATA.md`](docs/DATA.md).

### Weights

Twenty checkpoints are hosted at
[huggingface.co/nviva1/rbd-aim-epitope](https://huggingface.co/nviva1/rbd-aim-epitope).
They are retrieved automatically by `rbdaim score --from-hub`, or may be
obtained directly:

```bash
hf download nviva1/rbd-aim-epitope --local-dir weights/
```

For each pooling mode the release comprises five replicates that held out fold
0 (`NN_cv_0_{CLS,CDR}_{0..4}.pth`) and five models trained on all available
data (`NN_cv_{0..4}_{CLS,CDR}_full.pth`). Fold 0 is consequently the only fold
against which these weights can be scored without evaluating on data used in
their training.

## Usage

```bash
# Cross-validation training
rbdaim train data.path=data/dataset_production.pkl

# CDR pooling, selected folds, GPU execution
rbdaim train data.path=data/dataset_production.pkl \
    model=antiberty_cdr 'train.folds=[0,1]' train.device=cuda

# Inference model: trained on all non-holdout records; no validation score
rbdaim train data.path=data/dataset_production.pkl train.full_model=true

# Score the published checkpoints, retrieving them on demand
rbdaim score --from-hub --fold 0 data.path=data/dataset_production.pkl

# Score a local checkpoint directory
rbdaim score --checkpoints weights/ --fold 0 data.path=data/dataset_production.pkl

# Display the composed configuration
rbdaim config data.path=data/dataset_production.pkl
```

Configuration is managed by Hydra. Unrecognised keys raise an error rather than
being ignored; `train.epocs=10` therefore fails instead of training with the
default number of epochs.

### Scoring semantics

Checkpoint selection is fold-aware. Only checkpoints that held out the fold
under evaluation are used, so reported metrics are not computed on records seen
during training. A request for a fold with no eligible checkpoint raises an
error enumerating the folds for which one exists.

The `--allow-full-models` and `--allow-other-folds` options remove this
restriction. Metrics obtained with either option incorporate training data and
do not constitute held-out performance.

The `--hub-revision` option pins the retrieved revision. Unpinned retrievals
track the default branch of the Hub repository and may differ between runs.

### Interpretation of scores

The classifier is trained with per-class binary cross-entropy. Inference
accordingly reports independent per-class sigmoid scores, which do not form a
probability distribution over classes and are not calibrated. They are
appropriate as a ranking rather than as class probabilities.

## Development

```bash
make install     # uv sync --locked --extra dev
make lint        # ruff check and format verification
make typecheck   # mypy --strict
make test        # pytest, excluding tests marked slow
make test-all    # complete suite, including tests requiring ANARCI
make smoke DATA=data/dataset_production.pkl      # end-to-end run on real data
make score-hub DATA=data/dataset_production.pkl  # published weights, fold 0
```

Continuous integration performs linting, type checking, execution of the test
suite under Python 3.10, 3.11 and 3.12, and a packaging check on every push and
pull request. As the AntiBERTy weights are distributed within its wheel, tests
that construct a model execute in CI; the `slow` marker is reserved for tests
requiring ANARCI.

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

A machine-readable form is provided in [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0; see [LICENSE](LICENSE).

This work is derived from [tanishAI/rbd-aim](https://github.com/tanishAI/rbd-aim)
(MIT) and depends on [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy)
(MIT). The corresponding upstream notices are reproduced in [NOTICE](NOTICE).
