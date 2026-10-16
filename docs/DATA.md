# The dataset contract

`rbdaim` validates every dataframe on load (`src/rbdaim/schema.py`), so a
malformed input fails immediately with a typed error instead of a `KeyError`
somewhere inside the training loop. This document is that contract in prose.

## Required columns

| column | type | meaning |
| --- | --- | --- |
| `index` | int, unique | Stable antibody identifier, carried through predictions |
| `POS_class` | int in `[-1, 11]` | Epitope class; `-1` means unlabeled |
| `good` | `{-1, 0, 1}` or null | Curation status; `1` = PDB-validated, `0` = curated, `-1` = not curated |
| `heavy_anarci` | str | Gap-free heavy chain in Chothia numbering order |
| `light_anarci` | str | Gap-free light chain in Chothia numbering order |
| `heavy_ids_cdr` | array of int | **1-based** ordinals of heavy CDR residues in `heavy_anarci` |
| `light_ids_cdr` | array of int | **1-based** ordinals of light CDR residues in `light_anarci` |
| `our_test` | bool-like | Final holdout membership |
| `cross_val_test_{0..9}` | bool-like or null | Fold membership; null means "not in this fold" |

Unknown extra columns are allowed and ignored.

## Three label sets, which are not interchangeable

The production frame has 1,810 rows, and three different "labeled" subsets live
inside it. Mixing them up silently changes every number you report.

| subset | rule | n | used for |
| --- | --- | --- | --- |
| trainable | `POS_class >= 0` | 1,788 | training and fold scoring |
| curated | `good in {0, 1}` | 1,311 | **the default scoring subset** |
| PDB-validated | `good == 1` | 392 | the PDB-validated evaluation subset |
| unlabeled | `POS_class == -1` | 22 | **trained on** by default; never scored |

The 30 rows with a null `good` are exactly the `our_test` holdout rows, so the
curated rule also excludes the holdout. The trainer does not rely on that
coincidence — it excludes `our_test` explicitly.

The curated set is what `rbdaim score` uses by default, matching the subset the
original research code filters to before computing its cross-validation metrics.
Pass `--all-labeled` to score the larger trainable set instead; the two are not
interchangeable.

Unlabeled rows are excluded from **scoring** — there is no ground truth to be
right or wrong about — but by default they still **train**, because the original
research code trains on them. Their one-hot encoding here is all-zero, which the
per-class BCE loss reads as "negative for all 12 epitopes". Set
`train.exclude_unlabeled=true` to drop them.

Note the original is not all-zero on these rows: it builds targets with
`np.eye(12)[POS_class]`, and `np.eye(12)[-1]` is the one-hot for **class 11**
(F3) rather than a zero vector. So upstream trains its 22 unlabeled rows as
if they were F3 binders, while this code trains them as negative for every
class. Neither is a deliberate design choice — both follow from how `-1` is
indexed. The behaviour here is left as-is rather than "corrected" in either
direction, because changing it changes what the weights learn; it is recorded
here so the difference is known rather than discovered.

## CDR indices and the token offset

`heavy_ids_cdr` / `light_ids_cdr` are 1-based residue ordinals. The dataset
class shifts them by −1, while the tokenizer prepends a `[CLS]` token — so
residue *r* occupies token index *r*, and CDR pooling reads a window one residue
to the left.

This is inherited from the published model, not introduced here. The original
research code builds 1-based ordinals (`rbdaim.py:255`) and applies the same −1
alongside `add_special_tokens=True` (`epitope_model.py:41,45,67,76`). Correcting
the offset would invalidate every released CDR checkpoint, so it is preserved
and commented in `src/rbdaim/data.py`.

## Obtaining the data

The labeled Fab dataframe accompanies the paper rather than this repository;
see its Data Availability section. Once you have it, nothing needs configuring
beyond passing the path:

```bash
rbdaim train data.path=/path/to/dataset_production.pkl
```

There is deliberately no default path. A default that resolves only on one
machine turns a missing-file error into a confusing one.
