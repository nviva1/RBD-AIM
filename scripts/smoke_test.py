#!/usr/bin/env python
"""End-to-end smoke test on real data (CPU, tiny subset, fast).

Proves the plumbing wires together: load + validate data -> build dataset ->
instantiate model -> a few train steps -> save a checkpoint under the published
naming -> reload it through the predictor -> predict. Not a quality run; it is
the "does this work at all" gate. Quality lives in `rbdaim score`.

Usage:
    python scripts/smoke_test.py --data /path/to/dataset_production.pkl
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rbdaim import logging as rlog
from rbdaim.constants import NUM_CLASSES
from rbdaim.data import EpitopePairDataset, collate_fab, labeled_mask, load_dataframe
from rbdaim.model import AntiBertyPairClassifier
from rbdaim.predictor import EpitopePredictor
from rbdaim.torch_utils import resolve_device, seed_everything

log = rlog.get_logger("smoke")


def main() -> int:
    """Run the smoke test; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset_production.pkl")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--pooling", default="cls", choices=["cls", "cdr"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device("cpu")

    log.info("loading_data", path=args.data)
    df = load_dataframe(args.data)
    labeled = df[labeled_mask(df)]
    subset = labeled.head(args.n_train)
    log.info("data_ok", rows=len(df), labeled=len(labeled), using=len(subset))

    ds = EpitopePairDataset(subset, NUM_CLASSES)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fab)

    log.info("loading_antiberty_x2")
    model = AntiBertyPairClassifier(num_classes=NUM_CLASSES, pooling=args.pooling).to(
        device
    )
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for step, batch in enumerate(loader):
        inputs = {
            k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device))
            for k, v in batch.items()
            if k != "antibody_id"
        }
        loss = model(**inputs)["loss"]
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.info("train_step", step=step, loss=float(loss.item()))
        if step + 1 >= args.n_steps:
            break

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / f"NN_cv_0_{args.pooling.upper()}_full.pth"
        torch.save({"model_state": model.state_dict()}, ckpt)
        log.info("checkpoint_saved", path=str(ckpt))

        predictor = EpitopePredictor.from_dir(
            tmp, pooling=args.pooling, ensemble=False, device="cpu"
        )
        row = subset.iloc[0]
        # predict_numbered: the dataset rows are already numbered, so this
        # path needs no ANARCI install.
        pred = predictor.predict_numbered(row)
        log.info(
            "prediction_ok",
            predicted=pred.predicted_class,
            top_score=pred.ranked()[0][1],
            true_class=int(row["POS_class"]),
        )

    log.info("smoke_test_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
