"""Command-line entrypoints: ``rbdaim {train,score,predict,config}``.

A thin argv dispatcher in front of Hydra. The first token selects the command;
``key=value`` tokens are Hydra overrides and ``--flags`` are command options::

    rbdaim train data.path=data/dataset_production.pkl model=antiberty_cdr
    rbdaim score --from-hub data.path=data/dataset_production.pkl
    rbdaim score --checkpoints weights/ data.path=data/dataset_production.pkl
    rbdaim predict --heavy EVQ... --light DIQ... --checkpoints weights/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from rbdaim import logging as rlog
from rbdaim.config import Config, from_dictconfig

_CONF_DIRNAME = "conf"


def config_dir() -> Path:
    """Return the Hydra config directory.

    A ``./conf`` in the working directory wins (so a checkout can edit configs
    without reinstalling); otherwise the copy packaged inside ``rbdaim`` is
    used, which is what makes ``pip install rbdaim-classifier`` give a working
    CLI rather than one that dies composing its own defaults.
    """
    local = Path.cwd() / _CONF_DIRNAME
    if local.is_dir():
        return local
    return Path(__file__).resolve().parent / _CONF_DIRNAME


def load_config(overrides: list[str]) -> Config:
    """Compose the Hydra config with ``key=value`` CLI overrides."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = compose(config_name="config", overrides=overrides)
    return from_dictconfig(cfg)


def split_argv(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split argv into (argparse flags, Hydra ``key=value`` overrides).

    One function, one partition, so every token lands in exactly one bucket.
    Splitting with two independent filters — ``"=" in t`` for Hydra and
    ``"=" not in t`` for argparse — silently drops ``--flag=value``, which
    matches neither: ``--fold=3`` then scores fold 0 and says nothing.
    """
    flags: list[str] = []
    overrides: list[str] = []
    for token in tokens:
        if token.startswith("-"):
            flags.append(token)  # includes --flag=value
        elif "=" in token:
            overrides.append(token)
        else:
            flags.append(token)  # a value for the preceding flag
    return flags, overrides


def _reject_flags(prog: str, flags: list[str], usage: str) -> None:
    """Fail on any ``--flag`` for a command that takes only Hydra overrides.

    Without this, an unrecognised flag is silently dropped and the command runs
    with defaults — and ``rbdaim train --help`` starts a full training run
    instead of printing help.
    """
    if not flags:
        return
    if flags[0] in {"-h", "--help"}:
        print(usage)
        raise SystemExit(0)
    raise ValueError(
        f"{prog} takes no {flags[0]!r} option; it is configured with Hydra "
        f"key=value overrides.\n{usage}"
    )


_TRAIN_USAGE = (
    "usage: rbdaim train [hydra overrides]\n"
    "example: rbdaim train data.path=data/dataset_production.pkl "
    "model=antiberty_cdr train.epochs=4"
)

_CONFIG_USAGE = (
    "usage: rbdaim config [hydra overrides]\n"
    "example: rbdaim config data.path=data/dataset_production.pkl"
)


def _cmd_train(argv: list[str]) -> int:
    """Train cross-validation folds (or the full inference model)."""
    from rbdaim import trainer

    flags, overrides = split_argv(argv)
    _reject_flags("rbdaim train", flags, _TRAIN_USAGE)
    config = load_config(overrides)
    results = trainer.train(config)
    for r in results:
        score = (
            "n/a (full_model: no held-out split)"
            if r.best_val_f1_macro is None
            else f"{r.best_val_f1_macro:.4f}"
        )
        print(f"fold {r.fold}: {r.checkpoint_path}  val_f1_macro={score}")
    return 0


def _cmd_score(argv: list[str]) -> int:
    """Score checkpoints over the cross-validation folds of a real dataset."""
    from rbdaim.weights import DEFAULT_REPO_ID

    parser = argparse.ArgumentParser(prog="rbdaim score")
    parser.add_argument("--checkpoints", default=None, help="Directory of .pth files")
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Download the released checkpoints from the Hugging Face Hub "
        "instead of reading a local directory.",
    )
    parser.add_argument(
        "--hub-repo",
        default=DEFAULT_REPO_ID,
        help=f"Hub repo to download from (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--hub-revision",
        default=None,
        help="Pin the Hub revision (branch, tag or commit SHA). Unpinned "
        "downloads track the repo default and can change between runs.",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold to score against")
    parser.add_argument(
        "--pooling",
        default=None,
        choices=["cls", "cdr"],
        help="Override model.pooling (default: whatever the config composes to).",
    )
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument(
        "--allow-other-folds",
        action="store_true",
        help="Include checkpoints that held out a different fold. They trained "
        "on these rows, so the score will be inflated.",
    )
    parser.add_argument(
        "--allow-full-models",
        action="store_true",
        help="Include *_full.pth checkpoints. They trained on this fold's "
        "validation rows, so the score will be inflated.",
    )
    parser.add_argument(
        "--all-labeled",
        action="store_true",
        help="Score every labeled row, not just curated (good in {0,1}) ones. "
        "Not comparable to the curated-subset numbers.",
    )
    flags, overrides = split_argv(argv)
    args = parser.parse_args(flags)

    from rbdaim.scoring import score_checkpoints
    from rbdaim.weights import resolve_checkpoint_dir

    checkpoint_dir = resolve_checkpoint_dir(
        args.checkpoints,
        from_hub=args.from_hub,
        repo_id=args.hub_repo,
        revision=args.hub_revision,
    )
    config = load_config(overrides)
    report = score_checkpoints(
        config,
        checkpoint_dir=checkpoint_dir,
        fold=args.fold,
        pooling=args.pooling,
        ensemble=not args.no_ensemble,
        allow_full_models=args.allow_full_models,
        allow_other_folds=args.allow_other_folds,
        curated_only=not args.all_labeled,
    )
    print(report.render())
    return 0


def _cmd_predict(argv: list[str]) -> int:
    """Predict the epitope class for one Fab pair."""
    parser = argparse.ArgumentParser(prog="rbdaim predict")
    parser.add_argument("--heavy", required=True, help="Heavy-chain sequence")
    parser.add_argument("--light", required=True, help="Light-chain sequence")
    parser.add_argument("--checkpoints", required=True, help="Directory of .pth files")
    parser.add_argument("--pooling", default="cls", choices=["cls", "cdr"])
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--device", default="auto")
    flags, _ = split_argv(argv)
    args = parser.parse_args(flags)

    from rbdaim.predictor import EpitopePredictor

    predictor = EpitopePredictor.from_dir(
        args.checkpoints,
        pooling=args.pooling,
        ensemble=not args.no_ensemble,
        device=args.device,
    )
    pred = predictor.predict(args.heavy, args.light)
    print(f"predicted_epitope={pred.predicted_class}")
    print("# per-class sigmoid scores (independent, not a distribution)")
    for cls, score in pred.ranked():
        print(f"  {cls:>4}: {score:.4f}")
    return 0


def _cmd_config(argv: list[str]) -> int:
    """Print the composed configuration and exit."""
    flags, overrides = split_argv(argv)
    _reject_flags("rbdaim config", flags, _CONFIG_USAGE)
    print(OmegaConf.to_yaml(OmegaConf.structured(load_config(overrides))))
    return 0


_COMMANDS = {
    "train": _cmd_train,
    "score": _cmd_score,
    "predict": _cmd_predict,
    "config": _cmd_config,
}

# Errors whose message is written for a human and needs no traceback.
_EXPECTED_ERRORS = (
    FileNotFoundError,  # missing dataset / checkpoints (NoHonestCheckpointsError too)
    ValueError,  # bad config key, unscorable fold, unnumberable sequence
    RuntimeError,  # incompatible checkpoint, missing ANARCI, failed hub download
    OmegaConfBaseException,  # missing mandatory value, bad Hydra override
)

_USAGE = """usage: rbdaim <command> [options] [hydra overrides]

commands:
  train       Train cross-validation folds, or the full inference model
  score       Score checkpoints over a dataset fold (--from-hub downloads them)
  predict     Predict the epitope class for one heavy/light Fab pair
  config      Print the composed configuration

Run `rbdaim <command> --help` for a command's options."""


def main() -> int:
    """Entrypoint: dispatch on the first argv token."""
    rlog.configure()
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(_USAGE)
        return 0
    if argv[0] not in _COMMANDS:
        print(f"unknown command {argv[0]!r}\n\n{_USAGE}", file=sys.stderr)
        return 2
    try:
        return _COMMANDS[argv[0]](argv[1:])
    except _EXPECTED_ERRORS as exc:
        # These carry messages written for the user (missing dataset, missing
        # ANARCI, no honest checkpoint, metric regression). A traceback here
        # buries the message that was the point of raising.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
