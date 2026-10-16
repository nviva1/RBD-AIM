"""CLI argv handling and dispatch."""

from __future__ import annotations

import sys

import pytest

from rbdaim import cli


class TestSplitArgv:
    """argv must partition cleanly: every token lands in exactly one bucket."""

    def test_separates_flags_from_hydra_overrides(self):
        flags, overrides = cli.split_argv(
            [
                "--checkpoints",
                "weights/",
                "data.path=x.pkl",
                "--fold",
                "0",
                "train.epochs=2",
            ]
        )
        assert overrides == ["data.path=x.pkl", "train.epochs=2"]
        assert flags == ["--checkpoints", "weights/", "--fold", "0"]

    def test_flag_with_equals_goes_to_argparse_not_hydra(self):
        """Regression: `--fold=3` matched neither filter and vanished.

        It then silently scored fold 0 while reporting success.
        """
        flags, overrides = cli.split_argv(["--fold=3", "--pooling=cdr", "a.b=1"])
        assert flags == ["--fold=3", "--pooling=cdr"]
        assert overrides == ["a.b=1"]

    def test_no_token_is_ever_dropped(self):
        argv = ["--a", "1", "--b=2", "c.d=3", "e", "--f"]
        flags, overrides = cli.split_argv(argv)
        assert sorted(flags + overrides) == sorted(argv)


class TestScoreArgumentParsing:
    """`--flag=value` must actually reach the scoring call."""

    def _captured_kwargs(self, monkeypatch, tmp_path, argv):
        """Run `rbdaim score`, substituting a real directory for the `w/` token."""
        captured = {}

        def fake_score(config, **kwargs):
            captured.update(kwargs)
            raise SystemExit(0)  # stop before touching data

        import rbdaim.scoring

        monkeypatch.setattr(rbdaim.scoring, "score_checkpoints", fake_score)
        argv = [t.replace("w/", str(tmp_path)) for t in argv]
        with pytest.raises(SystemExit):
            cli._cmd_score(argv)
        # The resolved directory must reach the scoring call, not just parse.
        assert captured["checkpoint_dir"] == tmp_path
        return captured

    def test_equals_form_is_honoured(self, monkeypatch, tmp_path):
        kwargs = self._captured_kwargs(
            monkeypatch,
            tmp_path,
            ["--checkpoints=w/", "--fold=3", "--pooling=cdr", "data.path=x.pkl"],
        )
        assert kwargs["fold"] == 3
        assert kwargs["pooling"] == "cdr"

    def test_space_form_is_honoured(self, monkeypatch, tmp_path):
        kwargs = self._captured_kwargs(
            monkeypatch,
            tmp_path,
            [
                "--checkpoints",
                "w/",
                "--fold",
                "3",
                "--pooling",
                "cdr",
                "data.path=x.pkl",
            ],
        )
        assert kwargs["fold"] == 3
        assert kwargs["pooling"] == "cdr"

    def test_leak_guards_default_to_safe(self, monkeypatch, tmp_path):
        kwargs = self._captured_kwargs(
            monkeypatch, tmp_path, ["--checkpoints", "w/", "data.path=x.pkl"]
        )
        assert kwargs["allow_full_models"] is False
        assert kwargs["allow_other_folds"] is False
        assert kwargs["curated_only"] is True

    def test_pooling_defaults_to_config_not_hardcoded(self, monkeypatch, tmp_path):
        """`model=antiberty_cdr` must not be silently overridden by a CLI default."""
        kwargs = self._captured_kwargs(
            monkeypatch, tmp_path, ["--checkpoints", "w/", "data.path=x.pkl"]
        )
        assert kwargs["pooling"] is None  # defer to config.model.pooling


class TestDispatch:
    def test_known_commands_all_dispatch(self):
        for name, fn in cli._COMMANDS.items():
            assert callable(fn), name

    def test_help_exits_zero_and_prints_usage(self, capsys):
        sys.argv = ["rbdaim", "--help"]
        assert cli.main() == 0
        assert "usage: rbdaim" in capsys.readouterr().out

    def test_unknown_command_exits_nonzero(self, capsys):
        sys.argv = ["rbdaim", "frobnicate"]
        assert cli.main() == 2
        assert "unknown command" in capsys.readouterr().err

    def test_bare_invocation_prints_usage(self, capsys):
        sys.argv = ["rbdaim"]
        assert cli.main() == 0
        assert "usage: rbdaim" in capsys.readouterr().out

    def test_expected_errors_print_a_message_not_a_traceback(self, capsys):
        """A missing dataset must give the message, not a pandas stack trace."""
        sys.argv = ["rbdaim", "train", "data.path=/nonexistent/nope.pkl"]
        assert cli.main() == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ")
        assert "Traceback" not in err
        assert "README" in err

    def test_config_command_renders_composed_config(self, capsys):
        assert cli._cmd_config(["data.path=/tmp/x.pkl", "train.epochs=3"]) == 0
        out = capsys.readouterr().out
        assert "/tmp/x.pkl" in out
        assert "epochs: 3" in out

    def test_predict_requires_its_arguments(self):
        with pytest.raises(SystemExit):
            cli._cmd_predict(["--heavy", "EVQ"])
