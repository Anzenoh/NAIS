"""Offline public API and CLI boundary checks; no real binding or fit."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_query_adapter_low_rank_train_fit as cli
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as runtime


def test_public_entry_has_four_explicit_roots_and_no_synthetic_controls() -> None:
    assert list(inspect.signature(runtime.run_low_rank_train_fit).parameters) == [
        "repository_root", "t001d_run_root", "t002d_validation_root", "output_root"]
    assert list(inspect.signature(cli.main).parameters) == ["argv"]
    assert not hasattr(runtime, "generate") and not hasattr(runtime, "select_winner")


@pytest.mark.parametrize("extra", ["--test-root", "--rank", "--synthetic-test-shape", "--split", "--model"])
def test_cli_rejects_forbidden_overrides(extra: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--repository-root", "a", "--t001d-run-root", "b",
                  "--t002d-validation-root", "c", "--output-root", "d", extra, "x"])
    assert error.value.code == 2


def test_cli_help_runs_offline_without_importing_model_stack() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    command = [sys.executable, "scripts/run_query_adapter_low_rank_train_fit.py", "--help"]
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--repository-root" in completed.stdout and "--output-root" in completed.stdout
    assert "--test-root" not in completed.stdout and "--rank" not in completed.stdout


def test_production_routes_exclusively_through_public_binder(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    roots = [tmp_path / name for name in ("repo", "t001d", "t002d")]
    for path in roots:
        path.mkdir()
    output = tmp_path / "fresh"
    token = object()
    calls: list[object] = []

    def bind(*args):
        calls.append(("bind", args))
        return token

    def publish(bound, destination, **kwargs):
        calls.append(("publish", bound, destination, kwargs))
        return {"status": runtime.STATUS}

    monkeypatch.setattr(runtime.binding, "bind_low_rank_runtime", bind)
    monkeypatch.setattr(runtime, "_publish", publish)
    assert cli.main(["--repository-root", str(roots[0]), "--t001d-run-root", str(roots[1]),
                     "--t002d-validation-root", str(roots[2]), "--output-root", str(output)]) == 0
    assert calls == [("bind", tuple(roots)), ("publish", token, output, {})]
    assert runtime.STATUS in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize("kind", ["relative", "missing", "inside_repo", "collision", "source_overlap"])
def test_bad_roots_stop_before_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    roots = [tmp_path / name for name in ("repo", "t001d", "t002d")]
    for path in roots:
        path.mkdir()
    output = tmp_path / "new"
    if kind == "relative":
        output = Path("relative-output")
    elif kind == "missing":
        roots[1] = tmp_path / "absent"
    elif kind == "inside_repo":
        output = roots[0] / "new"
    elif kind == "collision":
        output.mkdir()
        (output / "keep").write_text("keep", encoding="utf-8")
    else:
        output = roots[1] / "new"
    monkeypatch.setattr(runtime.binding, "bind_low_rank_runtime",
                        lambda *args: pytest.fail("binder called after bad roots"))
    with pytest.raises(runtime.LowRankTrainFitError):
        runtime.run_low_rank_train_fit(*roots, output)
    if kind == "collision":
        assert (output / "keep").read_text(encoding="utf-8") == "keep"


def test_cli_reports_blocked_bad_root_without_fit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--repository-root", str(tmp_path / "missing"),
                     "--t001d-run-root", str(tmp_path),
                     "--t002d-validation-root", str(tmp_path),
                     "--output-root", str(tmp_path / "output")]) == 1
    assert "BLOCKED" in capsys.readouterr().err
