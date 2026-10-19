"""Offline public CLI and production-root boundary for T004G."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_query_adapter_low_rank_validation_metrics as cli
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_metrics_runtime as runtime


def test_public_signatures_have_only_fixed_roots() -> None:
    assert list(inspect.signature(runtime.run_low_rank_validation_metrics).parameters) == [
        "repository_root", "t001d_run_root", "t002d_validation_root", "t004f_fit_root", "output_root"]
    assert list(inspect.signature(binding.bind_low_rank_validation_runtime).parameters) == [
        "repository_root", "t001d_run_root", "t002d_validation_root"]
    assert not hasattr(runtime, "select_winner") and not hasattr(runtime, "generate")


@pytest.mark.parametrize("option", ["--test-root", "--rank", "--synthetic-shape", "--model", "--split"])
def test_cli_forbidden_options_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--repository-root", "a", "--t001d-run-root", "b",
                  "--t002d-validation-root", "c", "--t004f-fit-root", "d",
                  "--output-root", "e", option, "x"])
    assert error.value.code == 2


def test_help_is_offline_and_no_model_stack_import() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    command = [sys.executable, "scripts/run_query_adapter_low_rank_validation_metrics.py", "--help"]
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--t004f-fit-root" in completed.stdout and "--output-root" in completed.stdout
    assert "--test-root" not in completed.stdout and "--rank" not in completed.stdout


def test_cli_routes_exact_five_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    calls = []
    def run(*roots):
        calls.append(roots)
        return {"status": runtime.STATUS}
    monkeypatch.setattr(cli, "run_low_rank_validation_metrics", run)
    args = ["--repository-root", str(tmp_path / "repo"),
            "--t001d-run-root", str(tmp_path / "t001d"),
            "--t002d-validation-root", str(tmp_path / "t002d"),
            "--t004f-fit-root", str(tmp_path / "fit"),
            "--output-root", str(tmp_path / "out")]
    assert cli.main(args) == 0
    assert calls == [tuple(Path(args[index]) for index in (1, 3, 5, 7, 9))]
    assert runtime.STATUS in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["relative", "missing", "in_repo", "in_fit", "collision"])
def test_bad_roots_block_before_binder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       kind: str) -> None:
    roots = [tmp_path / item for item in ("repo", "t001d", "t002d", "fit")]
    for root in roots:
        root.mkdir()
    output = tmp_path / "out"
    if kind == "relative":
        output = Path("relative")
    elif kind == "missing":
        roots[2] = tmp_path / "absent"
    elif kind == "in_repo":
        output = roots[0] / "out"
    elif kind == "in_fit":
        output = roots[3] / "out"
    else:
        output.mkdir()
        (output / "keep").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(binding, "bind_low_rank_validation_runtime",
                        lambda *args: pytest.fail("binder called after bad root"))
    with pytest.raises(RuntimeError):
        runtime.run_low_rank_validation_metrics(*roots, output)
    if kind == "collision":
        assert (output / "keep").read_text(encoding="utf-8") == "keep"


def test_cli_reports_blocked_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--repository-root", str(tmp_path / "missing"),
                     "--t001d-run-root", str(tmp_path),
                     "--t002d-validation-root", str(tmp_path),
                     "--t004f-fit-root", str(tmp_path),
                     "--output-root", str(tmp_path / "out")])
    assert code == 1 and "BLOCKED" in capsys.readouterr().err
