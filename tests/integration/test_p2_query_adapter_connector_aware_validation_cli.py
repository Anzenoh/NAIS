"""CLI gates and deterministic fake execution, with no real model/root access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_query_adapter_connector_aware_validation as cli


def test_real_cli_requires_all_explicit_inputs(tmp_path: Path, monkeypatch, capsys) -> None:
    called = []
    monkeypatch.setattr(cli.binding_module, "preflight_runtime", lambda *args: called.append("preflight"))
    monkeypatch.setattr(cli.connector_module, "load_frozen_tinyllava_connector",
                        lambda **kwargs: called.append("model"))
    for name in ("t001d", "t002d", "cache"):
        (tmp_path / name).mkdir()
    base = ["--t001d-run-root", str(tmp_path / "t001d")]
    additions = [
        ["--t002d-validation-root", str(tmp_path / "t002d")],
        ["--cache-root", str(tmp_path / "cache")],
        ["--output-root", str(tmp_path / "output")],
        ["--device", "mps"],
    ]
    for index in range(len(additions)):
        args = base + [item for group in additions[:index] for item in group]
        assert cli.main(args) == 2
        assert json.loads(capsys.readouterr().err)["status"] == "BLOCKED"
    assert called == []


def test_real_preflight_precedes_model_load(tmp_path: Path, monkeypatch, capsys) -> None:
    for name in ("t001d", "t002d", "cache"):
        (tmp_path / name).mkdir()
    calls = []
    def stop_preflight(*args):
        calls.append("preflight")
        raise cli.binding_module.RuntimeBindingError("synthetic stop")
    monkeypatch.setattr(cli.binding_module, "preflight_runtime", stop_preflight)
    monkeypatch.setattr(cli.connector_module, "load_frozen_tinyllava_connector",
                        lambda **kwargs: calls.append("model"))
    assert cli.main([
        "--t001d-run-root", str(tmp_path / "t001d"),
        "--t002d-validation-root", str(tmp_path / "t002d"),
        "--cache-root", str(tmp_path / "cache"),
        "--output-root", str(tmp_path / "out"), "--device", "mps",
    ]) == 2
    assert calls == ["preflight"]
    assert "synthetic stop" in capsys.readouterr().err


def test_fake_mode_never_loads_model_or_runtime_roots_and_repeats(tmp_path: Path, monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("fake mode invoked a real boundary")
    monkeypatch.setattr(cli.binding_module, "preflight_runtime", forbidden)
    monkeypatch.setattr(cli.connector_module, "load_frozen_tinyllava_connector", forbidden)
    results = []
    for index in (1, 2):
        output = tmp_path / f"fake-{index}"
        assert cli.main(["--fake-smoke", str(output)]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["fake"] is True
        results.append(json.loads((output / "result.json").read_text(encoding="utf-8")))
    assert results[0] == results[1]
    assert results[0]["test_access_count"] == results[0]["test_metric_count"] == 0
    assert results[0]["generation_calls"] == results[0]["vision_calls"] == 0


def test_fake_mode_rejects_extra_real_arguments_and_output_collision(tmp_path: Path, capsys) -> None:
    output = tmp_path / "occupied"; output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")
    assert cli.main(["--fake-smoke", str(output)]) == 2
    assert "output root" in capsys.readouterr().err
    assert (output / "existing").read_text(encoding="utf-8") == "keep"
    assert cli.main(["--fake-smoke", str(tmp_path / "unused"), "--device", "cpu"]) == 2
    assert "fake smoke" in capsys.readouterr().err
