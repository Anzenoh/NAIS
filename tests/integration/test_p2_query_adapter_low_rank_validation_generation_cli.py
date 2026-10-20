"""Subprocess checks for the T004J fake-only CLI contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_query_adapter_low_rank_validation_generation.py"


def call(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                          text=True, capture_output=True, check=False)


def test_fake_smoke_subprocess_and_fresh_output(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic"
    result = call("--fake-smoke", str(output))
    assert result.returncode == 0, result.stderr
    message = json.loads(result.stdout)
    assert message["generation_counts"] == {"native": 2, "adapted": 8, "total": 10}
    assert message["synthetic_test_only"] is True
    saved = runtime.read_json(output / "result.json")
    assert saved["rows_sha256"] == runtime.sha_file(output / "rows.jsonl")
    assert saved["aggregate_sha256"] == runtime.sha_file(output / "aggregate.json")
    assert len(runtime.read_rows(output / "rows.jsonl")) == 10
    repeated = call("--fake-smoke", str(output))
    assert repeated.returncode == 2 and "fresh" in repeated.stderr


def test_fake_real_flags_never_mix_and_production_roots_required(tmp_path: Path) -> None:
    output = tmp_path / "mixed"
    mixed = call("--fake-smoke", str(output), "--device", "cuda:0")
    assert mixed.returncode == 2 and "cannot mix" in mixed.stderr
    assert not output.exists()
    missing = call("--repository-root", str(ROOT))
    assert missing.returncode == 2
    assert not output.exists()


def test_fake_mode_does_not_import_model_or_gpu_runtime(tmp_path: Path) -> None:
    code = ("import sys; from scripts import run_query_adapter_low_rank_validation_generation as c; "
            f"assert c.main(['--fake-smoke', {str(tmp_path / 'offline')!r}]) == 0; "
            "assert 'transformers' not in sys.modules; "
            "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
