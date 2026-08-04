from pathlib import Path

from vision_encoder_alignment.analysis.artifacts import deterministic_result_name


def test_windows_result_path_is_one_bounded_child(tmp_path: Path):
    name = deterministic_result_name("accepted_run", "1" * 40, "2" * 64)
    result = (tmp_path / name).resolve()
    assert result.parent == tmp_path.resolve()
    assert result.name == "accepted_run_analysis_111111111111_222222222222"
    assert not any(character in result.name for character in '<>:"/\\|?*')


def test_windows_runner_uses_only_native_entrypoints():
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts" / "run_p1_analysis_campaign.ps1").is_file()
    assert (root / "scripts" / "run_p1_analysis_campaign.py").is_file()
