from pathlib import Path
from types import SimpleNamespace

from vision_encoder_alignment.analysis.artifacts import STATE_SEQUENCE
from vision_encoder_alignment.analysis.campaign import AnalysisCampaign
from vision_encoder_alignment.analysis.config import load_analysis_config


def test_campaign_declares_exact_monotonic_sequence():
    assert STATE_SEQUENCE == ("PREFLIGHT", "BLIND_INPUT_INTEGRITY", "REPEAT_CONTROL", "T004_DEV", "T005_DEV", "T006_DEV", "EXTENSION_DEV", "FREEZE", "T007_TEST_UNSEAL", "REPORT", "SEAL", "VERIFY")
    assert hasattr(AnalysisCampaign, "run_all")


def test_blind_input_integrity_writes_frozen_config_bytes(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    config = load_analysis_config(repository / "configs" / "analysis" / "p1_analysis_campaign_v1.json")
    checksum_manifest = tmp_path / "input_252_file_sha256.txt"
    checksum_manifest.write_text("0" * 64 + "  fixture\n", encoding="ascii")
    result_root = tmp_path / "result"
    stage_root = tmp_path / "blind_input_integrity"
    result_root.mkdir()
    stage_root.mkdir()

    campaign = object.__new__(AnalysisCampaign)
    campaign.config = config
    campaign.input = SimpleNamespace(
        run_root=tmp_path / "input",
        evidence_root=tmp_path / "evidence",
        checksum_manifest_sha256="1" * 64,
        checksum_manifest_path=checksum_manifest,
        verified_file_count=252,
    )
    campaign.result_root = result_root

    summary = campaign._blind_input_integrity(stage_root)

    assert summary == {"status": "PASS", "verified_file_count": 252, "decoded_value_rows": 0}
    assert (result_root / "analysis_config.json").read_bytes() == config.canonical_bytes
