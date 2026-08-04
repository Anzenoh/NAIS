from vision_encoder_alignment.analysis.artifacts import STATE_SEQUENCE
from vision_encoder_alignment.analysis.campaign import AnalysisCampaign


def test_campaign_declares_exact_monotonic_sequence():
    assert STATE_SEQUENCE == ("PREFLIGHT", "BLIND_INPUT_INTEGRITY", "REPEAT_CONTROL", "T004_DEV", "T005_DEV", "T006_DEV", "EXTENSION_DEV", "FREEZE", "T007_TEST_UNSEAL", "REPORT", "SEAL", "VERIFY")
    assert hasattr(AnalysisCampaign, "run_all")
