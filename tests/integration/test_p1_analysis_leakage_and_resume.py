import json

from vision_encoder_alignment.analysis.artifacts import CampaignStore, STATE_SEQUENCE, atomic_write_bytes


def test_interrupted_test_resume_has_one_logical_unseal(tmp_path):
    store = CampaignStore(tmp_path, run_id="r", commit_sha="1" * 40, tree_sha="2" * 40, config_sha="3" * 64, input_checksum_manifest_sha="4" * 64)
    with store:
        for state in STATE_SEQUENCE[:8]:
            stage = store.begin_stage(state)
            atomic_write_bytes(stage / "x", state.encode())
            store.complete_stage(state, stage, {"status": "PASS"})
        binding = {
            "analysis_commit": "1" * 40,
            "analysis_tree": "2" * 40,
            "analysis_config_sha256": "3" * 64,
            "input_checksum_manifest_sha256": "4" * 64,
            "freeze_sha256": "5" * 64,
            "fitted_state_manifest_sha256": "6" * 64,
        }
        event_id = store.authorize_test_unseal(binding)
        interrupted = store.begin_stage("T007_TEST_UNSEAL")
        store.mark_test_read_started(event_id)
        atomic_write_bytes(interrupted / "partial", b"partial")
    with store:
        assert store.authorize_test_unseal(binding) == event_id
        resumed = store.begin_stage("T007_TEST_UNSEAL")
        assert resumed is not None
        store.complete_stage("T007_TEST_UNSEAL", resumed, {"status": "PASS"})
        assert store.load_state()["test_unseal_event_count"] == 1
        assert store.load_state()["test_unseal_authorization_count"] == 1
        assert store.load_state()["unseal_event_id"] == event_id
    events = [json.loads(line) for line in (tmp_path / "campaign_events.jsonl").read_text().splitlines()]
    assert sum(event.get("event") == "test_unseal" for event in events) == 1
    assert sum(event.get("event") == "test_unseal_authorized" for event in events) == 1
    assert sum(event.get("event") == "test_unseal_first_value_read" for event in events) == 1
