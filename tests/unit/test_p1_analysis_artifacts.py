import json

import pytest

from vision_encoder_alignment.analysis.artifacts import ArtifactError, CampaignStore, STATE_SEQUENCE, atomic_write_bytes


def _store(root):
    return CampaignStore(root, run_id="r", commit_sha="1" * 40, tree_sha="2" * 40, config_sha="3" * 64, input_checksum_manifest_sha="4" * 64)


def test_resume_no_clobber_tamper_and_concurrent_lock(tmp_path):
    store = _store(tmp_path / "run")
    store.acquire()
    with pytest.raises(ArtifactError):
        _store(tmp_path / "run").acquire()
    stage = store.begin_stage(STATE_SEQUENCE[0])
    atomic_write_bytes(stage / "value.txt", b"fixed\n")
    store.complete_stage(STATE_SEQUENCE[0], stage, {"status": "PASS"})
    store.release()
    (tmp_path / "run" / "preflight" / "value.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactError):
        _store(tmp_path / "run").acquire()
    assert not (tmp_path / "run" / ".campaign.lock").exists()


def test_output_collision(tmp_path):
    path = tmp_path / "x"
    atomic_write_bytes(path, b"one")
    with pytest.raises(ArtifactError):
        atomic_write_bytes(path, b"two")
