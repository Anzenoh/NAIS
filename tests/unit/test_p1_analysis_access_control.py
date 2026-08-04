import pytest

from vision_encoder_alignment.analysis.access_control import AccessError, AuthorizedFeatureReader


def test_span_split_prevents_whole_shard_loader():
    assert AuthorizedFeatureReader._spans(list(range(25)), 25) == ((0, 12), (12, 25))


def test_pre_freeze_test_authorization_rejected():
    reader = object.__new__(AuthorizedFeatureReader)
    reader.state_getter = lambda: "T004_DEV"
    with pytest.raises(AccessError, match="BLOCKED_LEAKAGE"):
        reader._authorize(("test",))
    reader.state_getter = lambda: "T007_TEST_UNSEAL"
    assert reader._authorize(("test",)) == "T007_TEST_UNSEAL"
