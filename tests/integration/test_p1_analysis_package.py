import io
import tarfile

import pytest

from vision_encoder_alignment.analysis.artifacts import ArtifactError, verify_review_bundle, verify_sealed_package
from tests._p1_analysis_fixtures import miniature_package


def test_two_clean_miniature_campaigns_are_byte_identical(tmp_path):
    first = miniature_package(tmp_path / "first")
    second = miniature_package(tmp_path / "second")
    assert first == second
    assert (tmp_path / "first" / "sealed_package.tar.gz").read_bytes() == (tmp_path / "second" / "sealed_package.tar.gz").read_bytes()
    assert (tmp_path / "first" / "review_bundle.zip").read_bytes() == (tmp_path / "second" / "review_bundle.zip").read_bytes()


def test_package_tamper_is_rejected(tmp_path):
    miniature_package(tmp_path / "run")
    archive = tmp_path / "run" / "sealed_package.tar.gz"
    payload = archive.read_bytes()
    archive.write_bytes(payload[: len(payload) // 2])
    with pytest.raises((ArtifactError, tarfile.TarError, EOFError)):
        verify_sealed_package(archive)


def test_review_bundle_embeds_and_verifies_sealed_package(tmp_path):
    miniature_package(tmp_path / "run")
    verified = verify_review_bundle(tmp_path / "run" / "review_bundle.zip")
    assert verified["review_bundle_covered_member_count"] == 4


def test_review_bundle_tamper_is_rejected(tmp_path):
    miniature_package(tmp_path / "run")
    bundle = tmp_path / "run" / "review_bundle.zip"
    payload = bundle.read_bytes()
    bundle.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(ArtifactError):
        verify_review_bundle(bundle)
