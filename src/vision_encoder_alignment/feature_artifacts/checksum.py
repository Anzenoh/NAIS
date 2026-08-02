"""SHA-256 helpers for feature artifact files."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from .schema import ArtifactValidationError


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise when a file's SHA-256 digest does not match metadata."""

    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ArtifactValidationError(
            f"checksum_value: SHA-256 mismatch, expected {expected}, found {actual}"
        )
