"""Deterministic synthetic fixtures for the sealed Phase 1 analysis lane."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from vision_encoder_alignment.analysis.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    build_review_bundle,
    seal_payload,
    verify_review_bundle,
    verify_sealed_package,
)
from vision_encoder_alignment.analysis.config import canonical_json_bytes


def paired(seed: int = 7, n: int = 64, d: int = 12) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    x = generator.standard_normal((n, d))
    q, _ = np.linalg.qr(generator.standard_normal((d, d)))
    return x, x @ q


def miniature_package(root: Path) -> dict[str, str | int]:
    payload = root / "payload"
    payload.mkdir(parents=True)
    generator = np.random.Generator(np.random.PCG64(20260804))
    encoders = [generator.standard_normal((24, 6)) for _ in range(3)]
    atomic_write_bytes(payload / "analysis_config.json", canonical_json_bytes({"seed": 20260804, "tensor_key_count": 18}))
    atomic_write_bytes(payload / "tables" / "miniature.json", canonical_json_bytes({
        "encoder_count": 3,
        "global_key_count": 14,
        "patch_key_count": 4,
        "fingerprints": [hashlib.sha256(value.tobytes()).hexdigest() for value in encoders],
        "stages": ["baseline", "linear_extension", "freeze", "t007", "report", "seal", "verify"],
    }))
    result = seal_payload(
        payload,
        root / "sealed_package.tar.gz",
        code_commit="1" * 40,
        code_tree="2" * 40,
        config_sha256="3" * 64,
        input_checksum_manifest_sha256="4" * 64,
        fixed_mtime=0,
    )
    assert verify_sealed_package(root / "sealed_package.tar.gz") == result
    atomic_write_bytes(
        root / "sealed_package.sha256",
        f"{result['sealed_package_sha256']}  sealed_package.tar.gz\n".encode("ascii"),
    )
    atomic_write_json(root / "delivery_summary.json", {
        "schema_id": "vea.p1.delivery_summary.v1",
        "sealed_package_sha256": result["sealed_package_sha256"],
    })
    atomic_write_bytes(root / "WINDOWS_VERIFY.txt", b"synthetic deterministic review fixture\n")
    review = build_review_bundle(root, root / "review_bundle.zip")
    verified_review = verify_review_bundle(root / "review_bundle.zip")
    assert verified_review["review_bundle_sha256"] == review["review_bundle_sha256"]
    return {**result, **review}
