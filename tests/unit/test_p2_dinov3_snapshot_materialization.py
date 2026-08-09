from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/materialize_dinov3_snapshot_from_hf_cache.py"
SPEC = importlib.util.spec_from_file_location("p2_dinov3_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


class Identity:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self.size_bytes = len(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"bytes": self.size_bytes, "filename": self.filename, "sha256": self.sha256}


class FakeBoundary:
    SNAPSHOT_MANIFEST_SCHEMA_ID = "vea.external_model_snapshot.v0"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, bool]] = []

    def validate_snapshot_root(self, root: Path, _profile: object, **kwargs: object) -> object:
        path = Path(root)
        staging = bool(kwargs.get("_allow_owned_staging", False))
        self.calls.append((path, staging))
        if set(item.name for item in path.iterdir()) != {"files", "snapshot-manifest.json"}:
            raise ValueError("invalid layout")
        files = path / "files"
        if set(item.name for item in files.iterdir()) != {
            "config.json", "preprocessor_config.json", "model.safetensors"
        }:
            raise ValueError("invalid files")
        if any(item.is_symlink() or not item.is_file() for item in files.iterdir()):
            raise ValueError("non-plain file")
        return object()


class MaterializerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.repository = base / "repository"
        self.cache = base / "cache"
        self.source = self.cache / "snapshot"
        self.runtime = base / "runtime"
        self.destination = base / "published" / materializer.SNAPSHOT_ID
        for directory in (self.repository, self.source, self.runtime):
            directory.mkdir(parents=True)
        self.payloads = {
            "config.json": b"config",
            "preprocessor_config.json": b"processor",
            "model.safetensors": b"weights",
        }
        self.identities = tuple(Identity(name, payload) for name, payload in self.payloads.items())
        for name, payload in self.payloads.items():
            (self.source / name).write_bytes(payload)
        self.profile = SimpleNamespace(
            files=self.identities,
            staging_name_prefix=f".{materializer.SNAPSHOT_ID}.staging-",
            source_sha256="a" * 64,
            snapshot_id=materializer.SNAPSHOT_ID,
            selected_total_bytes=sum(len(value) for value in self.payloads.values()),
            model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
            revision="5931719e67bbdb9737e363e781fb0c67687896bc",
            weight_format="single_file_safetensors",
            weight_file=self.identities[2],
        )
        self.boundary = FakeBoundary()
        self.plan = materializer.MaterializationPlan(
            self.repository.resolve(),
            self.cache.resolve(),
            self.source.resolve(),
            self.destination.resolve(strict=False),
            self.runtime.resolve(),
            self.repository / "profile.json",
            self.profile,
            self.boundary,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def enough_space(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=2 * 1024**3)

    @staticmethod
    def write_fake_manifest(_plan: object, staging: Path) -> None:
        path = staging / "snapshot-manifest.json"
        with path.open("xb") as stream:
            stream.write(b"{}\n")
            stream.flush()
            os.fsync(stream.fileno())

    def materialize(self, **kwargs: object) -> str:
        values = {"write_manifest": self.write_fake_manifest, "disk_usage": self.enough_space}
        values.update(kwargs)
        return materializer.materialize_snapshot(self.plan, **values)

    def staging_entries(self) -> list[Path]:
        parent = self.destination.parent
        return [] if not parent.exists() else list(parent.glob(self.profile.staging_name_prefix + "*"))


class ManifestAndBindingTests(unittest.TestCase):
    def test_exact_canonical_manifest_bytes_sha_and_profile_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cache = base / "cache"
            source = cache / materializer.SOURCE_SNAPSHOT_RELATIVE_PATH
            runtime = base / "runtime"
            source.mkdir(parents=True)
            runtime.mkdir()
            destination = base / "destination" / materializer.SNAPSHOT_ID
            plan = materializer.plan_materialization(
                repository_root=ROOT,
                source_cache_root=cache,
                source_snapshot_root=source,
                destination_root=destination,
                runtime_root=runtime,
            )
            payload = materializer.manifest_payload(plan)
            value = json.loads(payload)
            self.assertEqual(len(payload), 1129)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), materializer.MANIFEST_SHA256)
            self.assertEqual(value["repository_profile_sha256"], materializer.PROFILE_SHA256)
            self.assertEqual(value["provisioned_at_utc"], materializer.PROVISIONED_AT_UTC)

    def test_plan_rejects_a_different_cache_snapshot_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cache = base / "cache"
            source = cache / "different-snapshot"
            runtime = base / "runtime"
            source.mkdir(parents=True)
            runtime.mkdir()
            with self.assertRaisesRegex(materializer.MaterializationError, "exact Hugging Face"):
                materializer.plan_materialization(
                    repository_root=ROOT,
                    source_cache_root=cache,
                    source_snapshot_root=source,
                    destination_root=base / "destination" / materializer.SNAPSHOT_ID,
                    runtime_root=runtime,
                )

    def test_live_apple_profile_chain_is_coherent(self) -> None:
        apple = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
        digest = hashlib.sha256(apple.read_bytes()).hexdigest()
        apple_value = json.loads(apple.read_text(encoding="utf-8"))
        extraction = json.loads(
            (ROOT / "configs/extraction_profiles/p1_1000_apple_fp32_v0.json").read_text(encoding="utf-8")
        )
        source = (ROOT / "src/vision_encoder_alignment/p1_feature_extraction.py").read_text(encoding="utf-8")
        self.assertEqual(
            apple_value["identities"]["external_snapshot_manifest_sha256"],
            materializer.MANIFEST_SHA256,
        )
        self.assertEqual(extraction["runtime"]["dinov3"]["extractor_profile"]["sha256"], digest)
        self.assertIn(f'"sha256": "{digest}"', source)
        self.assertNotIn("7b83fd74033f359a5a7b4e7d052b8d1bd106c658355ae86499947d151b0176a9", source)

    def test_import_is_standard_library_only(self) -> None:
        code = f"""
import importlib.util, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('materializer_import_test', {str(SCRIPT)!r})
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
spec.loader.exec_module(module)
added=set(sys.modules)-before
forbidden=('huggingface_hub','torch','transformers','requests')
assert not any(name==root or name.startswith(root+'.') for name in added for root in forbidden), sorted(added)
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=ROOT,
            check=False, capture_output=True, text=True, shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SourceValidationTests(MaterializerFixture):
    def test_exact_three_file_allowlist_ignores_extra_cache_files(self) -> None:
        (self.source / "README.md").write_text("ignored", encoding="utf-8")
        selected = materializer.validate_source_files(self.plan)
        self.assertEqual(set(selected), set(self.payloads))

    def test_source_symlink_resolves_inside_cache(self) -> None:
        leaf = self.source / "config.json"
        target = self.cache / "blob-config"
        target.write_bytes(leaf.read_bytes())
        leaf.unlink()
        try:
            leaf.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        selected = materializer.validate_source_files(self.plan)
        self.assertEqual(selected["config.json"], target.resolve())

    def test_source_symlink_escape_is_rejected(self) -> None:
        leaf = self.source / "config.json"
        target = self.cache.parent / "outside-config"
        target.write_bytes(leaf.read_bytes())
        leaf.unlink()
        try:
            leaf.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        with self.assertRaisesRegex(materializer.MaterializationError, "escapes cache"):
            materializer.validate_source_files(self.plan)

    def test_incorrect_size_or_sha_fails_before_publication(self) -> None:
        (self.source / "config.json").write_bytes(b"wrong")
        with self.assertRaisesRegex(materializer.MaterializationError, "identity mismatch"):
            self.materialize()
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.staging_entries(), [])


class PublicationTests(MaterializerFixture):
    def test_plain_exact_destination_and_both_validator_calls(self) -> None:
        self.assertEqual(self.materialize(), "DINOV3_CONTRACT_SNAPSHOT_MATERIALIZED")
        self.assertEqual(set(item.name for item in self.destination.iterdir()), {"files", "snapshot-manifest.json"})
        files = tuple((self.destination / "files").iterdir())
        self.assertEqual({item.name for item in files}, set(self.payloads))
        self.assertTrue(all(item.is_file() and not item.is_symlink() for item in files))
        self.assertEqual([staging for _path, staging in self.boundary.calls], [True, False])

    def test_existing_invalid_destination_is_not_overwritten(self) -> None:
        self.destination.mkdir(parents=True)
        marker = self.destination / "do-not-overwrite"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(materializer.MaterializationCollisionError):
            self.materialize()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_existing_exact_destination_is_idempotently_reused(self) -> None:
        self.materialize()
        before = {path.name: path.read_bytes() for path in (self.destination / "files").iterdir()}
        self.assertEqual(self.materialize(), "DINOV3_CONTRACT_SNAPSHOT_ALREADY_VALID")
        after = {path.name: path.read_bytes() for path in (self.destination / "files").iterdir()}
        self.assertEqual(before, after)

    def test_copy_manifest_and_rename_failures_cleanup_staging(self) -> None:
        def fail_copy(_source: Path, _destination: Path) -> None:
            raise OSError("copy")

        def fail_manifest(_plan: object, _staging: Path) -> None:
            raise OSError("manifest")

        def fail_rename(_source: Path, _destination: Path) -> None:
            raise OSError("rename")

        for key, failure in (("copy_file", fail_copy), ("write_manifest", fail_manifest), ("rename", fail_rename)):
            with self.subTest(stage=key):
                with self.assertRaises(OSError):
                    self.materialize(**{key: failure})
                self.assertFalse(self.destination.exists())
                self.assertEqual(self.staging_entries(), [])

    def test_verify_command_uses_existing_validator(self) -> None:
        self.materialize()
        self.boundary.calls.clear()
        self.assertEqual(materializer.verify_snapshot(self.plan), "DINOV3_CONTRACT_SNAPSHOT_VERIFY_OK")
        self.assertEqual(self.boundary.calls, [(self.plan.destination_root, False)])


if __name__ == "__main__":
    unittest.main()
