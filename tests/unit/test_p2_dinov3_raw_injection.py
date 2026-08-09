from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat import dinov3_raw_injection as smoke
from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native


SCRIPT = ROOT / "scripts/run_dinov3_raw_injection.py"
PROFILE = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"


class FakePatch:
    def __init__(self, shape: tuple[int, ...] = (1, 196, 768), dtype: str = "float32", finite: bool = True) -> None:
        self.shape = shape
        self.dtype = dtype
        self.finite = finite


class FakeTorch:
    def __init__(self) -> None:
        self.source: object | None = None

    def from_numpy(self, value: object) -> object:
        self.source = value
        return value


class FakeModel:
    def __init__(self) -> None:
        self.vision_calls = 0

    def encode_images(self, value: object) -> object:
        self.vision_calls += 1
        return value


class GeometryAndDINOContractTests(unittest.TestCase):
    def test_exact_canonical_geometry_plan(self) -> None:
        self.assertEqual(
            smoke.canonical_geometry_plan(300, 600),
            smoke.GeometryPlan((224, 448), (0, 112, 224, 336)),
        )
        self.assertEqual(
            smoke.canonical_geometry_plan(600, 300),
            smoke.GeometryPlan((448, 224), (112, 0, 336, 224)),
        )

    def test_exact_dino_identity_profile_and_block11(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(value["model"]["model_id"], smoke.DINO_MODEL_ID)
        self.assertEqual(value["model"]["revision"], smoke.DINO_REVISION)
        self.assertEqual(value["runtime_profile"]["selector"], smoke.DINO_SELECTOR)
        self.assertIn(smoke.DINO_TAP, value["tap_catalog"])
        mapping = {item["name"]: item["source"] for item in value["tap_mapping"]}
        self.assertEqual(mapping[smoke.DINO_TAP], "hidden_states[11]")
        self.assertEqual(value["token_layout"]["special_token_indices"], [0, 1, 2, 3, 4])

    def test_dino_patch_contract_rejects_shape_dtype_and_nonfinite(self) -> None:
        finite = lambda value: value.finite
        smoke.validate_dinov3_block11_patch(FakePatch(), finite_check=finite)
        for value in (FakePatch((1, 196, 767)), FakePatch(dtype="float16"), FakePatch(finite=False)):
            with self.assertRaises(smoke.DINOv3RawContractError):
                smoke.validate_dinov3_block11_patch(value, finite_check=finite)

    def test_only_raw_numpy_conversion_precedes_connector(self) -> None:
        patch = FakePatch()
        fake_torch = FakeTorch()
        self.assertIs(smoke.raw_numpy_to_torch(patch, torch=fake_torch), patch)
        self.assertIs(fake_torch.source, patch)


class ExplicitAppleRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.repository = base / "repository"
        self.runtime = base / "runtime"
        self.snapshot = base / "snapshot"
        self.repository.mkdir()
        self.runtime.mkdir()
        self.snapshot.mkdir()
        (self.repository / smoke.DINO_RUNTIME_PROFILE).parent.mkdir(parents=True)
        (self.repository / smoke.DINO_RUNTIME_PROFILE).write_text("{}", encoding="utf-8")
        (self.repository / smoke.DINO_SNAPSHOT_PROFILE).parent.mkdir(parents=True)
        (self.repository / smoke.DINO_SNAPSHOT_PROFILE).write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_interpreter(self, name: str = "python") -> Path:
        interpreter = self.runtime / "bin" / name
        interpreter.parent.mkdir(exist_ok=True)
        interpreter.write_bytes(b"fake")
        interpreter.chmod(0o700)
        return interpreter

    def _probe(self, *, prefix: Path | None = None, returncode: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=returncode,
            stdout=str(self.runtime if prefix is None else prefix) + "\n",
        )

    def test_explicit_roots_validate_without_localappdata_or_default_resolvers(self) -> None:
        interpreter = self._make_interpreter()
        snapshot_calls: list[tuple[Path, Path, Path]] = []

        def validate_snapshot(root: Path, *, profile_path: Path, repository_root: Path, runtime_root: Path) -> None:
            snapshot_calls.append((root, repository_root, runtime_root))

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            smoke, "_load_pinned_runtime_root", return_value=self.runtime.resolve()
        ), mock.patch.object(
            smoke, "resolve_apple_dinov3_interpreter", return_value=interpreter
        ), mock.patch.object(
            smoke, "_validate_pinned_snapshot", side_effect=validate_snapshot
        ):
            bindings = smoke.validate_dinov3_roots(
                repository_root=self.repository,
                runtime_root=self.runtime,
                snapshot_root=self.snapshot,
            )
        self.assertEqual(bindings.runtime_root, self.runtime.resolve())
        self.assertEqual(bindings.snapshot_root, self.snapshot.resolve())
        self.assertEqual(snapshot_calls, [(self.snapshot.resolve(), self.repository.resolve(), self.runtime.resolve())])
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        self.assertNotIn("resolve_default_runtime_root", source)
        self.assertNotIn("resolve_default_snapshot_root", source)

    def test_runtime_profile_root_mismatch_is_rejected(self) -> None:
        other = self.runtime.parent / "other-runtime"
        with mock.patch.object(smoke, "_load_pinned_runtime_root", return_value=other):
            with self.assertRaisesRegex(smoke.DINOv3RawContractError, "pinned Apple profile"):
                smoke.validate_dinov3_roots(
                    repository_root=self.repository,
                    runtime_root=self.runtime,
                    snapshot_root=self.snapshot,
                )

    def test_direct_bin_python_is_accepted_and_prefix_probed(self) -> None:
        interpreter = self._make_interpreter()
        runner = mock.Mock(return_value=self._probe())
        resolved = smoke.resolve_apple_dinov3_interpreter(self.runtime, run=runner)
        self.assertEqual(resolved, interpreter.resolve())
        self.assertEqual(runner.call_args.args[0][0], str(interpreter.resolve()))
        self.assertIn("sys.prefix", runner.call_args.args[0][3])

    def test_bin_python3_is_the_only_fallback(self) -> None:
        interpreter = self._make_interpreter("python3")
        resolved = smoke.resolve_apple_dinov3_interpreter(
            self.runtime, run=mock.Mock(return_value=self._probe())
        )
        self.assertEqual(resolved, interpreter.resolve())

    def test_legacy_venv_bin_python_is_never_selected(self) -> None:
        legacy = self.runtime / "venv" / "bin" / "python"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"fake")
        legacy.chmod(0o700)
        runner = mock.Mock()
        with self.assertRaises(smoke.DINOv3RawContractError):
            smoke.resolve_apple_dinov3_interpreter(self.runtime, run=runner)
        runner.assert_not_called()

    def test_explicit_roots_are_bound_in_extractor_config(self) -> None:
        captured: dict[str, object] = {}

        class FakeConfig:
            def __init__(self, profile_path: Path, **kwargs: object) -> None:
                captured["profile_path"] = profile_path
                captured.update(kwargs)

        class FakeExtractor:
            def __init__(self, config: object) -> None:
                captured["config"] = config

        fake_module = SimpleNamespace(
            DINOv3ExtractorConfig=FakeConfig,
            DINOv3VisionEncoderExtractor=FakeExtractor,
        )
        bindings = smoke.DINOv3RootBindings(
            self.repository.resolve(), self.runtime.resolve(), self.snapshot.resolve(), self.runtime / "bin/python"
        )
        profile = self.repository / "extractor.json"
        runtime_profile = self.repository / "runtime.json"
        with mock.patch.dict(
            sys.modules,
            {"vision_encoder_alignment.extractors.dinov3": fake_module},
        ):
            smoke._construct_dinov3_extractor(
                bindings,
                profile_path=profile,
                runtime_profile_path=runtime_profile,
            )
        self.assertIs(captured["runtime_root"], bindings.runtime_root)
        self.assertIs(captured["snapshot_root"], bindings.snapshot_root)
        self.assertEqual(captured["execution_selector"], smoke.DINO_SELECTOR)
        self.assertEqual(captured["apple_runtime_profile_path"], runtime_profile)

    def test_compatibility_context_restores_references_after_success(self) -> None:
        originals = tuple(lambda *args, index=index, **kwargs: (index, args, kwargs) for index in range(3))
        modules = tuple(SimpleNamespace(runtime_python_path=value) for value in originals)
        bindings = smoke.DINOv3RootBindings(
            self.repository.resolve(), self.runtime.resolve(), self.snapshot.resolve(), self.runtime / "bin/python"
        )
        with mock.patch.object(smoke, "_runtime_python_modules", return_value=modules):
            with smoke.apple_dinov3_interpreter_compatibility(bindings):
                for module in modules:
                    self.assertEqual(
                        module.runtime_python_path(self.runtime, interpreter_layout="linux_bin"),
                        bindings.interpreter,
                    )
            self.assertEqual(tuple(module.runtime_python_path for module in modules), originals)

    def test_compatibility_context_restores_references_after_exception(self) -> None:
        originals = tuple(lambda *args, index=index, **kwargs: index for index in range(3))
        modules = tuple(SimpleNamespace(runtime_python_path=value) for value in originals)
        bindings = smoke.DINOv3RootBindings(
            self.repository.resolve(), self.runtime.resolve(), self.snapshot.resolve(), self.runtime / "bin/python"
        )
        with mock.patch.object(smoke, "_runtime_python_modules", return_value=modules):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with smoke.apple_dinov3_interpreter_compatibility(bindings):
                    raise RuntimeError("boom")
            self.assertEqual(tuple(module.runtime_python_path for module in modules), originals)


class ReplayAndResultTests(unittest.TestCase):
    def test_raw_replay_bypasses_clip_vision_and_restores_method(self) -> None:
        model = FakeModel()
        original = model.encode_images
        raw_patch = FakePatch()
        connector_calls: list[object] = []

        def connector(value: object) -> object:
            connector_calls.append(value)
            return "connector-output"

        injection = native.ReplayInjection(model, raw_patch, connector)
        with injection:
            self.assertEqual(model.encode_images("placeholder"), "connector-output")
        self.assertEqual(model.vision_calls, 0)
        self.assertEqual(connector_calls, [raw_patch])
        self.assertTrue(injection.restored)
        self.assertIs(model.encode_images.__func__, original.__func__)

    def test_differing_answers_are_a_successful_diagnostic(self) -> None:
        result = smoke.build_result(
            status=smoke.STATUS,
            repository_commit="a" * 40,
            image_sha256="b" * 64,
            geometry_id=smoke.GEOMETRY_ID,
            tinyllava={},
            dinov3={},
            native={"answer": "native", "token_ids": [[1]]},
            raw_dinov3={"answer": "different", "token_ids": [[2]], "answer_equal_to_native": False},
            diagnostics={},
            artifacts={},
        )
        self.assertEqual(result["status"], smoke.STATUS)
        self.assertFalse(result["raw_dinov3"]["answer_equal_to_native"])
        self.assertEqual(result["schema_id"], smoke.RESULT_SCHEMA_ID)


class ImportAndCliTests(unittest.TestCase):
    def test_runtime_dependencies_absent_at_import_time(self) -> None:
        code = """
import sys
before=set(sys.modules)
import vision_encoder_alignment.vlm_compat.dinov3_raw_injection
added=set(sys.modules)-before
forbidden=('torch','transformers','safetensors','PIL','vision_encoder_alignment.extractors.dinov3')
assert not any(name==root or name.startswith(root+'.') for name in added for root in forbidden), sorted(added)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
            check=False, capture_output=True, text=True, shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_help_is_import_safe(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], cwd=ROOT,
            check=False, capture_output=True, text=True, shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--tinyllava-config", completed.stdout)
        self.assertIn("--dinov3-runtime-root", completed.stdout)
        self.assertIn("--dinov3-snapshot-root", completed.stdout)

    def test_cli_requires_both_explicit_dino_roots(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT,
            check=False, capture_output=True, text=True, shell=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--dinov3-runtime-root", completed.stderr)
        self.assertIn("--dinov3-snapshot-root", completed.stderr)


if __name__ == "__main__":
    unittest.main()
