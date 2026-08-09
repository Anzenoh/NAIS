from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
