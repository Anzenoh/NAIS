from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import vision_encoder_alignment.vlm_compat.tinyllava_native_replay as replay


CONFIG = ROOT / "configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json"
SCRIPT = ROOT / "scripts/run_tinyllava_native_replay.py"


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], values: tuple[float, ...], dtype: str = "float32") -> None:
        self.shape = shape
        self.values = values
        self.dtype = dtype

    def detach(self) -> "FakeTensor":
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.shape, self.values, self.dtype)


class HookHandle:
    def __init__(self, callbacks: list[object], callback: object) -> None:
        self.callbacks = callbacks
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.callbacks:
            self.callbacks.remove(self.callback)


class HookModule:
    def __init__(self, output: FakeTensor) -> None:
        self.output = output
        self.pre_hooks: list[object] = []
        self.hooks: list[object] = []

    def register_forward_pre_hook(self, callback: object) -> HookHandle:
        self.pre_hooks.append(callback)
        return HookHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback: object) -> HookHandle:
        self.hooks.append(callback)
        return HookHandle(self.hooks, callback)

    def __call__(self, value: object) -> FakeTensor:
        args = (value,)
        for callback in tuple(self.pre_hooks):
            callback(self, args)  # type: ignore[operator]
        for callback in tuple(self.hooks):
            callback(self, args, self.output)  # type: ignore[operator]
        return self.output


class TokenTensor:
    def detach(self) -> "TokenTensor":
        return self

    def cpu(self) -> "TokenTensor":
        return self

    def tolist(self) -> list[list[int]]:
        return [[1, 2, 3]]


class FakeModel:
    def __init__(self) -> None:
        self.vision_calls = 0

    def encode_images(self, value: object) -> object:
        self.vision_calls += 1
        return value

    def generate(self) -> TokenTensor:
        return TokenTensor()


class PinnedChatModel:
    def chat(
        self,
        prompt: str,
        tokenizer: object | None = None,
        image: str | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        top_p: float | None = None,
        temperature: float = 0,
    ) -> str:
        return prompt


class MissingTemperatureChatModel:
    def chat(
        self,
        prompt: str,
        tokenizer: object | None = None,
        image: str | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        top_p: float | None = None,
    ) -> str:
        return prompt


def _finite(value: object) -> bool:
    return all(math.isfinite(item) for item in value.values)  # type: ignore[attr-defined]


class AnchorConfigTests(unittest.TestCase):
    def test_strict_config_validation(self) -> None:
        config = replay.load_anchor_config(CONFIG)
        self.assertEqual(config.schema_id, replay.ANCHOR_SCHEMA_ID)
        value = config.to_dict()
        value["unknown"] = True
        with self.assertRaises(replay.AnchorConfigError):
            replay.AnchorConfig.from_mapping(value)

    def test_invalid_model_identity_rejected(self) -> None:
        value = replay.load_anchor_config(CONFIG).to_dict()
        value["model_id"] = "other/model"
        with self.assertRaisesRegex(replay.AnchorConfigError, "model_id"):
            replay.AnchorConfig.from_mapping(value)

    def test_duplicate_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"schema_id":"x","schema_id":"y"}', encoding="utf-8")
            with self.assertRaises(replay.AnchorConfigError):
                replay.load_anchor_config(path)


class TensorContractTests(unittest.TestCase):
    def test_patch_shape_rejected(self) -> None:
        value = FakeTensor((1, 195, 768), (0.0,))
        with self.assertRaisesRegex(replay.ReplayContractError, "shape"):
            replay.validate_tensor_contract(
                value,
                expected_shape=(1, 196, 768),
                expected_dtype="float32",
                finite_check=_finite,
                label="patch",
            )

    def test_connector_shape_rejected(self) -> None:
        value = FakeTensor((1, 196, 768), (0.0,))
        with self.assertRaisesRegex(replay.ReplayContractError, "shape"):
            replay.validate_tensor_contract(
                value,
                expected_shape=(1, 196, 1536),
                expected_dtype="float32",
                finite_check=_finite,
                label="connector",
            )

    def test_nonfinite_rejected(self) -> None:
        value = FakeTensor((1, 196, 768), (float("nan"),))
        with self.assertRaisesRegex(replay.ReplayContractError, "NaN"):
            replay.validate_tensor_contract(
                value,
                expected_shape=(1, 196, 768),
                expected_dtype="float32",
                finite_check=_finite,
                label="patch",
            )

    def test_fake_serialization_exact_round_trip(self) -> None:
        original = FakeTensor((1, 196, 768), (0.0, 1.0))
        reloaded = FakeTensor((1, 196, 768), (0.0, 1.0))
        result = replay.validate_serialization_round_trip(
            original,
            reloaded,
            finite_check=_finite,
            max_abs_diff=lambda left, right: max(
                abs(a - b) for a, b in zip(left.values, right.values)  # type: ignore[attr-defined]
            ),
        )
        self.assertEqual(result["max_abs_diff"], 0.0)


class ReplayBoundaryTests(unittest.TestCase):
    def test_native_call_accounting(self) -> None:
        vision = HookModule(FakeTensor((1, 197, 768), (0.0,)))
        connector = HookModule(FakeTensor((1, 196, 1536), (0.0,)))
        model = FakeModel()

        def chat() -> str:
            vision("pixels")
            connector(FakeTensor((1, 196, 768), (0.0,)))
            model.generate()
            return "ok"

        state, token_ids, text = replay._run_chat_capture(  # type: ignore[attr-defined]
            model=model,
            vision_tower=vision,
            connector=connector,
            chat=chat,
        )
        self.assertEqual(state.vision_calls, 1)
        self.assertEqual(state.connector_calls, 1)
        self.assertEqual(token_ids, [[1, 2, 3]])
        self.assertEqual(text, "ok")

    def test_replay_bypasses_vision_and_restores_after_success(self) -> None:
        model = FakeModel()
        original = model.encode_images
        calls: list[object] = []

        def connector(value: object) -> object:
            calls.append(value)
            return "connected"

        patch = FakeTensor((1, 196, 768), (0.0,))
        injection = replay.ReplayInjection(model, patch, connector)
        with injection:
            self.assertEqual(model.encode_images("ignored"), "connected")
        self.assertEqual(model.vision_calls, 0)
        self.assertEqual(calls, [patch])
        self.assertTrue(injection.restored)
        self.assertIs(model.encode_images.__func__, original.__func__)


class PinnedChatAbiTests(unittest.TestCase):
    def test_pinned_chat_kwargs_are_exact(self) -> None:
        config = replay.load_anchor_config(CONFIG)
        tokenizer = object()
        image_path = Path("relative-image.jpg")
        kwargs = replay._chat_kwargs(  # type: ignore[attr-defined]
            model=PinnedChatModel(),
            tokenizer=tokenizer,
            image_path=image_path,
            config=config,
        )
        self.assertEqual(
            kwargs,
            {
                "prompt": config.prompt,
                "tokenizer": tokenizer,
                "image": str(image_path),
                "max_new_tokens": 32,
                "num_beams": 1,
                "top_p": None,
                "temperature": 0,
            },
        )
        self.assertIsInstance(kwargs["image"], str)
        self.assertNotIn("do_sample", kwargs)
        self.assertNotIn("generation_config", kwargs)
        self.assertNotIn("image_processor", kwargs)

    def test_missing_temperature_is_rejected(self) -> None:
        with self.assertRaisesRegex(replay.RuntimeABIError, "temperature"):
            replay._chat_kwargs(  # type: ignore[attr-defined]
                model=MissingTemperatureChatModel(),
                tokenizer=object(),
                image_path=Path("image.jpg"),
                config=replay.load_anchor_config(CONFIG),
            )

    def test_original_method_restored_after_exception(self) -> None:
        model = FakeModel()
        original = model.encode_images

        def connector(_value: object) -> object:
            raise RuntimeError("expected")

        injection = replay.ReplayInjection(
            model,
            FakeTensor((1, 196, 768), (0.0,)),
            connector,
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with injection:
                model.encode_images("ignored")
        self.assertTrue(injection.restored)
        self.assertIs(model.encode_images.__func__, original.__func__)


class ResultAndImportTests(unittest.TestCase):
    def test_result_schema_construction(self) -> None:
        value = replay.build_result(
            status="PASS_NATIVE_REPLAY",
            repository_commit="a" * 40,
            model_id="model",
            model_revision="b" * 40,
            runtime_package_versions={},
            cuda_identity={},
            checkpoint_snapshot_identity={},
            image_path_basename="image.jpg",
            image_sha256="c" * 64,
            runtime_resolved_model_configuration={},
            trainable_parameter_count=0,
            patch_shape=[1, 196, 768],
            connector_shape=[1, 196, 1536],
            native_call_counts={"vision_tower": 1, "connector": 1},
            replay_call_counts={"vision_tower": 0, "connector": 1},
            serialization_comparison={"max_abs_diff": 0},
            connector_comparison={"max_abs_diff": 0, "cosine_similarity": 1},
            native_token_ids=[[1]],
            replay_token_ids=[[1]],
            native_decoded_text="x",
            replay_decoded_text="x",
            restoration_status={"encode_images_restored": True},
            artifacts={},
        )
        self.assertEqual(value["schema_id"], replay.RESULT_SCHEMA_ID)
        json.loads(replay.canonical_json_bytes(value))

    def test_runtime_dependencies_absent_at_import_time(self) -> None:
        code = """
import sys
before=set(sys.modules)
import vision_encoder_alignment.vlm_compat.tinyllava_native_replay
added=set(sys.modules)-before
forbidden=('torch','transformers','safetensors','sentencepiece','PIL')
assert not any(name==root or name.startswith(root+'.') for name in added for root in forbidden), sorted(added)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_help_is_import_safe(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--validate-config-only", completed.stdout)

    def test_cli_config_validation_is_import_safe(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--validate-config-only",
                "--config",
                str(CONFIG),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("STATUS=PASS_CONFIG", completed.stdout)

    def test_cli_refuses_existing_output_without_writing_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"not-decoded-before-output-check")
            cache = root / "cache"
            cache.mkdir()
            output = root / "existing-output"
            output.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    "--config",
                    str(CONFIG),
                    "--image-path",
                    str(image),
                    "--cache-root",
                    str(cache),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--dtype",
                    "float16",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
