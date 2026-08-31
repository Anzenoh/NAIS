from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_generation as boundary
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def to(self, **_kwargs):
        return self


class FakeTorch:
    @staticmethod
    @contextlib.contextmanager
    def inference_mode():
        yield

    @staticmethod
    def from_numpy(value: np.ndarray) -> FakeTensor:
        return FakeTensor(value)


class FakeParameter:
    device = "fake"
    dtype = "float16"


class FakeConnector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, tensor: FakeTensor) -> FakeTensor:
        self.calls += 1
        return tensor

    def parameters(self):
        return iter((FakeParameter(),))


class FakeModel:
    def __init__(self) -> None:
        self.original_calls = 0

    def encode_images(self, value):
        self.original_calls += 1
        return value


class ContractGenerator(boundary.FrozenTinyLLaVAGenerator):
    def _capture(self, _image_path: Path):
        injected = "encode_images" in self.model.__dict__
        if injected:
            self.model.encode_images(FakeTensor(np.zeros((1, 2, 3), dtype=np.float32)))
            state = SimpleNamespace(vision_calls=0, connector_calls=1)
            return state, [[7, 8]], "adapted"
        state = SimpleNamespace(vision_calls=1, connector_calls=1)
        return state, [[7, 8]], "native"


def generator() -> tuple[ContractGenerator, FakeModel]:
    model = FakeModel()
    config = SimpleNamespace(expected_patch_count=2, expected_vision_hidden_size=3, expected_connector_output_size=4)
    return ContractGenerator(model, object(), object(), FakeConnector(), FakeTorch(), config, "fake", "float16"), model


def test_native_and_adapted_paths_are_distinct_and_injection_restores(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"fake")
    instance, model = generator()
    native = instance.generate_native(image)
    original = model.encode_images
    adapted = instance.generate_adapted(image, np.ones((1, 2, 3), dtype=np.float32))
    assert native.vision_calls == 1 and native.injection_calls == 0
    assert adapted.vision_calls == 0 and adapted.injection_calls == 1 and adapted.injection_restored is True
    assert model.encode_images.__self__ is original.__self__ and model.encode_images.__func__ is original.__func__
    assert instance.native_calls == instance.adapted_calls == 1


def test_generation_metrics_reuse_t001d_token_semantics() -> None:
    metrics = t001d.generation_metrics([1, 2, 2], [1, 2, 3])
    assert metrics["exact_match"] == 0
    assert metrics["token_unigram_F1"] == 2 / 3


def test_generation_module_import_is_heavy_runtime_safe() -> None:
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.query_adapter_tinyllava_generation; "
        "assert not {'torch','transformers','huggingface_hub'}.intersection(sys.modules)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
