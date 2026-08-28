from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_connector as boundary


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    def to(self, **_kwargs):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    @contextlib.contextmanager
    def inference_mode():
        yield

    @staticmethod
    def from_numpy(value: np.ndarray) -> FakeTensor:
        return FakeTensor(value)


class FakeConnector:
    def __init__(self, *, bad: bool = False) -> None:
        self.bad = bad
        self.calls = 0

    def __call__(self, tensor: FakeTensor) -> FakeTensor:
        self.calls += 1
        width = 1535 if self.bad else 1536
        return FakeTensor(np.zeros((tensor.shape[0], 196, width), dtype=np.float32))

    def parameters(self):
        return []


class FakeModel:
    training = False

    def parameters(self):
        return []


def connector(*, bad: bool = False) -> boundary.FrozenTinyLLaVAConnector:
    return boundary.FrozenTinyLLaVAConnector(
        model=FakeModel(), connector=FakeConnector(bad=bad), torch=FakeTorch(), device="fake", dtype=FakeTorch.float16
    )


def test_connector_boundary_invokes_only_frozen_connector_batches() -> None:
    value = np.ones((3, 196, 768), dtype=np.float32)
    instance = connector()
    output = instance.apply(value, batch_size=2)
    assert output.shape == (3, 196, 1536)
    assert instance.connector_calls == 2
    assert instance.vision_calls == instance.generation_calls == 0


def test_connector_boundary_rejects_bad_connector_output() -> None:
    with pytest.raises(boundary.FrozenConnectorError, match="output contract"):
        connector(bad=True).apply(np.ones((1, 196, 768), dtype=np.float32), batch_size=1)


def test_connector_module_import_is_heavy_runtime_safe() -> None:
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.query_adapter_tinyllava_connector; "
        "assert not {'torch','transformers','huggingface_hub'}.intersection(sys.modules)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr


def test_connector_module_does_not_call_vision_or_generation_apis() -> None:
    source = Path(boundary.__file__).read_text(encoding="utf-8")
    assert ".generate(" not in source
    assert "get_vision_tower(" not in source
