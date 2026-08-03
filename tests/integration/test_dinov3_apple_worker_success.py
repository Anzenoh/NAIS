"""Apple DINO worker success regression without Apple, CUDA, or model loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from vision_encoder_alignment.dinov3_extraction_boundary import (
    DINOV3_SELECTOR,
    MODEL_ID,
    MODEL_KEY,
    REVISION,
    load_dinov3_extractor_profile,
    validate_dinov3_extraction_request,
    write_dinov3_extraction_request,
)
from vision_encoder_alignment.extractors.dinov3 import _APPLE_EXPECTED_WORKER_POLICY


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts/dinov3_extraction_worker.py"
APPLE_PROFILE = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
APPLE_RUNTIME = ROOT / "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
SNAPSHOT_PROFILE = ROOT / "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
CANONICAL_SNAPSHOT_SHA256 = "d1326e9358930d826eaae6c18b44e384fa38b854140304951210347d7ecdac0d"


_CHILD = r"""
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

worker_path, extractor_profile, runtime_profile_path, snapshot_profile_path, runtime_root, snapshot_root, workspace = map(Path, sys.argv[1:])

real_distribution = importlib.metadata.distribution
real_find_spec = importlib.util.find_spec
def isolated_distribution(name):
    if str(name).replace('_', '-').lower() == 'torchvision':
        raise importlib.metadata.PackageNotFoundError(name)
    return real_distribution(name)
def isolated_find_spec(name, package=None):
    if str(name).split('.', 1)[0] == 'torchvision':
        return None
    return real_find_spec(name, package)
importlib.metadata.distribution = isolated_distribution
importlib.util.find_spec = isolated_find_spec

spec = importlib.util.spec_from_file_location('vea_apple_worker_success', worker_path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

import vision_encoder_alignment.snapshot_boundary as snapshot_boundary
import vision_encoder_alignment.runtime_boundary as runtime_boundary
import vision_encoder_alignment.production_runtime_contract as runtime_contract

def legacy_called(*args, **kwargs):
    raise AssertionError('legacy validator was called')
runtime_boundary.load_runtime_profile = legacy_called
runtime_boundary.validate_runtime_root = legacy_called
snapshot_boundary.validate_snapshot_root = legacy_called
snapshot_boundary.validate_apple_hf_snapshot = lambda *args, **kwargs: SimpleNamespace(
    source_sha256='d1326e9358930d826eaae6c18b44e384fa38b854140304951210347d7ecdac0d'
)
real_runtime_loader = runtime_contract.load_production_runtime_profile
runtime_contract.load_production_runtime_profile = lambda path: replace(
    real_runtime_loader(path), runtime_root=runtime_root.as_posix()
)

class FakeModel:
    training = False
    def parameters(self):
        return ()
    def state_dict(self):
        return {}

worker._load_model = lambda *args, **kwargs: (FakeModel(), {})
worker._state_sha256 = lambda model: worker.MODEL_STATE_SHA256

class FakeTensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
    def detach(self):
        return self
    def cpu(self):
        return self
    def contiguous(self):
        return self
    def numpy(self):
        return self.array

class FakeRngState:
    def clone(self):
        return self

class FakeTorch:
    class random:
        @staticmethod
        def get_rng_state():
            return FakeRngState()
    @staticmethod
    def equal(left, right):
        return True

# Run the unmodified import-light guard before installing the test-only hardware
# double.  This avoids importing the Windows host Torch runtime after the worker
# has installed its network socket guard.
real_import_light = worker._assert_import_light
def import_light_then_install_fake_torch():
    real_import_light()
    sys.modules['torch'] = FakeTorch
worker._assert_import_light = import_light_then_install_fake_torch

def fake_forward(model, normalized, selector):
    import numpy as np
    sequence = FakeTensor(np.zeros((normalized.shape[0], 201, 768), dtype=np.float32))
    pooler = FakeTensor(np.zeros((normalized.shape[0], 768), dtype=np.float32))
    return {'final_norm': sequence}, pooler
worker._one_forward = fake_forward

response = worker.run_worker(
    extractor_profile_path=extractor_profile,
    runtime_profile_path=runtime_profile_path,
    snapshot_profile_path=snapshot_profile_path,
    runtime_root=runtime_root,
    snapshot_root=snapshot_root,
    workspace=workspace,
    request_path=Path('request.json'),
    response_path=Path('response.json'),
    execution_selector=worker.APPLE_EXECUTION_SELECTOR,
)
print(json.dumps(response, sort_keys=True))
"""


def test_apple_worker_publishes_validated_success_response_without_legacy_validators(
    tmp_path: Path,
) -> None:
    """The real Apple selector reaches response validation and `_write_response`."""
    workspace = tmp_path / "workspace"
    runtime_root = tmp_path / "runtime"
    snapshot_root = tmp_path / "cache" / "models--facebook--dinov3-vitb16-pretrain-lvd1689m" / "snapshots" / REVISION
    for path in (workspace, runtime_root, snapshot_root):
        path.mkdir(parents=True)
    images = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    np.save(workspace / "input.npy", images, allow_pickle=False)
    raw_sha = hashlib.sha256(images.tobytes(order="C")).hexdigest()
    profile = load_dinov3_extractor_profile(APPLE_PROFILE, execution_selector=DINOV3_SELECTOR)
    request = validate_dinov3_extraction_request(
        {
            "expected_worker_policy": dict(_APPLE_EXPECTED_WORKER_POLICY),
            "extractor_profile_sha256": profile.sha256,
            "image_ids": ["image-1"],
            "input_dtype": "uint8",
            "input_file_sha256": hashlib.sha256((workspace / "input.npy").read_bytes()).hexdigest(),
            "input_shape": [1, 224, 224, 3],
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "operation": "extract_canonical_features",
            "raw_input_sha256": raw_sha,
            "requested_tap_names": ["final_norm"],
            "revision": REVISION,
            "run_id": "apple-worker-success",
            "schema_id": "vea.dinov3_extraction_request.v0",
            "workspace_relative_input_path": "input.npy",
            "workspace_relative_output_directory": "output",
        },
        execution_selector=DINOV3_SELECTOR,
    )
    write_dinov3_extraction_request(
        workspace / "request.json", request, workspace_root=workspace,
        execution_selector=DINOV3_SELECTOR,
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(workspace / "hf-home"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "PATH": os.environ.get("PATH", ""),
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "TEMP": str(workspace / "temp"),
        "TMP": str(workspace / "tmp"),
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(workspace / "xdg-cache"),
        "DO_NOT_TRACK": "1",
    }
    for key in ("HF_HOME", "TEMP", "TMP", "XDG_CACHE_HOME"):
        Path(environment[key]).mkdir()
    result = subprocess.run(
        [
            sys.executable, "-B", "-c", _CHILD,
            str(WORKER), str(APPLE_PROFILE), str(APPLE_RUNTIME), str(SNAPSHOT_PROFILE),
            str(runtime_root), str(snapshot_root), str(workspace),
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        shell=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    returned = json.loads(result.stdout)
    published = json.loads((workspace / "response.json").read_text(encoding="ascii"))
    assert published == returned
    assert published["status"] == "success"
    assert published["identities"]["external_snapshot_manifest_sha256"] == CANONICAL_SNAPSHOT_SHA256
