#!/usr/bin/env python3
"""Run T003F Validation-only frozen TinyLLaVA generation confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_runtime_binding as binding_module
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_validation_generation as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_generation as generation_boundary


class FakeGenerator:
    """In-memory deterministic boundary used only by the explicit fake smoke."""
    def __init__(self) -> None:
        self.device = "cpu"; self.native_calls = 0; self.adapted_calls = 0

    def generate_native(self, image_path: str | Path) -> generation_boundary.GenerationOutput:
        self.native_calls += 1
        index = int(Path(image_path).stem.split("-")[-1])
        return generation_boundary.GenerationOutput((index, 10), f"native-{index}", 1, 1, 0, True)

    def generate_adapted(self, image_path: str | Path, patch: np.ndarray) -> generation_boundary.GenerationOutput:
        self.adapted_calls += 1
        if patch.shape != (1, 196, 768) or patch.dtype != np.float32:
            raise AssertionError("synthetic adapted patch contract differs")
        index = int(Path(image_path).stem.split("-")[-1])
        return generation_boundary.GenerationOutput((index, 10), f"adapted-{index}", 0, 1, 1, True)


def _fake_inputs(config: dict) -> tuple[binding_module.BoundInputs, dict[str, runtime.LoadedLane], list[str]]:
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    count = 2
    ids = tuple(f"synthetic-validation-{index}" for index in range(count))
    target = torch.randn((count, 196, 768), generator=generator, dtype=torch.float32)
    connector = torch.cat((target.half(), target.half()), dim=-1).float()
    batches = {}
    for source_index, source in enumerate(core.SOURCES):
        batches[source] = core.FeatureBatch(target + (source_index + 1) * 0.01, target.clone(), connector.clone(),
                                            ids, ids, ids, "validation", synthetic_fixture=True)
    inputs = binding_module.BoundInputs({}, batches, {}, {
        "policy": binding_module.DENIED_POLICY, "synthetic_fixture": True,
    })
    identity = np.eye(768, dtype=np.float32)
    zero = np.zeros(768, dtype=np.float32)
    lanes = {}
    for key, spec in config["lanes"].items():
        adapter = core.AffineAdapter(identity, zero)
        lanes[key] = runtime.LoadedLane(key, spec["source"], spec["lane"], spec["artifact_condition"],
                                        spec["selected_epoch"], spec["affine_sha256"], adapter)
    return inputs, lanes, [f"synthetic-{index}.bin" for index in range(count)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake-smoke", type=Path, metavar="OUTPUT_ROOT")
    mode.add_argument("--t001d-run-root", type=Path)
    parser.add_argument("--t002d-validation-root", type=Path)
    parser.add_argument("--t003e-output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    return parser


def _directory(path: Path | None, label: str) -> Path:
    if path is None or not path.is_absolute() or not path.is_dir():
        raise runtime.ValidationGenerationError(f"{label} must be an explicit existing absolute directory")
    return path.resolve()


def _new_output(path: Path | None) -> Path:
    if path is None or not path.is_absolute() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise runtime.ValidationGenerationError("output root must be an explicit new or empty absolute directory")
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = runtime.load_config(ROOT)
        if args.fake_smoke is not None:
            if any(value is not None for value in (args.t002d_validation_root, args.t003e_output_root,
                                                    args.cache_root, args.output_root, args.device)):
                raise runtime.ValidationGenerationError("fake smoke accepts only its output root")
            output = _new_output(args.fake_smoke)
            inputs, lanes, paths = _fake_inputs(config)
            result = runtime.run_validation_generation(config=config, inputs=inputs, lanes=lanes,
                                                       generator=FakeGenerator(), image_paths=paths,
                                                       output_root=output, device="cpu", fake=True)
        else:
            device = runtime.validation_runtime.canonical_real_device(args.device)
            t001d = _directory(args.t001d_run_root, "T001D run root")
            t002d = _directory(args.t002d_validation_root, "T002D Validation root")
            t003e = _directory(args.t003e_output_root, "T003E output root")
            cache = _directory(args.cache_root, "local TinyLLaVA cache")
            output = _new_output(args.output_root)
            lanes = runtime.verify_t003e_selected_artifacts(config, t003e)
            inputs = binding_module.preflight_runtime(ROOT, t001d, t002d)
            paths = runtime.load_validation_image_paths(t001d, inputs)
            generator = generation_boundary.load_frozen_tinyllava_generator(
                config_path=ROOT / runtime.validation_runtime.ANCHOR_PATH, cache_root=cache,
                device=device, dtype_name="float16",
            )
            result = runtime.run_validation_generation(config=config, inputs=inputs, lanes=lanes, generator=generator,
                                                       image_paths=paths, output_root=output, device=device)
    except (runtime.ValidationGenerationError, binding_module.RuntimeBindingError,
            generation_boundary.FrozenGenerationError, core.ConnectorAwareLinearError,
            OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "generation_counts": result["generation_counts"],
                      "fake": args.fake_smoke is not None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
