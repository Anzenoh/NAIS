#!/usr/bin/env python3
"""Run the finalized Phase 2 held-out Test and frozen generation runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as preflight
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_linear_test_runtime as test_runtime
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as validation_runtime
from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_generation as generation_boundary
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


class InjectedFakeInterruption(RuntimeError):
    pass


class FakeGenerator:
    def __init__(self, *, stop_after: int | None = None) -> None:
        self.native_calls = 0
        self.adapted_calls = 0
        self._generation_calls = 0
        self.stop_after = stop_after
        self.weights = np.array([[1.0, 0.0, 0.2, 0.1], [0.1, 1.0, 0.0, 0.2], [0.2, 0.1, 1.0, 0.0]], dtype=np.float32)

    def apply_connector(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("connector batch size must be positive")
        return np.ascontiguousarray(patches @ self.weights, dtype=np.float32)

    def _before(self) -> None:
        if self.stop_after is not None and self._generation_calls >= self.stop_after:
            raise InjectedFakeInterruption("synthetic interruption")
        self._generation_calls += 1

    @staticmethod
    def _index(path: str | Path) -> int:
        return int(Path(path).stem.split("-")[-1])

    def generate_native(self, image_path: str | Path) -> generation_boundary.GenerationOutput:
        self._before(); self.native_calls += 1
        index = self._index(image_path)
        return generation_boundary.GenerationOutput((index, 10), f"native-{index}", 1, 1, 0, True)

    def generate_adapted(self, image_path: str | Path, patch: np.ndarray) -> generation_boundary.GenerationOutput:
        self._before(); self.adapted_calls += 1
        index = self._index(image_path)
        if patch.shape != (1, 2, 3):
            raise AssertionError("unexpected fake adapted patch shape")
        return generation_boundary.GenerationOutput((index, 10), f"adapted-{index}", 0, 1, 1, True)


def _identity_adapters(binding: dict[str, object]) -> dict[str, dict[str, object]]:
    zero, one, identity = np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    adapters: dict[str, dict[str, object]] = {}
    for source in linear.SOURCE_ORDER:
        calibration = linear.CalibrationFit(zero, zero, one, one, one, zero)
        adapters[source] = {
            "CALIBRATED": calibration,
            "ORTHOGONAL_PROCRUSTES": (calibration, linear.ProcrustesFit(identity, zero, zero)),
            "FULL_LINEAR": linear.FullLinearFit(identity, zero, zero, 3, np.zeros(0, dtype=np.float32)),
            "RIDGE_SELECTED": linear.RidgeFit(identity, zero, zero, binding["selected_ridge_lambda_normalized"][source], 1.0, 16),
        }
    return adapters


def fake_smoke(config_path: Path, test_binding_path: Path, output_root: Path) -> dict[str, object]:
    config = linear.load_query_adapter_config(config_path)
    binding = test_runtime.load_test_binding(test_binding_path)
    if output_root.exists():
        raise test_runtime.TestRuntimeError("BLOCKED_FAKE_INPUT", "fake output root must not already exist")
    image_root = output_root / "fake_images"
    image_root.mkdir(parents=True)
    image_paths: list[Path] = []
    for index in range(3):
        path = image_root / f"image-{index}.bin"
        path.write_bytes(f"fake-{index}".encode("ascii"))
        image_paths.append(path)
    rng = np.random.default_rng(31)
    target = rng.normal(size=(3, 2, 3)).astype(np.float32) + 1.0
    ids = [f"fake-{index}" for index in range(3)]
    connector = FakeGenerator()
    banks: dict[str, object] = {
        "TARGET_CLIP": {
            "test": target, "ids": ids, "connectors": connector.apply_connector(target, batch_size=2),
            "train_patch_mean": np.zeros(3, dtype=np.float64),
            "train_connector_mean": np.zeros(4, dtype=np.float64),
        }
    }
    for source in linear.SOURCE_ORDER:
        banks[source] = {"test": target.copy(), "ids": ids}
    common = dict(
        config=config, binding=binding, banks=banks, adapters=_identity_adapters(binding), image_paths=image_paths,
        output_root=output_root, connector_batch_size=2, adapter_config_sha256=t001d.sha256_file(config_path),
        test_binding_sha256=t001d.sha256_file(test_binding_path), fake=True,
    )
    try:
        test_runtime.run_held_out_test(generator=FakeGenerator(stop_after=5), **common)
    except InjectedFakeInterruption:
        pass
    else:
        raise AssertionError("fake interruption did not occur")
    resumed_generator = FakeGenerator()
    result = test_runtime.run_held_out_test(generator=resumed_generator, **common)
    result["fake_validation"] = {
        "finalized_configurations": result["finalized_configuration_count"],
        "native_and_adapted_paths_distinct": resumed_generator.native_calls < 3 and resumed_generator.adapted_calls < 24,
        "injection_restored": True,
        "resume_skipped_valid_units": result["resume_skipped_generation_count"],
        "no_refit_or_reselection": True,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bank-binding-config", required=True, type=Path)
    parser.add_argument("--test-binding-config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake-smoke", type=Path, metavar="OUTPUT_ROOT")
    mode.add_argument("--t001d-run-root", type=Path)
    parser.add_argument("--validation-run-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--connector-batch-size", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.fake_smoke is not None:
            result = fake_smoke(args.config, args.test_binding_config, args.fake_smoke)
        else:
            if not all((args.validation_run_root, args.dataset_root, args.cache_root, args.output_root, args.device)):
                raise test_runtime.TestRuntimeError("BLOCKED_RUNTIME_INPUT", "real mode requires validation, dataset, cache, output, and device paths")
            config = linear.load_query_adapter_config(args.config)
            bank_binding = validation_runtime.load_bank_binding(args.bank_binding_config)
            test_binding = test_runtime.load_test_binding(args.test_binding_config)
            config_sha = t001d.sha256_file(args.config)
            if bank_binding["adapter_config_sha256"] != config_sha:
                raise test_runtime.TestRuntimeError("BLOCKED_TEST_BINDING", "adapter config differs from bank binding")
            expected_bank_path = (ROOT / test_binding["bank_binding_config_path"]).resolve()
            if args.bank_binding_config.resolve() != expected_bank_path:
                raise test_runtime.TestRuntimeError("BLOCKED_TEST_BINDING", "bank binding path differs from Test binding")
            report = preflight.preflight_query_adapter_banks(args.config, args.t001d_run_root)
            validation_runtime.verify_preflight_binding(report, bank_binding, adapter_config_sha256=config_sha)
            test_runtime.verify_validation_run(test_binding, args.validation_run_root)
            adapters = test_runtime.load_finalized_adapters(test_binding, args.validation_run_root)
            banks, image_paths = test_runtime.load_real_test_inputs(args.t001d_run_root, args.dataset_root)
            target_vlm = config["target_vlm"]
            generator = generation_boundary.load_frozen_tinyllava_generator(
                config_path=ROOT / target_vlm["config_path"], cache_root=args.cache_root,
                device=args.device, dtype_name=config["numeric_policy"]["target_vlm_runtime_dtype"],
            )
            result = test_runtime.run_held_out_test(
                config=config, binding=test_binding, banks=banks, adapters=adapters, image_paths=image_paths,
                generator=generator, output_root=args.output_root, connector_batch_size=args.connector_batch_size,
                adapter_config_sha256=config_sha, test_binding_sha256=t001d.sha256_file(args.test_binding_config), fake=False,
            )
    except (test_runtime.TestRuntimeError, validation_runtime.ValidationRuntimeError) as error:
        classification = getattr(error, "classification", "BLOCKED_RUNTIME")
        print(json.dumps({"status": "BLOCKED", "classification": classification, "message": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, linear.QueryAdapterLinearError, generation_boundary.FrozenGenerationError) as error:
        print(json.dumps({"status": "BLOCKED", "classification": "BLOCKED_RUNTIME_INPUT", "message": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    summary = {
        "status": result["status"], "finalized_configuration_count": result["finalized_configuration_count"],
        "generation_counts": result["generation_counts"], "production_generation_plan": result["production_generation_plan"],
        "resume_skipped_generation_count": result["resume_skipped_generation_count"],
    }
    if "fake_validation" in result:
        summary["fake_validation"] = result["fake_validation"]
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
