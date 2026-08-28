#!/usr/bin/env python3
"""Run the Phase 2 direct linear-adapter Train/Validation boundary only."""

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
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_connector as connector_module
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


class FakeConnector:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights.astype(np.float32)
        self.generation_calls = 0
        self.vision_calls = 0
        self.connector_calls = 0

    def apply(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.connector_calls += (patches.shape[0] + batch_size - 1) // batch_size
        return np.ascontiguousarray(patches @ self.weights, dtype=np.float32)


def _fake_banks() -> tuple[dict[str, object], FakeConnector]:
    rng = np.random.default_rng(20260829)
    train_count, validation_count, patch_count, dim, connector_dim = 8, 4, 2, 3, 4
    total = train_count + validation_count
    target = rng.normal(size=(total, patch_count, dim)).astype(np.float32) + 0.5
    connector_weights = rng.normal(size=(dim, connector_dim)).astype(np.float32)
    connector = FakeConnector(connector_weights)
    ids = [f"fake-{index:03d}" for index in range(total)]
    banks: dict[str, object] = {
        "TARGET_CLIP": {
            "train": target[:train_count],
            "validation": target[train_count:],
            "ids": {"train": ids[:train_count], "validation": ids[train_count:]},
            "connectors": {
                "train": connector.apply(target[:train_count], batch_size=3),
                "validation": connector.apply(target[train_count:], batch_size=3),
            },
        }
    }
    for index, source in enumerate(linear.SOURCE_ORDER):
        matrix = np.eye(dim, dtype=np.float32) * (1.0 + index * 0.1)
        values = np.ascontiguousarray(target @ matrix + (index + 1) * 0.05, dtype=np.float32)
        banks[source] = {"train": values[:train_count], "validation": values[train_count:], "ids": {"train": ids[:train_count], "validation": ids[train_count:]}}
    connector.connector_calls = 0
    return banks, connector


def fake_smoke(config_path: Path, binding_path: Path, output_root: Path) -> dict[str, object]:
    config = linear.load_query_adapter_config(config_path)
    binding = runtime.load_bank_binding(binding_path)
    banks, connector = _fake_banks()
    return runtime.run_train_validation(
        config=config,
        binding=binding,
        banks=banks,
        connector=connector,
        output_root=output_root,
        connector_batch_size=3,
        adapter_config_sha256=t001d.sha256_file(config_path),
        binding_config_sha256=t001d.sha256_file(binding_path),
        fake=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--binding-config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake-smoke", type=Path, metavar="OUTPUT_ROOT")
    mode.add_argument("--t001d-run-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--connector-batch-size", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.fake_smoke is not None:
            result = fake_smoke(args.config, args.binding_config, args.fake_smoke)
        else:
            if not all((args.cache_root, args.output_root, args.device)):
                raise runtime.ValidationRuntimeError("BLOCKED_RUNTIME_INPUT", "real mode requires cache-root, output-root, and device")
            config = linear.load_query_adapter_config(args.config)
            binding = runtime.load_bank_binding(args.binding_config)
            config_sha = t001d.sha256_file(args.config)
            if binding["adapter_config_sha256"] != config_sha:
                raise runtime.ValidationRuntimeError("BLOCKED_BANK_BINDING_CONFIG", "adapter config SHA differs from binding")
            report = preflight.preflight_query_adapter_banks(args.config, args.t001d_run_root)
            runtime.verify_preflight_binding(report, binding, adapter_config_sha256=config_sha)
            banks = runtime.load_train_validation_banks(args.t001d_run_root)
            target_vlm = config["target_vlm"]
            connector = connector_module.load_frozen_tinyllava_connector(
                config_path=ROOT / target_vlm["config_path"], model_id=target_vlm["model_id"], revision=target_vlm["revision"], cache_root=args.cache_root,
                device=args.device, dtype_name=config["numeric_policy"]["target_vlm_runtime_dtype"],
            )
            result = runtime.run_train_validation(
                config=config, binding=binding, banks=banks, connector=connector, output_root=args.output_root,
                connector_batch_size=args.connector_batch_size, adapter_config_sha256=config_sha,
                binding_config_sha256=t001d.sha256_file(args.binding_config), fake=False,
            )
    except runtime.ValidationRuntimeError as error:
        print(json.dumps({"status": "BLOCKED", "classification": error.classification, "message": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, linear.QueryAdapterLinearError, connector_module.FrozenConnectorError) as error:
        print(json.dumps({"status": "BLOCKED", "classification": "BLOCKED_RUNTIME_INPUT", "message": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
