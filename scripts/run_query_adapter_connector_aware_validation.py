#!/usr/bin/env python3
"""Run the T003D connector-aware Train/Validation runtime or its offline smoke."""

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
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_validation_runtime as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_connector as connector_module


class _FakeConnectorModule(torch.nn.Module):
    """Parameter-free local fixture with the frozen connector's output shape."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float16))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return torch.cat((patches, patches), dim=-1) * self.scale


def _fake_inputs(binding: dict) -> tuple[binding_module.BoundInputs, connector_module.FrozenTinyLLaVAConnector]:
    generator = torch.Generator(device="cpu").manual_seed(20261001)
    target = {
        "train": torch.randn((1, 196, 768), generator=generator, dtype=torch.float32),
        "validation": torch.randn((2, 196, 768), generator=generator, dtype=torch.float32),
    }
    ids = {"train": ("synthetic-train-0",),
           "validation": ("synthetic-validation-0", "synthetic-validation-1")}
    connector_targets = {
        split: torch.cat((values.half(), values.half()), dim=-1).float()
        for split, values in target.items()
    }
    partitions: dict[str, dict[str, core.FeatureBatch]] = {"train": {}, "validation": {}}
    initializations: dict[str, dict[str, np.ndarray]] = {}
    for source_index, source in enumerate(core.SOURCES):
        shift = float(source_index + 1) * 0.05
        for split in ("train", "validation"):
            partitions[split][source] = core.FeatureBatch(
                target[split] + shift, target[split].clone(), connector_targets[split].clone(),
                ids[split], ids[split], ids[split], split, synthetic_fixture=True,
            )
        initializations[source] = {
            "weights": np.eye(768, dtype=np.float32),
            "source_mean": np.zeros(768, dtype=np.float32),
            "target_mean": np.zeros(768, dtype=np.float32),
            "bias": np.zeros(768, dtype=np.float32),
        }
    module = _FakeConnectorModule().eval()
    inference = connector_module.FrozenTinyLLaVAConnector(
        model=None, connector=module, torch=torch, device="cpu", dtype=torch.float16,
    )
    return binding_module.BoundInputs(
        partitions["train"], partitions["validation"], initializations,
        {"binding": binding, "policy": binding_module.DENIED_POLICY, "synthetic_fixture": True},
    ), inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake-smoke", type=Path, metavar="OUTPUT_ROOT")
    mode.add_argument("--t001d-run-root", type=Path)
    parser.add_argument("--t002d-validation-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    return parser


def _absolute_directory(path: Path | None, label: str) -> Path:
    if path is None or not path.is_absolute() or not path.is_dir():
        raise runtime.ValidationError(f"{label} must be an explicit existing absolute directory")
    return path.resolve()


def _output_available(path: Path) -> None:
    if not path.is_absolute() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise runtime.ValidationError("output root must be an explicit new or empty absolute directory")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, binding, anchor = runtime.load_accepted_inputs(ROOT)
        if args.fake_smoke is not None:
            if any(value is not None for value in (
                args.t002d_validation_root, args.cache_root, args.output_root, args.device
            )):
                raise runtime.ValidationError("fake smoke accepts only its output root")
            _output_available(args.fake_smoke)
            inputs, inference = _fake_inputs(binding)
            result = runtime.run_validation(
                config=config, binding=binding, anchor=anchor, inputs=inputs,
                inference_connector=inference, output_root=args.fake_smoke, device="cpu", fake=True,
            )
        else:
            device = runtime.canonical_real_device(args.device)
            t001d_root = _absolute_directory(args.t001d_run_root, "T001D_RUN_ROOT")
            t002d_root = _absolute_directory(args.t002d_validation_root, "T002D_VALIDATION_ROOT")
            cache_root = _absolute_directory(args.cache_root, "cache root")
            if args.output_root is None:
                raise runtime.ValidationError("real mode requires --output-root")
            _output_available(args.output_root)
            # Identity, bank and Full Linear checks complete before model load.
            inputs = binding_module.preflight_runtime(ROOT, t001d_root, t002d_root)
            if inputs.provenance.get("binding") != binding:
                raise runtime.ValidationError("T003C preflight binding identity differs")
            inference = connector_module.load_frozen_tinyllava_connector(
                config_path=ROOT / runtime.ANCHOR_PATH,
                model_id=anchor["model_id"], revision=anchor["revision"],
                cache_root=cache_root, device=device, dtype_name="float16",
            )
            result = runtime.run_validation(
                config=config, binding=binding, anchor=anchor, inputs=inputs,
                inference_connector=inference, output_root=args.output_root, device=device,
            )
    except (runtime.ValidationError, binding_module.RuntimeBindingError,
            core.ConnectorAwareLinearError, connector_module.FrozenConnectorError,
            OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "runtime_identity_sha256": result["runtime_identity_sha256"],
                      "fake": result["fake"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
