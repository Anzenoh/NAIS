#!/usr/bin/env python3
"""Run the frozen T004J Validation diagnostic, or an explicit offline fake smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as fit
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime


class FakeGenerator:
    """Deterministic boundary double; never imports a model or opens feature banks."""

    device = "cpu"

    def __init__(self) -> None:
        self.native_calls = 0
        self.adapted_calls = 0

    def generate_native(self, image_path: Path) -> object:
        from vision_encoder_alignment.vlm_compat.query_adapter_tinyllava_generation import GenerationOutput
        self.native_calls += 1
        index = int(image_path.stem[-1])
        return GenerationOutput((index, 10), f"native-{index}", 1, 1, 0, True)

    def generate_adapted(self, image_path: Path, patch: np.ndarray) -> object:
        from vision_encoder_alignment.vlm_compat.query_adapter_tinyllava_generation import GenerationOutput
        if patch.shape != (1, 196, 768) or patch.dtype != np.float32 or not patch.flags.c_contiguous:
            raise runtime.DiagnosticError("fake adapted patch differs")
        self.adapted_calls += 1
        index = int(image_path.stem[-1])
        return GenerationOutput((index, 10), f"adapted-{index}", 0, 1, 1, True)


def fake_inputs(root: Path) -> tuple[object, ...]:
    ids = ("synthetic-validation-0", "synthetic-validation-1")
    images = []
    for index, image_id in enumerate(ids):
        path = root / f"synthetic-{index}.jpg"
        path.write_bytes(b"\xff\xd8synthetic-" + bytes([index]) + b"\xff\xd9")
        images.append(runtime.ValidationImage(image_id, path, path.name, runtime.sha_file(path)))
    arrays = {source: np.full((2, 196, 768), index + 1, dtype=np.float32)
              for index, source in enumerate(binding.SOURCE_ORDER)}
    identity = np.eye(768, dtype=np.float32)
    zero = np.zeros(768, dtype=np.float32)
    ols = {source: binding.AcceptedOLS(identity, zero, zero, hashlib.sha256(source.encode()).hexdigest())
           for source in binding.SOURCE_ORDER}
    factors = {source: core.LowRankFactors(256, np.zeros((768, 256), dtype=np.float32),
                                          np.zeros((768, 256), dtype=np.float32), zero,
                                          2 * 768 * 256 + 768, 196, 768, True)
               for source in binding.SOURCE_ORDER}
    hashes = {source: {"OLS_REFERENCE": ols[source].artifact_sha256,
                       "LOW_RANK_256": hashlib.sha256((source + "256").encode()).hexdigest()}
              for source in binding.SOURCE_ORDER}
    ranks = {source: {method: {image_id: index + 1 for index, image_id in enumerate(ids)}
                      for method in ("OLS_REFERENCE", "LOW_RANK_256")}
             for source in binding.SOURCE_ORDER}
    return ids, images, arrays, ols, factors, hashes, ranks, {"synthetic_fixture": True}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake-smoke", type=Path, metavar="OUTPUT_ROOT")
    mode.add_argument("--repository-root", type=Path)
    for name in ("t001d-run-root", "t002d-validation-root", "t004f-fit-root",
                 "t004h-validation-root", "cache-root", "output-root"):
        value.add_argument(f"--{name}", type=Path)
    value.add_argument("--device")
    return value


def directory(path: Path | None, label: str) -> Path:
    if path is None:
        raise runtime.DiagnosticError(f"{label} required")
    return fit._input_root(path, label)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.fake_smoke is not None:
            if any(getattr(args, name) is not None for name in
                   ("t001d_run_root", "t002d_validation_root", "t004f_fit_root",
                    "t004h_validation_root", "cache_root", "output_root", "device")):
                raise runtime.DiagnosticError("fake smoke cannot mix real-root or device flags")
            config = runtime.load_config(ROOT)
            output = fit._output_root(args.fake_smoke, ROOT.resolve(), ())
            with tempfile.TemporaryDirectory(prefix="vea-t004j-fake-") as temporary:
                inputs = fake_inputs(Path(temporary))
                result = runtime.run_diagnostic(config=config, ids=inputs[0], images=inputs[1],
                                                arrays=inputs[2], ols=inputs[3], factors=inputs[4],
                                                hashes=inputs[5], ranks=inputs[6], provenance=inputs[7],
                                                generator=FakeGenerator(), output_root=output,
                                                device="cpu", fake=True)
        else:
            repository = directory(args.repository_root, "repository_root")
            if repository != ROOT.resolve():
                raise runtime.DiagnosticError("repository root differs from this script checkout")
            t001d = directory(args.t001d_run_root, "t001d_run_root")
            t002d = directory(args.t002d_validation_root, "t002d_validation_root")
            fit_root = directory(args.t004f_fit_root, "t004f_fit_root")
            h_root = directory(args.t004h_validation_root, "t004h_validation_root")
            cache = directory(args.cache_root, "cache_root")
            output = fit._output_root(args.output_root, repository,
                                      (t001d, t002d, fit_root, h_root, cache))
            if args.device != "cuda:0":
                raise runtime.DiagnosticError("real device must be indexed cuda:0")
            config = runtime.load_config(repository)
            ids, images, arrays, ols, factors, hashes, ranks, provenance = runtime.production_inputs(
                repository, t001d, t002d, fit_root, h_root)
            from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_generation as boundary
            generator = boundary.load_frozen_tinyllava_generator(
                config_path=repository / "configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json",
                cache_root=cache, device=args.device, dtype_name="float16")
            result = runtime.run_diagnostic(config=config, ids=ids, images=images,
                                            arrays=arrays, ols=ols, factors=factors, hashes=hashes,
                                            ranks=ranks, provenance=provenance,
                                            generator=generator, output_root=output, device=args.device)
    except (runtime.DiagnosticError, binding.LowRankRuntimeBindingError,
            core.LowRankLinearError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "generation_counts": result["generation_counts"],
                      "synthetic_test_only": result["synthetic_test_only"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
