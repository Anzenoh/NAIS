#!/usr/bin/env python3
"""CLI for the read-only Phase 2 T001D feature-bank preflight."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as bank


FAKE_EXPECTATIONS = bank.PreflightExpectations(
    image_count=4,
    split_counts=(2, 1, 1),
    patch_shape=(2, 3),
    connector_shape=(2, 4),
)
FAKE_SENTINEL = ".vea-p2-t002c-fake-owned"


def _fresh_directory(path: Path) -> Path:
    if path.exists():
        sentinel = path / FAKE_SENTINEL
        if not sentinel.is_file() or sentinel.read_text(encoding="utf-8").strip() != bank.SCHEMA_ID:
            raise bank.BankPreflightError(
                "BLOCKED_BANK_PREFLIGHT_CONFIG",
                "refusing to replace a fake-smoke directory without the T002C ownership sentinel",
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / FAKE_SENTINEL).write_text(bank.SCHEMA_ID, encoding="utf-8")
    return path


def fake_smoke(config_path: Path, root: Path) -> dict[str, object]:
    """Exercise valid, corrupt, and misordered tiny fixtures outside the repo."""

    if not root.is_absolute():
        raise bank.BankPreflightError("BLOCKED_BANK_PREFLIGHT_CONFIG", "--fake-smoke must be an absolute path")
    config = linear.load_query_adapter_config(config_path)
    root = root.resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise bank.BankPreflightError("BLOCKED_BANK_PREFLIGHT_CONFIG", "--fake-smoke must be outside the repository")
    root = _fresh_directory(root)
    plan = bank.synthetic_t001d_plan(config, FAKE_EXPECTATIONS)

    valid_root = root / "valid"
    bank.publish_synthetic_t001d_fixture(valid_root, plan)
    result = bank.preflight_query_adapter_banks(config_path, valid_root, expectations=FAKE_EXPECTATIONS)

    corrupt_root = root / "corrupt"
    bank.publish_synthetic_t001d_fixture(corrupt_root, plan)
    corrupt_artifact = corrupt_root / plan["expected_units"][0]["artifact_relative_path"]
    with corrupt_artifact.open("ab") as stream:
        stream.write(b"corruption")
    try:
        bank.preflight_query_adapter_banks(config_path, corrupt_root, expectations=FAKE_EXPECTATIONS)
    except bank.BankPreflightError as error:
        corruption_rejection = error.classification == "BLOCKED_T001D_ARTIFACT_HASH"
    else:
        corruption_rejection = False
    if not corruption_rejection:
        raise bank.BankPreflightError("BLOCKED_T001D_ARTIFACT_HASH", "fake corruption was not rejected")

    order_root = root / "order"
    first_siglip = next(
        unit for unit in plan["expected_units"]
        if unit["encoder"] == "SIGLIP2" and unit["chunk_index"] == 0
    )
    bank.publish_synthetic_t001d_fixture(
        order_root,
        plan,
        payload_order_override={"SIGLIP2": list(reversed(first_siglip["image_ids"]))},
    )
    try:
        bank.preflight_query_adapter_banks(config_path, order_root, expectations=FAKE_EXPECTATIONS)
    except bank.BankPreflightError as error:
        order_rejection = error.classification == "BLOCKED_BANK_IMAGE_ORDER"
    else:
        order_rejection = False
    if not order_rejection:
        raise bank.BankPreflightError("BLOCKED_BANK_IMAGE_ORDER", "fake order mismatch was not rejected")

    result["fake_validation"] = {
        "multiple_chunks": True,
        "all_three_banks": True,
        "corruption_rejection": True,
        "order_mismatch_rejection": True,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--t001d-run-root", type=Path)
    mode.add_argument("--fake-smoke", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            fake_smoke(args.config, args.fake_smoke)
            if args.fake_smoke is not None
            else bank.preflight_query_adapter_banks(args.config, args.t001d_run_root)
        )
    except bank.BankPreflightError as error:
        print(json.dumps(error.as_dict(), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, linear.QueryAdapterLinearError) as error:
        blocked = bank.BankPreflightError("BLOCKED_BANK_PREFLIGHT_CONFIG", f"preflight input rejected: {error}")
        print(json.dumps(blocked.as_dict(), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
