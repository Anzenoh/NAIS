#!/usr/bin/env python3
"""One-process entry point for the resumable sealed Phase 1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_R2_STATE_SEQUENCE = (
    "TAKEOVER_PREPARED", "BRANCH_ADOPTED", "VALIDATED", "COMMITTED",
    "DELIVERY_RECORDED", "DETACHED_VALIDATED", "INPUT_VERIFIED", "DEV",
    "FROZEN", "TEST_UNSEAL_AUTHORIZED", "TEST_UNSEALED", "SEALED", "VERIFIED",
)


def _sanitize(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")[:1000]


def _emit(block: str) -> None:
    sys.stdout.write(block)
    sys.stdout.flush()
    tty = Path("/dev/tty")
    if not sys.stdout.isatty() and tty.exists() and os.access(tty, os.W_OK):
        try:
            with tty.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(block)
        except OSError:
            pass


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {_sanitize(result.stderr)}")
    return result.stdout.strip()


class _RecoveryLifecycle:
    """Atomically mirror irreversible campaign boundaries into the R2 state ledger."""

    def __init__(self, state_path: str | Path, result_root: Path) -> None:
        self.state_path = Path(state_path).resolve(strict=True)
        self.digest_path = Path(f"{self.state_path}.sha256")
        self.result_root = result_root.resolve()

    @staticmethod
    def _canonical(value: object) -> bytes:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _load(self) -> tuple[dict[str, object], str]:
        payload = self.state_path.read_bytes()
        digest = self._sha256(payload)
        line = self.digest_path.read_text(encoding="ascii")
        if line != f"{digest}  {self.state_path.name}\n":
            raise RuntimeError("BLOCKED_RECOVERY_STATE: state sibling digest mismatch")
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or value.get("schema_id") != "vea.codex.recovery_state.v2"
            or value.get("task_id") != "VEA-P1-T004A-LOCAL-R2"
            or value.get("current_stage") not in _R2_STATE_SEQUENCE
        ):
            raise RuntimeError("BLOCKED_RECOVERY_STATE: state schema or task identity mismatch")
        configured = value.get("result_root")
        if configured is not None and Path(str(configured)).resolve() != self.result_root:
            raise RuntimeError("BLOCKED_RECOVERY_STATE: result-root identity mismatch")
        return value, digest

    def _decoded_counts(self) -> tuple[int, int]:
        ledger = self.result_root / "data_access_ledger.jsonl"
        if not ledger.exists():
            return 0, 0
        events = 0
        values = 0
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            images = row.get("image_count")
            tap = row.get("tap")
            if type(images) is not int or images < 1 or not isinstance(tap, str):
                raise RuntimeError("BLOCKED_RECOVERY_STATE: access ledger count fields are invalid")
            events += 1
            values += images * 768 * (196 if tap.endswith(".patch") else 1)
        return events, values

    def _atomic_replace(self, path: Path, payload: bytes) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        if temporary.exists():
            raise RuntimeError("BLOCKED_RECOVERY_STATE: state temporary-file collision")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write(self, state: dict[str, object], predecessor: str) -> None:
        state["sequence"] = int(state["sequence"]) + 1
        state["predecessor_state_sha256"] = predecessor
        payload = self._canonical(state)
        digest = self._sha256(payload)
        self._atomic_replace(self.state_path, payload)
        self._atomic_replace(
            self.digest_path,
            f"{digest}  {self.state_path.name}\n".encode("ascii"),
        )
        history_root = Path(str(state["validation_root"])) / "state_history"
        history = history_root / f"{int(state['sequence']):04d}_{state['current_stage']}.json"
        history_digest = Path(f"{history}.sha256")
        if history.exists() or history_digest.exists():
            raise RuntimeError("BLOCKED_RECOVERY_STATE: state-history collision")
        self._atomic_replace(history, payload)
        self._atomic_replace(history_digest, f"{digest}  {history.name}\n".encode("ascii"))

    @staticmethod
    def _assert_later_identity(state: dict[str, object], stage: str, payload: dict[str, object]) -> None:
        if stage == "FROZEN" and state.get("frozen_binding") != payload:
            raise RuntimeError("BLOCKED_RECOVERY_STATE: frozen binding changed")
        if stage in {"TEST_UNSEAL_AUTHORIZED", "TEST_UNSEALED"} and state.get("unseal_event_id") != payload.get("unseal_event_id"):
            raise RuntimeError("BLOCKED_RECOVERY_STATE: unseal event identity changed")
        if stage == "SEALED" and state.get("sealed_package_sha256") != payload.get("sealed_package_sha256"):
            raise RuntimeError("BLOCKED_RECOVERY_STATE: sealed package identity changed")
        if stage == "VERIFIED" and state.get("review_bundle_sha256") != payload.get("review_bundle_sha256"):
            raise RuntimeError("BLOCKED_RECOVERY_STATE: verified bundle identity changed")

    def __call__(self, stage: str, supplied: dict[str, object]) -> None:
        predecessors = {
            "FROZEN": "DEV",
            "TEST_UNSEAL_AUTHORIZED": "FROZEN",
            "TEST_UNSEALED": "TEST_UNSEAL_AUTHORIZED",
            "SEALED": "TEST_UNSEALED",
            "VERIFIED": "SEALED",
        }
        if stage not in predecessors:
            raise RuntimeError("BLOCKED_RECOVERY_STATE: unknown campaign lifecycle stage")
        state, predecessor_sha = self._load()
        current = str(state["current_stage"])
        if _R2_STATE_SEQUENCE.index(current) >= _R2_STATE_SEQUENCE.index(stage):
            self._assert_later_identity(state, stage, supplied)
            return
        if current != predecessors[stage]:
            raise RuntimeError(f"BLOCKED_RECOVERY_STATE: expected {predecessors[stage]} before {stage}, got {current}")
        events, values = self._decoded_counts()
        state["current_stage"] = stage
        state["result_root"] = str(self.result_root)
        state["decoded_value_read_event_count"] = events
        state["decoded_value_count"] = values
        lifecycle = dict(state.get("campaign_lifecycle", {}))
        lifecycle[stage] = supplied
        state["campaign_lifecycle"] = lifecycle
        if stage == "FROZEN":
            state["frozen_binding"] = supplied
        elif stage == "TEST_UNSEAL_AUTHORIZED":
            if supplied.get("test_unseal_authorization_count") != 1:
                raise RuntimeError("BLOCKED_RECOVERY_STATE: invalid unseal authorization count")
            state["test_unseal_authorization_count"] = 1
            state["unseal_event_id"] = supplied["unseal_event_id"]
        elif stage == "TEST_UNSEALED":
            if supplied.get("test_unseal_count") != 1 or supplied.get("unseal_event_id") != state.get("unseal_event_id"):
                raise RuntimeError("BLOCKED_RECOVERY_STATE: invalid logical test unseal")
            state["test_unseal_count"] = 1
        elif stage == "SEALED":
            state["sealed_package"] = supplied["sealed_package"]
            state["sealed_package_sha256"] = supplied["sealed_package_sha256"]
            state["payload_manifest_sha256"] = supplied["payload_manifest_sha256"]
        elif stage == "VERIFIED":
            if supplied.get("ending_input_checksum_status") != "PASS_IDENTICAL_252_FILES":
                raise RuntimeError("BLOCKED_RECOVERY_STATE: ending input checksum status failed")
            state["review_bundle_sha256"] = supplied["review_bundle_sha256"]
            state["ending_input_checksum_status"] = supplied["ending_input_checksum_status"]
        self._write(state, predecessor_sha)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--result-parent", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis" / "p1_analysis_campaign_v1.json"))
    parser.add_argument("--recovery-state", required=True)
    parser.add_argument("--platform-mode", choices=("windows_local", "apple"), required=True)
    parser.add_argument("--remote-branch", default="origin/codex/vea-p1-t004a-local-r2")
    parser.add_argument("--resume-or-start", action="store_true", required=True)
    parser.add_argument("--run-all", action="store_true", required=True)
    args = parser.parse_args(argv)
    repository = Path(args.repository_root).resolve()
    result_root: Path | None = None
    failed_gate = "PREFLIGHT"
    try:
        required = {
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        }
        if any(os.environ.get(key) != value for key, value in required.items()):
            raise RuntimeError("BLOCKED_HOST_RUNTIME: deterministic thread environment is not exact")
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) or not re.fullmatch(r"[0-9a-f]{40}", args.expected_tree):
            raise RuntimeError("BLOCKED_SOURCE_IDENTITY: expected commit/tree must be full 40-hex identities")
        if args.remote_branch != "origin/codex/vea-p1-t004a-local-r2":
            raise RuntimeError("BLOCKED_SOURCE_IDENTITY: remote branch identity mismatch")
        head = _git(repository, "rev-parse", "HEAD^{commit}")
        tree = _git(repository, "rev-parse", "HEAD^{tree}")
        if head != args.expected_commit or tree != args.expected_tree:
            raise RuntimeError("BLOCKED_SOURCE_IDENTITY: checkout commit or tree mismatch")
        if _git(repository, "diff", "--name-only") or _git(repository, "diff", "--cached", "--name-only"):
            raise RuntimeError("BLOCKED_SOURCE_IDENTITY: analysis checkout has content changes")
        sys.path.insert(0, str(repository / "src"))
        from vision_encoder_alignment.analysis.access_control import validate_analysis_input
        from vision_encoder_alignment.analysis.artifacts import deterministic_result_name
        from vision_encoder_alignment.analysis.campaign import AnalysisCampaign
        from vision_encoder_alignment.analysis.config import load_analysis_config

        config = load_analysis_config(args.config)
        failed_gate = "BLIND_INPUT_INTEGRITY"
        validated = validate_analysis_input(
            run_root=args.run_root,
            evidence_root=args.evidence_root,
            repository_root=repository,
            config=config,
        )
        result_parent = Path(args.result_parent).resolve()
        result_parent.mkdir(parents=True, exist_ok=True)
        result_root = result_parent / deterministic_result_name(
            config.document["input"]["run_id"], head, config.sha256,
        )
        lifecycle = _RecoveryLifecycle(args.recovery_state, result_root)
        failed_gate = "CAMPAIGN"
        campaign = AnalysisCampaign(
            repository_root=repository,
            result_root=result_root,
            validated_input=validated,
            config=config,
            commit_sha=head,
            tree_sha=tree,
            remote_branch=args.remote_branch,
            platform_mode=args.platform_mode,
            lifecycle_hook=lifecycle,
        )
        result = campaign.run_all()
        block = "\n".join([
            f"final_status={result['final_status']}",
            f"commit_sha={result['commit_sha']}",
            f"remote_branch={result['remote_branch']}",
            f"result_root={result['result_root']}",
            f"sealed_package={result['sealed_package']}",
            f"sealed_package_sha256={result['sealed_package_sha256']}",
            f"review_bundle={result['review_bundle']}",
            f"review_bundle_sha256={result['review_bundle_sha256']}",
        ]) + "\n"
        _emit(block)
        return 0
    except Exception as exc:
        message = _sanitize(exc)
        failure_class = message.split(":", 1)[0] if message.startswith(("BLOCKED_", "FAILED_")) else "FAILED_STAGE_EXECUTION"
        evidence = result_root if result_root is not None else Path(args.result_parent).resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        resume = result_root is not None and (result_root / "campaign_state.json").is_file()
        block = "\n".join([
            f"final_status={failure_class}",
            f"failed_gate={failed_gate}",
            f"first_root_cause={message}",
            f"evidence_path={evidence}",
            f"resume_possible={'true' if resume else 'false'}",
        ]) + "\n"
        try:
            (evidence / "final_status.txt").write_text(block, encoding="utf-8", newline="\n")
        except OSError:
            pass
        _emit(block)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
