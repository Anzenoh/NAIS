"""Static contract tests for the Stage A Apple runbook.

These tests intentionally inspect only Markdown fenced bash blocks and Git blobs.
They do not execute the runbook or access Apple resources.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md"
PROFILE = ROOT / "configs/extraction_profiles/p1_1000_apple_fp32_v0.json"
OBSOLETE_COMMIT = "e518a51ba83f70b42611cc4386a9f55dc9e97de3"


def _section(markdown: str, heading: str, next_heading: str | None = None) -> str:
    start = markdown.index(heading)
    end = markdown.index(next_heading, start) if next_heading else len(markdown)
    return markdown[start:end]


def _bash_blocks(markdown: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)


def _binding_rows(block_name: str) -> list[tuple[str, str]]:
    text = RUNBOOK.read_text(encoding="utf-8")
    match = re.search(
        rf"done <<'{re.escape(block_name)}'\n(.*?)\n{re.escape(block_name)}",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing {block_name} heredoc"
    rows: list[tuple[str, str]] = []
    for line in match.group(1).splitlines():
        fields = line.split()
        if len(fields) != 2:
            rows.append(("<malformed>", line))
        else:
            rows.append((fields[0], fields[1]))
    return rows


def _git_blob(relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _binding_errors(block_name: str, *, required_member: str | None = None) -> list[str]:
    errors: list[str] = []
    rows = _binding_rows(block_name)
    if len(rows) != 6:
        errors.append(f"{block_name}: expected 6 rows, found {len(rows)}")
    seen: set[str] = set()
    for expected, relative in rows:
        if expected == "<malformed>":
            errors.append(f"{block_name}: malformed row {relative!r}")
            continue
        if relative in seen:
            errors.append(f"{block_name}: duplicate path {relative}")
        seen.add(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{block_name}: malformed digest for {relative}")
        candidate = ROOT / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not candidate.is_file()
        ):
            errors.append(f"{block_name}: invalid regular repository path {relative}")
            continue
        blob = _git_blob(relative)
        if blob is None:
            errors.append(f"{block_name}: Git blob missing for {relative}")
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(
                f"{block_name}: SHA-256 mismatch for {relative}: {expected} != {actual}"
            )
    if required_member is not None and required_member not in seen:
        errors.append(f"{block_name}: required member missing: {required_member}")
    return errors


def test_approved_commit_gate_is_required_lowercase_and_exact_head() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    setup = _section(text, "## 1.", "## 2.")
    checks = _section(text, "## 2.", "## 3.")
    assert ': "${VEA_APPROVED_REPO_COMMIT:?set to the independently approved exact repository commit}"' in _bash_blocks(setup)[0]
    commands = "\n".join(_bash_blocks(checks))
    assert 'test "${#VEA_APPROVED_REPO_COMMIT}" -eq 40' in commands
    assert 'case "$VEA_APPROVED_REPO_COMMIT" in *[!0-9a-f]*) exit 2;; esac' in commands
    assert 'test "$(git rev-parse HEAD)" = "$VEA_APPROVED_REPO_COMMIT"' in commands
    stage_a = "\n".join(_bash_blocks(_section(text, "## 3.", "## 4.")))
    assert "export VEA_APPROVED_REPO_COMMIT" in stage_a


def test_complete_git_cleanliness_precedes_evidence_root_creation() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    checks = "\n".join(_bash_blocks(_section(text, "## 2.", "## 3.")))
    mkdir_at = checks.index('mkdir -m 700 -- "$EVIDENCE_ROOT"')
    required = (
        'git diff --cached --name-only',
        'git diff --name-only',
        'git ls-files --others --exclude-standard',
        'git -c core.whitespace=cr-at-eol diff --check',
    )
    errors = [command for command in required if command not in checks or checks.index(command) >= mkdir_at]
    assert not errors, f"pre-evidence cleanliness failures: {errors}"


def test_executable_bash_has_no_fixed_branch_gate_or_obsolete_identity() -> None:
    executable = "\n".join(_bash_blocks(RUNBOOK.read_text(encoding="utf-8")))
    assert "git branch --show-current" not in executable
    assert "git checkout" not in executable
    assert OBSOLETE_COMMIT not in executable


def test_extraction_profile_digest_matches_current_bytes() -> None:
    executable = "\n".join(_bash_blocks(_section(RUNBOOK.read_text(encoding="utf-8"), "## 2.", "## 3.")))
    match = re.search(r"printf '%s  %s\\n' ([0-9a-f]{64}) \"\$PROFILE\" \| sha256sum --check --strict", executable)
    assert match is not None, "missing profile SHA-256 binding"
    assert match.group(1) == hashlib.sha256(PROFILE.read_bytes()).hexdigest()


def test_source_bindings_are_six_unique_regular_git_matched_files() -> None:
    errors = _binding_errors("SOURCE_BINDINGS")
    assert not errors, "\n".join(errors)


def test_stage_a_implementation_bindings_are_six_unique_git_matched_files() -> None:
    errors = _binding_errors(
        "STAGE_A_IMPLEMENTATION",
        required_member="scripts/validate_p1_apple_runtime.py",
    )
    assert not errors, "\n".join(errors)


def test_stage_a_evidence_contract_is_exact_and_stops_before_stage_b() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    stage_a = _section(text, "## 3.", "## 4.")
    commands = "\n".join(_bash_blocks(stage_a))
    expected = {
        "00-runtime-live.json",
        "01-checkpoint-inventory.json",
        "02-plan.json",
        "03-preflight.json",
        "04-readiness.txt",
        "05-preflight-manifest.txt",
    }
    observed = set(re.findall(r'\$EVIDENCE_ROOT/([0-9]{2}-[A-Za-z0-9.-]+)', commands))
    assert observed == expected
    assert "02-plan.stdout.json" not in commands
    assert "03-preflight.stdout.json" not in commands
    assert "Stop here and submit Stage A evidence for independent approval." in stage_a


def test_stage_a_plan_separates_outer_workers_from_dino_model_child() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    stage_a = _section(text, "## 3.", "## 4.")
    setup = _section(text, "## 1.", "## 2.")
    assert "MAIN_PYTHON` launches the outer" in setup
    assert "Only the model-bearing DINOv3 child uses `DINO_PYTHON`" in setup
    assert "supersedes the ambiguous R3R1 wording" in setup
    for required in (
        'plan.get("commands") != expected',
        'profile["runtime"]["main"]["interpreter"] != main_python',
        'profile["runtime"]["dinov3"]["interpreter"] != dino_python',
        '"clip_vit_b16", "siglip2_base_p16_224", "dinov3_vit_b16"',
        'f"{repository_root}/scripts/extract_features.py"',
    ):
        assert required in stage_a


def test_stage_b_is_separately_approved_derived_and_never_full_extraction() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    stage_b = _section(text, "## 4.", "## 5.")
    assert "Continue only after explicit independent approval of all Stage A evidence." in stage_b
    commands = "\n".join(_bash_blocks(stage_b))
    assert 'APPROVED_COMMIT_PREFIX=${VEA_APPROVED_REPO_COMMIT:0:7}' in commands
    assert "three_encoder_${APPROVED_COMMIT_PREFIX}" in commands
    assert all("--mode full" not in block for block in _bash_blocks(text))


def test_stage_b_reports_final_pass_only_after_evidence_manifest() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    stage_b = _section(text, "## 4.", "## 5.")
    final_evidence = _section(text, "## 5.")
    assert "final_status=complete" not in stage_b
    assert "extraction_status=complete" in stage_b
    assert "final_status=failed" in stage_b
    assert "exit_code=%s" in stage_b
    assert "15-smoke-evidence-manifest.txt" in final_evidence
    assert "final_status=PASS" in final_evidence
    assert final_evidence.index("15-smoke-evidence-manifest.txt") < final_evidence.index("final_status=PASS")


def test_stage_b_failure_trap_stays_armed_through_final_evidence_creation() -> None:
    """Exercise the real Stage B/final-evidence blocks, not just status strings."""
    text = RUNBOOK.read_text(encoding="utf-8")
    stage_b_blocks = _bash_blocks(_section(text, "## 4.", "## 5."))
    final_blocks = _bash_blocks(_section(text, "## 5."))
    assert len(stage_b_blocks) == len(final_blocks) == 1
    stage_b, final_evidence = stage_b_blocks[0], final_blocks[0]
    combined = stage_b + "\n" + final_evidence

    assert re.search(r"stage_b_failure\(\) \{\n  stage_b_exit=\$\?\n", stage_b)
    assert combined.count("trap 'stage_b_failure' ERR") == 1
    assert combined.count("trap - ERR") == 2  # recursion guard plus final success disarm
    assert combined.count("final_status=failed") == 1
    assert combined.count("exit_code=%s") == 1
    assert combined.count("report_path=%s") == 3
    assert "set +e" in stage_b
    assert "exit \"$stage_b_exit\"" in stage_b

    expected_paths = (
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/10-smoke-run.json"',
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/11-smoke-validation.json"',
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/12-run-file-sha256.txt"',
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/13-review-bundle-no-tensors.tar.gz"',
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/14-readiness.txt"',
        'STAGE_B_REPORT_PATH="$EVIDENCE_ROOT/15-smoke-evidence-manifest.txt"',
    )
    positions = [combined.index(path) for path in expected_paths]
    assert positions == sorted(positions)
    manifest_write = '> "$EVIDENCE_ROOT/15-smoke-evidence-manifest.txt"'
    final_disarm = final_evidence.rindex("trap - ERR")
    assert final_evidence.index(manifest_write) < final_disarm < final_evidence.index("final_status=PASS")
    assert stage_b.index("extraction_status=complete") > stage_b.index(
        '--mode smoke --batch-size 1 > "$EVIDENCE_ROOT/10-smoke-run.json"'
    )
    assert 'SHARD_DIRECTORY=$(find "$OUTPUT_PARENT/$RUN_ID/models" -type d -name shards -print -quit)' in stage_b
    assert 'SHARD_FIND_EXIT=$?' in stage_b
    assert 'test "$SHARD_FIND_EXIT" -eq 0' in stage_b


def test_repository_root_is_required_clean_and_checked_before_evidence_creation() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    setup = _section(text, "## 1.", "## 2.")
    checks = _section(text, "## 2.", "## 3.")
    setup_commands = "\n".join(_bash_blocks(setup))
    check_commands = "\n".join(_bash_blocks(checks))
    errors: list[str] = []
    required_gate = (
        ': "${VEA_REPO_ROOT:?set VEA_REPO_ROOT to the canonical clean Apple '
        'repository root}"'
    )
    if required_gate not in setup_commands:
        errors.append("missing required VEA_REPO_ROOT environment gate")
    assignments = re.findall(r'^REPO_ROOT=.*$', setup_commands, flags=re.MULTILINE)
    if assignments != ['REPO_ROOT="$VEA_REPO_ROOT"']:
        errors.append(f"REPO_ROOT assignment must be exact, found {assignments!r}")
    executable = "\n".join(_bash_blocks(text))
    if "REPO_ROOT=/home/chltjsdh/vea/repo" in executable:
        errors.append("fixed Apple REPO_ROOT assignment remains")
    fallbacks = (
        "VEA_REPO_ROOT:-",
        "VEA_REPO_ROOT:=",
        "VEA_REPO_ROOT-",
        "${VEA_REPO_ROOT+",
    )
    for fallback in fallbacks:
        if fallback in executable:
            errors.append(f"repository-root fallback remains: {fallback}")
    mkdir = 'mkdir -m 700 -- "$EVIDENCE_ROOT"'
    root_check = 'assert_nonlink_existing "$REPO_ROOT" directory'
    required_following_checks = (
        'test "$(git rev-parse HEAD)" = "$VEA_APPROVED_REPO_COMMIT"',
        'git diff --cached --name-only',
        'git diff --name-only',
        'git ls-files --others --exclude-standard',
        'git -c core.whitespace=cr-at-eol diff --check',
    )
    if root_check not in check_commands:
        errors.append("selected repository root is not checked as a directory")
    elif mkdir not in check_commands or check_commands.index(root_check) >= check_commands.index(mkdir):
        errors.append("selected repository root check does not precede evidence mkdir")
    selected_at = text.index('REPO_ROOT="$VEA_REPO_ROOT"')
    for command in required_following_checks:
        if command not in check_commands:
            errors.append(f"missing post-root contract command: {command}")
        elif text.index(command) <= selected_at:
            errors.append(f"post-root contract command occurs before REPO_ROOT: {command}")
    assert not errors, "\n".join(errors)
