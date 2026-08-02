"""Runtime readiness checks kept separate from structural validation."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .validation import validate_experiment_config


class ReadinessStatus(str, Enum):
    """Overall readiness states exposed by the public API."""

    READY = "ready"
    NOT_READY = "not_ready"
    BLOCKED_BY_USER_ACTION = "blocked_by_user_action"


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    """One runtime condition that prevents a ready result."""

    code: str
    field_path: str
    reason: str
    requires_user_action: bool


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Structured readiness result independent of config validity."""

    status: ReadinessStatus
    issues: tuple[ReadinessIssue, ...]


def assess_runtime_readiness(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    repository_root: str | Path,
    environ: Mapping[str, str] | None = None,
    cached_model_keys: Collection[str] | None = None,
) -> ReadinessResult:
    """Assess local inputs and user gates without network or model loading."""

    validate_experiment_config(config, registry)
    environment = os.environ if environ is None else environ
    root = Path(repository_root)
    issues: list[ReadinessIssue] = []

    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        issues.append(
            ReadinessIssue(
                code="dataset_manifest_missing",
                field_path="dataset.manifest",
                reason="The configured dataset manifest does not exist.",
                requires_user_action=True,
            )
        )

    root_env = dataset["root_env"]
    dataset_root_value = environment.get(root_env)
    if not dataset_root_value:
        issues.append(
            ReadinessIssue(
                code="dataset_root_env_unset",
                field_path="dataset.root_env",
                reason=f"Environment variable {root_env!r} is not set.",
                requires_user_action=True,
            )
        )
    elif not Path(dataset_root_value).is_dir():
        issues.append(
            ReadinessIssue(
                code="dataset_root_missing",
                field_path="dataset.root_env",
                reason="The configured dataset root is not an existing directory.",
                requires_user_action=True,
            )
        )

    registry_models = registry["models"]
    selected_model_keys = config["models"]
    known_cached_keys = None if cached_model_keys is None else set(cached_model_keys)
    for model_key in selected_model_keys:
        model = registry_models[model_key]
        model_path = f"models.{model_key}"
        if model["revision_status"] == "unresolved":
            issues.append(
                ReadinessIssue(
                    code="checkpoint_revision_unresolved",
                    field_path=f"{model_path}.revision",
                    reason="An exact checkpoint revision has not been resolved.",
                    requires_user_action=False,
                )
            )
        if model.get("access") != "public":
            issues.append(
                ReadinessIssue(
                    code="gated_model_access_unconfirmed",
                    field_path=f"{model_path}.access",
                    reason="Model access or license confirmation must be supplied by the user.",
                    requires_user_action=True,
                )
            )
        if known_cached_keys is None:
            issues.append(
                ReadinessIssue(
                    code="checkpoint_cache_unverified",
                    field_path=f"{model_path}.model_id",
                    reason="Local checkpoint cache availability was not supplied.",
                    requires_user_action=False,
                )
            )
        elif model_key not in known_cached_keys:
            issues.append(
                ReadinessIssue(
                    code="checkpoint_not_cached",
                    field_path=f"{model_path}.model_id",
                    reason="The checkpoint is not present in the declared local cache set.",
                    requires_user_action=False,
                )
            )

    if any(issue.requires_user_action for issue in issues):
        status = ReadinessStatus.BLOCKED_BY_USER_ACTION
    elif issues:
        status = ReadinessStatus.NOT_READY
    else:
        status = ReadinessStatus.READY
    return ReadinessResult(status=status, issues=tuple(issues))
