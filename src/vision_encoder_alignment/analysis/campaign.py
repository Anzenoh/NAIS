"""Resumable sealed T004--T007 campaign orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from itertools import combinations
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from vision_encoder_alignment.metrics import (
    MultiwayLatentModel,
    RidgeMapping,
    alignment_metrics,
    bootstrap_mapping_metric,
    cka_image_bootstrap,
    closed_cycle_error,
    compose_linear_maps,
    cosine_retrieval,
    covariance_geometry,
    fit_multiway_maxvar,
    fit_orthogonal_procrustes,
    fit_position_template,
    knn_overlap,
    linear_cka,
    patch_coordinate_retrieval,
    patch_state,
    permutation_calibration,
    rsa_cosine_spearman,
    select_multiway_dimension,
    select_reduced_rank_mapping,
    select_ridge_mapping,
    spectrum_matched_row_null,
    ucentered_cka,
)
from vision_encoder_alignment.metrics.statistics import empirical_pvalue, holm_bonferroni

from .access_control import AuthorizedFeatureReader, ValidatedInput, sha256_file
from .artifacts import (
    CampaignStore,
    STATE_SEQUENCE,
    atomic_write_bytes,
    atomic_write_json,
    build_review_bundle,
    seal_payload,
    verify_review_bundle,
    verify_sealed_package,
)
from .config import AnalysisConfig, canonical_json_bytes
from .reporting import heatmap_svg, write_csv, write_reports


class CampaignError(RuntimeError):
    """Raised when a scientific or state-machine stage cannot complete."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"unable to read campaign artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"campaign artifact is not an object: {path.name}")
    return value


def _pair_key(first: str, second: str) -> str:
    return f"{first}__{second}"


def _direction_key(source: str, source_tap: str, target: str, target_tap: str) -> str:
    return f"{source}__{source_tap}__to__{target}__{target_tap}"


def _global_key(tap: str) -> str:
    return f"backbone.{tap}.global_mean"


def _patch_key(tap: str) -> str:
    return f"backbone.{tap}.patch"


def _save_mapping(path: Path, mapping: RidgeMapping, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "weights": np.asarray(mapping.weights, dtype=np.float64),
        "source_mean": np.asarray(mapping.source_mean, dtype=np.float64),
        "target_mean": np.asarray(mapping.target_mean, dtype=np.float64),
    }, str(path))
    atomic_write_json(path.with_suffix(".json"), {
        "schema_id": "vea.p1.fitted_ridge.v1",
        "relative_alpha": mapping.relative_alpha,
        "absolute_alpha": mapping.absolute_alpha,
        "training_sample_count": mapping.training_sample_count,
        "source_name": mapping.source_name,
        "target_name": mapping.target_name,
        "condition_number": mapping.condition_number,
        "rank": mapping.rank,
        "safetensors_file": path.name,
        "safetensors_sha256": sha256_file(path),
        **dict(metadata),
    })


def _load_mapping(path: Path) -> tuple[RidgeMapping, dict[str, Any]]:
    metadata = _json(path.with_suffix(".json"))
    if metadata.get("safetensors_sha256") != sha256_file(path):
        raise CampaignError("fitted mapping checksum mismatch")
    with safe_open(str(path), framework="np") as handle:
        if set(handle.keys()) != {"weights", "source_mean", "target_mean"}:
            raise CampaignError("fitted mapping tensor inventory mismatch")
        weights = np.asarray(handle.get_tensor("weights"), dtype=np.float64)
        source_mean = np.asarray(handle.get_tensor("source_mean"), dtype=np.float64)
        target_mean = np.asarray(handle.get_tensor("target_mean"), dtype=np.float64)
    return RidgeMapping(
        weights=weights,
        source_mean=source_mean,
        target_mean=target_mean,
        relative_alpha=float(metadata["relative_alpha"]),
        absolute_alpha=float(metadata["absolute_alpha"]),
        training_sample_count=int(metadata["training_sample_count"]),
        source_name=str(metadata["source_name"]),
        target_name=str(metadata["target_name"]),
        condition_number=float(metadata["condition_number"]),
        rank=metadata["rank"],
    ), metadata


def _save_multiway(path: Path, model: MultiwayLatentModel, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for index, name in enumerate(model.encoder_names):
        tensors[f"mean_{index:02d}"] = np.asarray(model.means[index], dtype=np.float64)
        tensors[f"encoder_{index:02d}"] = np.asarray(model.encoders[index], dtype=np.float64)
        tensors[f"decoder_{index:02d}"] = np.asarray(model.decoders[index], dtype=np.float64)
    save_file(tensors, str(path))
    atomic_write_json(path.with_suffix(".json"), {
        "schema_id": "vea.p1.fitted_multiway.v1",
        "encoder_names": list(model.encoder_names),
        "relative_alpha": model.relative_alpha,
        "absolute_alphas": list(model.absolute_alphas),
        "latent_dimension": model.latent_dimension,
        "training_sample_count": model.training_sample_count,
        "safetensors_file": path.name,
        "safetensors_sha256": sha256_file(path),
        **dict(metadata),
    })


def _load_multiway(path: Path) -> tuple[MultiwayLatentModel, dict[str, Any]]:
    metadata = _json(path.with_suffix(".json"))
    if metadata.get("safetensors_sha256") != sha256_file(path):
        raise CampaignError("fitted multiway checksum mismatch")
    names = tuple(metadata["encoder_names"])
    expected = {
        f"{kind}_{index:02d}"
        for index in range(len(names))
        for kind in ("mean", "encoder", "decoder")
    }
    with safe_open(str(path), framework="np") as handle:
        if set(handle.keys()) != expected:
            raise CampaignError("fitted multiway tensor inventory mismatch")
        means = tuple(np.asarray(handle.get_tensor(f"mean_{index:02d}"), dtype=np.float64) for index in range(len(names)))
        encoders = tuple(np.asarray(handle.get_tensor(f"encoder_{index:02d}"), dtype=np.float64) for index in range(len(names)))
        decoders = tuple(np.asarray(handle.get_tensor(f"decoder_{index:02d}"), dtype=np.float64) for index in range(len(names)))
    return MultiwayLatentModel(
        encoder_names=names,
        means=means,
        encoders=encoders,
        decoders=decoders,
        relative_alpha=float(metadata["relative_alpha"]),
        absolute_alphas=tuple(float(value) for value in metadata["absolute_alphas"]),
        latent_dimension=int(metadata["latent_dimension"]),
        training_sample_count=int(metadata["training_sample_count"]),
    ), metadata


class AnalysisCampaign:
    """Execute the exact frozen campaign against one validated P1 input."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        result_root: str | Path,
        validated_input: ValidatedInput,
        config: AnalysisConfig,
        commit_sha: str,
        tree_sha: str,
        remote_branch: str = "origin/codex/vea-p1-t004a-local-r2",
        platform_mode: str = "windows_local",
        lifecycle_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.result_root = Path(result_root)
        self.input = validated_input
        self.config = config
        self.commit_sha = commit_sha
        self.tree_sha = tree_sha
        self.remote_branch = remote_branch
        self.lifecycle_hook = lifecycle_hook
        if platform_mode not in {"windows_local", "apple"}:
            raise CampaignError("unsupported platform mode")
        self.platform_mode = platform_mode
        self.store = CampaignStore(
            self.result_root,
            run_id=config.document["input"]["run_id"],
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            config_sha=config.sha256,
            input_checksum_manifest_sha=validated_input.checksum_manifest_sha256,
        )
        self.reader = AuthorizedFeatureReader(
            validated_input,
            config=config,
            state_getter=lambda: self.store.current_state,
            ledger_path=self.result_root / "data_access_ledger.jsonl",
            on_first_test_read=self._on_first_test_read,
            unseal_event_getter=lambda: self.store.unseal_event_id,
        )

    def _notify_lifecycle(self, stage: str, payload: Mapping[str, Any]) -> None:
        if self.lifecycle_hook is not None:
            self.lifecycle_hook(stage, dict(payload))

    def _unseal_binding(self) -> dict[str, str]:
        freeze_path = self.result_root / "freeze" / "freeze.json"
        freeze = _json(freeze_path)
        frozen_members = freeze.get("frozen_members")
        if not isinstance(frozen_members, list):
            raise CampaignError("BLOCKED_LEAKAGE: frozen fitted-state manifest is missing")
        return {
            "analysis_commit": self.commit_sha,
            "analysis_tree": self.tree_sha,
            "analysis_config_sha256": self.config.sha256,
            "input_checksum_manifest_sha256": self.input.checksum_manifest_sha256,
            "freeze_sha256": sha256_file(freeze_path),
            "fitted_state_manifest_sha256": hashlib.sha256(canonical_json_bytes(frozen_members)).hexdigest(),
        }

    def _sync_lifecycle(self) -> None:
        state = self.store.load_state()
        completed = set(state["completed_states"])
        binding: dict[str, str] | None = None
        if "FREEZE" in completed:
            binding = self._unseal_binding()
            self._notify_lifecycle("FROZEN", binding)
        event_id = state.get("unseal_event_id")
        if state.get("test_unseal_authorization_count") == 1:
            if binding is None or not isinstance(event_id, str) or not event_id:
                raise CampaignError("BLOCKED_LEAKAGE: unseal authorization binding is incomplete")
            self._notify_lifecycle("TEST_UNSEAL_AUTHORIZED", {
                **binding,
                "unseal_event_id": event_id,
                "test_unseal_authorization_count": 1,
            })
        if state.get("test_unseal_first_read_started"):
            self._notify_lifecycle("TEST_UNSEALED", {
                "unseal_event_id": event_id,
                "test_unseal_count": 1,
            })
        if "SEAL" in completed:
            seal_result = _json(self.result_root / "seal_stage" / "seal_result.json")
            self._notify_lifecycle("SEALED", {
                "sealed_package": str((self.result_root / "sealed_package.tar.gz").resolve()),
                "sealed_package_sha256": seal_result["sealed_package_sha256"],
                "payload_manifest_sha256": seal_result["payload_manifest_sha256"],
            })
        if "VERIFY" in completed:
            verification = _json(self.result_root / "verify_stage" / "verification.json")
            self._notify_lifecycle("VERIFIED", {
                "sealed_package_sha256": verification["sealed_package_sha256"],
                "review_bundle_sha256": verification["review_bundle_sha256"],
                "ending_input_checksum_status": verification["ending_input_checksum_status"],
            })

    def _on_first_test_read(self, event_id: str) -> None:
        self.store.mark_test_read_started(event_id)
        self._sync_lifecycle()

    def run_all(self) -> dict[str, str]:
        with self.store:
            self.store.assert_resume_integrity()
            self._sync_lifecycle()
            handlers = {
                "PREFLIGHT": self._preflight,
                "BLIND_INPUT_INTEGRITY": self._blind_input_integrity,
                "REPEAT_CONTROL": self._repeat_control,
                "T004_DEV": self._t004,
                "T005_DEV": self._t005,
                "T006_DEV": self._t006,
                "EXTENSION_DEV": self._extension,
                "FREEZE": self._freeze,
                "T007_TEST_UNSEAL": self._t007,
                "REPORT": self._report,
                "SEAL": self._seal,
                "VERIFY": self._verify,
            }
            while self.store.current_state != "COMPLETE":
                state = self.store.current_state
                if state not in handlers:
                    raise CampaignError(f"unknown campaign state: {state}")
                if state == "T007_TEST_UNSEAL":
                    self._sync_lifecycle()
                    event_id = self.store.authorize_test_unseal(self._unseal_binding())
                    if self.store.unseal_event_id != event_id:
                        raise CampaignError("BLOCKED_LEAKAGE: unseal event identity mismatch")
                    self._sync_lifecycle()
                temporary = self.store.begin_stage(state)
                if temporary is None:
                    continue
                summary = handlers[state](temporary)
                self.store.complete_stage(state, temporary, summary)
                self._sync_lifecycle()
            self.store.assert_resume_integrity()
            self._sync_lifecycle()
        delivery = _json(self.result_root / "delivery_summary.json")
        review_sha = (self.result_root / "review_bundle.sha256").read_text(encoding="ascii").split()[0]
        return {
            "final_status": "PASS_P1_ANALYSIS_CAMPAIGN_SEALED",
            "commit_sha": self.commit_sha,
            "remote_branch": self.remote_branch,
            "result_root": str(self.result_root.resolve()),
            "sealed_package": str((self.result_root / delivery["sealed_package"]).resolve()),
            "sealed_package_sha256": str(delivery["sealed_package_sha256"]),
            "review_bundle": str((self.result_root / "review_bundle.zip").resolve()),
            "review_bundle_sha256": review_sha,
        }

    def _preflight(self, directory: Path) -> dict[str, Any]:
        required_environment = {
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        }
        mismatches = {key: os.environ.get(key) for key, expected in required_environment.items() if os.environ.get(key) != expected}
        if mismatches:
            raise CampaignError(f"BLOCKED_HOST_RUNTIME: deterministic environment mismatch {sorted(mismatches)}")
        if self.platform_mode == "windows_local" and os.name != "nt":
            raise CampaignError("BLOCKED_HOST_RUNTIME: Windows-local mode requires Windows")
        document = {
            "schema_id": "vea.p1.analysis_preflight.v1",
            "platform_mode": self.platform_mode,
            "operating_system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "analysis_backend": "numpy_cpu",
            "thread_environment": required_environment,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "config_sha256": self.config.sha256,
            "model_loaded": False,
            "network_used_after_checkout": False,
        }
        atomic_write_json(directory / "preflight.json", document)
        logs = self.result_root / "logs"
        logs.mkdir(exist_ok=True)
        if not (logs / "runtime.json").exists():
            atomic_write_json(logs / "runtime.json", {
                "schema_id": "vea.p1.runtime_log.v1",
                "platform_mode": self.platform_mode,
                "analysis_backend": "numpy_cpu",
                "thread_environment": required_environment,
                "durations_recorded": False,
            })
        return {"status": "PASS", "backend": "numpy_cpu"}

    def _blind_input_integrity(self, directory: Path) -> dict[str, Any]:
        input_identity = {
            "schema_id": "vea.p1.analysis_input_identity.v1",
            "run_id": self.config.document["input"]["run_id"],
            "extraction_commit": self.config.document["input"]["extraction_commit"],
            "extraction_tree": self.config.document["input"]["extraction_tree"],
            "run_manifest_sha256": self.config.document["input"]["run_manifest_sha256"],
            "image_manifest_sha256": self.config.document["input"]["image_manifest_sha256"],
            "split_sha256": self.config.document["input"]["split_sha256"],
            "input_checksum_manifest_sha256": self.input.checksum_manifest_sha256,
            "verified_file_count": self.input.verified_file_count,
            "value_rows_read": 0,
            "run_root": str(self.input.run_root),
            "evidence_root": str(self.input.evidence_root),
        }
        atomic_write_json(directory / "input_identity.json", input_identity)
        identity_path = self.result_root / "input_identity.json"
        if not identity_path.exists():
            atomic_write_json(identity_path, input_identity)
        config_path = self.result_root / "analysis_config.json"
        if not config_path.exists():
            atomic_write_bytes(config_path, self.config.canonical_bytes)
        checksum_copy = self.result_root / "input_252_file_sha256.txt"
        if not checksum_copy.exists():
            atomic_write_bytes(checksum_copy, self.input.checksum_manifest_path.read_bytes())
        return {"status": "PASS", "verified_file_count": self.input.verified_file_count, "decoded_value_rows": 0}

    def _repeat_control(self, directory: Path) -> dict[str, Any]:
        if self.platform_mode == "windows_local":
            status = "NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE"
            document = {
                "schema_id": "vea.p1.repeat_control.v1",
                "status": status,
                "reason": "accepted extractor runtimes, checkpoints, images, and CUDA inference host are unavailable in the Windows-local analysis lane",
                "tensor_values_compared": False,
                "checksum_integrity_is_repeat": False,
                "input_checksum_integrity_status": "PASS_SEPARATE_CONTROL",
            }
        else:
            raise CampaignError("BLOCKED_REPEAT_CONTROL: Apple repeat must run through the verified extraction checkout boundary")
        atomic_write_json(directory / "repeat_control.json", document)
        return {"status": status, "checksum_integrity_is_repeat": False}

    def _global(self, model: str, tap: str, split: str, purpose: str) -> np.ndarray:
        _, value = self.reader.read_array(model, _global_key(tap), split=split, purpose=purpose)
        return value

    def _t004(self, directory: Path) -> dict[str, Any]:
        encoders = self.config.encoder_keys
        block_taps = tuple(f"block_{index:02d}" for index in range(1, 13))
        selections: list[dict[str, Any]] = []
        atlas_summaries: list[dict[str, Any]] = []
        for first, second in combinations(encoders, 2):
            first_train = {tap: self._global(first, tap, "train", "t004_train_atlas") for tap in block_taps}
            second_train = {tap: self._global(second, tap, "train", "t004_train_atlas") for tap in block_taps}
            matrix = [[linear_cka(first_train[x], second_train[y]) for y in block_taps] for x in block_taps]
            ranked = sorted(
                ((-matrix[i][j], block_taps[i], block_taps[j], i, j) for i in range(12) for j in range(12)),
                key=lambda item: (item[0], item[1], item[2]),
            )
            candidate_rows: list[dict[str, Any]] = []
            for _, source_tap, target_tap, i, j in ranked[:8]:
                source_validation = self._global(first, source_tap, "validation", "t004_validation_selection")
                target_validation = self._global(second, target_tap, "validation", "t004_validation_selection")
                candidate_rows.append({
                    "source_tap": source_tap,
                    "target_tap": target_tap,
                    "train_cka": matrix[i][j],
                    "validation_cka": linear_cka(source_validation, target_validation),
                })
            selected = sorted(candidate_rows, key=lambda row: (-row["validation_cka"], row["source_tap"], row["target_tap"]))[0]
            x = first_train[selected["source_tap"]]
            y = second_train[selected["target_tap"]]
            calibration = permutation_calibration(
                x,
                y,
                permutations=int(self.config.document["resampling"]["global_permutations"]),
                seed=self.config.seed(f"t004:{first}:{second}:permutation"),
            )
            bootstrap = cka_image_bootstrap(
                x,
                y,
                replicates=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                seed=self.config.seed(f"t004:{first}:{second}:bootstrap"),
            )
            spectrum_x = spectrum_matched_row_null(x, seed=self.config.seed(f"t004:{first}:spectrum"))
            spectrum_y = spectrum_matched_row_null(y, seed=self.config.seed(f"t004:{second}:spectrum"))
            selection = {
                "pair": [first, second],
                **selected,
                "candidate_count": 8,
                "candidate_table": candidate_rows,
                "train_metrics": {
                    "linear_cka": linear_cka(x, y),
                    "ucentered_cka": ucentered_cka(x, y),
                    "rsa_cosine_spearman": rsa_cosine_spearman(x, y),
                    "spectrum_matched_null_cka": linear_cka(spectrum_x, spectrum_y),
                    "same_encoder_positive_controls": [linear_cka(x, x), linear_cka(y, y)],
                    "permutation_calibration": calibration,
                    "bootstrap": bootstrap,
                    "knn_overlap": knn_overlap(x, y, k_values=tuple(self.config.document["mapping"]["knn_values"])),
                    "source_geometry": covariance_geometry(x),
                    "target_geometry": covariance_geometry(y),
                },
            }
            selections.append(selection)
            pair = _pair_key(first, second)
            atomic_write_json(directory / f"atlas_{pair}.json", {
                "schema_id": "vea.p1.t004_atlas.v1",
                "pair": [first, second],
                "row_taps": list(block_taps),
                "column_taps": list(block_taps),
                "linear_cka": matrix,
            })
            atomic_write_bytes(directory / f"atlas_{pair}.svg", heatmap_svg(
                matrix, row_labels=block_taps, column_labels=block_taps,
                title=f"T004 train linear CKA: {first} vs {second}",
            ))
            feature_class_rows = []
            for tap in ("embedding_output", "final_norm"):
                a = self._global(first, tap, "train", "t004_feature_class_table")
                b = self._global(second, tap, "train", "t004_feature_class_table")
                feature_class_rows.append({"tap": tap, "linear_cka": linear_cka(a, b)})
            atomic_write_json(directory / f"feature_classes_{pair}.json", {
                "schema_id": "vea.p1.t004_feature_classes.v1",
                "pair": [first, second],
                "rows": feature_class_rows,
            })
            atlas_summaries.append({"pair": [first, second], "maximum_train_cka": -ranked[0][0]})
        atomic_write_json(directory / "selected_endpoints.json", {
            "schema_id": "vea.p1.t004_selection.v1",
            "selection_rule": "top-8-train-discovery_then_max-validation_then_lexical",
            "selections": selections,
        })
        return {"status": "PASS", "pair_count": 3, "selected_endpoint_count": len(selections), "atlases": atlas_summaries}

    def _t005(self, directory: Path) -> dict[str, Any]:
        selections = _json(self.result_root / "t004_dev" / "selected_endpoints.json")["selections"]
        results: list[dict[str, Any]] = []
        fit_directory = directory / "fit"
        relative_alphas = tuple(float(item) for item in self.config.document["mapping"]["ridge_relative_alphas"])
        for selected in selections:
            first, second = selected["pair"]
            endpoints = ((first, selected["source_tap"]), (second, selected["target_tap"]))
            for (source, source_tap), (target, target_tap) in (endpoints, endpoints[::-1]):
                x_train = self._global(source, source_tap, "train", "t005_mapping_fit")
                y_train = self._global(target, target_tap, "train", "t005_mapping_fit")
                x_validation = self._global(source, source_tap, "validation", "t005_validation_selection")
                y_validation = self._global(target, target_tap, "validation", "t005_validation_selection")
                direction = _direction_key(source, source_tap, target, target_tap)
                ridge, alpha_table = select_ridge_mapping(
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    relative_alphas=relative_alphas,
                    bootstrap_count=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                    seed=self.config.seed(f"t005:{direction}:alpha"),
                    source_name=f"{source}:{source_tap}",
                    target_name=f"{target}:{target_tap}",
                    condition_number_limit=float(self.config.document["numerics"]["condition_number_limit"]),
                )
                procrustes = fit_orthogonal_procrustes(
                    x_train,
                    y_train,
                    source_name=f"{source}:{source_tap}",
                    target_name=f"{target}:{target_tap}",
                )
                ridge_prediction = ridge.transform(x_validation)
                procrustes_prediction = procrustes.transform(x_validation)
                ridge_metrics = alignment_metrics(y_validation, ridge_prediction, target_train_mean=ridge.target_mean)
                procrustes_metrics = alignment_metrics(y_validation, procrustes_prediction, target_train_mean=procrustes.target_mean)
                ridge_metrics["retrieval"] = cosine_retrieval(ridge_prediction, y_validation)
                procrustes_metrics["retrieval"] = cosine_retrieval(procrustes_prediction, y_validation)
                ridge_metrics["per_image_mse"] = np.mean(
                    (np.asarray(y_validation, dtype=np.float64) - ridge_prediction) ** 2,
                    axis=1,
                    dtype=np.float64,
                ).tolist()
                ridge_metrics["bootstrap_intervals"] = bootstrap_mapping_metric(
                    y_validation,
                    ridge_prediction,
                    target_train_mean=ridge.target_mean,
                    replicates=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                    seed=self.config.seed(f"t005:{direction}:bootstrap"),
                )
                null_count = int(self.config.document["resampling"]["mapping_shuffled_refits"])
                generator = np.random.Generator(np.random.PCG64(self.config.seed(f"t005:{direction}:shuffled-refit")))
                x_centered = np.asarray(x_train, dtype=np.float64) - ridge.source_mean
                y_centered = np.asarray(y_train, dtype=np.float64) - ridge.target_mean
                system = x_centered.T @ x_centered + ridge.absolute_alpha * np.eye(x_centered.shape[1])
                projection = (np.asarray(x_validation, dtype=np.float64) - ridge.source_mean) @ np.linalg.solve(system, x_centered.T)
                null = np.empty(null_count, dtype=np.float64)
                for index in range(null_count):
                    order = generator.permutation(y_centered.shape[0])
                    prediction = projection @ y_centered[order] + ridge.target_mean
                    null[index] = float(alignment_metrics(y_validation, prediction, target_train_mean=ridge.target_mean)["macro_r2_train_target_mean"])
                observed_r2 = float(ridge_metrics["macro_r2_train_target_mean"])
                shuffled_summary = {
                    "count": null_count,
                    "mean_macro_r2": float(np.mean(null, dtype=np.float64)),
                    "standard_deviation_macro_r2": float(np.std(null, ddof=1)),
                    "empirical_pvalue_greater": empirical_pvalue(observed_r2, null, alternative="greater"),
                }
                mapping_path = fit_directory / f"{direction}.safetensors"
                _save_mapping(mapping_path, ridge, {
                    "source_model": source,
                    "source_tap": source_tap,
                    "target_model": target,
                    "target_tap": target_tap,
                })
                results.append({
                    "direction": direction,
                    "source_model": source,
                    "source_tap": source_tap,
                    "target_model": target,
                    "target_tap": target_tap,
                    "ridge_alpha_selection": alpha_table,
                    "ridge_validation": ridge_metrics,
                    "procrustes_validation": procrustes_metrics,
                    "controls": {
                        "train_target_mean": {
                            "macro_r2_train_target_mean": 0.0,
                            "nmse_train_target_mean": 1.0,
                            "cosine": "INDETERMINATE_ZERO_NORM_CONSTANT_PREDICTION",
                        },
                        "identity_descriptive": alignment_metrics(
                            y_validation,
                            x_validation,
                            target_train_mean=ridge.target_mean,
                        ),
                        "same_encoder_positive": alignment_metrics(
                            x_validation,
                            x_validation,
                            target_train_mean=np.mean(x_train, axis=0, dtype=np.float64),
                        ),
                        "repeat_ceiling": "NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE",
                    },
                    "shuffled_refit_null": shuffled_summary,
                    "fit_path": f"fit/{direction}.safetensors",
                })
        atomic_write_json(directory / "mapping_results.json", {
            "schema_id": "vea.p1.t005_results.v1",
            "directions": results,
        })
        return {"status": "PASS", "direction_count": len(results), "frozen_map_count": len(results)}

    def _position_template(self, model: str, tap: str, directory: Path) -> tuple[np.ndarray, np.ndarray]:
        _, train = self.reader.read_array(model, _patch_key(tap), split="train", purpose="t006_train_position_template")
        template = fit_position_template(train)
        path = directory / "templates" / f"{model}__{tap}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({
            "patch_mean": np.asarray(template.patch_mean, dtype=np.float64),
            "global_mean": np.asarray(template.global_mean, dtype=np.float64),
        }, str(path))
        return template.patch_mean, template.global_mean

    def _load_template(self, directory: Path, model: str, tap: str):
        from vision_encoder_alignment.metrics.patch_correspondence import PositionTemplate

        path = directory / "templates" / f"{model}__{tap}.safetensors"
        with safe_open(str(path), framework="np") as handle:
            patch_mean = np.asarray(handle.get_tensor("patch_mean"), dtype=np.float64)
            global_mean = np.asarray(handle.get_tensor("global_mean"), dtype=np.float64)
        return PositionTemplate(patch_mean, global_mean, 800)

    def _t006(self, directory: Path) -> dict[str, Any]:
        encoders = self.config.encoder_keys
        taps = tuple(self.config.document["patch"]["taps"])
        for model in encoders:
            for tap in taps:
                self._position_template(model, tap, directory)
        atlases: list[dict[str, Any]] = []
        selected_endpoints: list[dict[str, Any]] = []
        for first, second in combinations(encoders, 2):
            matrices = {state: np.empty((4, 4), dtype=np.float64) for state in ("X", "W", "D")}
            validation_cache: dict[tuple[str, str], np.ndarray] = {}
            for row, first_tap in enumerate(taps):
                x = self.reader.read_array(first, _patch_key(first_tap), split="validation", purpose="t006_patch_atlas")[1]
                first_template = self._load_template(directory, first, first_tap)
                for column, second_tap in enumerate(taps):
                    cache_key = (second, second_tap)
                    if cache_key not in validation_cache:
                        validation_cache[cache_key] = self.reader.read_array(second, _patch_key(second_tap), split="validation", purpose="t006_patch_atlas")[1]
                    y = validation_cache[cache_key]
                    second_template = self._load_template(directory, second, second_tap)
                    for state in ("X", "W", "D"):
                        xs = patch_state(x, state, template=first_template).reshape(-1, 768)
                        ys = patch_state(y, state, template=second_template).reshape(-1, 768)
                        matrices[state][row, column] = linear_cka(xs, ys)
            pair = _pair_key(first, second)
            atlas = {
                "schema_id": "vea.p1.t006_patch_atlas.v1",
                "pair": [first, second],
                "row_taps": list(taps),
                "column_taps": list(taps),
                "states": {state: matrix.tolist() for state, matrix in matrices.items()},
                "partition": "validation",
            }
            atomic_write_json(directory / f"patch_atlas_{pair}.json", atlas)
            atlases.append(atlas)
            diagonal = [(float(matrices["D"][index, index]), taps[index], index) for index in range(4)]
            _, selected_tap, selected_index = sorted(diagonal, key=lambda item: (-item[0], item[1]))[0]
            x = self.reader.read_array(first, _patch_key(selected_tap), split="validation", purpose="t006_selected_patch_endpoint")[1]
            y = self.reader.read_array(second, _patch_key(selected_tap), split="validation", purpose="t006_selected_patch_endpoint")[1]
            xt = self._load_template(directory, first, selected_tap)
            yt = self._load_template(directory, second, selected_tap)
            xd = patch_state(x, "D", template=xt)
            yd = patch_state(y, "D", template=yt)
            observed = linear_cka(xd.reshape(-1, 768), yd.reshape(-1, 768))
            generator = np.random.Generator(np.random.PCG64(self.config.seed(f"t006:{pair}:{selected_tap}:patch-permutation")))
            null_count = int(self.config.document["resampling"]["patch_permutations"])
            null = np.empty(null_count, dtype=np.float64)
            for index in range(null_count):
                permuted = np.empty_like(yd)
                for image in range(yd.shape[0]):
                    permuted[image] = yd[image, generator.permutation(yd.shape[1])]
                null[index] = linear_cka(xd.reshape(-1, 768), permuted.reshape(-1, 768))
            retrieval = patch_coordinate_retrieval(xd, yd)
            selected_endpoints.append({
                "pair": [first, second],
                "tap": selected_tap,
                "state": "D",
                "validation_linear_cka": observed,
                "patch_permutation_count": null_count,
                "patch_permutation_mean": float(np.mean(null, dtype=np.float64)),
                "patch_permutation_standard_deviation": float(np.std(null, ddof=1)),
                "patch_permutation_empirical_pvalue": empirical_pvalue(observed, null, alternative="greater"),
                "unmapped_retrieval": retrieval,
                "global_map_reuse": "NOT_APPLICABLE_PATCH_TAP_UNAVAILABLE",
                "patch_procrustes": "NOT_SELECTED_BASELINE_CONTROL",
                "patch_ridge": "NOT_SELECTED_BASELINE_CONTROL",
            })
        atomic_write_json(directory / "selected_patch_endpoints.json", {
            "schema_id": "vea.p1.t006_selection.v1",
            "selection_rule": "max_validation_D_same_tap_then_lexical",
            "endpoints": selected_endpoints,
        })
        return {"status": "PASS", "atlas_count": len(atlases), "selected_endpoint_count": len(selected_endpoints)}

    def _extension(self, directory: Path) -> dict[str, Any]:
        encoders = self.config.encoder_keys
        compatible_taps = tuple(self.config.document["patch"]["taps"])
        node_rows = []
        node_cache: dict[tuple[str, str, str], np.ndarray] = {}
        for candidate_tap in compatible_taps:
            for partition in ("train", "validation"):
                for model in encoders:
                    node_cache[(model, candidate_tap, partition)] = self._global(
                        model, candidate_tap, partition, "extension_joint_node_selection",
                    )
            train_score = float(np.mean([
                linear_cka(node_cache[(first, candidate_tap, "train")], node_cache[(second, candidate_tap, "train")])
                for first, second in combinations(encoders, 2)
            ], dtype=np.float64))
            validation_score = float(np.mean([
                linear_cka(node_cache[(first, candidate_tap, "validation")], node_cache[(second, candidate_tap, "validation")])
                for first, second in combinations(encoders, 2)
            ], dtype=np.float64))
            node_rows.append({"tap": candidate_tap, "train_encoder_balanced_cka": train_score, "validation_encoder_balanced_cka": validation_score})
        discovered = sorted(node_rows, key=lambda row: (-row["train_encoder_balanced_cka"], row["tap"]))[:2]
        tap = sorted(discovered, key=lambda row: (-row["validation_encoder_balanced_cka"], row["tap"]))[0]["tap"]
        for row in node_rows:
            row["train_discovery_candidate"] = row in discovered
            row["selected"] = row["tap"] == tap
        train = [node_cache[(model, tap, "train")] for model in encoders]
        validation = [node_cache[(model, tap, "validation")] for model in encoders]
        relative_alphas = tuple(float(item) for item in self.config.document["mapping"]["ridge_relative_alphas"])
        direct: dict[str, RidgeMapping] = {}
        direct_results: list[dict[str, Any]] = []
        fit_directory = directory / "fit"
        for source_index, source in enumerate(encoders):
            for target_index, target in enumerate(encoders):
                if source == target:
                    continue
                direction = _direction_key(source, tap, target, tap)
                mapping, alpha_table = select_ridge_mapping(
                    train[source_index], train[target_index], validation[source_index], validation[target_index],
                    relative_alphas=relative_alphas,
                    bootstrap_count=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                    seed=self.config.seed(f"extension:{direction}:alpha"),
                    source_name=f"{source}:{tap}", target_name=f"{target}:{tap}",
                    condition_number_limit=float(self.config.document["numerics"]["condition_number_limit"]),
                )
                valid_ranks = [
                    int(rank) for rank in self.config.document["mapping"]["reduced_rank_candidates"]
                    if int(rank) <= min(mapping.source_dimension, mapping.target_dimension, mapping.training_sample_count - 1)
                ]
                selected_mapping, rank_rows = select_reduced_rank_mapping(
                    mapping,
                    validation[source_index],
                    validation[target_index],
                    ranks=valid_ranks,
                    bootstrap_count=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                    seed=self.config.seed(f"extension:{direction}:rank"),
                )
                selected_rank = int(selected_mapping.rank)
                direct[direction] = selected_mapping
                path = fit_directory / f"{direction}.safetensors"
                _save_mapping(path, selected_mapping, {
                    "source_model": source, "source_tap": tap,
                    "target_model": target, "target_tap": tap,
                    "extension_family": "reduced_rank_direct",
                })
                direct_results.append({
                    "direction": direction,
                    "alpha_selection": alpha_table,
                    "rank_candidates": rank_rows,
                    "selected_rank": selected_rank,
                    "validation": alignment_metrics(
                        validation[target_index], selected_mapping.transform(validation[source_index]),
                        target_train_mean=selected_mapping.target_mean,
                    ),
                    "fit_path": f"fit/{direction}.safetensors",
                })
        cycles = []
        direct_vs_composed = []
        for start in range(3):
            forward_order = (start, (start + 1) % 3, (start + 2) % 3, start)
            reverse_order = (start, (start - 1) % 3, (start - 2) % 3, start)
            for label, order in (("forward", forward_order), ("reverse", reverse_order)):
                maps = [direct[_direction_key(encoders[a], tap, encoders[b], tap)] for a, b in zip(order[:-1], order[1:], strict=True)]
                cycles.append({"start": encoders[start], "direction": label, **closed_cycle_error(validation[start], maps)})
        for source_index, source in enumerate(encoders):
            for target_index, target in enumerate(encoders):
                if source == target:
                    continue
                intermediate_index = next(index for index in range(3) if index not in {source_index, target_index})
                intermediate = encoders[intermediate_index]
                direct_map = direct[_direction_key(source, tap, target, tap)]
                first_map = direct[_direction_key(source, tap, intermediate, tap)]
                second_map = direct[_direction_key(intermediate, tap, target, tap)]
                weights, intercept = compose_linear_maps(first_map, second_map)
                composed_prediction = validation[source_index] @ weights + intercept
                direct_vs_composed.append({
                    "source": source,
                    "intermediate": intermediate,
                    "target": target,
                    "direct_validation": alignment_metrics(
                        validation[target_index], direct_map.transform(validation[source_index]),
                        target_train_mean=direct_map.target_mean,
                    ),
                    "composed_validation": alignment_metrics(
                        validation[target_index], composed_prediction,
                        target_train_mean=second_map.target_mean,
                    ),
                })
        latent_candidates = [
            int(item) for item in self.config.document["mapping"]["multiway_latent_candidates"]
            if int(item) <= min(799, *(matrix.shape[1] for matrix in train))
        ]
        multiway, latent_table = select_multiway_dimension(
            train,
            validation,
            encoder_names=encoders,
            latent_candidates=latent_candidates,
            relative_alpha=1.0e-3,
            bootstrap_count=int(self.config.document["resampling"]["global_primary_bootstraps"]),
            seed=self.config.seed("extension:multiway:selection"),
        )
        shared_private = multiway.shared_private(validation)
        multiway_path = fit_directory / "symmetric_multiway.safetensors"
        _save_multiway(multiway_path, multiway, {
            "common_exact_node": tap,
            "extension_family": "symmetric_maxvar_gcca",
        })
        learning_curve = []
        for count in (100, 200, 400, 800):
            model = fit_multiway_maxvar(
                [value[:count] for value in train],
                encoder_names=encoders,
                latent_dimension=min(multiway.latent_dimension, count - 1),
                relative_alpha=1.0e-3,
            )
            reconstructed = model.reconstruct(validation)
            loss = float(np.mean([
                np.mean((observed - estimate) ** 2, dtype=np.float64)
                for observed, estimate in zip(validation, reconstructed, strict=True)
            ]))
            learning_curve.append({"nested_train_count": count, "validation_encoder_balanced_mse": loss})
        atomic_write_json(directory / "extension_results.json", {
            "schema_id": "vea.p1.linear_canonical_extension.v1",
            "authorized": True,
            "confirmatory": True,
            "common_exact_node": tap,
            "joint_node_selection": node_rows,
            "direct_reduced_rank": direct_results,
            "direct_vs_composed": direct_vs_composed,
            "closed_cycles": cycles,
            "multiway_selection": latent_table,
            "selected_multiway_latent_dimension": multiway.latent_dimension,
            "multiway_fit_path": "fit/symmetric_multiway.safetensors",
            "shared_private_validation": shared_private,
            "validation_learning_curve": learning_curve,
            "nonlinear_methods": {"mlp": False, "cross_attention": False},
        })
        return {"status": "PASS", "direct_map_count": 6, "cycle_count": len(cycles), "latent_dimension": multiway.latent_dimension}

    def _freeze(self, directory: Path) -> dict[str, Any]:
        ledger = self.result_root / "data_access_ledger.jsonl"
        if not ledger.is_file():
            raise CampaignError("BLOCKED_LEAKAGE: missing data-access ledger")
        entries = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        if any(entry.get("split") == "test" for entry in entries):
            raise CampaignError("BLOCKED_LEAKAGE: pre-freeze test value read recorded")
        protected_paths = [
            self.result_root / "t004_dev" / "selected_endpoints.json",
            self.result_root / "t005_dev" / "mapping_results.json",
            self.result_root / "t006_dev" / "selected_patch_endpoints.json",
            self.result_root / "extension_dev" / "extension_results.json",
        ]
        protected_paths.extend(sorted((self.result_root / "t005_dev" / "fit").glob("*")))
        protected_paths.extend(sorted((self.result_root / "t006_dev" / "templates").glob("*")))
        protected_paths.extend(sorted((self.result_root / "extension_dev" / "fit").glob("*")))
        manifest = [
            {
                "path": path.relative_to(self.result_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in protected_paths if path.is_file()
        ]
        freeze = {
            "schema_id": "vea.p1.analysis_freeze.v1",
            "analysis_commit": self.commit_sha,
            "analysis_tree": self.tree_sha,
            "analysis_config_sha256": self.config.sha256,
            "input_checksum_manifest_sha256": self.input.checksum_manifest_sha256,
            "split_sha256": self.config.document["input"]["split_sha256"],
            "data_access_ledger_sha256": sha256_file(ledger),
            "pre_freeze_test_value_read_count": 0,
            "frozen_member_count": len(manifest),
            "frozen_members": manifest,
        }
        atomic_write_json(directory / "freeze.json", freeze)
        return {"status": "PASS", "frozen_member_count": len(manifest), "pre_freeze_test_value_read_count": 0}

    def _verify_freeze(self) -> dict[str, Any]:
        freeze = _json(self.result_root / "freeze" / "freeze.json")
        for item in freeze["frozen_members"]:
            path = self.result_root.joinpath(*Path(item["path"]).parts)
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise CampaignError("BLOCKED_LEAKAGE: frozen member changed after freeze")
        if freeze["analysis_commit"] != self.commit_sha or freeze["analysis_tree"] != self.tree_sha or freeze["analysis_config_sha256"] != self.config.sha256:
            raise CampaignError("BLOCKED_LEAKAGE: frozen code or config identity mismatch")
        return freeze

    def _t007(self, directory: Path) -> dict[str, Any]:
        self._verify_freeze()
        t004 = _json(self.result_root / "t004_dev" / "selected_endpoints.json")["selections"]
        global_results = []
        global_pvalues = []
        for selected in t004:
            first, second = selected["pair"]
            x = self._global(first, selected["source_tap"], "test", "t007_frozen_global_endpoint")
            y = self._global(second, selected["target_tap"], "test", "t007_frozen_global_endpoint")
            calibration = permutation_calibration(
                x, y,
                permutations=int(self.config.document["resampling"]["global_permutations"]),
                seed=self.config.seed(f"t007:{first}:{second}:permutation"),
            )
            global_pvalues.append(float(calibration["empirical_pvalue_greater"]))
            global_results.append({
                "pair": [first, second],
                "source_tap": selected["source_tap"],
                "target_tap": selected["target_tap"],
                "linear_cka": linear_cka(x, y),
                "ucentered_cka": ucentered_cka(x, y),
                "rsa_cosine_spearman": rsa_cosine_spearman(x, y),
                "permutation_calibration": calibration,
                "bootstrap": cka_image_bootstrap(
                    x, y,
                    replicates=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                    seed=self.config.seed(f"t007:{first}:{second}:bootstrap"),
                ),
            })
        global_rejections = holm_bonferroni(global_pvalues)
        for result, rejected in zip(global_results, global_rejections, strict=True):
            result["holm_reject_null"] = rejected

        mapping_results = []
        t005 = _json(self.result_root / "t005_dev" / "mapping_results.json")["directions"]
        for item in t005:
            mapping_path = self.result_root / "t005_dev" / item["fit_path"]
            mapping, metadata = _load_mapping(mapping_path)
            source = self._global(metadata["source_model"], metadata["source_tap"], "test", "t007_frozen_mapping_endpoint")
            target = self._global(metadata["target_model"], metadata["target_tap"], "test", "t007_frozen_mapping_endpoint")
            prediction = mapping.transform(source)
            metrics = alignment_metrics(target, prediction, target_train_mean=mapping.target_mean)
            metrics["retrieval"] = cosine_retrieval(prediction, target)
            metrics["bootstrap_intervals"] = bootstrap_mapping_metric(
                target, prediction, target_train_mean=mapping.target_mean,
                replicates=int(self.config.document["resampling"]["global_primary_bootstraps"]),
                seed=self.config.seed(f"t007:{item['direction']}:bootstrap"),
            )
            mapping_results.append({"direction": item["direction"], "metrics": metrics})

        patch_results = []
        t006 = _json(self.result_root / "t006_dev" / "selected_patch_endpoints.json")["endpoints"]
        for item in t006:
            first, second = item["pair"]
            tap = item["tap"]
            x = self.reader.read_array(first, _patch_key(tap), split="test", purpose="t007_frozen_patch_endpoint")[1]
            y = self.reader.read_array(second, _patch_key(tap), split="test", purpose="t007_frozen_patch_endpoint")[1]
            xt = self._load_template(self.result_root / "t006_dev", first, tap)
            yt = self._load_template(self.result_root / "t006_dev", second, tap)
            xd = patch_state(x, "D", template=xt)
            yd = patch_state(y, "D", template=yt)
            patch_results.append({
                "pair": [first, second],
                "tap": tap,
                "state": "D",
                "linear_cka": linear_cka(xd.reshape(-1, 768), yd.reshape(-1, 768)),
                "unmapped_retrieval": patch_coordinate_retrieval(xd, yd),
                "frozen_validation_patch_permutation_pvalue": item["patch_permutation_empirical_pvalue"],
            })

        extension_dev = _json(self.result_root / "extension_dev" / "extension_results.json")
        extension_results = []
        extension_maps: dict[str, RidgeMapping] = {}
        for item in extension_dev["direct_reduced_rank"]:
            path = self.result_root / "extension_dev" / item["fit_path"]
            mapping, metadata = _load_mapping(path)
            extension_maps[item["direction"]] = mapping
            source = self._global(metadata["source_model"], metadata["source_tap"], "test", "t007_frozen_extension_endpoint")
            target = self._global(metadata["target_model"], metadata["target_tap"], "test", "t007_frozen_extension_endpoint")
            extension_results.append({
                "direction": item["direction"],
                "selected_rank": item["selected_rank"],
                "metrics": alignment_metrics(target, mapping.transform(source), target_train_mean=mapping.target_mean),
            })
        composed_results = []
        extension_tap = extension_dev["common_exact_node"]
        extension_test = {
            model: self._global(model, extension_tap, "test", "t007_frozen_extension_composed_endpoint")
            for model in self.config.encoder_keys
        }
        for row in extension_dev["direct_vs_composed"]:
            source, intermediate, target = row["source"], row["intermediate"], row["target"]
            direct = extension_maps[_direction_key(source, extension_tap, target, extension_tap)]
            first_map = extension_maps[_direction_key(source, extension_tap, intermediate, extension_tap)]
            second_map = extension_maps[_direction_key(intermediate, extension_tap, target, extension_tap)]
            weights, intercept = compose_linear_maps(first_map, second_map)
            composed = extension_test[source] @ weights + intercept
            composed_results.append({
                "source": source,
                "intermediate": intermediate,
                "target": target,
                "direct_metrics": alignment_metrics(
                    extension_test[target], direct.transform(extension_test[source]),
                    target_train_mean=direct.target_mean,
                ),
                "composed_metrics": alignment_metrics(
                    extension_test[target], composed,
                    target_train_mean=second_map.target_mean,
                ),
            })
        multiway, multiway_metadata = _load_multiway(
            self.result_root / "extension_dev" / extension_dev["multiway_fit_path"],
        )
        ordered_test = [extension_test[name] for name in multiway.encoder_names]
        multiway_test = {
            "common_exact_node": multiway_metadata["common_exact_node"],
            "latent_dimension": multiway.latent_dimension,
            "shared_private": multiway.shared_private(ordered_test),
        }
        result = {
            "schema_id": "vea.p1.t007_sealed_evaluation.v1",
            "test_unseal_event_count": 1,
            "global": global_results,
            "mapping": mapping_results,
            "patch": patch_results,
            "linear_canonical_extension": extension_results,
            "linear_canonical_direct_vs_composed": composed_results,
            "linear_canonical_multiway": multiway_test,
            "descriptive_all_1000_atlas_executed": False,
        }
        atomic_write_json(directory / "t007_results.json", result)
        return {"status": "PASS", "global_endpoint_count": len(global_results), "mapping_endpoint_count": len(mapping_results), "patch_endpoint_count": len(patch_results), "extension_endpoint_count": len(extension_results)}

    def _report(self, directory: Path) -> dict[str, Any]:
        result = _json(self.result_root / "t007_test" / "t007_results.json")
        global_confirmed = all(item["holm_reject_null"] and item["permutation_calibration"]["null_centered_effect"] > 0 for item in result["global"])
        mapping_positive = [
            item["metrics"]["bootstrap_intervals"]["macro_r2_train_target_mean"][0] > 0
            for item in result["mapping"]
        ]
        mapping_confirmed = sum(mapping_positive) >= 4
        spatial_confirmed = sum(item["frozen_validation_patch_permutation_pvalue"] <= 0.05 and item["linear_cka"] > 0 for item in result["patch"]) >= 2
        extension_direction_count = sum(
            item["metrics"]["macro_r2_train_target_mean"] > 0
            for item in result["linear_canonical_extension"]
        )
        extension_confirmed = (
            extension_direction_count >= 4
            and result["linear_canonical_multiway"]["shared_private"]["encoder_balanced_shared_variance_fraction"] > 0
        )
        summary = {
            "campaign_status": "SCIENTIFIC_EVALUATION_COMPLETE_PENDING_SEAL",
            "repeat_control_status": "NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE",
            "strongest_supporting_evidence": "maximum frozen test null-calibrated global or reduced-rank linear endpoint",
            "strongest_counterevidence": "minimum frozen test mapping R2 or spatial retrieval endpoint",
            "pairwise_global_summary": f"{sum(item['holm_reject_null'] for item in result['global'])}/{len(result['global'])} Holm-significant frozen pairs",
            "mapping_summary": f"{sum(mapping_positive)}/{len(mapping_positive)} directions have bootstrap lower R2 above zero",
            "spatial_summary": f"{sum(item['linear_cka'] > 0 for item in result['patch'])}/{len(result['patch'])} frozen D endpoints are positive",
            "canonical_extension_summary": f"{extension_direction_count}/{len(result['linear_canonical_extension'])} reduced-rank directions have positive test R2; symmetric multiway shared fraction={result['linear_canonical_multiway']['shared_private']['encoder_balanced_shared_variance_fraction']}",
            "conclusion_evidence": {
                "all_frozen_endpoints_complete": True,
                "global_confirmed": global_confirmed,
                "mapping_confirmed": mapping_confirmed,
                "spatial_confirmed": spatial_confirmed,
                "canonical_extension_confirmed": extension_confirmed,
                "strong_counterevidence": any(item["metrics"]["macro_r2_train_target_mean"] < 0 for item in result["mapping"]),
            },
        }
        paths = write_reports(
            self.result_root,
            summary=summary,
            detailed_sections={
                "T004/T007 global evidence": result["global"],
                "T005/T007 directional mapping evidence": result["mapping"],
                "T006/T007 spatial evidence": result["patch"],
                "Linear canonical extension": result["linear_canonical_extension"],
                "Linear direct versus composed": result["linear_canonical_direct_vs_composed"],
                "Linear symmetric multiway": result["linear_canonical_multiway"],
            },
        )
        tables = self.result_root / "tables"
        tables.mkdir(exist_ok=True)
        write_csv(tables / "global_test.csv", [
            {
                "pair": "--".join(item["pair"]),
                "source_tap": item["source_tap"],
                "target_tap": item["target_tap"],
                "linear_cka": item["linear_cka"],
                "null_effect": item["permutation_calibration"]["null_centered_effect"],
                "pvalue": item["permutation_calibration"]["empirical_pvalue_greater"],
                "holm_reject": item["holm_reject_null"],
            }
            for item in result["global"]
        ], fieldnames=("pair", "source_tap", "target_tap", "linear_cka", "null_effect", "pvalue", "holm_reject"))
        figures = self.result_root / "figures"
        figures.mkdir(exist_ok=True)
        for path in (self.result_root / "t004_dev").glob("*.svg"):
            shutil.copyfile(path, figures / path.name)
        fit_artifacts = self.result_root / "fit_artifacts"
        fit_artifacts.mkdir()
        shutil.copytree(self.result_root / "t005_dev" / "fit", fit_artifacts / "t005_global")
        shutil.copytree(self.result_root / "t006_dev" / "templates", fit_artifacts / "t006_position_templates")
        shutil.copytree(self.result_root / "extension_dev" / "fit", fit_artifacts / "linear_canonical_extension")
        payload = self.result_root / "payload_staging"
        if payload.exists():
            raise CampaignError("payload staging collision")
        payload.mkdir()
        for name in ("analysis_config.json", "input_identity.json", "input_252_file_sha256.txt"):
            if name == "input_identity.json":
                live = _json(self.result_root / name)
                live["run_root"] = "$VEA_INPUT_ROOT"
                live["evidence_root"] = "$VEA_EVIDENCE_ROOT"
                atomic_write_json(payload / name, live)
            else:
                shutil.copyfile(self.result_root / name, payload / name)
        for name in ("preflight", "repeat_control", "t004_dev", "t005_dev", "t006_dev", "extension_dev", "freeze", "t007_test", "fit_artifacts", "tables", "figures", "reports"):
            shutil.copytree(self.result_root / name, payload / name)
        (payload / "logs").mkdir()
        shutil.copyfile(self.result_root / "logs" / "runtime.json", payload / "logs" / "runtime.json")
        audit = payload / "audit"
        audit.mkdir()
        shutil.copyfile(self.result_root / "campaign_state.json", audit / "campaign_state_at_report.json")
        shutil.copyfile(self.result_root / "campaign_events.jsonl", audit / "campaign_events_through_report.jsonl")
        shutil.copyfile(self.result_root / "data_access_ledger.jsonl", audit / "data_access_ledger.jsonl")
        atomic_write_json(directory / "report_stage.json", {
            "schema_id": "vea.p1.report_stage.v1",
            "summary": summary,
            "report_paths": {key: path.relative_to(self.result_root).as_posix() for key, path in paths.items()},
        })
        return {"status": "PASS", "payload_staging_created": True, "report_count": len(paths)}

    def _seal(self, directory: Path) -> dict[str, Any]:
        archive = self.result_root / "sealed_package.tar.gz"
        result = seal_payload(
            self.result_root / "payload_staging",
            archive,
            code_commit=self.commit_sha,
            code_tree=self.tree_sha,
            config_sha256=self.config.sha256,
            input_checksum_manifest_sha256=self.input.checksum_manifest_sha256,
            fixed_mtime=int(self.config.document["execution"]["fixed_archive_mtime"]),
        )
        shutil.copyfile(self.result_root / "payload_staging" / "payload_manifest.json", self.result_root / "payload_manifest.json")
        shutil.copyfile(self.result_root / "payload_staging" / "seal.json", self.result_root / "seal.json")
        atomic_write_json(directory / "seal_result.json", {"schema_id": "vea.p1.seal_result.v1", **result})
        return {"status": "PASS", **result}

    def _verify(self, directory: Path) -> dict[str, Any]:
        archive = self.result_root / "sealed_package.tar.gz"
        verification = verify_sealed_package(archive)
        state = self.store.load_state()
        if state["test_unseal_event_count"] != 1:
            raise CampaignError("FAILED_PACKAGE: expected exactly one test unseal")
        ending_manifest = self.result_root / "ending_input_252_file_sha256.txt"
        lines = []
        for relative in sorted(
            path.relative_to(self.input.run_root).as_posix()
            for path in self.input.run_root.rglob("*") if path.is_file()
        ):
            lines.append(f"{sha256_file(self.input.run_root.joinpath(*Path(relative).parts))}  {relative}\n")
        atomic_write_bytes(ending_manifest, "".join(lines).encode("utf-8"))
        start_map = {}
        prefix = f"/home/chltjsdh/vea/artifacts/p1_t003/{self.config.document['input']['run_id']}/"
        for line in self.input.checksum_manifest_path.read_text(encoding="utf-8").splitlines():
            digest, absolute = line.split("  ", 1)
            start_map[absolute[len(prefix):]] = digest
        end_map = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in ending_manifest.read_text(encoding="utf-8").splitlines()}
        if start_map != end_map:
            raise CampaignError("BLOCKED_INPUT_IDENTITY: ending input checksums differ")
        delivery = {
            "schema_id": "vea.p1.delivery_summary.v1",
            "analysis_config_sha256": self.config.sha256,
            "input_checksum_manifest_sha256": self.input.checksum_manifest_sha256,
            "payload_manifest_sha256": verification["payload_manifest_sha256"],
            "test_unseal_event_count": 1,
            "sealed_package": archive.name,
            "sealed_package_sha256": verification["sealed_package_sha256"],
            "ending_input_checksum_status": "PASS_IDENTICAL_252_FILES",
            "same_setting_repeat_status": "NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE",
            "checksum_integrity_is_repeat": False,
        }
        atomic_write_json(self.result_root / "delivery_summary.json", delivery)
        atomic_write_bytes(
            self.result_root / "sealed_package.sha256",
            f"{verification['sealed_package_sha256']}  {archive.name}\n".encode("ascii"),
        )
        atomic_write_bytes(self.result_root / "WINDOWS_VERIFY.txt", (
            "Use scripts/verify_p1_analysis_package.py with --package sealed_package.tar.gz "
            "and compare sealed_package.sha256. No source tensors are contained in the package.\n"
        ).encode("utf-8"))
        review = build_review_bundle(self.result_root, self.result_root / "review_bundle.zip")
        independent_review = verify_review_bundle(self.result_root / "review_bundle.zip")
        if review != {
            key: independent_review[key]
            for key in ("review_bundle_sha256", "review_bundle_manifest_sha256", "review_bundle_covered_member_count")
        }:
            raise CampaignError("FAILED_PACKAGE: review bundle verification identity mismatch")
        atomic_write_bytes(
            self.result_root / "review_bundle.sha256",
            f"{review['review_bundle_sha256']}  review_bundle.zip\n".encode("ascii"),
        )
        final_lines = [
            "final_status=PASS_P1_ANALYSIS_CAMPAIGN_SEALED",
            f"commit_sha={self.commit_sha}",
            f"remote_branch={self.remote_branch}",
            f"result_root={self.result_root.resolve()}",
            f"sealed_package={archive.resolve()}",
            f"sealed_package_sha256={verification['sealed_package_sha256']}",
            f"review_bundle={(self.result_root / 'review_bundle.zip').resolve()}",
            f"review_bundle_sha256={review['review_bundle_sha256']}",
        ]
        atomic_write_bytes(self.result_root / "final_status.txt", ("\n".join(final_lines) + "\n").encode("utf-8"))
        atomic_write_json(directory / "verification.json", {
            "schema_id": "vea.p1.package_verification.v1",
            **verification,
            **review,
            "ending_input_checksum_status": "PASS_IDENTICAL_252_FILES",
        })
        return {"status": "PASS", **verification, "ending_input_checksum_status": "PASS_IDENTICAL_252_FILES"}


__all__ = ["AnalysisCampaign", "CampaignError"]
