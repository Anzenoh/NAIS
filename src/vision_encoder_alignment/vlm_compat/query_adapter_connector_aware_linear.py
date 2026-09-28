"""Offline core for the frozen T003A connector-aware affine experiment.

This module accepts caller-supplied tensors and a frozen connector. It does not
resolve feature banks, load a model, or authorize Test or generation access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


CONFIG_PATH = Path("configs/vlm_compat/p2_query_adapter_connector_aware_linear_v1.json")
CONFIG_SHA256 = "4684a1ae43a124725df93d15c2287a30aa31ae38f6c58bd2f05b83e749f8494b"
CANONICAL_CONFIG_SHA256 = "66c7d0b277bd1f3840b2fda9d45c92e4a2b51c8c7f7a5f27a05e0f589f84c940"
CONFIG_SCHEMA_ID = "vea.p2.query_adapter_connector_aware_linear_config.v1"
EXPERIMENT_ID = "vea.p2.query_adapter_connector_aware_linear.v1"
SOURCES = ("SIGLIP2", "DINOV3")
CONDITIONS = (
    "OLS_REFERENCE", "PATCH_ONLY_CONTROL", "PATCH_CONNECTOR_0P25",
    "PATCH_CONNECTOR_1P0", "PATCH_CONNECTOR_4P0",
)
CONNECTOR_WEIGHTS = {"PATCH_ONLY_CONTROL": 0.0, "PATCH_CONNECTOR_0P25": 0.25,
                     "PATCH_CONNECTOR_1P0": 1.0, "PATCH_CONNECTOR_4P0": 4.0}
OPEN_BINDING = "OPEN_PENDING_RUNTIME_IDENTITY_BINDING"


class ConnectorAwareLinearError(ValueError):
    """A frozen design or offline training boundary was violated."""


def _fail(message: str) -> None:
    raise ConnectorAwareLinearError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConnectorAwareLinearError("config is not finite JSON") from error


def validate_config(value: Any) -> dict[str, Any]:
    """Reject every change to the accepted, scientifically load-bearing design."""

    if not isinstance(value, Mapping):
        _fail("config must be an object")
    config = dict(value)
    if config.get("schema_id") != CONFIG_SCHEMA_ID or config.get("experiment_id") != EXPERIMENT_ID:
        _fail("config schema or experiment drift")
    if hashlib.sha256(_canonical_bytes(config)).hexdigest() != CANONICAL_CONFIG_SHA256:
        _fail("frozen config drift")
    bindings = config["runtime_bindings"]
    if any(bindings[key] != OPEN_BINDING for key in (
        "source_feature_bank_paths_and_hashes", "target_feature_bank_path_and_hash",
        "accepted_full_linear_artifact_paths_and_hashes",
    )):
        _fail("runtime identities must remain unbound")
    return config


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
            _fail("accepted config SHA-256 mismatch")
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConnectorAwareLinearError("accepted config cannot be loaded") from error
    config = validate_config(value)
    for binding in config["provenance"].values():
        try:
            observed = hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
        except OSError as error:
            raise ConnectorAwareLinearError("predecessor identity is unavailable") from error
        if observed != binding["sha256"]:
            _fail("predecessor SHA-256 mismatch")
    return config


def _float32_tensor(value: Any, label: str, shape_tail: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        _fail(f"{label} must be a float32 tensor")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail or value.shape[0] == 0:
        _fail(f"{label} shape mismatch")
    if not bool(torch.isfinite(value).all()):
        _fail(f"{label} contains nonfinite values")
    return value


def _ids(value: Sequence[str], count: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count or any(
        not isinstance(item, str) or not item for item in value
    ):
        _fail(f"{label} image IDs must be nonempty strings")
    if len(set(value)) != count:
        _fail(f"{label} image IDs mismatch")
    return tuple(value)


@dataclass(frozen=True)
class FeatureBatch:
    source: torch.Tensor
    target: torch.Tensor
    target_connector: torch.Tensor
    source_image_ids: Sequence[str]
    target_image_ids: Sequence[str]
    connector_image_ids: Sequence[str]
    split: str
    synthetic_fixture: bool = False


def validate_batch(batch: FeatureBatch, config: Mapping[str, Any]) -> FeatureBatch:
    config = validate_config(config)
    if batch.split not in ("train", "validation"):
        _fail("Test or unknown split access is forbidden")
    x = _float32_tensor(batch.source, "source", (196, 768))
    y = _float32_tensor(batch.target, "target", (196, 768))
    z = _float32_tensor(batch.target_connector, "target connector", (196, 1536))
    if x.shape != y.shape or x.shape[0] != z.shape[0] or x.device != y.device or x.device != z.device:
        _fail("source/target shape or device mismatch")
    if y.requires_grad or z.requires_grad:
        _fail("frozen target tensors may not require gradients")
    source_ids = _ids(batch.source_image_ids, x.shape[0], "source")
    target_ids = _ids(batch.target_image_ids, x.shape[0], "target")
    connector_ids = _ids(batch.connector_image_ids, x.shape[0], "target connector")
    if source_ids != target_ids or target_ids != connector_ids:
        _fail("source/target/connector image order mismatch")
    if not batch.synthetic_fixture and x.shape[0] != config["split"][batch.split]["image_count"]:
        _fail("partition image count mismatch")
    return batch


def validate_train_validation_pair(train: FeatureBatch, validation: FeatureBatch,
                                   config: Mapping[str, Any]) -> None:
    validate_batch(train, config)
    validate_batch(validation, config)
    if train.split != "train" or validation.split != "validation":
        _fail("Train/Validation role mismatch")
    if set(train.source_image_ids) & set(validation.source_image_ids):
        _fail("Train/Validation image-ID overlap")


@dataclass
class AccessGuard:
    test_access_count: int = 0
    generation_calls: int = 0

    def require_clean(self) -> None:
        if self.test_access_count != 0 or self.generation_calls != 0:
            _fail("Test access and generation must remain zero")


class AffineAdapter(nn.Module):
    """One per-source affine map with exactly W and b trainable."""

    def __init__(self, weights: Any, bias: Any) -> None:
        super().__init__()
        w = torch.as_tensor(weights)
        b = torch.as_tensor(bias)
        if w.shape != (768, 768) or b.shape != (768,) or w.dtype != torch.float32 or b.dtype != torch.float32:
            _fail("accepted Full Linear W/b shape or dtype mismatch")
        if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(b).all()):
            _fail("accepted Full Linear W/b contains nonfinite values")
        self.W = nn.Parameter(w.clone())
        self.b = nn.Parameter(b.clone())
        if tuple(self.state_dict()) != ("W", "b") or sum(p.numel() for p in self.parameters()) != 590592:
            _fail("affine trainable scalar count mismatch")

    @classmethod
    def from_full_linear(cls, weights: Any, source_mean: Any, target_mean: Any) -> AffineAdapter:
        w = torch.as_tensor(weights)
        sx = torch.as_tensor(source_mean)
        ty = torch.as_tensor(target_mean)
        if w.shape != (768, 768) or sx.shape != (768,) or ty.shape != (768,):
            _fail("accepted Full Linear fit shape mismatch")
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in (w, sx, ty)):
            _fail("accepted Full Linear fit must be finite float32")
        bias = (ty.double() - sx.double() @ w.double()).float()
        return cls(w, bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = _float32_tensor(source, "source", (196, 768))
        if x.device != self.W.device:
            _fail("affine source/parameter device mismatch")
        result = x @ self.W + self.b
        if not bool(torch.isfinite(result).all()):
            _fail("affine output contains nonfinite values")
        return result


class FrozenDifferentiableConnector:
    """A separate autograd path; the existing inference connector is untouched."""

    def __init__(self, connector: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if not callable(connector):
            _fail("connector must be callable")
        if not isinstance(connector, nn.Module) and hasattr(connector, "parameters"):
            _fail("stateful connector callable must be a Torch module")
        self.connector = connector
        if isinstance(connector, nn.Module):
            connector.eval()
            for parameter in connector.parameters():
                parameter.requires_grad_(False)
            if any(value.is_floating_point() and value.dtype != torch.float16 for value in connector.state_dict().values()):
                _fail("connector floating state must use production float16")
            self._frozen_state = {name: tensor.clone() for name, tensor in connector.state_dict().items()}
        else:
            self._frozen_state = None
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if not isinstance(self.connector, nn.Module):
            return
        if self.connector.training or any(p.requires_grad for p in self.connector.parameters()):
            _fail("connector is not frozen in eval mode")
        state = self.connector.state_dict()
        if set(state) != set(self._frozen_state) or any(
            not torch.equal(value, self._frozen_state[name]) for name, value in state.items()
        ):
            _fail("frozen connector parameter or buffer changed")

    def __call__(self, candidate: torch.Tensor) -> torch.Tensor:
        self.assert_unchanged()
        value = _float32_tensor(candidate, "candidate", (196, 768))
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or not value.requires_grad:
            _fail("candidate connector path must remain differentiable")
        result = self.connector(value.to(dtype=torch.float16))
        if not isinstance(result, torch.Tensor) or result.shape != (value.shape[0], 196, 1536):
            _fail("connector output shape mismatch")
        if result.dtype != torch.float16 or not result.requires_grad:
            _fail("connector output dtype or gradient mismatch")
        output = result.to(dtype=torch.float32)
        if not bool(torch.isfinite(output).all()):
            _fail("connector output contains nonfinite values")
        self.assert_unchanged()
        return output


@dataclass(frozen=True)
class TrainNormalizers:
    patch: torch.Tensor
    connector: torch.Tensor


def compute_train_normalizers(batch: FeatureBatch, config: Mapping[str, Any]) -> TrainNormalizers:
    validate_batch(batch, config)
    if batch.split != "train":
        _fail("loss normalizers require Train only")
    patch = (batch.target - batch.target.mean(dim=(0, 1), keepdim=True)).square().mean()
    connector = (batch.target_connector - batch.target_connector.mean(dim=(0, 1), keepdim=True)).square().mean()
    for name, value in (("patch", patch), ("connector", connector)):
        if value.dtype != torch.float32 or not bool(torch.isfinite(value)) or float(value) <= 0.0:
            _fail(f"{name} Train denominator must be finite and positive")
    return TrainNormalizers(patch.clone(), connector.clone())


def condition_weight(condition: str, config: Mapping[str, Any]) -> float:
    validate_config(config)
    if condition not in CONNECTOR_WEIGHTS or config["conditions"][condition]["connector_loss_weight"] != CONNECTOR_WEIGHTS[condition]:
        _fail("unsupported or drifted gradient condition")
    return CONNECTOR_WEIGHTS[condition]


def compute_losses(
    candidate: torch.Tensor, target: torch.Tensor, target_connector: torch.Tensor,
    boundary: FrozenDifferentiableConnector, normalizers: TrainNormalizers,
    condition: str, config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    weight = condition_weight(condition, config)
    y = _float32_tensor(target, "target", (196, 768))
    z = _float32_tensor(target_connector, "target connector", (196, 1536))
    if candidate.shape != y.shape or z.shape[0] != y.shape[0] or y.requires_grad or z.requires_grad:
        _fail("loss target identity or shape mismatch")
    for denominator in (normalizers.patch, normalizers.connector):
        if denominator.dtype != torch.float32 or denominator.requires_grad or not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            _fail("loss denominator must be a fixed finite positive float32 scalar")
    projected = boundary(candidate)
    patch = (candidate - y).square().mean() / normalizers.patch
    connector = (projected - z).square().mean() / normalizers.connector
    total = patch + weight * connector
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value)) for value in (patch, connector, total)):
        _fail("loss is nonfinite or not float32")
    return {"L_patch": patch, "L_connector": connector, "L_total": total}


def make_optimizer(adapter: AffineAdapter, config: Mapping[str, Any]) -> torch.optim.AdamW:
    config = validate_config(config)
    if tuple(name for name, p in adapter.named_parameters() if p.requires_grad) != ("W", "b"):
        _fail("only W and b may be trainable")
    spec = config["gradient_training"]
    return torch.optim.AdamW(adapter.parameters(), lr=spec["learning_rate"], betas=tuple(spec["betas"]),
                             eps=spec["eps"], weight_decay=spec["weight_decay"])


def train_step(
    adapter: AffineAdapter, batch: FeatureBatch, boundary: FrozenDifferentiableConnector,
    normalizers: TrainNormalizers, condition: str, optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any], guard: AccessGuard,
) -> dict[str, float]:
    validate_batch(batch, config)
    guard.require_clean()
    if batch.split != "train":
        _fail("parameter updates require Train")
    spec = config["gradient_training"]
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        _fail("optimizer must be frozen AdamW")
    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [id(adapter.W), id(adapter.b)]:
        _fail("optimizer contains a non-adapter parameter")
    if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]) != (
        spec["learning_rate"], tuple(spec["betas"]), spec["eps"], spec["weight_decay"]
    ):
        _fail("optimizer settings differ from frozen budget")
    optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(adapter(batch.source), batch.target, batch.target_connector,
                            boundary, normalizers, condition, config)
    losses["L_total"].backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), config["gradient_training"]["gradient_clip_global_norm"])
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()):
        _fail("adapter gradient is missing or nonfinite")
    optimizer.step()
    boundary.assert_unchanged()
    guard.require_clean()
    return {key: float(value.item()) for key, value in losses.items()}


def image_batches(image_count: int, epoch: int, config: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    config = validate_config(config)
    if not isinstance(image_count, int) or image_count <= 0 or epoch not in range(1, 9):
        _fail("training epoch or image count is invalid")
    spec = config["gradient_training"]
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + epoch)
    order = torch.randperm(image_count, generator=generator).tolist()
    width = spec["batch_size_images"]
    return tuple(tuple(order[start:start + width]) for start in range(0, image_count, width))


def train_epoch(
    adapter: AffineAdapter, train: FeatureBatch, boundary: FrozenDifferentiableConnector,
    normalizers: TrainNormalizers, condition: str, optimizer: torch.optim.Optimizer,
    epoch: int, config: Mapping[str, Any], guard: AccessGuard,
) -> list[dict[str, float]]:
    validate_batch(train, config)
    if train.split != "train":
        _fail("training epoch requires Train")
    records = []
    for indices in image_batches(train.source.shape[0], epoch, config):
        ids = tuple(train.source_image_ids[index] for index in indices)
        subset = FeatureBatch(train.source[list(indices)], train.target[list(indices)],
                              train.target_connector[list(indices)], ids, ids, ids, "train", True)
        records.append(train_step(adapter, subset, boundary, normalizers, condition, optimizer, config, guard))
    return records


def _validation_rows(records: Sequence[Mapping[str, Any]], source: str, condition: str) -> list[dict[str, Any]]:
    if source not in SOURCES or condition not in CONNECTOR_WEIGHTS or len(records) != 9:
        _fail("Validation checkpoint set must contain epochs 0..8")
    by_epoch: dict[int, dict[str, Any]] = {}
    for item in records:
        row = dict(item)
        if row.get("split") != "validation" or row.get("source") != source or row.get("condition") != condition:
            _fail("Validation checkpoint identity mismatch")
        if any(key.lower().startswith("test") for key in row):
            _fail("Test selector input is forbidden")
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch not in range(9) or epoch in by_epoch:
            _fail("Validation epoch is missing, duplicate, or invalid")
        for metric in ("L_patch", "L_total"):
            value = row.get(metric)
            if type(value) not in (int, float) or not np.isfinite(value):
                _fail("Validation loss is missing or nonfinite")
        by_epoch[epoch] = row
    if set(by_epoch) != set(range(9)):
        _fail("Validation checkpoint set must contain epochs 0..8")
    return [by_epoch[index] for index in range(9)]


def select_checkpoint(records: Sequence[Mapping[str, Any]], source: str, condition: str,
                      config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    rows = _validation_rows(records, source, condition)
    metric = "L_patch" if condition == "PATCH_ONLY_CONTROL" else "L_total"
    return dict(min(rows, key=lambda row: (float(row[metric]), row["epoch"])))


def select_connector_weight(records: Sequence[Mapping[str, Any]], source: str,
                            config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    if source not in SOURCES or len(records) != 3:
        _fail("all connector-aware Validation candidates are required")
    by_weight: dict[float, dict[str, Any]] = {}
    for item in records:
        row = dict(item)
        if row.get("split") != "validation" or row.get("source") != source or any(
            key.lower().startswith("test") for key in row
        ):
            _fail("connector-aware selection requires Validation only")
        condition = row.get("condition")
        if condition not in CONDITIONS[2:]:
            _fail("connector-aware candidate condition mismatch")
        weight = condition_weight(condition, config)
        if row.get("connector_loss_weight") != weight or weight in by_weight:
            _fail("connector-aware weight is missing or duplicate")
        for metric in ("same_image_recall_at_1", "same_image_recall_at_5", "connector_cosine"):
            value = row.get(metric)
            if type(value) not in (int, float) or not np.isfinite(value):
                _fail("connector-aware selector metric is missing or nonfinite")
        by_weight[weight] = row
    if set(by_weight) != {0.25, 1.0, 4.0}:
        _fail("connector-aware grid is incomplete")
    return dict(min(by_weight.values(), key=lambda row: (
        -float(row["same_image_recall_at_1"]), -float(row["same_image_recall_at_5"]),
        -float(row["connector_cosine"]), float(row["connector_loss_weight"]),
    )))


def save_selected_adapter(directory: str | Path, adapter: AffineAdapter, *, source: str,
                          condition: str, epoch: int, guard: AccessGuard) -> dict[str, Any]:
    guard.require_clean()
    if source not in SOURCES or condition not in CONDITIONS or type(epoch) is not int or epoch not in range(9):
        _fail("selected adapter identity is invalid")
    folder = Path(directory)
    if folder.exists():
        _fail("selected adapter output already exists")
    folder.mkdir(parents=True)
    archive = folder / "affine.npz"
    with archive.open("xb") as stream:
        np.savez(stream, W=adapter.W.cpu().detach().numpy(), b=adapter.b.cpu().detach().numpy())
    metadata = {
        "schema_id": "vea.p2.query_adapter_connector_aware_affine_artifact.v1",
        "config_sha256": CONFIG_SHA256, "source": source, "condition": condition,
        "selected_epoch": epoch, "affine_npz_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "test_access_count": 0, "generation_calls": 0,
        "runtime_identity_binding": OPEN_BINDING,
    }
    (folder / "metadata.json").write_bytes(_canonical_bytes(metadata))
    return metadata


def load_selected_adapter(directory: str | Path) -> tuple[AffineAdapter, dict[str, Any]]:
    folder = Path(directory)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_id") != "vea.p2.query_adapter_connector_aware_affine_artifact.v1" or metadata.get("config_sha256") != CONFIG_SHA256:
            _fail("selected adapter metadata identity mismatch")
        if metadata.get("source") not in SOURCES or metadata.get("condition") not in CONDITIONS or metadata.get("selected_epoch") not in range(9):
            _fail("selected adapter metadata selection mismatch")
        if metadata.get("test_access_count") != 0 or metadata.get("generation_calls") != 0 or metadata.get("runtime_identity_binding") != OPEN_BINDING:
            _fail("selected adapter metadata policy mismatch")
        archive = folder / "affine.npz"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata.get("affine_npz_sha256"):
            _fail("selected adapter artifact hash mismatch")
        with np.load(archive, allow_pickle=False) as arrays:
            if set(arrays.files) != {"W", "b"}:
                _fail("selected adapter archive fields mismatch")
            adapter = AffineAdapter(arrays["W"], arrays["b"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, ConnectorAwareLinearError):
            raise
        raise ConnectorAwareLinearError("selected adapter artifact cannot be loaded") from error
    return adapter, metadata
