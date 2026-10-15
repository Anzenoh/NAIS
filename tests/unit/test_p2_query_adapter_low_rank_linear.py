from __future__ import annotations

import copy
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as low_rank


CONFIG_PATH = Path("configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json")


def _config() -> dict[str, object]:
    return low_rank.load_low_rank_config(CONFIG_PATH)


def _fixture(
    source: np.ndarray, target: np.ndarray, *, sizes: tuple[int, ...] | None = None
) -> tuple[low_rank.SyntheticTestShape, object]:
    images, patches, dim = source.shape
    assert target.shape == source.shape
    if sizes is None:
        sizes = (images,)
    assert sum(sizes) == images
    ids = tuple(f"synthetic-{index}" for index in range(images))

    def factory() -> object:
        offset = 0
        for size in sizes:
            next_offset = offset + size
            yield low_rank.TrainChunk(
                source[offset:next_offset].copy(),
                target[offset:next_offset].copy(),
                ids[offset:next_offset],
                ids[offset:next_offset],
                "train",
            )
            offset = next_offset

    return low_rank.SyntheticTestShape(patches, dim, images), factory


def _fit(
    source: np.ndarray, target: np.ndarray, W: np.ndarray, ranks: tuple[int, ...], *, sizes: tuple[int, ...] | None = None
) -> low_rank.RRRResult:
    geometry, factory = _fixture(source, target, sizes=sizes)
    return low_rank.fit_low_rank_rrr(
        factory,
        W,
        source.mean(axis=(0, 1), dtype=np.float64),
        target.mean(axis=(0, 1), dtype=np.float64),
        _config(),
        synthetic_test_shape=geometry,
        synthetic_test_ranks=ranks,
    )


def test_frozen_config_acceptance_and_exact_production_counts() -> None:
    config = _config()
    assert config["schema_id"] == "vea.p2.query_adapter_low_rank_linear_config.v1"
    assert [low_rank.production_parameter_count(rank, config) for rank in low_rank.RANK_GRID] == [
        13056, 25344, 49920, 99072, 197376, 393984
    ]
    with pytest.raises(low_rank.LowRankLinearError, match="frozen production grid"):
        low_rank.production_parameter_count(7, config)
    with pytest.raises(low_rank.LowRankLinearError, match="frozen production grid"):
        low_rank.production_parameter_count(True, config)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("adapter", "rank_grid_ordered"), [8, 16, 32]),
        (("closed_form_fit", "method"), "SVD_of_W"),
        (("closed_form_fit", "psd_tolerance"), "no_check"),
        (("closed_form_fit", "tie_policy"), "raw_eigh_output"),
        (("closed_form_fit", "sign_policy"), "arbitrary"),
        (("closed_form_fit", "partition"), "validation"),
        (("closed_form_fit", "max_patch_rows_per_streaming_chunk"), 99999),
        (("complexity", "hard_max_stored_float32_scalars_per_source"), 500000),
        (("latency", "hard_max_candidate_to_matched_OLS_p50_ratio"), 2.0),
        (("test_policy", "new_test_access_authorized"), True),
        (("test_policy", "design_and_train_validation_development_access"), 1),
        (("adapter", "connector_aware_loss_forbidden"), False),
        (("adapter", "optimizer"), "AdamW"),
        (("ols_control", "exact_artifact_sha256_by_source", "SIGLIP2"), "0" * 64),
    ],
)
def test_material_config_drift_is_rejected(path: tuple[str, ...], replacement: object) -> None:
    config = copy.deepcopy(_config())
    node: dict[str, object] = config
    for key in path[:-1]:
        node = node[key]  # type: ignore[assignment,index]
    node[path[-1]] = replacement
    with pytest.raises(low_rank.LowRankLinearError, match="frozen config drift"):
        low_rank.validate_low_rank_config(config)


def test_other_frozen_config_field_and_file_digest_drift_are_rejected(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["selection"]["mean_token_unigram_f1_minimum"] = "candidate >= OLS - 0.20 absolute"  # type: ignore[index]
    with pytest.raises(low_rank.LowRankLinearError, match="canonical T004A design"):
        low_rank.validate_low_rank_config(config)
    changed_file = tmp_path / "changed.json"
    changed_file.write_text(CONFIG_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(low_rank.LowRankLinearError, match="SHA-256 mismatch"):
        low_rank.load_low_rank_config(changed_file)


def test_known_rank_mapping_matches_independent_train_prediction_svd() -> None:
    rng = np.random.default_rng(43)
    source = rng.normal(size=(20, 2, 4))
    left = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    right = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.5]])
    W = left @ right.T
    offset = np.array([1.5, -2.0, 0.25, 3.0])
    target = source @ W + offset
    fitted = _fit(source, target, W, (1, 2, 4), sizes=(3, 7, 10))
    assert fitted.train_image_count == 20 and fitted.train_row_count == 40
    assert fitted.positive_eigenvalue_count == 2
    assert isinstance(fitted.by_rank[4], low_rank.DegenerateRank)
    rank2 = fitted.by_rank[2]
    assert isinstance(rank2, low_rank.LowRankFactors)
    adapted = low_rank.apply_low_rank(source, rank2)
    np.testing.assert_allclose(adapted, target, rtol=2e-6, atol=2e-6)
    assert adapted.dtype == np.float32 and adapted.flags.c_contiguous
    mean_x = source.mean(axis=(0, 1))
    mean_y = target.mean(axis=(0, 1))
    np.testing.assert_allclose(low_rank.apply_low_rank(mean_x.reshape(1, 1, 4).repeat(2, axis=1), rank2)[0, 0], mean_y, atol=2e-6)
    rank1 = fitted.by_rank[1]
    assert isinstance(rank1, low_rank.LowRankFactors)
    centered = source.reshape(-1, 4) - mean_x
    predictions = centered @ W
    independent_u, independent_s, independent_vt = np.linalg.svd(predictions, full_matrices=False)
    best_rank1 = independent_u[:, :1] @ np.diag(independent_s[:1]) @ independent_vt[:1]
    achieved = centered @ rank1.U.astype(np.float64) @ rank1.V.astype(np.float64).T
    np.testing.assert_allclose(achieved, best_rank1, rtol=2e-6, atol=2e-6)
    assert np.linalg.norm(predictions - achieved) ** 2 == pytest.approx(float(np.sum(independent_s[1:] ** 2)), rel=2e-6)
    assert rank1.U.dtype == rank1.V.dtype == rank1.bias.dtype == np.float32
    assert rank1.U.flags.c_contiguous and rank1.V.flags.c_contiguous and rank1.bias.flags.c_contiguous
    assert not rank1.U.flags.writeable
    assert set(vars(rank1)) == {
        "rank", "U", "V", "bias", "scalar_count", "patches_per_image", "feature_dim",
        "synthetic_test_only", "input_feature_class", "output_feature_class",
    }


def test_full_synthetic_numerical_rank_reconstructs_ols_without_changing_production_grid() -> None:
    rng = np.random.default_rng(19)
    source = rng.normal(size=(12, 3, 3))
    W = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0], [2.0, 0.5, 1.0]])
    target = source @ W + np.array([1.0, -0.5, 2.0])
    original = W.copy()
    fitted = _fit(source, target, W, (3,))
    factors = fitted.by_rank[3]
    assert isinstance(factors, low_rank.LowRankFactors)
    np.testing.assert_allclose(low_rank.apply_low_rank(source, factors), target, atol=2e-6)
    np.testing.assert_array_equal(W, original)
    assert 3 not in low_rank.RANK_GRID and factors.synthetic_test_only


def test_two_pass_covariance_is_chunk_independent_and_uses_source_covariance() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(10, 2, 3))
    W = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    target = source @ W + np.array([0.5, -1.0, 1.5])
    one = _fit(source, target, W, (1, 2, 3), sizes=(10,))
    many = _fit(source, target, W, (1, 2, 3), sizes=(1, 2, 3, 4))
    centered = source.reshape(-1, 3) - source.mean(axis=(0, 1))
    covariance = centered.T @ centered
    predicted_covariance = W.T @ covariance @ W
    values, vectors = np.linalg.eigh(predicted_covariance)
    top = vectors[:, np.argmax(values)]
    for result in (one, many):
        rank1 = result.by_rank[1]
        assert isinstance(rank1, low_rank.LowRankFactors)
        np.testing.assert_allclose(rank1.V.astype(np.float64) @ rank1.V.astype(np.float64).T, np.outer(top, top), atol=2e-7)
    np.testing.assert_allclose(
        low_rank.apply_low_rank(source, one.by_rank[1]),  # type: ignore[arg-type]
        low_rank.apply_low_rank(source, many.by_rank[1]),  # type: ignore[arg-type]
        atol=2e-6,
    )


def test_second_pass_changed_values_and_length_fail_closed() -> None:
    source = np.arange(12.0).reshape(3, 2, 2)
    target = source.copy()
    ids = ("a", "b", "c")
    calls = 0

    def changed_factory() -> object:
        nonlocal calls
        calls += 1
        observed = source.copy()
        if calls == 2:
            observed[0, 0, 0] += 1.0
        yield low_rank.TrainChunk(observed, target.copy(), ids, ids, "train")

    with pytest.raises(low_rank.LowRankLinearError, match="changed between passes"):
        low_rank.fit_low_rank_rrr(
            changed_factory, np.eye(2), source.mean(axis=(0, 1)), target.mean(axis=(0, 1)),
            _config(), synthetic_test_shape=low_rank.SyntheticTestShape(2, 2, 3), synthetic_test_ranks=(1,)
        )

    calls = 0

    def short_factory() -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield low_rank.TrainChunk(source.copy(), target.copy(), ids, ids, "train")

    with pytest.raises(low_rank.LowRankLinearError, match="changed length"):
        low_rank.fit_low_rank_rrr(
            short_factory, np.eye(2), source.mean(axis=(0, 1)), target.mean(axis=(0, 1)),
            _config(), synthetic_test_shape=low_rank.SyntheticTestShape(2, 2, 3), synthetic_test_ranks=(1,)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: dataclasses.replace(c, target_image_ids=("b", "a")), "image-ID order"),
        (lambda c: dataclasses.replace(c, split="validation"), "split='train'"),
        (lambda c: dataclasses.replace(c, source=np.full_like(c.source, np.nan)), "NaN"),
        (lambda c: dataclasses.replace(c, target=c.target[:, :, :1]), "shape"),
    ],
)
def test_malformed_chunk_rejected(mutate: object, message: str) -> None:
    values = np.ones((2, 2, 2))
    chunk = low_rank.TrainChunk(values, values, ("a", "b"), ("a", "b"), "train")
    changed = mutate(chunk)  # type: ignore[operator]
    with pytest.raises(low_rank.LowRankLinearError, match=message):
        low_rank.fit_low_rank_rrr(
            lambda: iter((changed,)), np.eye(2), np.ones(2), np.ones(2), _config(),
            synthetic_test_shape=low_rank.SyntheticTestShape(2, 2, 2), synthetic_test_ranks=(1,)
        )


def test_duplicate_ids_across_chunks_and_mean_mismatch_rejected() -> None:
    values = np.ones((1, 2, 2))
    repeated = lambda: iter((
        low_rank.TrainChunk(values, values, ("a",), ("a",), "train"),
        low_rank.TrainChunk(values, values, ("a",), ("a",), "train"),
    ))
    with pytest.raises(low_rank.LowRankLinearError, match="duplicate Train image ID"):
        low_rank.fit_low_rank_rrr(
            repeated, np.eye(2), np.ones(2), np.ones(2), _config(),
            synthetic_test_shape=low_rank.SyntheticTestShape(2, 2, 2), synthetic_test_ranks=(1,)
        )
    geometry, factory = _fixture(values, values)
    with pytest.raises(low_rank.LowRankLinearError, match="accepted source mean"):
        low_rank.fit_low_rank_rrr(
            factory, np.eye(2), np.zeros(2), np.ones(2), _config(),
            synthetic_test_shape=geometry, synthetic_test_ranks=(1,)
        )
    with pytest.raises(low_rank.LowRankLinearError, match="accepted target mean"):
        low_rank.fit_low_rank_rrr(
            factory, np.eye(2), np.ones(2), np.zeros(2), _config(),
            synthetic_test_shape=geometry, synthetic_test_ranks=(1,)
        )


def test_chunk_row_ceiling_and_production_image_row_gate() -> None:
    oversized = np.zeros((1366, 3, 2))
    ids = tuple(f"i-{index}" for index in range(1366))
    with pytest.raises(low_rank.LowRankLinearError, match="4096"):
        low_rank.fit_low_rank_rrr(
            lambda: iter((low_rank.TrainChunk(oversized, oversized, ids, ids, "train"),)),
            np.eye(2), np.zeros(2), np.zeros(2), _config(),
            synthetic_test_shape=low_rank.SyntheticTestShape(3, 2, 1366), synthetic_test_ranks=(1,)
        )
    production_one = np.zeros((1, 196, 768), dtype=np.float32)
    with pytest.raises(low_rank.LowRankLinearError, match="image or flattened row count"):
        low_rank.fit_low_rank_rrr(
            lambda: iter((low_rank.TrainChunk(production_one, production_one, ("p-0",), ("p-0",), "train"),)),
            np.eye(768, dtype=np.float32), np.zeros(768), np.zeros(768), _config()
        )


def test_tie_projector_sign_rank_cut_and_repeated_eigensolve_are_deterministic() -> None:
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    matrix = rotation @ np.diag([3.0, 3.0, 1.0]) @ rotation.T
    first_values, first_vectors, tolerance = low_rank._symmetric_eigh(matrix, "M")
    first = low_rank._canonical_positive_basis(first_values, first_vectors, tolerance)
    for _ in range(3):
        values, vectors, observed_tolerance = low_rank._symmetric_eigh(matrix, "M")
        repeated = low_rank._canonical_positive_basis(values, vectors, observed_tolerance)
        np.testing.assert_allclose(repeated, first, atol=1e-13)
    order = np.argsort(-first_values, kind="stable")
    rotated_vectors = first_vectors.copy()
    tied = order[:2]
    rotated_vectors[:, tied] = first_vectors[:, tied] @ rotation[:2, :2]
    rotated = low_rank._canonical_positive_basis(first_values, rotated_vectors, tolerance)
    np.testing.assert_allclose(rotated, first, atol=1e-13)
    np.testing.assert_allclose(first[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-13)
    assert first[np.argmax(np.abs(first[:, 0])), 0] > 0

    source = np.concatenate((np.eye(3), -np.eye(3))).reshape(6, 1, 3)
    W = np.diag([1.0, 1.0, 0.5])
    fitted = _fit(source, source @ W, W, (1, 2))
    rank_cut = fitted.by_rank[1]
    assert isinstance(rank_cut, low_rank.LowRankFactors)
    np.testing.assert_allclose(rank_cut.V[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-7)
    assert isinstance(fitted.by_rank[2], low_rank.LowRankFactors)


def test_psd_near_zero_clamps_and_material_negative_rejected() -> None:
    values, _vectors, tolerance = low_rank._symmetric_eigh(np.diag([1.0, -1e-15]), "M")
    assert tolerance > 1e-15 and values.tolist() == [0.0, 1.0]
    with pytest.raises(low_rank.LowRankLinearError, match="materially negative"):
        low_rank._symmetric_eigh(np.diag([1.0, -1e-4]), "M")


def test_rank_deficient_and_zero_spectrum_return_explicit_degenerate_status() -> None:
    source = np.arange(24.0).reshape(4, 2, 3)
    W = np.diag([1.0, 0.0, 0.0])
    rank_deficient = _fit(source, source @ W, W, (1, 2, 3))
    assert isinstance(rank_deficient.by_rank[1], low_rank.LowRankFactors)
    assert isinstance(rank_deficient.by_rank[2], low_rank.DegenerateRank)
    assert isinstance(rank_deficient.by_rank[3], low_rank.DegenerateRank)
    zero = _fit(source, np.ones_like(source), np.zeros((3, 3)), (1, 2, 3))
    assert zero.positive_eigenvalue_count == 0
    assert all(isinstance(item, low_rank.DegenerateRank) for item in zero.by_rank.values())


def test_rejects_unsupported_ranks_bad_ols_and_apply_geometry_or_nonfinite() -> None:
    source = np.arange(12.0).reshape(3, 2, 2)
    target = source.copy()
    geometry, factory = _fixture(source, target)
    kwargs = {"synthetic_test_shape": geometry}
    with pytest.raises(low_rank.LowRankLinearError, match="duplicate"):
        low_rank.fit_low_rank_rrr(factory, np.eye(2), source.mean((0, 1)), target.mean((0, 1)), _config(), synthetic_test_ranks=(1, 1), **kwargs)
    with pytest.raises(low_rank.LowRankLinearError, match="unsupported"):
        low_rank.fit_low_rank_rrr(factory, np.eye(2), source.mean((0, 1)), target.mean((0, 1)), _config(), synthetic_test_ranks=(3,), **kwargs)
    with pytest.raises(low_rank.LowRankLinearError, match="weight or mean shape"):
        low_rank.fit_low_rank_rrr(factory, np.eye(3), source.mean((0, 1)), target.mean((0, 1)), _config(), synthetic_test_ranks=(1,), **kwargs)
    fitted = _fit(source, target, np.eye(2), (1,))
    factors = fitted.by_rank[1]
    assert isinstance(factors, low_rank.LowRankFactors)
    with pytest.raises(low_rank.LowRankLinearError, match="geometry"):
        low_rank.apply_low_rank(source[:, :1], factors)
    with pytest.raises(low_rank.LowRankLinearError, match="NaN"):
        low_rank.apply_low_rank(np.full_like(source, np.nan), factors)
    with pytest.raises(low_rank.LowRankLinearError, match="nondegenerate"):
        low_rank.apply_low_rank(source, low_rank.DegenerateRank(2))  # type: ignore[arg-type]
    bad_factors = dataclasses.replace(factors, U=np.full_like(factors.U, np.inf))
    with pytest.raises(low_rank.LowRankLinearError, match="compact factor"):
        low_rank.apply_low_rank(source, bad_factors)


def test_import_has_no_heavy_runtime_or_generation_modules() -> None:
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.query_adapter_low_rank_linear as m; "
        "assert not {'torch','transformers','huggingface_hub'}.intersection(sys.modules); "
        "assert not any(name.startswith('scripts.run_') for name in sys.modules); "
        "assert not hasattr(m, 'generate') and not hasattr(m, 'select_winner')"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False, env=env)
    assert completed.returncode == 0, completed.stderr
