"""Repository-level T003C binding contract, entirely offline."""

from pathlib import Path

import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_runtime_binding as binding


ROOT = Path(__file__).resolve().parents[2]


def test_repository_binding_is_exact_and_contains_no_machine_root() -> None:
    value = binding.load_binding(ROOT)
    assert value["experiment_id"] == binding.core.EXPERIMENT_ID
    assert value["runtime_roots"] == {
        "t001d": "EXPLICIT_T001D_RUN_ROOT", "t002d": "EXPLICIT_T002D_VALIDATION_ROOT"
    }
    assert value["policy"] == binding.DENIED_POLICY
    assert set(value["t002d"]["full_linear"]) == set(binding.core.SOURCES)


def test_public_preflight_requires_explicit_absolute_roots(tmp_path: Path) -> None:
    with pytest.raises(binding.RuntimeBindingError, match="absolute directory"):
        binding.preflight_runtime(ROOT, "relative-t001d", tmp_path)
