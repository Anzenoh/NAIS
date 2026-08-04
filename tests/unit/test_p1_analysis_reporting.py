import pytest

from vision_encoder_alignment.analysis.reporting import ReportingError, determine_conclusion, heatmap_svg


def test_conclusion_gates_and_svg_are_deterministic():
    evidence = dict(all_frozen_endpoints_complete=True, global_confirmed=True, mapping_confirmed=True, spatial_confirmed=True, canonical_extension_confirmed=True, strong_counterevidence=False)
    assert determine_conclusion(evidence) == "A"
    assert heatmap_svg([[1.0]], row_labels=["r"], column_labels=["c"], title="x") == heatmap_svg([[1.0]], row_labels=["r"], column_labels=["c"], title="x")
    with pytest.raises(ReportingError):
        determine_conclusion({})
