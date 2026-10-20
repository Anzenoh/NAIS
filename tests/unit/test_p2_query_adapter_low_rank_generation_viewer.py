"""Two-image offline viewer fixture and adversarial input checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_low_rank_generation_viewer as viewer
from scripts import run_query_adapter_low_rank_validation_generation as cli
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime

ROOT = Path(__file__).resolve().parents[2]


def miniature(tmp_path: Path):
    config = runtime.load_config(ROOT)
    inputs = cli.fake_inputs(tmp_path)
    result = runtime.run_diagnostic(config=config, ids=inputs[0], images=inputs[1],
                                    arrays=inputs[2], ols=inputs[3], factors=inputs[4],
                                    hashes=inputs[5], ranks=inputs[6], provenance=inputs[7],
                                    generator=cli.FakeGenerator(), output_root=tmp_path / "result",
                                    device="cpu", fake=True)
    result["result_sha256"] = runtime.sha_file(tmp_path / "result/result.json")
    rows = runtime.read_rows(tmp_path / "result/rows.jsonl")
    return rows, inputs[1], result


def test_two_image_html_is_offline_and_escapes_untrusted_text(tmp_path: Path) -> None:
    rows, images, result = miniature(tmp_path)
    rows[0]["decoded_text"] = '</script><script>alert("test")</script>'
    rows[1]["decoded_text"] = '<img src=x onerror=alert(1)>'
    html = viewer.render_html(rows, images, result)
    assert html.count("data:image/jpeg;base64,") == 2
    assert '</script><script>alert' not in html
    assert '<img src=x onerror' not in html
    assert "\\u003c/script\\u003e" in html
    assert viewer.STORAGE_KEY in html
    assert "vea-t002e-human-audit-v2-0to5" not in html
    assert "vea-t002e-image-bookmarks-v1" not in html
    for required in ("Bookmark", "Filter image ID", "Previous", "Next", "Unrated",
                     '"0","1","2","3","4","5"', "viewer_scores.v1", "rating_range:[0,5]",
                     "Export ratings and notes JSON", "LOW_RANK_256", "OLS_REFERENCE"):
        assert required in html
    assert "http://" not in html and "https://" not in html


def test_builder_rejects_wrong_sha_and_synthetic_result(tmp_path: Path) -> None:
    rows, images, result = miniature(tmp_path)
    with pytest.raises(runtime.DiagnosticError, match="SHA"):
        viewer.build(repository=ROOT, result_root=tmp_path / "result",
                     result_sha256="0" * 64, t001d_root=tmp_path,
                     output_file=tmp_path / "viewer.html")
    with pytest.raises(runtime.DiagnosticError, match="synthetic_test_only"):
        viewer.build(repository=ROOT, result_root=tmp_path / "result",
                     result_sha256=result["result_sha256"], t001d_root=tmp_path,
                     output_file=tmp_path / "viewer.html")
    assert not (tmp_path / "viewer.html").exists()


def test_viewer_refuses_test_namespace_and_duplicate_conditions(tmp_path: Path) -> None:
    rows, images, result = miniature(tmp_path)
    config = runtime.load_config(ROOT)
    altered = list(rows)
    altered[1] = dict(altered[1], image_id="test-0")
    with pytest.raises(runtime.DiagnosticError):
        runtime.validate_rows(altered, tuple(image.image_id for image in images),
                              images, result["artifact_sha256"],
                              {source: {method: {image.image_id: index + 1 for index, image in enumerate(images)}
                                        for method in ("OLS_REFERENCE", "LOW_RANK_256")}
                               for source in ("SIGLIP2", "DINOV3")}, config)
    altered = list(rows)
    altered[2] = dict(altered[1])
    with pytest.raises(runtime.DiagnosticError):
        runtime.validate_rows(altered, tuple(image.image_id for image in images),
                              images, result["artifact_sha256"],
                              {source: {method: {image.image_id: index + 1 for index, image in enumerate(images)}
                                        for method in ("OLS_REFERENCE", "LOW_RANK_256")}
                               for source in ("SIGLIP2", "DINOV3")}, config)


def test_viewer_output_collision_and_input_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "viewer.html"
    assert viewer.fresh_html_path(output, (source,)) == output
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="fresh external"):
        viewer.fresh_html_path(output, (source,))
    with pytest.raises(runtime.DiagnosticError, match="fresh external"):
        viewer.fresh_html_path(source / "inside.html", (source,))
