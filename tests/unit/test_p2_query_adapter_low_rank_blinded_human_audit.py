"""Offline T004M blinding, backup, and descriptive analysis adversaries."""

from __future__ import annotations

import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_low_rank_blinded_human_viewer as blind
from scripts import summarize_low_rank_blinded_human_scores as analysis
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime

ROOT = Path(__file__).resolve().parents[2]
IDS = ("validation-one", "validation-two")


def miniature(tmp_path: Path):
    protocol = blind.load_protocol(ROOT)
    images = []
    for index, image_id in enumerate(IDS):
        path = tmp_path / f"image-{index}.jpg"
        path.write_bytes(b"\xff\xd8fixture-" + bytes([index]) + b"\xff\xd9")
        images.append(runtime.ValidationImage(image_id, path, path.name, runtime.sha_file(path)))
    rows = [{"image_id": image_id, "lane_key": lane,
             "decoded_text": f"Caption content {image_index} {lane_index}"}
            for image_index, image_id in enumerate(IDS)
            for lane_index, lane in enumerate(runtime.LANES)]
    assignment = blind.make_assignment(IDS, protocol)
    return protocol, images, rows, assignment


def set_lane_scores(value: dict, assignment: dict, by_image_lane: list[dict[str, int]]) -> None:
    for entry, mapping, desired in zip(value["entries"], assignment["assignments"], by_image_lane, strict=True):
        for label, lane in mapping["display_to_lane"].items():
            entry["ratings"][label] = desired[lane]


def test_protocol_exact_rubric_and_result_pin() -> None:
    protocol = blind.load_protocol(ROOT)
    assert protocol["source_t004k_result_sha256"] == blind.T004K_SHA
    assert protocol["purpose"] == "post_hoc_exploratory_absolute_caption_quality_not_selection"
    assert protocol["partition"] == "validation" and protocol["image_count"] == 100
    assert protocol["condition_count"] == 5 and protocol["blinding_seed"] == 43
    assert protocol["selection_status"] == "NO_RANK_ACCEPTED"
    assert protocol["rubric"] == blind.RUBRIC
    assert tuple(protocol["rubric"]) == ("5", "4", "3", "2", "1", "0")


def test_stable_per_image_shuffle_and_assignment_bijection(tmp_path: Path) -> None:
    protocol, _, _, assignment = miniature(tmp_path)
    again = blind.make_assignment(IDS, protocol)
    random.seed(999)
    for _ in range(100):
        random.random()
    reversed_order = blind.make_assignment(tuple(reversed(IDS)), protocol)
    assert again == assignment
    assert assignment["assignments"][0] == reversed_order["assignments"][1]
    assert assignment["assignments"][1] == reversed_order["assignments"][0]
    permutations = [tuple(entry["display_to_lane"][label] for label in blind.LABELS)
                    for entry in assignment["assignments"]]
    assert all(set(order) == set(runtime.LANES) for order in permutations)
    assert permutations[0] != permutations[1]
    blind.validate_assignment(assignment, IDS, protocol)
    changed = copy.deepcopy(assignment)
    changed["assignments"][0]["display_to_lane"]["A"] = changed["assignments"][0]["display_to_lane"]["B"]
    with pytest.raises(runtime.DiagnosticError, match="assignment"):
        blind.validate_assignment(changed, IDS, protocol)
    changed = copy.deepcopy(assignment)
    changed["mapping_sha256"] = "0" * 64
    with pytest.raises(runtime.DiagnosticError, match="assignment"):
        blind.validate_assignment(changed, IDS, protocol)


def test_blind_html_has_only_anonymous_captions_and_escaped_text(tmp_path: Path) -> None:
    protocol, images, rows, assignment = miniature(tmp_path)
    rows[0]["decoded_text"] = '</script><script>alert("bad")</script>'
    rows[1]["decoded_text"] = "<img src=x onerror=alert(1)>"
    html = blind.render_blind_html(rows, images, assignment, protocol)
    assert html.count("data:image/jpeg;base64,") == 2
    assert "\\u003c/script\\u003e" in html
    assert '</script><script>alert' not in html and '<img src=x onerror' not in html
    for forbidden in ("NATIVE", "OLS_REFERENCE", "LOW_RANK_256", "SIGLIP2", "DINOV3",
                      "lane_key", "token_unigram_f1", "same_image_rank", "native_exact_match",
                      "vea-t002e", "vea-t004j"):
        assert forbidden not in html
    for label in blind.LABELS:
        assert f"Caption {label}" not in html  # DOM text is assembled at runtime.
        assert f'\"id\":\"{label}\"' in html
    for definition in blind.RUBRIC.values():
        assert definition in html
    assert "Unrated" in html and "INCOMPLETE" in html
    assert "vea-t004m-blind-validation-v1-" in html
    assert assignment["mapping_sha256"] in html
    assert "http://" not in html and "https://" not in html
    assert "Reveal methods" not in html
    assert "textContent=caption.text" in html


def test_blind_browser_script_parses_when_node_available(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for optional browser-script syntax check")
    protocol, images, rows, assignment = miniature(tmp_path)
    html = blind.render_blind_html(rows, images, assignment, protocol)
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "viewer.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([node, "--check", str(path)], text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_empty_caption_is_not_auto_scored(tmp_path: Path) -> None:
    protocol, images, rows, assignment = miniature(tmp_path)
    rows[0]["decoded_text"] = "   "
    html = blind.render_blind_html(rows, images, assignment, protocol)
    assert "   " in html
    value = blind.empty_scores(assignment)
    assert value["entries"][0]["ratings"] == {label: None for label in blind.LABELS}
    assert blind.validate_scores(value, assignment, protocol)["rated_slots"] == 0


def test_score_export_roundtrip_zero_vs_null_and_partial_coverage(tmp_path: Path) -> None:
    protocol, _, _, assignment = miniature(tmp_path)
    value = blind.empty_scores(assignment)
    value["entries"][0]["ratings"]["A"] = 0
    value["entries"][0]["notes"] = "Image-grounded note"
    backup = tmp_path / "scores.json"
    backup.write_text(json.dumps(value), encoding="utf-8")
    loaded = runtime.read_json(backup)
    coverage = blind.validate_scores(loaded, assignment, protocol)
    assert loaded == value
    assert coverage["rated_slots"] == 1 and coverage["total_slots"] == 10
    assert coverage["by_anonymous_id"]["A"] == 1 and not coverage["complete"]
    summary = analysis.summarize(loaded, assignment, protocol)
    assert summary["status"] == "INCOMPLETE_NO_FINAL_INFERENCE"
    assert "per_lane" not in summary and "within_source_low_rank_minus_ols" not in summary
    assert "NATIVE" not in json.dumps(summary)
    assert "lane_key" not in json.dumps(value)
    assert "LOW_RANK_256" not in json.dumps(value)


@pytest.mark.parametrize("mutation", [
    "missing_entry", "duplicate_entry", "reordered", "test_id", "missing_rating",
    "extra_rating", "six", "negative", "boolean", "float", "bad_schema",
    "old_namespace", "bad_result", "bad_rubric", "bad_mapping", "extra_field",
])
def test_score_export_adversaries_rejected(tmp_path: Path, mutation: str) -> None:
    protocol, _, _, assignment = miniature(tmp_path)
    value = blind.empty_scores(assignment)
    if mutation == "missing_entry":
        value["entries"].pop()
    elif mutation == "duplicate_entry":
        value["entries"][1] = copy.deepcopy(value["entries"][0])
    elif mutation == "reordered":
        value["entries"].reverse()
    elif mutation == "test_id":
        value["entries"][0]["image_id"] = "test-image"
    elif mutation == "missing_rating":
        del value["entries"][0]["ratings"]["A"]
    elif mutation == "extra_rating":
        value["entries"][0]["ratings"]["F"] = 2
    elif mutation == "six":
        value["entries"][0]["ratings"]["A"] = 6
    elif mutation == "negative":
        value["entries"][0]["ratings"]["A"] = -1
    elif mutation == "boolean":
        value["entries"][0]["ratings"]["A"] = False
    elif mutation == "float":
        value["entries"][0]["ratings"]["A"] = 2.0
    elif mutation == "bad_schema":
        value["schema_id"] = "unknown"
    elif mutation == "old_namespace":
        value["schema_id"] = "vea-t002e-human-audit-v2-0to5"
    elif mutation == "bad_result":
        value["source_t004k_result_sha256"] = "0" * 64
    elif mutation == "bad_rubric":
        value["rubric_sha256"] = "0" * 64
    elif mutation == "bad_mapping":
        value["mapping_sha256"] = "0" * 64
    else:
        value["unexpected"] = True
    with pytest.raises(runtime.DiagnosticError):
        blind.validate_scores(value, assignment, protocol)


def test_complete_same_image_descriptive_scores_and_bootstrap(tmp_path: Path) -> None:
    protocol, _, _, assignment = miniature(tmp_path)
    value = blind.empty_scores(assignment)
    by_image = [
        {"NATIVE": 2, "SIGLIP2/OLS_REFERENCE": 3, "SIGLIP2/LOW_RANK_256": 4,
         "DINOV3/OLS_REFERENCE": 4, "DINOV3/LOW_RANK_256": 2},
        {"NATIVE": 3, "SIGLIP2/OLS_REFERENCE": 4, "SIGLIP2/LOW_RANK_256": 4,
         "DINOV3/OLS_REFERENCE": 5, "DINOV3/LOW_RANK_256": 3},
    ]
    set_lane_scores(value, assignment, by_image)
    summary = analysis.summarize(value, assignment, protocol)
    assert summary["status"] == "COMPLETE_EXPLORATORY_DESCRIPTIVE_ONLY"
    assert summary["coverage"]["rated_slots"] == 10
    assert summary["native_absolute_image_grounded_quality"]["mean"] == 2.5
    assert summary["per_lane"]["SIGLIP2/LOW_RANK_256"]["score_distribution_0_to_5"]["4"] == 2
    sig = summary["within_source_low_rank_minus_ols"]["SIGLIP2"]
    assert sig["mean_paired_delta"] == 0.5
    assert (sig["win_count"], sig["tie_count"], sig["loss_count"]) == (1, 1, 0)
    assert sig["descriptive_paired_image_percentile_bootstrap_ci_95"] == [0.0, 1.0]
    dino = summary["within_source_low_rank_minus_ols"]["DINOV3"]
    assert dino["mean_paired_delta"] == -2
    assert (dino["win_count"], dino["tie_count"], dino["loss_count"]) == (0, 0, 2)
    assert summary["equivalence_or_significance_claim"] is None
    assert analysis.summarize(value, assignment, protocol) == summary


def test_output_collision_and_pinned_production_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "blind-output"
    assert blind.output_directory(output, (source,)) == output
    output.mkdir()
    with pytest.raises(runtime.DiagnosticError, match="fresh"):
        blind.output_directory(output, (source,))
    with pytest.raises(runtime.DiagnosticError, match="overlaps"):
        blind.output_directory(source / "nested", (source,))
    json_path = tmp_path / "summary.json"
    assert analysis.fresh_json_path(json_path, (source,)) == json_path
    json_path.write_text("existing", encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="fresh"):
        analysis.fresh_json_path(json_path, (source,))
    with pytest.raises(runtime.DiagnosticError, match="outside"):
        analysis.fresh_json_path(source / "inside.json", (source,))
    monkeypatch.setattr(blind.prior_viewer, "load_verified",
                        lambda **_kwargs: pytest.fail("wrong SHA reached loader"))
    with pytest.raises(runtime.DiagnosticError, match="SHA"):
        blind.build(repository=ROOT, result_root=tmp_path, result_sha256="0" * 64,
                    t001d_root=tmp_path, output_root=tmp_path / "new")


def test_production_entry_rejects_synthetic_miniature_before_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, images, rows, _ = miniature(tmp_path)
    result = {"ordered_image_ids": list(IDS), "selection_status": "NO_RANK_ACCEPTED",
              "test_generation_count": 0, "synthetic_test_only": True}
    monkeypatch.setattr(blind.prior_viewer, "load_verified",
                        lambda **_kwargs: (result, rows, images))
    with pytest.raises(runtime.DiagnosticError, match="production identity"):
        blind.build(repository=ROOT, result_root=tmp_path,
                    result_sha256=blind.T004K_SHA, t001d_root=tmp_path,
                    output_root=tmp_path / "blind-output")
    assert not (tmp_path / "blind-output").exists()
    protocol_path = ROOT / blind.PROTOCOL_PATH
    with pytest.raises(runtime.DiagnosticError, match="production identity"):
        analysis.analyze(repository=ROOT, result_root=tmp_path,
                         result_sha256=blind.T004K_SHA, t001d_root=tmp_path,
                         protocol_path=protocol_path, assignment_path=tmp_path / "absent.json",
                         scores_path=tmp_path / "absent-scores.json")


def test_protocol_drift_and_no_heavy_runtime_import(tmp_path: Path) -> None:
    source = ROOT / blind.PROTOCOL_PATH
    destination = tmp_path / blind.PROTOCOL_PATH
    destination.parent.mkdir(parents=True)
    value = runtime.read_json(source)
    value["rubric"]["0"] = "changed"
    destination.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="rubric"):
        blind.load_protocol(tmp_path)
    code = ("import sys; "
            "import scripts.build_low_rank_blinded_human_viewer; "
            "import scripts.summarize_low_rank_blinded_human_scores; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules")
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True,
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
