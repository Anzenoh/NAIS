"""Dependency-free Phase 1 tables, SVG figures, and bounded reports."""

from __future__ import annotations

import csv
import html
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_bytes, atomic_write_json


class ReportingError(RuntimeError):
    """Raised when report inputs are incomplete or ambiguous."""


def determine_conclusion(evidence: Mapping[str, Any]) -> str:
    required = {
        "all_frozen_endpoints_complete", "global_confirmed", "mapping_confirmed",
        "spatial_confirmed", "canonical_extension_confirmed", "strong_counterevidence",
    }
    if set(evidence) != required or any(type(evidence[key]) is not bool for key in required):
        raise ReportingError("conclusion evidence must contain exact boolean gates")
    if not evidence["all_frozen_endpoints_complete"]:
        return "INDETERMINATE"
    global_positive = evidence["global_confirmed"]
    mapping_positive = evidence["mapping_confirmed"]
    spatial_positive = evidence["spatial_confirmed"]
    extension_positive = evidence["canonical_extension_confirmed"]
    counter = evidence["strong_counterevidence"]
    if global_positive and mapping_positive and spatial_positive and extension_positive and not counter:
        return "A"
    if global_positive and mapping_positive and extension_positive:
        return "B"
    if (mapping_positive or extension_positive) and not global_positive:
        return "C"
    if spatial_positive and not (global_positive and mapping_positive):
        return "D"
    if not any((global_positive, mapping_positive, spatial_positive, extension_positive)):
        return "E"
    return "INDETERMINATE"


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def heatmap_svg(
    matrix: Sequence[Sequence[float]],
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
) -> bytes:
    rows = [list(row) for row in matrix]
    if not rows or any(len(row) != len(column_labels) for row in rows) or len(rows) != len(row_labels):
        raise ReportingError("heatmap matrix/label shape mismatch")
    flat = [float(value) for row in rows for value in row]
    low, high = min(flat), max(flat)
    span = high - low
    cell = 24
    left = 150
    top = 90
    width = left + cell * len(column_labels) + 20
    height = top + cell * len(row_labels) + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="10" y="22" font-family="sans-serif" font-size="15">{html.escape(title)}</text>',
    ]
    for column, label in enumerate(column_labels):
        x = left + column * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top - 6}" text-anchor="end" transform="rotate(-45 {x} {top - 6})" font-family="sans-serif" font-size="9">{html.escape(label)}</text>')
    for row, (label, values) in enumerate(zip(row_labels, rows, strict=True)):
        y = top + row * cell
        parts.append(f'<text x="{left - 5}" y="{y + 16}" text-anchor="end" font-family="sans-serif" font-size="9">{html.escape(label)}</text>')
        for column, value in enumerate(values):
            scaled = 0.5 if span == 0.0 else (float(value) - low) / span
            red = int(round(245 - 180 * scaled))
            green = int(round(245 - 90 * scaled))
            blue = int(round(255 - 20 * scaled))
            x = left + column * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="white" stroke-width="1"/>')
    parts.append('</svg>\n')
    return "".join(parts).encode("utf-8")


def _bounded_claim() -> str:
    return (
        "The conclusion applies only to the three fixed checkpoints, the frozen COCO "
        "1,000-image 224x224 condition, and encoder-only Phase 1."
    )


def write_reports(
    result_root: str | Path,
    *,
    summary: Mapping[str, Any],
    detailed_sections: Mapping[str, Any],
) -> dict[str, Path]:
    root = Path(result_root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    conclusion = determine_conclusion(summary["conclusion_evidence"])
    repeat_status = summary["repeat_control_status"]
    executive = "\n".join([
        "# Phase 1 Vision Encoder Alignment — Executive Report",
        "",
        f"- Campaign status: `{summary['campaign_status']}`",
        f"- Conclusion type: `{conclusion}`",
        f"- Same-setting repeat: `{repeat_status}`",
        f"- Strongest supporting evidence: {summary['strongest_supporting_evidence']}",
        f"- Strongest counterevidence: {summary['strongest_counterevidence']}",
        "",
        "## Evidence families",
        "",
        f"- Pairwise/global baseline: {summary['pairwise_global_summary']}",
        f"- Directional mapping baseline: {summary['mapping_summary']}",
        f"- Spatial/patch baseline: {summary['spatial_summary']}",
        f"- Linear canonical extension: {summary['canonical_extension_summary']}",
        "",
        "## Bounded claim",
        "",
        _bounded_claim(),
        "",
        "No causal claim about pretraining objective is made. The result is not a universal "
        "encoder claim, and it does not establish any nonlinear alignment conclusion.",
        "",
        "Checksum integrity is an input-preservation control and is not labeled as a repeat.",
        "",
    ])
    executive_path = atomic_write_bytes(reports / "executive.md", executive.encode("utf-8"))
    detailed_lines = [
        "# Phase 1 Vision Encoder Alignment — Detailed Report",
        "",
        f"Conclusion: `{conclusion}`",
        "",
        _bounded_claim(),
        "",
    ]
    for title, content in detailed_sections.items():
        detailed_lines.extend([
            f"## {title}",
            "",
            "```json",
            __import__("json").dumps(content, ensure_ascii=False, sort_keys=True, indent=2),
            "```",
            "",
        ])
    detailed_lines.extend([
        "## Limitations",
        "",
        "- No causal attribution to a pretraining objective.",
        "- No universal claim beyond the fixed three encoders and fixed input condition.",
        "- No MLP, nonlinear adapter, cross-attention, projector, or language model was tested.",
        f"- Same-setting repeat status: `{repeat_status}`.",
        "- Starting and ending input checksums are preservation evidence, not repeat evidence.",
        "",
    ])
    detailed_path = atomic_write_bytes(reports / "detailed.md", "\n".join(detailed_lines).encode("utf-8"))
    summary_path = atomic_write_json(reports / "report_summary.json", {
        "schema_id": "vea.p1.analysis_report_summary.v1",
        "conclusion_type": conclusion,
        **dict(summary),
    })
    return {"executive": executive_path, "detailed": detailed_path, "summary": summary_path}


__all__ = [
    "ReportingError", "determine_conclusion", "heatmap_svg", "write_csv",
    "write_reports",
]
