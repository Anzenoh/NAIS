"""Public contracts and deterministic mock vision encoder extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    CanonicalImageBatch,
    CanonicalTapFeatures,
    ExtractionResult,
    ExtractorDescriptor,
    ExtractorValidationError,
    NumericNormalizationSpec,
    TapSpec,
    TokenLayout,
    VisionEncoderExtractor,
    build_vit_tap_specs,
    resolve_normalized_block_taps,
    resolve_requested_taps,
    select_patch_tokens,
)
from .mock import MockExtractorConfig, MockVisionEncoderExtractor

if TYPE_CHECKING:
    from .clip import CLIPExtractorConfig, CLIPVisionEncoderExtractor
    from .dinov3 import DINOv3ExtractorConfig, DINOv3VisionEncoderExtractor
    from .siglip2 import SigLIP2ExtractorConfig, SigLIP2VisionEncoderExtractor


def __getattr__(name: str) -> Any:
    if name in {"CLIPExtractorConfig", "CLIPVisionEncoderExtractor"}:
        from .clip import CLIPExtractorConfig, CLIPVisionEncoderExtractor

        return {
            "CLIPExtractorConfig": CLIPExtractorConfig,
            "CLIPVisionEncoderExtractor": CLIPVisionEncoderExtractor,
        }[name]
    if name in {"SigLIP2ExtractorConfig", "SigLIP2VisionEncoderExtractor"}:
        from .siglip2 import SigLIP2ExtractorConfig, SigLIP2VisionEncoderExtractor

        return {
            "SigLIP2ExtractorConfig": SigLIP2ExtractorConfig,
            "SigLIP2VisionEncoderExtractor": SigLIP2VisionEncoderExtractor,
        }[name]
    if name in {"DINOv3ExtractorConfig", "DINOv3VisionEncoderExtractor"}:
        from .dinov3 import DINOv3ExtractorConfig, DINOv3VisionEncoderExtractor

        return {
            "DINOv3ExtractorConfig": DINOv3ExtractorConfig,
            "DINOv3VisionEncoderExtractor": DINOv3VisionEncoderExtractor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "CanonicalImageBatch",
    "CanonicalTapFeatures",
    "CLIPExtractorConfig",
    "CLIPVisionEncoderExtractor",
    "DINOv3ExtractorConfig",
    "DINOv3VisionEncoderExtractor",
    "ExtractionResult",
    "ExtractorDescriptor",
    "ExtractorValidationError",
    "MockExtractorConfig",
    "MockVisionEncoderExtractor",
    "NumericNormalizationSpec",
    "SigLIP2ExtractorConfig",
    "SigLIP2VisionEncoderExtractor",
    "TapSpec",
    "TokenLayout",
    "VisionEncoderExtractor",
    "build_vit_tap_specs",
    "resolve_normalized_block_taps",
    "resolve_requested_taps",
    "select_patch_tokens",
]
