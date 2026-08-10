"""Static contracts shared by QC metric producers, gates and reports."""

from __future__ import annotations

from typing import Any, TypedDict


class SimilarityReport(TypedDict, total=False):
    ncc_before: float | None
    ncc_after: float | None
    ncc_gain: float | None
    nmi_before: float | None
    nmi_after: float | None
    nmi_gain: float | None
    n_voxels_evaluated: int
    n_voxels_available: int
    subsampled: bool


class JacobianStats(TypedDict, total=False):
    available: bool
    folding_fraction: float | None
    det_min: float | None
    det_max: float | None
    det_mean: float | None


class LinearAnalysis(TypedDict, total=False):
    translation_norm_mm: float | None
    max_scale_deviation: float | None
    determinant: float | None
    includes_initialization: bool


class LandmarkReport(TypedDict, total=False):
    tre_mean_mm: float | None
    tre_before_mean_mm: float | None
    n_landmarks: int
    per_landmark_mm: list[float]


class StageSummary(TypedDict, total=False):
    stage: str
    metric: str
    metric_elastix: str
    final_metric: float | None
    transform: str
    seconds: float


class InitializationReport(TypedDict, total=False):
    mode: str
    requested: str
    chosen: str
    n_candidates: int
    ambiguous: bool


class OrganOverlapEntry(TypedDict, total=False):
    dice: float | None
    hd95_mm: float | None
    msd_mm: float | None
    available: bool
    reason: str


def require_known_schema(name: str, value: dict[str, Any] | None, keys: set[str]) -> None:
    """Reject a non-empty report whose keys no longer match its declared contract."""
    if value and not keys.intersection(value):
        raise ValueError(
            f"unexpected {name} schema: expected at least one of {sorted(keys)}, received {sorted(value)}"
        )
