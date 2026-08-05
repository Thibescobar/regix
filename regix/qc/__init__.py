"""Quality control: metrics, acceptance gates, report."""

from regix.qc.gates import Check, GateResult, evaluate_gates
from regix.qc.metrics import (
    dice,
    displacement_statistics,
    hausdorff95,
    jacobian_statistics,
    mean_surface_distance,
    normalized_cross_correlation,
    normalized_mutual_information,
    organ_overlap_report,
    similarity_report,
    target_registration_error,
)
from regix.qc.report import (
    build_html_report,
    checkerboard_figure,
    contour_figure,
    jacobian_figure,
    overlay_figure,
)

__all__ = [
    "dice",
    "displacement_statistics",
    "hausdorff95",
    "jacobian_statistics",
    "mean_surface_distance",
    "normalized_cross_correlation",
    "normalized_mutual_information",
    "organ_overlap_report",
    "similarity_report",
    "target_registration_error",
    "Check",
    "GateResult",
    "evaluate_gates",
    "build_html_report",
    "checkerboard_figure",
    "contour_figure",
    "jacobian_figure",
    "overlay_figure",
]
