"""Similarity and plausibility metrics.

An important methodological point: these metrics are computed **independently**
of elastix, on the full-resolution images, inside the QC mask. An optimiser's
internal score is not a quality measure -- it is computed on a subsample, at the
working resolution, with exactly the criterion that was optimised. Using it to
judge the result is marking your own homework.

What actually matters in practice, in decreasing order of reliability:
1. TRE on landmarks identified by a radiologist;
2. Dice / surface distance on independently segmented organs;
3. Jacobian determinant (detects non-physical deformations);
4. intensity metrics (NCC, NMI) -- the least conclusive, because they are the
   ones that were optimised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("qc.metrics")


# --------------------------------------------------------------------------- #
# Intensities
# --------------------------------------------------------------------------- #
def _paired_arrays(
    a: sitk.Image, b: sitk.Image, mask: sitk.Image | None
) -> tuple[np.ndarray, np.ndarray]:
    if a.GetSize() != b.GetSize():
        raise ValueError(f"different grids: {a.GetSize()} vs {b.GetSize()}")
    # GetArrayFromImage (a copy), not GetArrayViewFromImage: a view onto a temporary
    # image points at memory freed as soon as the temporary is collected, which causes
    # an access violation -- silent or fatal depending on timing.
    arr_a = sitk.GetArrayFromImage(sitk.Cast(a, sitk.sitkFloat32)).astype(np.float64).ravel()
    arr_b = sitk.GetArrayFromImage(sitk.Cast(b, sitk.sitkFloat32)).astype(np.float64).ravel()
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask is not None:
        if mask.GetSize() != a.GetSize():
            raise ValueError(f"mask {mask.GetSize()} incompatible with image {a.GetSize()}")
        valid &= sitk.GetArrayViewFromImage(mask).astype(bool).ravel()
    return arr_a[valid], arr_b[valid]


def normalized_cross_correlation(
    fixed: sitk.Image, moving: sitk.Image, mask: sitk.Image | None = None
) -> float:
    """Pearson NCC inside the mask. 1 = identical, 0 = uncorrelated."""
    a, b = _paired_arrays(fixed, moving, mask)
    if a.size < 64:
        log.warning("NCC over only %d voxels: not a meaningful value", a.size)
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def normalized_mutual_information(
    fixed: sitk.Image, moving: sitk.Image, mask: sitk.Image | None = None, bins: int = 64
) -> float:
    """Studholme NMI: (H(A) + H(B)) / H(A, B). 1 = independent, ~2 = identical.

    This is the reference metric for judging a multimodal registration when
    neither landmarks nor segmentations are available.
    """
    a, b = _paired_arrays(fixed, moving, mask)
    if a.size < 256:
        return float("nan")
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / hist.sum()
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)

    def _h(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h_ab = _h(pab.ravel())
    if h_ab <= 0:
        return float("nan")
    return (_h(pa) + _h(pb)) / h_ab


def similarity_report(
    fixed: sitk.Image,
    moving_before: sitk.Image | None,
    moving_after: sitk.Image,
    mask: sitk.Image | None = None,
) -> dict[str, Any]:
    """NCC and NMI before/after, and their gains."""
    out: dict[str, Any] = {
        "ncc_after": _round(normalized_cross_correlation(fixed, moving_after, mask)),
        "nmi_after": _round(normalized_mutual_information(fixed, moving_after, mask)),
        "n_voxels_evaluated": int(
            np.count_nonzero(sitk.GetArrayViewFromImage(mask))
            if mask is not None
            else np.prod(fixed.GetSize())
        ),
    }
    if moving_before is not None:
        out["ncc_before"] = _round(normalized_cross_correlation(fixed, moving_before, mask))
        out["nmi_before"] = _round(normalized_mutual_information(fixed, moving_before, mask))
        for key in ("ncc", "nmi"):
            before, after = out.get(f"{key}_before"), out.get(f"{key}_after")
            if before is not None and after is not None and np.isfinite(before) and np.isfinite(after):
                out[f"{key}_gain"] = _round(after - before)
    return out


def _round(value: float, digits: int = 4) -> float:
    return float(round(value, digits)) if value is not None and np.isfinite(value) else float("nan")


# --------------------------------------------------------------------------- #
# Structure overlap
# --------------------------------------------------------------------------- #
def dice(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    a = sitk.GetArrayFromImage(sitk.Cast(mask_a, sitk.sitkUInt8)).astype(bool)
    b = sitk.GetArrayFromImage(sitk.Cast(mask_b, sitk.sitkUInt8)).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"masks of different sizes: {a.shape} vs {b.shape}")
    total = a.sum() + b.sum()
    if total == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / total)


def _surface_distances(mask_a: sitk.Image, mask_b: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    a = sitk.Cast(mask_a, sitk.sitkUInt8)
    b = sitk.Cast(mask_b, sitk.sitkUInt8)
    if sitk.GetArrayViewFromImage(a).sum() == 0 or sitk.GetArrayViewFromImage(b).sum() == 0:
        raise ValueError("surface distance undefined: one mask is empty")
    surf_a = sitk.LabelContour(a, False)
    surf_b = sitk.LabelContour(b, False)
    # Signed distance maps -> absolute value sampled on the opposite surface
    dist_a = sitk.Abs(sitk.SignedMaurerDistanceMap(a, squaredDistance=False, useImageSpacing=True))
    dist_b = sitk.Abs(sitk.SignedMaurerDistanceMap(b, squaredDistance=False, useImageSpacing=True))
    d_ab = sitk.GetArrayViewFromImage(dist_b)[sitk.GetArrayViewFromImage(surf_a) > 0]
    d_ba = sitk.GetArrayViewFromImage(dist_a)[sitk.GetArrayViewFromImage(surf_b) > 0]
    return np.asarray(d_ab, dtype=float), np.asarray(d_ba, dtype=float)


def hausdorff95(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    """95th-percentile Hausdorff distance (mm): robust to isolated voxels."""
    try:
        d_ab, d_ba = _surface_distances(mask_a, mask_b)
    except ValueError as exc:
        log.debug("HD95 unavailable: %s", exc)
        return float("nan")
    if d_ab.size == 0 or d_ba.size == 0:
        return float("nan")
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def mean_surface_distance(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    """Symmetric mean surface distance (mm)."""
    try:
        d_ab, d_ba = _surface_distances(mask_a, mask_b)
    except ValueError:
        return float("nan")
    if d_ab.size == 0 or d_ba.size == 0:
        return float("nan")
    return float((d_ab.mean() + d_ba.mean()) / 2.0)


def organ_overlap_report(
    fixed_labelmap: sitk.Image,
    warped_labelmap: sitk.Image,
    label_names: dict[int, str],
    organs: Sequence[str] | None = None,
    with_surface: bool = True,
) -> dict[str, dict[str, float]]:
    """Per-organ Dice / HD95 / MSD after registration.

    ``warped_labelmap`` must have been resampled onto the fixed grid with
    nearest-neighbour interpolation using the final transform.
    """
    out: dict[str, dict[str, float]] = {}
    wanted = set(organs) if organs else None
    f_arr = sitk.GetArrayViewFromImage(fixed_labelmap)
    w_arr = sitk.GetArrayViewFromImage(warped_labelmap)
    for label, name in sorted(label_names.items()):
        if wanted is not None and name not in wanted:
            continue
        if not np.any(f_arr == label) or not np.any(w_arr == label):
            continue
        m_fixed = sitk.Cast(sitk.Equal(fixed_labelmap, int(label)), sitk.sitkUInt8)
        m_warp = sitk.Cast(sitk.Equal(warped_labelmap, int(label)), sitk.sitkUInt8)
        entry = {"dice": _round(dice(m_fixed, m_warp))}
        if with_surface:
            entry["hd95_mm"] = _round(hausdorff95(m_fixed, m_warp), 2)
            entry["msd_mm"] = _round(mean_surface_distance(m_fixed, m_warp), 2)
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #
def target_registration_error(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    transform_fixed_to_moving,
) -> dict[str, Any]:
    """TRE (mm): the only measure that is genuinely independent of the optimised criterion.

    ``transform_fixed_to_moving`` is either a ``sitk.Transform`` or a callable
    mapping a point from the fixed frame to the moving frame (elastix
    convention). The TRE is then the distance between the measured moving point
    and the image of the corresponding fixed point.
    """
    f = np.asarray(fixed_points, dtype=float).reshape(-1, 3)
    m = np.asarray(moving_points, dtype=float).reshape(-1, 3)
    if f.shape != m.shape:
        raise ValueError(f"{len(f)} fixed landmarks against {len(m)} moving ones")
    if callable(transform_fixed_to_moving) and not hasattr(transform_fixed_to_moving, "TransformPoint"):
        mapped = np.asarray(transform_fixed_to_moving(f), dtype=float).reshape(-1, 3)
    else:
        mapped = np.asarray(
            [transform_fixed_to_moving.TransformPoint([float(v) for v in p]) for p in f], dtype=float
        )
    errors = np.linalg.norm(mapped - m, axis=1)
    initial = np.linalg.norm(f - m, axis=1)
    return {
        "n_landmarks": int(len(errors)),
        "tre_mean_mm": _round(float(errors.mean()), 2),
        "tre_median_mm": _round(float(np.median(errors)), 2),
        "tre_p95_mm": _round(float(np.percentile(errors, 95)), 2),
        "tre_max_mm": _round(float(errors.max()), 2),
        "tre_before_mean_mm": _round(float(initial.mean()), 2),
        "per_landmark_mm": [round(float(v), 2) for v in errors],
    }


# --------------------------------------------------------------------------- #
# Deformation-field plausibility
# --------------------------------------------------------------------------- #
def jacobian_statistics(
    displacement_field: sitk.Image, mask: sitk.Image | None = None
) -> dict[str, Any]:
    """Jacobian determinant of the displacement field.

    How to read it:
    * det < 0  : the field folds, anatomy turns inside out -> invalid result;
    * det ~ 1  : volume locally preserved;
    * det >> 1 or << 1 : strong local expansion/compression, plausible during
      breathing, suspicious beyond a factor of two on a solid organ.
    """
    field = sitk.Cast(displacement_field, sitk.sitkVectorFloat64)
    jac = sitk.DisplacementFieldJacobianDeterminant(field)
    arr = sitk.GetArrayFromImage(jac).astype(np.float64)
    if mask is not None:
        m = sitk.GetArrayViewFromImage(mask).astype(bool)
        if m.shape != arr.shape:
            log.debug("mask incompatible with the field: statistics over the whole volume")
        else:
            arr = arr[m]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"available": False}
    folding = int(np.count_nonzero(arr <= 0))
    return {
        "available": True,
        "n_voxels": int(arr.size),
        "det_min": _round(float(arr.min()), 4),
        "det_max": _round(float(arr.max()), 4),
        "det_mean": _round(float(arr.mean()), 4),
        "det_std": _round(float(arr.std()), 4),
        "det_p1": _round(float(np.percentile(arr, 1)), 4),
        "det_p99": _round(float(np.percentile(arr, 99)), 4),
        "folding_voxels": folding,
        "folding_fraction": _round(folding / arr.size, 6),
    }


def displacement_statistics(
    displacement_field: sitk.Image, mask: sitk.Image | None = None
) -> dict[str, Any]:
    """Displacement magnitude (mm): compare against the expected physiological motion."""
    arr = sitk.GetArrayFromImage(displacement_field).astype(np.float64)
    magnitude = np.linalg.norm(arr, axis=-1)
    if mask is not None:
        m = sitk.GetArrayViewFromImage(mask).astype(bool)
        if m.shape == magnitude.shape:
            magnitude = magnitude[m]
    magnitude = magnitude[np.isfinite(magnitude)]
    if magnitude.size == 0:
        return {"available": False}
    return {
        "available": True,
        "mean_mm": _round(float(magnitude.mean()), 3),
        "p95_mm": _round(float(np.percentile(magnitude, 95)), 3),
        "max_mm": _round(float(magnitude.max()), 3),
    }
