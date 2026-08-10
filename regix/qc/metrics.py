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
from regix.preprocess.geometry import same_grid

log = get_logger("qc.metrics")


# --------------------------------------------------------------------------- #
# Intensities
# --------------------------------------------------------------------------- #
def _paired_arrays(
    a: sitk.Image,
    b: sitk.Image,
    mask: sitk.Image | None,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not same_grid(a, b):
        raise ValueError(
            "different grids: size, spacing, origin and direction must all agree "
            f"({a.GetSize()} vs {b.GetSize()})"
        )
    # GetArrayFromImage (a copy), not GetArrayViewFromImage: a view onto a temporary
    # image points at memory freed as soon as the temporary is collected, which causes
    # an access violation -- silent or fatal depending on timing.
    arr_a = sitk.GetArrayFromImage(sitk.Cast(a, sitk.sitkFloat32)).ravel()
    arr_b = sitk.GetArrayFromImage(sitk.Cast(b, sitk.sitkFloat32)).ravel()
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask is not None:
        if not same_grid(mask, a):
            raise ValueError(f"mask grid {mask.GetSize()} incompatible with image grid {a.GetSize()}")
        valid &= sitk.GetArrayViewFromImage(mask).astype(bool).ravel()
    indices = np.flatnonzero(valid)
    available = int(indices.size)
    if max_samples is not None and indices.size > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    return arr_a[indices], arr_b[indices], available


def _ncc_arrays(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 64:
        log.warning("NCC over only %d voxels: not a meaningful value", a.size)
        return float("nan")
    a = a - a.mean(dtype=np.float64)
    b = b - b.mean(dtype=np.float64)
    denom = np.sqrt(np.sum(a * a, dtype=np.float64) * np.sum(b * b, dtype=np.float64))
    return float(np.sum(a * b, dtype=np.float64) / denom) if denom > 0 else float("nan")


def _nmi_arrays(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
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
    return (_h(pa) + _h(pb)) / h_ab if h_ab > 0 else float("nan")


def normalized_cross_correlation(
    fixed: sitk.Image, moving: sitk.Image, mask: sitk.Image | None = None
) -> float:
    """Pearson NCC inside the mask. 1 = identical, 0 = uncorrelated."""
    a, b, _ = _paired_arrays(fixed, moving, mask)
    return _ncc_arrays(a, b)


def normalized_mutual_information(
    fixed: sitk.Image, moving: sitk.Image, mask: sitk.Image | None = None, bins: int = 64
) -> float:
    """Studholme NMI: (H(A) + H(B)) / H(A, B). 1 = independent, ~2 = identical.

    This is the reference metric for judging a multimodal registration when
    neither landmarks nor segmentations are available.
    """
    a, b, _ = _paired_arrays(fixed, moving, mask)
    return _nmi_arrays(a, b, bins=bins)


def similarity_report(
    fixed: sitk.Image,
    moving_before: sitk.Image | None,
    moving_after: sitk.Image,
    mask: sitk.Image | None = None,
    max_samples: int | None = 2_000_000,
    seed: int = 0,
) -> dict[str, Any]:
    """NCC and NMI before/after, and their gains."""
    after_fixed, after_moving, available = _paired_arrays(
        fixed,
        moving_after,
        mask,
        max_samples=max_samples,
        seed=seed,
    )
    out: dict[str, Any] = {
        "ncc_after": _round(_ncc_arrays(after_fixed, after_moving)),
        "nmi_after": _round(_nmi_arrays(after_fixed, after_moving)),
        "n_voxels_evaluated": int(after_fixed.size),
        "n_voxels_available": available,
        "subsampled": bool(after_fixed.size < available),
    }
    if moving_before is not None:
        before_fixed, before_moving, _ = _paired_arrays(
            fixed,
            moving_before,
            mask,
            max_samples=max_samples,
            seed=seed,
        )
        out["ncc_before"] = _round(_ncc_arrays(before_fixed, before_moving))
        out["nmi_before"] = _round(_nmi_arrays(before_fixed, before_moving))
        for key in ("ncc", "nmi"):
            before, after = out.get(f"{key}_before"), out.get(f"{key}_after")
            if before is not None and after is not None and np.isfinite(before) and np.isfinite(after):
                out[f"{key}_gain"] = _round(after - before)
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    return float(round(value, digits)) if value is not None and np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Structure overlap
# --------------------------------------------------------------------------- #
def dice(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    if not same_grid(mask_a, mask_b):
        raise ValueError("masks use different physical grids")
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


def _union_labels(labelmap: sitk.Image, labels: Sequence[int]) -> sitk.Image:
    """Binary union for one canonical organ that may occupy several labels."""
    if not labels:
        return sitk.Image(labelmap.GetSize(), sitk.sitkUInt8)
    mask = sitk.Cast(sitk.Equal(labelmap, int(labels[0])), sitk.sitkUInt8)
    for label in labels[1:]:
        mask = sitk.Or(mask, sitk.Cast(sitk.Equal(labelmap, int(label)), sitk.sitkUInt8))
    mask.CopyInformation(labelmap)
    return mask


def _labels_by_name(label_names: dict[int, str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label, name in sorted(label_names.items()):
        out.setdefault(name, []).append(int(label))
    return out


def organ_overlap_report(
    fixed_labelmap: sitk.Image,
    warped_labelmap: sitk.Image,
    label_names: dict[int, str] | None = None,
    organs: Sequence[str] | None = None,
    with_surface: bool = True,
    *,
    fixed_label_names: dict[int, str] | None = None,
    moving_label_names: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-organ Dice / HD95 / MSD after registration.

    ``warped_labelmap`` must have been resampled onto the fixed grid with
    nearest-neighbour interpolation using the final transform.
    """
    if fixed_label_names is None:
        fixed_label_names = label_names or {}
    if moving_label_names is None:
        moving_label_names = label_names or fixed_label_names
    fixed_by_name = _labels_by_name(fixed_label_names)
    moving_by_name = _labels_by_name(moving_label_names)
    out: dict[str, dict[str, Any]] = {}
    wanted = set(organs) if organs else None
    f_arr = sitk.GetArrayViewFromImage(fixed_labelmap)
    w_arr = sitk.GetArrayViewFromImage(warped_labelmap)
    names = sorted(wanted or (set(fixed_by_name) | set(moving_by_name)))
    for name in names:
        if wanted is not None and name not in wanted:
            continue
        fixed_labels = fixed_by_name.get(name, [])
        moving_labels = moving_by_name.get(name, [])
        fixed_present = any(np.any(f_arr == label) for label in fixed_labels)
        moving_present = any(np.any(w_arr == label) for label in moving_labels)
        if not fixed_present or not moving_present:
            missing = "fixed" if not fixed_present else "moving"
            out[name] = {"dice": 0.0, "available": False, "reason": f"organ missing from {missing}"}
            continue
        m_fixed = _union_labels(fixed_labelmap, fixed_labels)
        m_warp = _union_labels(warped_labelmap, moving_labels)
        entry: dict[str, Any] = {"dice": _round(dice(m_fixed, m_warp)), "available": True}
        if with_surface:
            try:
                d_ab, d_ba = _surface_distances(m_fixed, m_warp)
            except ValueError:
                d_ab = d_ba = np.asarray([], dtype=float)
            entry["hd95_mm"] = (
                _round(float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))), 2)
                if d_ab.size and d_ba.size
                else None
            )
            entry["msd_mm"] = (
                _round(float((d_ab.mean() + d_ba.mean()) / 2.0), 2) if d_ab.size and d_ba.size else None
            )
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #
def target_registration_error(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    transform_fixed_to_moving=None,
    *,
    mapped_points: np.ndarray | None = None,
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
    if (transform_fixed_to_moving is None) == (mapped_points is None):
        raise ValueError("provide exactly one of transform_fixed_to_moving or mapped_points")
    if mapped_points is not None:
        mapped = np.asarray(mapped_points, dtype=float).reshape(-1, 3)
    elif callable(transform_fixed_to_moving) and not hasattr(transform_fixed_to_moving, "TransformPoint"):
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
def jacobian_statistics(displacement_field: sitk.Image, mask: sitk.Image | None = None) -> dict[str, Any]:
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
        if not same_grid(displacement_field, mask):
            return {"available": False, "reason": "mask grid incompatible with displacement field"}
        m = sitk.GetArrayViewFromImage(mask).astype(bool)
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


def displacement_statistics(displacement_field: sitk.Image, mask: sitk.Image | None = None) -> dict[str, Any]:
    """Displacement magnitude (mm): compare against the expected physiological motion."""
    arr = sitk.GetArrayFromImage(displacement_field).astype(np.float64)
    magnitude = np.linalg.norm(arr, axis=-1)
    if mask is not None:
        if not same_grid(displacement_field, mask):
            return {"available": False, "reason": "mask grid incompatible with displacement field"}
        m = sitk.GetArrayViewFromImage(mask).astype(bool)
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


def linear_jacobian_statistics(matrix: np.ndarray, n_voxels: int) -> dict[str, Any]:
    """Exact constant Jacobian statistics for a 4x4 affine transform."""
    determinant = float(np.linalg.det(np.asarray(matrix, dtype=float)[:3, :3]))
    folding = int(n_voxels if determinant <= 0 else 0)
    value = _round(determinant, 4)
    return {
        "available": True,
        "analytic": True,
        "n_voxels": int(n_voxels),
        "det_min": value,
        "det_max": value,
        "det_mean": value,
        "det_std": 0.0,
        "det_p1": value,
        "det_p99": value,
        "folding_voxels": folding,
        "folding_fraction": 1.0 if determinant <= 0 else 0.0,
    }


def linear_displacement_statistics(
    matrix: np.ndarray,
    reference: sitk.Image,
    mask: sitk.Image | None = None,
    max_samples: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Displacement magnitudes of an affine without allocating a dense vector field."""
    if mask is not None and not same_grid(reference, mask):
        return {"available": False, "reason": "mask grid incompatible with reference"}
    if mask is None:
        size = np.asarray(reference.GetSize(), dtype=int)
        count = int(np.prod(size))
        rng = np.random.default_rng(seed)
        n = min(count, max_samples)
        flat = rng.choice(count, size=n, replace=False) if n < count else np.arange(count)
        x = flat % size[0]
        y = (flat // size[0]) % size[1]
        z = flat // (size[0] * size[1])
        indices = np.column_stack((x, y, z)).astype(np.float64)
    else:
        indices = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)[:, ::-1].astype(np.float64)
        count = len(indices)
        if count > max_samples:
            rng = np.random.default_rng(seed)
            indices = indices[rng.choice(count, size=max_samples, replace=False)]
    if not len(indices):
        return {"available": False}
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.asarray(reference.GetOrigin(), dtype=float)
    points = origin + (indices * spacing) @ direction.T
    affine = np.asarray(matrix, dtype=float)
    mapped = points @ affine[:3, :3].T + affine[:3, 3]
    magnitude = np.linalg.norm(mapped - points, axis=1)
    return {
        "available": True,
        "analytic": True,
        "n_samples": int(len(magnitude)),
        "n_voxels_available": int(count),
        "mean_mm": _round(float(magnitude.mean()), 3),
        "p95_mm": _round(float(np.percentile(magnitude, 95)), 3),
        "max_mm": _round(float(magnitude.max()), 3),
    }
