"""Criterion masks, organ centroids and ROI cropping.

Three distinct uses of masks, often conflated:

* **criterion mask** (``combined_mask``): where elastix samples. Dilated, because
  a mask that is too tight prevents the optimiser from "seeing" the misalignment;
* **initialization mask**: not dilated, used to compute centroids and principal
  axes;
* **computation ROI** (``plan_roi``): physical cropping of the volumes so that a
  whole-body registration is not paid for when only the pancreas matters.
  Typical gain: a factor of 5 to 20 in runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.io.volume import Volume
from regix.logging_utils import get_logger
from regix.organs.labels import merged_profile, resolve_targets
from regix.organs.segmenter import OrganSegmentation
from regix.preprocess.geometry import (
    body_mask,
    crop_to_mask,
    dilate_mask_mm,
)

log = get_logger("organs.roi")


def combined_mask(
    volume: Volume,
    segmentation: OrganSegmentation | None,
    targets: Sequence[str] | None = None,
    dilate_mm: float = 8.0,
    fallback_body_mask: bool = True,
) -> sitk.Image | None:
    """Criterion mask for elastix.

    Priority: target organs -> all available labels -> body mask (threshold plus
    morphology) -> no mask.
    """
    if segmentation is not None:
        seg = (
            segmentation
            if _same_grid(segmentation.labelmap, volume.image)
            else segmentation.resampled_to(volume.image)
        )
        wanted = resolve_targets(list(targets)) if targets else None
        try:
            mask = seg.mask_for(wanted)
        except ValueError:
            log.warning("target organs missing: falling back to the union of all labels")
            mask = seg.mask_for(None)
        return dilate_mask_mm(mask, dilate_mm) if dilate_mm > 0 else mask

    if not fallback_body_mask:
        return None
    log.debug("no segmentation: automatic body mask")
    return body_mask(volume.image, volume.modality)


def organ_centroids(
    segmentation: OrganSegmentation, organs: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Physical centroid (mm) of each requested and non-empty organ."""
    lm = segmentation.labelmap
    arr = sitk.GetArrayViewFromImage(lm)
    wanted = resolve_targets(list(organs)) if organs else segmentation.organs
    out: dict[str, np.ndarray] = {}
    for organ in wanted:
        label = segmentation.label_of(organ)
        if label is None:
            continue
        idx = np.argwhere(arr == label)
        if idx.size == 0:
            log.debug("organ %s is empty in the segmentation", organ)
            continue
        mean_zyx = idx.mean(axis=0)
        index_xyz = [float(mean_zyx[2]), float(mean_zyx[1]), float(mean_zyx[0])]
        out[organ] = np.asarray(lm.TransformContinuousIndexToPhysicalPoint(index_xyz), dtype=float)
    return out


def organ_volumes_ml(segmentation: OrganSegmentation) -> dict[str, float]:
    """Volume of each organ, in mL. Used for plausibility checks."""
    arr = sitk.GetArrayViewFromImage(segmentation.labelmap)
    voxel_ml = float(np.prod(segmentation.labelmap.GetSpacing())) / 1000.0
    out: dict[str, float] = {}
    for label, name in segmentation.label_names.items():
        count = int(np.count_nonzero(arr == label))
        if count:
            out[name] = round(count * voxel_ml, 2)
    return out


@dataclass
class OrganROI:
    """Volumes cropped to the region of interest, with the means to go back."""

    fixed: Volume
    moving: Volume
    fixed_region: tuple[list[int], list[int]] | None = None
    moving_region: tuple[list[int], list[int]] | None = None
    info: dict[str, Any] = field(default_factory=dict)


def plan_roi(
    fixed: Volume,
    moving: Volume,
    fixed_segmentation: OrganSegmentation | None,
    moving_segmentation: OrganSegmentation | None,
    targets: Sequence[str],
    margin_mm: float | None = None,
) -> OrganROI:
    """Crop both volumes around the target organs.

    Each volume is cropped to **its own** organ: that is what allows registering a
    whole-body CT against an abdominal MR without the optimiser drowning in the
    non-shared field of view. Physical coordinates are unchanged, so the resulting
    transform remains valid for the full volumes.
    """
    resolved = resolve_targets(list(targets))
    if not resolved:
        return OrganROI(fixed=fixed, moving=moving, info={"applied": False, "reason": "no target"})
    if fixed_segmentation is None or moving_segmentation is None:
        log.warning("ROI cropping requested but a segmentation is missing: skipped")
        return OrganROI(fixed=fixed, moving=moving, info={"applied": False, "reason": "missing segmentation"})

    margin = margin_mm if margin_mm is not None else merged_profile(resolved).roi_margin_mm
    out: dict[str, Any] = {"applied": True, "targets": resolved, "margin_mm": margin}

    def _crop(volume: Volume, seg: OrganSegmentation, side: str):
        seg = seg if _same_grid(seg.labelmap, volume.image) else seg.resampled_to(volume.image)
        try:
            mask = seg.mask_for(resolved)
        except ValueError as exc:
            log.warning("%s: %s -> no cropping on this side", side, exc)
            return volume, None
        try:
            cropped, region = crop_to_mask(volume.image, mask, margin)
        except ValueError as exc:
            log.warning("%s: %s", side, exc)
            return volume, None
        out[f"{side}_size_before"] = list(volume.size)
        out[f"{side}_size_after"] = list(cropped.GetSize())
        return volume.with_image(cropped), region

    f_vol, f_region = _crop(fixed, fixed_segmentation, "fixed")
    m_vol, m_region = _crop(moving, moving_segmentation, "moving")

    before = np.prod(fixed.size) + np.prod(moving.size)
    after = np.prod(f_vol.size) + np.prod(m_vol.size)
    out["speedup_estimate"] = round(float(before) / max(float(after), 1.0), 1)
    log.info(
        "ROI %s: fixed %s -> %s, moving %s -> %s (%.1fx fewer voxels)",
        resolved,
        fixed.size,
        f_vol.size,
        moving.size,
        m_vol.size,
        out["speedup_estimate"],
    )
    return OrganROI(fixed=f_vol, moving=m_vol, fixed_region=f_region, moving_region=m_region, info=out)


def roi_overlap_report(
    fixed: Volume, moving: Volume, fixed_mask: sitk.Image | None, moving_mask: sitk.Image | None
) -> dict[str, Any]:
    """Field-of-view overlap before registration.

    A low overlap is the primary cause of silent failure: better to measure and
    report it than to let the optimiser find an absurd local minimum.
    """
    f_box = _physical_box(fixed.image)
    m_box = _physical_box(moving.image)
    inter_lo = np.maximum(f_box[0], m_box[0])
    inter_hi = np.minimum(f_box[1], m_box[1])
    extent = np.maximum(inter_hi - inter_lo, 0.0)
    inter_vol = float(np.prod(extent))
    f_vol = float(np.prod(f_box[1] - f_box[0]))
    m_vol = float(np.prod(m_box[1] - m_box[0]))
    report: dict[str, Any] = {
        "fixed_extent_mm": [round(v, 1) for v in (f_box[1] - f_box[0])],
        "moving_extent_mm": [round(v, 1) for v in (m_box[1] - m_box[0])],
        "fov_overlap_fraction_fixed": round(inter_vol / f_vol, 3) if f_vol else 0.0,
        "fov_overlap_fraction_moving": round(inter_vol / m_vol, 3) if m_vol else 0.0,
    }
    for name, mask in (("fixed", fixed_mask), ("moving", moving_mask)):
        if mask is not None:
            arr = sitk.GetArrayViewFromImage(mask)
            report[f"{name}_mask_ml"] = round(
                float(arr.sum()) * float(np.prod(mask.GetSpacing())) / 1000.0, 1
            )
    if min(report["fov_overlap_fraction_fixed"], report["fov_overlap_fraction_moving"]) < 0.25:
        log.warning(
            "low field-of-view overlap (%.0f %% / %.0f %%): organ-based initialization "
            "is strongly recommended (init.mode=organ_centroid)",
            100 * report["fov_overlap_fraction_fixed"],
            100 * report["fov_overlap_fraction_moving"],
        )
    return report


def _physical_box(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """Physical bounding box of the grid (all 8 corners, direction included)."""
    size = np.asarray(image.GetSize(), dtype=float) - 1.0
    corners = []
    for ix in (0, size[0]):
        for iy in (0, size[1]):
            for iz in (0, size[2]):
                corners.append(image.TransformContinuousIndexToPhysicalPoint([ix, iy, iz]))
    pts = np.asarray(corners, dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def _same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )
