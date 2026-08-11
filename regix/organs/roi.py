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
from typing import Any, Literal

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
    same_grid,
)

log = get_logger("organs.roi")


@dataclass(frozen=True)
class Mask:
    """A mask together with the role that determines whether dilation is valid."""

    image: sitk.Image
    role: Literal["criterion", "initialization", "qc", "body"]
    dilated_mm: float = 0.0
    source: str = "unknown"


def combined_mask(
    volume: Volume,
    segmentation: OrganSegmentation | None,
    targets: Sequence[str] | None = None,
    dilate_mm: float = 8.0,
    fallback_body_mask: bool = True,
    role: Literal["criterion", "initialization", "qc", "body"] = "criterion",
) -> Mask | None:
    """Criterion mask for elastix.

    Priority: target organs -> all available labels -> body mask (threshold plus
    morphology) -> no mask.
    """
    if segmentation is not None:
        seg = (
            segmentation
            if same_grid(segmentation.labelmap, volume.image)
            else segmentation.resampled_to(volume.image)
        )
        wanted = resolve_targets(list(targets)) if targets else None
        try:
            mask = seg.mask_for(wanted)
        except ValueError:
            if not fallback_body_mask:
                return None
            log.warning("target organs missing: falling back to the native body mask")
            return Mask(
                image=body_mask(volume.image, volume.modality),
                role=role,
                dilated_mm=0.0,
                source="native-body-mask-fallback",
            )
        image = dilate_mask_mm(mask, dilate_mm) if dilate_mm > 0 else mask
        return Mask(image=image, role=role, dilated_mm=max(0.0, dilate_mm), source="segmentation")

    if not fallback_body_mask:
        return None
    log.debug("no segmentation: automatic body mask")
    return Mask(
        image=body_mask(volume.image, volume.modality),
        role=role,
        dilated_mm=0.0,
        source="native-body-mask",
    )


def organ_centroids(
    segmentation: OrganSegmentation, organs: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Physical centroid (mm) of each requested and non-empty organ."""
    lm = segmentation.labelmap
    arr = sitk.GetArrayViewFromImage(lm)
    wanted = resolve_targets(list(organs)) if organs else segmentation.organs
    out: dict[str, np.ndarray] = {}
    for organ in wanted:
        labels = segmentation.labels_of(organ)
        if not labels:
            continue
        idx = np.argwhere(np.isin(arr, labels))
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
            out[name] = out.get(name, 0.0) + count * voxel_ml
    return {name: round(value, 2) for name, value in out.items()}


@dataclass
class OrganROI:
    """Volumes cropped to the region of interest, with the means to go back."""

    fixed: Volume
    moving: Volume
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
        seg = seg if same_grid(seg.labelmap, volume.image) else seg.resampled_to(volume.image)
        try:
            mask = seg.mask_for(resolved)
        except ValueError as exc:
            log.warning("%s: %s -> no cropping on this side", side, exc)
            return volume
        try:
            cropped, _ = crop_to_mask(volume.image, mask, margin)
        except ValueError as exc:
            log.warning("%s: %s", side, exc)
            return volume
        out[f"{side}_size_before"] = list(volume.size)
        out[f"{side}_size_after"] = list(cropped.GetSize())
        return volume.with_image(cropped)

    f_vol = _crop(fixed, fixed_segmentation, "fixed")
    m_vol = _crop(moving, moving_segmentation, "moving")

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
    return OrganROI(fixed=f_vol, moving=m_vol, info=out)


def roi_overlap_report(
    fixed: Volume, moving: Volume, fixed_mask: sitk.Image | None, moving_mask: sitk.Image | None
) -> dict[str, Any]:
    """Field-of-view overlap before registration.

    A low overlap is the primary cause of silent failure: better to measure and
    report it than to let the optimiser find an absurd local minimum.
    """
    f_box = _physical_box(fixed.image)
    m_box = _physical_box(moving.image)
    report: dict[str, Any] = {
        "fixed_extent_mm": [round(v, 1) for v in (f_box[1] - f_box[0])],
        "moving_extent_mm": [round(v, 1) for v in (m_box[1] - m_box[0])],
        "fov_overlap_fraction_fixed": round(_fraction_inside(fixed.image, moving.image), 3),
        "fov_overlap_fraction_moving": round(_fraction_inside(moving.image, fixed.image), 3),
        "overlap_method": "deterministic Monte Carlo in oriented grids",
    }
    for name, mask in (("fixed", fixed_mask), ("moving", moving_mask)):
        if mask is not None:
            arr = sitk.GetArrayViewFromImage(mask)
            report[f"{name}_mask_ml"] = round(
                float(arr.sum()) * float(np.prod(mask.GetSpacing())) / 1000.0, 1
            )
    return report


def _fraction_inside(source: sitk.Image, target: sitk.Image, n: int = 8192) -> float:
    """Fraction of an oriented source grid that lies inside an oriented target grid."""
    if source.GetDimension() != 3 or target.GetDimension() != 3:
        return 0.0
    rng = np.random.default_rng(20250101)
    source_size = np.maximum(np.asarray(source.GetSize(), dtype=float) - 1.0, 0.0)
    source_idx = rng.random((n, 3)) * source_size
    source_scaled = source_idx * np.asarray(source.GetSpacing(), dtype=float)
    source_direction = np.asarray(source.GetDirection(), dtype=float).reshape(3, 3)
    points = np.asarray(source.GetOrigin(), dtype=float) + source_scaled @ source_direction.T

    target_direction = np.asarray(target.GetDirection(), dtype=float).reshape(3, 3)
    target_scaled = np.linalg.solve(
        target_direction,
        (points - np.asarray(target.GetOrigin(), dtype=float)).T,
    ).T
    target_idx = target_scaled / np.asarray(target.GetSpacing(), dtype=float)
    upper = np.asarray(target.GetSize(), dtype=float) - 1.0
    inside = np.all((target_idx >= 0.0) & (target_idx <= upper), axis=1)
    return float(np.mean(inside))


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
