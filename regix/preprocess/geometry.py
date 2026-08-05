"""Geometry: orientation, resampling, body masks, ROI cropping.

Two invariants hold throughout Regix:

1. the final output grid is **always** that of the original fixed image; the
   working resolution is only an optimisation accelerator;
2. a mask is never resampled with anything other than nearest neighbour (or
   linear interpolation followed by a 0.5 threshold for fuzzy masks), otherwise
   labels that do not exist get invented.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("preprocess.geometry")

_INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "gaussian": sitk.sitkGaussian,
    "lanczos": sitk.sitkLanczosWindowedSinc,
}


def interpolator_by_name(name: str) -> int:
    if name not in _INTERPOLATORS:
        raise ValueError(f"unknown interpolator '{name}': {sorted(_INTERPOLATORS)}")
    return _INTERPOLATORS[name]


def reorient(image: sitk.Image, code: str = "LPS") -> sitk.Image:
    """Reorient the axes to an anatomical code (permutations/flips only).

    No interpolation: this is a re-indexing. Useful so that numpy axes have a
    stable meaning in visualisations and networks.
    """
    if image.GetDimension() != 3:
        return image
    try:
        return sitk.DICOMOrient(image, code.upper())
    except Exception as exc:  # pragma: no cover
        log.warning("could not reorient to %s (%s): image left unchanged", code, exc)
        return image


def resample_to_spacing(
    image: sitk.Image,
    spacing: float | tuple[float, float, float],
    interpolator: str = "linear",
    default_value: float | None = None,
    is_mask: bool = False,
) -> sitk.Image:
    """Resample to a target spacing while preserving the physical extent."""
    target = (float(spacing),) * 3 if np.isscalar(spacing) else tuple(float(s) for s in spacing)  # type: ignore[arg-type]
    src_spacing = np.asarray(image.GetSpacing(), dtype=float)
    src_size = np.asarray(image.GetSize(), dtype=float)
    if np.allclose(src_spacing, target, atol=1e-4):
        return image

    new_size = np.maximum(1, np.round(src_size * src_spacing / np.asarray(target))).astype(int)
    interp = sitk.sitkNearestNeighbor if is_mask else interpolator_by_name(interpolator)
    if default_value is None:
        default_value = 0.0 if is_mask else _background_value(image)

    out = sitk.Resample(
        image,
        [int(v) for v in new_size],
        sitk.Transform(),
        interp,
        image.GetOrigin(),
        [float(v) for v in target],
        image.GetDirection(),
        float(default_value),
        image.GetPixelID(),
    )
    log.debug(
        "resampled %s @ %s -> %s @ %s",
        tuple(int(v) for v in src_size),
        tuple(round(v, 3) for v in src_spacing),
        tuple(out.GetSize()),
        tuple(round(v, 3) for v in out.GetSpacing()),
    )
    return out


def resample_like(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform | None = None,
    interpolator: str = "linear",
    default_value: float | None = None,
    is_mask: bool = False,
) -> sitk.Image:
    """Resample ``image`` onto the grid of ``reference``.

    ``transform`` follows the ITK convention: it maps points of the output grid
    (reference) into the space of ``image``. That is also the elastix convention,
    which avoids any inversion here.
    """
    interp = sitk.sitkNearestNeighbor if is_mask else interpolator_by_name(interpolator)
    if default_value is None:
        default_value = 0.0 if is_mask else _background_value(image)
    return sitk.Resample(
        image,
        reference,
        transform if transform is not None else sitk.Transform(),
        interp,
        float(default_value),
        image.GetPixelID(),
    )


def _background_value(image: sitk.Image) -> float:
    """Out-of-field fill value: the image minimum (air in CT, 0 otherwise)."""
    try:
        f = sitk.MinimumMaximumImageFilter()
        f.Execute(sitk.Cast(image, sitk.sitkFloat32))
        return float(f.GetMinimum())
    except Exception:  # pragma: no cover
        return 0.0


def pad_to_multiple(array: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Pad a 3D array so that every dimension is a multiple of ``multiple``.

    Required by U-Nets with 4 downsampling levels (anatomix). Returns the padded
    array and the padding applied, so it can be cropped back.
    """
    pads = []
    for dim in array.shape[-3:]:
        extra = (-dim) % multiple
        pads.append((extra // 2, extra - extra // 2))
    full_pads = ((0, 0),) * (array.ndim - 3) + tuple(pads)
    if not any(sum(p) for p in pads):
        return array, tuple(pads)
    return np.pad(array, full_pads, mode="edge"), tuple(pads)


def unpad(array: np.ndarray, pads: tuple[tuple[int, int], ...]) -> np.ndarray:
    slices = [slice(None)] * (array.ndim - 3)
    for lo, hi in pads:
        slices.append(slice(lo, array.shape[len(slices)] - hi if hi else None))
    return array[tuple(slices)]


def body_mask(
    image: sitk.Image,
    modality: str = "CT",
    closing_radius_mm: float = 5.0,
    keep_largest: bool = True,
) -> sitk.Image:
    """Coarse patient mask (excludes air and the table).

    Useful by default: masking the criterion to the body markedly improves
    rigid/affine registration when the fields of view differ (typically whole-body
    CT against abdominal MR).
    """
    img = sitk.Cast(image, sitk.sitkFloat32)
    if modality.upper() in ("CT", "CBCT"):
        f = sitk.MinimumMaximumImageFilter()
        f.Execute(img)
        # -300 HU separates air from tissue correctly; if the image has already
        # been normalised, fall back to Otsu.
        mask = (
            sitk.BinaryThreshold(img, -300.0, 4000.0, 1, 0)
            if f.GetMinimum() < -200
            else sitk.OtsuThreshold(img, 0, 1, 128)
        )
    else:
        mask = sitk.OtsuThreshold(img, 0, 1, 128)

    radius = [max(1, int(round(closing_radius_mm / s))) for s in image.GetSpacing()]
    mask = sitk.BinaryMorphologicalClosing(mask, radius, sitk.sitkBall)
    mask = sitk.BinaryFillhole(mask)
    if keep_largest:
        mask = keep_largest_component(mask)
    volume_ml = float(sitk.GetArrayViewFromImage(mask).sum()) * float(np.prod(image.GetSpacing())) / 1000.0
    log.debug("body mask: %.0f mL", volume_ml)
    return sitk.Cast(mask, sitk.sitkUInt8)


def keep_largest_component(mask: sitk.Image) -> sitk.Image:
    labelled = sitk.ConnectedComponent(sitk.Cast(mask, sitk.sitkUInt8))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labelled)
    labels = stats.GetLabels()
    if not labels:
        return sitk.Cast(mask, sitk.sitkUInt8)
    biggest = max(labels, key=stats.GetNumberOfPixels)
    return sitk.Cast(sitk.Equal(labelled, biggest), sitk.sitkUInt8)


def dilate_mask_mm(mask: sitk.Image, millimeters: float) -> sitk.Image:
    """Isotropic dilation in millimetres (the radius is converted per axis)."""
    if millimeters <= 0:
        return sitk.Cast(mask, sitk.sitkUInt8)
    radius = [max(1, int(round(millimeters / s))) for s in mask.GetSpacing()]
    dilated = sitk.BinaryDilate(sitk.Cast(mask, sitk.sitkUInt8), radius, sitk.sitkBall)
    return sitk.Cast(dilated, sitk.sitkUInt8)


def erode_mask_mm(mask: sitk.Image, millimeters: float) -> sitk.Image:
    if millimeters <= 0:
        return sitk.Cast(mask, sitk.sitkUInt8)
    radius = [max(1, int(round(millimeters / s))) for s in mask.GetSpacing()]
    return sitk.Cast(sitk.BinaryErode(sitk.Cast(mask, sitk.sitkUInt8), radius, sitk.sitkBall), sitk.sitkUInt8)


def mask_bounding_box_mm(mask: sitk.Image, margin_mm: float = 0.0) -> tuple[list[int], list[int]]:
    """Bounding box of a mask, in index space, enlarged by ``margin_mm``.

    Returns (start index, size), usable by ``sitk.RegionOfInterest``.
    """
    m = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(m)
    if 1 not in stats.GetLabels():
        raise ValueError("empty mask: no bounding box")
    bbox = stats.GetBoundingBox(1)  # (x, y, z, sx, sy, sz)
    ndim = m.GetDimension()
    start = list(bbox[:ndim])
    size = list(bbox[ndim:])
    spacing = m.GetSpacing()
    image_size = m.GetSize()
    for axis in range(ndim):
        pad = int(round(margin_mm / spacing[axis]))
        new_start = max(0, start[axis] - pad)
        new_end = min(image_size[axis], start[axis] + size[axis] + pad)
        start[axis] = new_start
        size[axis] = max(1, new_end - new_start)
    return start, size


def crop_to_mask(
    image: sitk.Image, mask: sitk.Image, margin_mm: float = 20.0
) -> tuple[sitk.Image, tuple[list[int], list[int]]]:
    """Crop ``image`` to the mask bounding box. Physical geometry is preserved."""
    if not _same_grid(image, mask):
        mask = resample_like(mask, image, is_mask=True)
    start, size = mask_bounding_box_mm(mask, margin_mm)
    cropped = sitk.RegionOfInterest(image, size, start)
    log.debug("cropped %s -> %s (%.0f mm margin)", image.GetSize(), cropped.GetSize(), margin_mm)
    return cropped, (start, size)


def _same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def image_center_physical(image: sitk.Image) -> np.ndarray:
    """Geometric centre of the grid, in physical coordinates (mm)."""
    size = np.asarray(image.GetSize(), dtype=float)
    return np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(((size - 1.0) / 2.0).tolist()), dtype=float
    )


def center_of_mass_physical(
    image: sitk.Image, mask: sitk.Image | None = None, threshold: float | None = None
) -> np.ndarray:
    """Intensity-weighted centre of mass (or the barycentre of a mask).

    Negative intensities (HU) are shifted so they remain valid weights.
    """
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32)).astype(np.float64)
    if mask is not None:
        if not _same_grid(image, mask):
            mask = resample_like(mask, image, is_mask=True)
        weights = (sitk.GetArrayViewFromImage(mask) > 0).astype(np.float64)
        arr = weights
    else:
        if threshold is not None:
            arr = np.where(arr >= threshold, arr, 0.0)
        vmin = arr.min()
        if vmin < 0:
            arr = arr - vmin
    total = arr.sum()
    if total <= 0:
        log.warning("zero weights for the centre of mass: falling back to the geometric centre")
        return image_center_physical(image)
    grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in arr.shape], indexing="ij")
    com_zyx = [float((g * arr).sum() / total) for g in grids]
    index_xyz = com_zyx[::-1]  # numpy (z, y, x) -> ITK (x, y, z)
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(index_xyz), dtype=float)


def principal_axes(mask: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentre, principal axes (as columns) and inertia lengths of a mask.

    The basis of organ-moment initialization: the organ's inertia is aligned
    before the optimiser is even started.
    """
    m = sitk.Cast(mask, sitk.sitkUInt8)
    arr = sitk.GetArrayViewFromImage(m)
    idx = np.argwhere(arr > 0)
    if idx.size == 0:
        raise ValueError("empty mask: principal axes undefined")
    points = np.asarray(
        [m.TransformContinuousIndexToPhysicalPoint([float(i[2]), float(i[1]), float(i[0])]) for i in idx],
        dtype=np.float64,
    )
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = centred.T @ centred / max(1, len(points) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order], np.sqrt(np.maximum(eigvals[order], 0.0))
