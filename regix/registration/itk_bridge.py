"""Bridge between SimpleITK (used everywhere else in Regix) and ITK (the elastix binding).

Why two libraries? Because of how the ecosystem actually is:

* **SimpleITK** is the comfortable tool for DICOM I/O, morphology, statistics and
  transforms. But since SimpleITK 2.x the official wheels **no longer bundle
  elastix**: ``SimpleElastix`` is not distributed for recent Python versions.
  Check it rather than assume it -- ``hasattr(SimpleITK, "ElastixImageFilter")``
  is ``False``.
* **itk-elastix** is the binding maintained by the elastix authors. It provides
  ``ElastixRegistrationMethod`` and ``TransformixFilter``, with multi-metric
  multi-channel registration, masks and ``-t0`` chaining.

Regix therefore keeps SimpleITK as its working type and only converts to ITK for
the elastix call. The conversions explicitly preserve spacing, origin **and
direction cosines**: forgetting the direction is the classic mistake that makes
a registration silently fail on oblique slices.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("registration.itk_bridge")

_ENGINE_HINT = (
    "itk-elastix is required as the registration engine: pip install itk-elastix. "
    "(SimpleITK has not bundled elastix since version 2.x.)"
)


def require_itk():
    """Import itk-elastix or raise an explicit error."""
    try:
        import itk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_ENGINE_HINT) from exc
    if not hasattr(itk, "ElastixRegistrationMethod"):  # pragma: no cover
        raise RuntimeError(_ENGINE_HINT)
    return itk


def engine_available() -> tuple[bool, str]:
    try:
        itk = require_itk()
    except RuntimeError as exc:
        return False, str(exc)
    return True, f"itk {itk.Version.GetITKVersion()}"


def image_types():
    """(3D float image type, 3D uchar image type, instantiated ElastixRegistrationMethod)."""
    itk = require_itk()
    image_f = itk.Image[itk.F, 3]
    image_uc = itk.Image[itk.UC, 3]
    return image_f, image_uc, itk.ElastixRegistrationMethod[image_f, image_f]


def sitk_to_itk(image: sitk.Image, as_mask: bool = False):
    """SimpleITK -> ITK, full geometry preserved."""
    itk = require_itk()
    if image.GetDimension() != 3:
        raise ValueError(f"expected a 3D volume, got {image.GetDimension()}D")
    array = sitk.GetArrayFromImage(image)
    if as_mask:
        array = (array > 0).astype(np.uint8)
    else:
        array = array.astype(np.float32, copy=False)
    out = itk.GetImageFromArray(np.ascontiguousarray(array))
    out.SetSpacing([float(v) for v in image.GetSpacing()])
    out.SetOrigin([float(v) for v in image.GetOrigin()])
    out.SetDirection(itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)))
    return out


def itk_to_sitk(image, is_vector: bool = False) -> sitk.Image:
    """ITK -> SimpleITK, full geometry preserved."""
    itk = require_itk()
    array = itk.GetArrayFromImage(image)
    out = sitk.GetImageFromArray(np.ascontiguousarray(array), isVector=is_vector)
    out.SetSpacing([float(v) for v in image.GetSpacing()])
    out.SetOrigin([float(v) for v in image.GetOrigin()])
    out.SetDirection([float(v) for v in np.asarray(itk.array_from_matrix(image.GetDirection())).reshape(-1)])
    return out


def itk_transform_to_sitk(itk_transform, work_dir: str | Path | None = None) -> sitk.Transform:
    """Convert an ITK transform (including composite / B-spline) to a ``sitk.Transform``.

    Going through an HDF5 file is deliberate: it is the only path that exactly
    preserves a heterogeneous composition (Euler + affine + B-spline). Verified
    experimentally -- zero discrepancy on the probed points.
    """
    itk = require_itk()
    directory = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="regix_tf_"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "combined_transform.h5"
    itk.transformwrite([itk_transform], str(path))
    transform = sitk.ReadTransform(str(path))

    # Regression guard: the conversion must be exact.
    probes = [(0.0, 0.0, 0.0), (57.0, -31.0, 93.0), (-42.0, 68.0, -17.0)]
    errors = []
    for probe in probes:
        try:
            a = np.asarray(transform.TransformPoint(probe), dtype=float)
            b = np.asarray(itk_transform.TransformPoint(list(probe)), dtype=float)
            errors.append(float(np.linalg.norm(a - b)))
        except Exception:  # pragma: no cover - point outside the B-spline support
            continue
    if errors and max(errors) > 1e-3:
        log.warning(
            "ITK -> SimpleITK conversion is imprecise (max discrepancy %.4f mm): "
            "outputs will be computed by transformix instead",
            max(errors),
        )
        raise ValueError(f"unreliable transform conversion ({max(errors):.4f} mm discrepancy)")
    log.debug(
        "transform converted: %s (max discrepancy %.2e mm)",
        type(transform).__name__,
        max(errors) if errors else 0.0,
    )
    return transform
