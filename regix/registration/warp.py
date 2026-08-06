"""Uniform application of the final transform, whatever the engine.

The pipeline must not need to know whether the transform came from elastix (a
chained parameter file) or from an ITK composition (linear + dense field from
instance optimisation). ``AppliedTransform`` therefore exposes a single
interface: resample an image, resample a label map, materialise a displacement
field, transform points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger
from regix.preprocess.geometry import resample_like
from regix.registration.convexadam import displacement_field_from_transform
from regix.registration.engine import apply_transform

log = get_logger("registration.warp")


class AppliedTransform(ABC):
    """Final transform, ready to be applied. Convention: fixed -> moving."""

    kind: str = "abstract"

    @abstractmethod
    def resample(
        self,
        moving: sitk.Image,
        reference: sitk.Image,
        is_label: bool = False,
        default_value: float | None = None,
    ) -> sitk.Image: ...

    @abstractmethod
    def displacement_field(self, reference: sitk.Image) -> sitk.Image | None: ...

    @abstractmethod
    def transform_points(self, points: np.ndarray) -> np.ndarray | None:
        """Map points from the fixed frame to the moving frame (mm)."""

    def as_sitk_transform(self) -> sitk.Transform | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind}


# --------------------------------------------------------------------------- #
class ElastixAppliedTransform(AppliedTransform):
    """An elastix chain carried by a ``TransformParameters.0.txt`` file."""

    kind = "elastix"

    def __init__(
        self,
        parameter_file: str | Path,
        work_dir: str | Path | None = None,
        linear_transform: sitk.Transform | None = None,
    ):
        self.parameter_file = Path(parameter_file)
        if not self.parameter_file.exists():
            raise FileNotFoundError(self.parameter_file)
        self.work_dir = Path(work_dir) if work_dir else None
        self._linear = linear_transform

    def resample(self, moving, reference, is_label=False, default_value=None):
        result, _ = apply_transform(
            self.parameter_file,
            moving,
            reference=reference,
            is_label=is_label,
            default_value=default_value,
            work_dir=self.work_dir,
        )
        if is_label:
            return sitk.Cast(result, moving.GetPixelID())
        return result

    def displacement_field(self, reference):
        # Trick: transformix only computes the field when given a moving image;
        # we hand it the reference itself, since only the geometry matters.
        try:
            _, field = apply_transform(
                self.parameter_file,
                reference,
                reference=reference,
                work_dir=self.work_dir,
                compute_deformation_field=True,
            )
            return field
        except Exception as exc:
            log.warning("displacement field unavailable through transformix: %s", exc)
            if self._linear is not None:
                return displacement_field_from_transform(self._linear, reference)
            return None

    def transform_points(self, points):
        if self._linear is not None:
            return np.asarray(
                [self._linear.TransformPoint([float(v) for v in p]) for p in points], dtype=float
            )
        log.debug("non-linear point transport: going through the displacement field")
        return None

    def as_sitk_transform(self):
        return self._linear

    def describe(self):
        return {
            "kind": self.kind,
            "parameter_file": str(self.parameter_file),
            "linear_available": self._linear is not None,
        }


# --------------------------------------------------------------------------- #
class SitkAppliedTransform(AppliedTransform):
    """An ITK/SimpleITK transform (linear, dense, or a composition of both)."""

    kind = "sitk"

    def __init__(self, transform: sitk.Transform, label: str = "composite"):
        self.transform = transform
        self.label = label

    def resample(self, moving, reference, is_label=False, default_value=None):
        return resample_like(
            moving,
            reference,
            transform=self.transform,
            interpolator="nearest" if is_label else "linear",
            default_value=default_value,
            is_mask=is_label,
        )

    def displacement_field(self, reference):
        return displacement_field_from_transform(self.transform, reference)

    def transform_points(self, points):
        return np.asarray([self.transform.TransformPoint([float(v) for v in p]) for p in points], dtype=float)

    def as_sitk_transform(self):
        return self.transform

    def describe(self):
        return {"kind": self.kind, "label": self.label}


# --------------------------------------------------------------------------- #
def transform_points_via_field(displacement_field: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Map points fixed -> moving by interpolating the displacement field.

    An exact fallback when the transform is not linear: the field *is* the
    transform, we merely read ``u(p)`` from it by linear interpolation, and
    ``p_moving = p + u(p)``.
    """
    transform = sitk.DisplacementFieldTransform(sitk.Cast(displacement_field, sitk.sitkVectorFloat64))
    return np.asarray(
        [transform.TransformPoint([float(v) for v in p]) for p in np.asarray(points).reshape(-1, 3)],
        dtype=float,
    )


def warp_landmarks_moving_to_fixed(
    applied: AppliedTransform, points_moving: np.ndarray, reference: sitk.Image
) -> np.ndarray | None:
    """Map points from the moving frame to the fixed frame (inverse of the convention).

    For a linear transform this is an exact inversion. For a dense transform we do
    not invent an inverse: we return None rather than a silent approximation -- an
    invisible 3 mm error is more dangerous than no result at all.
    """
    t = applied.as_sitk_transform()
    if t is None:
        log.info("inversion unavailable for a non-linear transform")
        return None
    try:
        inverse = t.GetInverse()
    except Exception as exc:
        log.warning("inversion failed: %s", exc)
        return None
    return np.asarray([inverse.TransformPoint([float(v) for v in p]) for p in points_moving], dtype=float)
