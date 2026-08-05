"""The ``Volume`` type: a SimpleITK image plus the clinical context that goes with it.

Regix never manipulates a bare numpy array: geometry (origin, spacing,
direction) is the only thing that makes a registration correct, and it is
precisely what an ndarray loses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger, pseudonymize

log = get_logger("io.volume")

_IMAGE_SUFFIXES = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd", ".img", ".hdr", ".gipl", ".vtk"}


@dataclass
class Volume:
    """A 3D volume and the metadata relevant to registration."""

    image: sitk.Image
    modality: str = "UNKNOWN"
    role: str = "image"                       # image | mask | labelmap | features
    source: Path | None = None
    subject_id: str = "unknown"               # pseudonymised when runtime.pseudonymize
    series_uid: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- geometry --------------------------------------------------------- #
    @property
    def size(self) -> tuple[int, ...]:
        return tuple(self.image.GetSize())

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(self.image.GetSpacing())

    @property
    def origin(self) -> tuple[float, ...]:
        return tuple(self.image.GetOrigin())

    @property
    def direction(self) -> tuple[float, ...]:
        return tuple(self.image.GetDirection())

    @property
    def physical_extent_mm(self) -> tuple[float, ...]:
        return tuple(s * sp for s, sp in zip(self.size, self.spacing, strict=False))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.size))

    def array(self, dtype=None) -> np.ndarray:
        """Numpy view in reversed ITK order (z, y, x). Use for statistics only."""
        arr = sitk.GetArrayFromImage(self.image)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def same_grid_as(self, other: Volume, tol: float = 1e-4) -> bool:
        return (
            self.size == other.size
            and np.allclose(self.spacing, other.spacing, atol=tol)
            and np.allclose(self.origin, other.origin, atol=tol)
            and np.allclose(self.direction, other.direction, atol=tol)
        )

    # -- non-destructive transformations ---------------------------------- #
    def with_image(self, image: sitk.Image, **overrides: Any) -> Volume:
        return replace(self, image=image, **overrides)

    def cast(self, pixel_type=sitk.sitkFloat32) -> Volume:
        if self.image.GetPixelID() == pixel_type:
            return self
        return self.with_image(sitk.Cast(self.image, pixel_type))

    def describe(self) -> dict[str, Any]:
        arr = self.array(np.float32)
        finite = arr[np.isfinite(arr)]
        return {
            "subject_id": self.subject_id,
            "modality": self.modality,
            "role": self.role,
            "size": list(self.size),
            "spacing_mm": [round(s, 4) for s in self.spacing],
            "extent_mm": [round(s, 1) for s in self.physical_extent_mm],
            "origin": [round(o, 3) for o in self.origin],
            "orientation": orientation_code(self.image),
            "intensity": {
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "p1": float(np.percentile(finite, 1)) if finite.size else None,
                "p99": float(np.percentile(finite, 99)) if finite.size else None,
                "nan_voxels": int(np.count_nonzero(~np.isfinite(arr))),
            },
            "source": str(self.source) if self.source else None,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Volume({self.modality}, size={self.size}, "
            f"spacing={tuple(round(s, 2) for s in self.spacing)}, "
            f"orient={orientation_code(self.image)})"
        )


def orientation_code(image: sitk.Image) -> str:
    """Three-letter anatomical orientation code (e.g. 'LPS', 'RAS')."""
    try:
        return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
    except Exception:  # pragma: no cover - older SimpleITK versions
        return "???"


def load_volume(
    path: str | Path,
    modality: str | None = None,
    role: str = "image",
    pseudonymize_ids: bool = True,
    salt: str | None = None,
) -> Volume:
    """Load a volume from an image file or a DICOM directory.

    The format is inferred: a directory -> DICOM series, otherwise the ITK reader.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"volume not found: {p}")

    if p.is_dir():
        from regix.io.dicom import list_series, load_series

        series = list_series(p)
        if not series:
            raise ValueError(f"no DICOM series found in {p}")
        if len(series) > 1:
            log.warning(
                "%d DICOM series in %s; selecting the largest one (%s, %d slices). "
                "Use `regix inspect` and pass the UID explicitly to remove the ambiguity.",
                len(series),
                p,
                series[0].modality,
                series[0].n_files,
            )
        return load_series(series[0], pseudonymize_ids=pseudonymize_ids, salt=salt, role=role)

    image = sitk.ReadImage(str(p))
    if image.GetDimension() == 4:
        log.warning("4D volume detected (%s): extracting the first time point", p.name)
        image = image[..., 0]
    if image.GetNumberOfComponentsPerPixel() > 1 and role != "features":
        raise ValueError(
            f"{p.name} has {image.GetNumberOfComponentsPerPixel()} components per voxel; "
            "Regix expects a scalar volume"
        )

    subject = p.name.split(".")[0]
    return Volume(
        image=image,
        modality=(modality or _modality_from_metadata(image) or "UNKNOWN").upper(),
        role=role,
        source=p,
        subject_id=pseudonymize(subject, salt) if pseudonymize_ids else subject,
        meta={"loader": "itk", "file": p.name},
    )


def _modality_from_metadata(image: sitk.Image) -> str | None:
    for key in ("0008|0060", "modality", "Modality"):
        if image.HasMetaDataKey(key):
            value = image.GetMetaData(key).strip()
            if value:
                return value
    return None
