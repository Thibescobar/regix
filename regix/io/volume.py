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


@dataclass
class Volume:
    """A 3D volume and the metadata relevant to registration."""

    image: sitk.Image
    modality: str = "UNKNOWN"
    role: str = "image"  # image | mask | labelmap | features
    source: Path | None = None
    subject_id: str = "unknown"  # pseudonymised when runtime.pseudonymize
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

    # -- non-destructive transformations ---------------------------------- #
    def with_image(self, image: sitk.Image, **overrides: Any) -> Volume:
        return replace(self, image=image, meta=dict(self.meta), **overrides)

    def cast(self, pixel_type=sitk.sitkFloat32) -> Volume:
        if self.image.GetPixelID() == pixel_type:
            return self
        return self.with_image(sitk.Cast(self.image, pixel_type))

    def describe(self) -> dict[str, Any]:
        arr = sitk.GetArrayViewFromImage(self.image)
        finite_mask = np.isfinite(arr)
        n_bad = int(arr.size - np.count_nonzero(finite_mask))
        if n_bad:
            finite = np.asarray(arr[finite_mask], dtype=np.float32)
            minimum = float(finite.min()) if finite.size else None
            maximum = float(finite.max()) if finite.size else None
        else:
            finite = np.asarray(arr).reshape(-1)
            stats = sitk.StatisticsImageFilter()
            stats.Execute(self.image)
            minimum = float(stats.GetMinimum())
            maximum = float(stats.GetMaximum())
        max_samples = 1_000_000
        if finite.size > max_samples:
            rng = np.random.default_rng(20250101)
            sample = finite[rng.choice(finite.size, size=max_samples, replace=False)]
        else:
            sample = finite
        percentiles = np.percentile(sample, (1, 99)) if sample.size else (None, None)
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
                "min": minimum,
                "max": maximum,
                "p1": float(percentiles[0]) if sample.size else None,
                "p99": float(percentiles[1]) if sample.size else None,
                "nan_voxels": n_bad,
                "percentile_sample_voxels": int(sample.size),
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
    series_uid: str | None = None,
) -> Volume:
    """Load a volume from an image file or a DICOM directory.

    The format is inferred: a directory -> DICOM series, otherwise the ITK reader.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"volume not found: {p}")

    if p.is_dir():
        from regix.io.dicom import list_series, load_series

        candidates = list_series(p)
        if not candidates:
            raise ValueError(f"no DICOM series found in {p}")
        if series_uid is not None:
            matches = [series for series in candidates if series.series_uid == series_uid]
            if not matches:
                raise ValueError(f"DICOM series {series_uid} is not present in {p}")
            selected = matches[0]
        else:
            selected = candidates[0]
        if len(candidates) > 1 and series_uid is None:
            log.warning(
                "%d DICOM series in %s; selecting the largest one (%s, %d slices). "
                "Use `regix inspect` and pass the UID explicitly to remove the ambiguity.",
                len(candidates),
                p,
                selected.modality,
                selected.n_files,
            )
        volume = load_series(selected, pseudonymize_ids=pseudonymize_ids, salt=salt, role=role)
        if len(candidates) > 1 and series_uid is None:
            volume.meta["series_ambiguity"] = {
                "selected": selected.series_uid,
                "available": [series.series_uid for series in candidates],
            }
        return volume

    image = sitk.ReadImage(str(p))
    if image.GetDimension() == 4:
        log.warning("4D volume detected (%s): extracting the first time point", p.name)
        size = list(image.GetSize())
        size[3] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])
    if image.GetDimension() != 3:
        raise ValueError(
            f"{p.name} is {image.GetDimension()}D; Regix accepts 3D volumes and 4D volumes "
            "from which the first 3D time point can be extracted"
        )
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
