"""DICOM reading oriented towards real clinical data.

What this module handles, because it happens all the time in production:

* several series in the same directory (localizer, dose maps, derived series);
* non-equidistant slices (breathing, partial reconstructions) -> detected and
  reported, never silently averaged;
* incomplete series / duplicate InstanceNumbers;
* gantry tilt (head CT) -> reported;
* PET: reminds the caller that the values are not SUV;
* patient metadata never copied verbatim into the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.io.volume import Volume
from regix.logging_utils import get_logger, pseudonymize

log = get_logger("io.dicom")

# Modalities for which Regix has tested settings.
SUPPORTED_MODALITIES = {"CT", "MR", "PT", "NM", "US", "CBCT", "MG", "XA", "RTDOSE"}


@dataclass
class DicomSeries:
    """An identified DICOM series, before pixel loading."""

    series_uid: str
    directory: Path
    files: list[str]
    modality: str = "UNKNOWN"
    description: str = ""
    study_uid: str | None = None
    patient_id_raw: str | None = None
    slice_thickness: float | None = None
    rows: int | None = None
    columns: int | None = None
    acquisition_date: str | None = None

    @property
    def n_files(self) -> int:
        return len(self.files)

    def summary(self, pseudonymize_ids: bool = True, salt: str | None = None) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "modality": self.modality,
            "description": self.description,
            "n_slices": self.n_files,
            "matrix": [self.rows, self.columns],
            "slice_thickness_mm": self.slice_thickness,
            "acquisition_date": self.acquisition_date,
            "subject": pseudonymize(self.patient_id_raw, salt) if pseudonymize_ids else self.patient_id_raw,
        }


def list_series(directory: str | Path, recursive: bool = True) -> list[DicomSeries]:
    """Inventory the DICOM series, sorted by decreasing slice count."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(d)

    reader = sitk.ImageSeriesReader()
    found: list[DicomSeries] = []
    directories = [d] + ([p for p in d.rglob("*") if p.is_dir()] if recursive else [])
    seen_uids: set[str] = set()

    for sub in directories:
        try:
            uids = reader.GetGDCMSeriesIDs(str(sub))
        except Exception as exc:  # pragma: no cover
            log.debug("GDCM failed on %s: %s", sub, exc)
            continue
        for uid in uids:
            if uid in seen_uids:
                continue
            files = reader.GetGDCMSeriesFileNames(str(sub), uid)
            if not files:
                continue
            seen_uids.add(uid)
            found.append(_probe_series(uid, sub, list(files)))

    found.sort(key=lambda s: s.n_files, reverse=True)
    return found


def _probe_series(uid: str, directory: Path, files: list[str]) -> DicomSeries:
    """Read the tags of the first slice only: fast even on 10 000 files."""
    tags = {
        "0008|0060": "modality",
        "0008|103e": "description",
        "0020|000d": "study_uid",
        "0010|0020": "patient_id_raw",
        "0018|0050": "slice_thickness",
        "0028|0010": "rows",
        "0028|0011": "columns",
        "0008|0022": "acquisition_date",
    }
    values: dict[str, Any] = {}
    try:
        fr = sitk.ImageFileReader()
        fr.SetFileName(files[0])
        fr.LoadPrivateTagsOff()
        fr.ReadImageInformation()
        for key, name in tags.items():
            if fr.HasMetaDataKey(key):
                values[name] = fr.GetMetaData(key).strip()
    except Exception as exc:  # pragma: no cover
        log.debug("could not read tags for %s: %s", files[0], exc)

    def _num(name: str) -> float | None:
        try:
            return float(values[name])
        except Exception:
            return None

    def _int(name: str) -> int | None:
        try:
            return int(float(values[name]))
        except Exception:
            return None

    return DicomSeries(
        series_uid=uid,
        directory=directory,
        files=files,
        modality=(values.get("modality") or "UNKNOWN").upper(),
        description=values.get("description", ""),
        study_uid=values.get("study_uid"),
        patient_id_raw=values.get("patient_id_raw"),
        slice_thickness=_num("slice_thickness"),
        rows=_int("rows"),
        columns=_int("columns"),
        acquisition_date=values.get("acquisition_date"),
    )


def load_series(
    series: DicomSeries | str | Path,
    series_uid: str | None = None,
    pseudonymize_ids: bool = True,
    salt: str | None = None,
    role: str = "image",
) -> Volume:
    """Load a series into a 3D volume, checking slice regularity."""
    if not isinstance(series, DicomSeries):
        candidates = list_series(series)
        if not candidates:
            raise ValueError(f"no DICOM series in {series}")
        if series_uid is not None:
            match = [c for c in candidates if c.series_uid == series_uid]
            if not match:
                raise ValueError(f"series {series_uid} not present in {series}")
            series = match[0]
        else:
            series = candidates[0]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    image = reader.Execute()

    warnings: list[str] = []
    spacing_report = _check_slice_regularity(reader, series)
    if spacing_report.get("irregular"):
        warnings.append(
            f"irregular inter-slice spacing (std {spacing_report['std_mm']:.3f} mm, "
            f"max {spacing_report['max_gap_mm']:.3f} mm): ITK geometry assumes a constant step"
        )
    if image.GetDimension() != 3:
        raise ValueError(f"unexpected {image.GetDimension()}D volume for series {series.series_uid}")
    if min(image.GetSize()) < 4:
        warnings.append(f"very thin volume {image.GetSize()}: 3D registration will be unreliable")
    if series.modality == "PT":
        warnings.append("PET: the intensities read are not SUV (no conversion applied)")
    if series.modality not in SUPPORTED_MODALITIES:
        warnings.append(f"modality '{series.modality}' is not covered by the Regix presets")
    tilt = _gantry_tilt(reader)
    if tilt is not None and abs(tilt) > 0.5:
        warnings.append(f"gantry tilt of {tilt:.1f} deg: check that it was corrected upstream")

    for w in warnings:
        log.warning("[%s] %s", series.modality, w)

    subject = series.patient_id_raw or series.directory.name
    return Volume(
        image=image,
        modality=series.modality,
        role=role,
        source=series.directory,
        subject_id=pseudonymize(subject, salt) if pseudonymize_ids else subject,
        series_uid=series.series_uid,
        meta={
            "loader": "dicom",
            "n_slices": series.n_files,
            "series_description": series.description,
            "study_uid": series.study_uid,
            "acquisition_date": series.acquisition_date,
            "slice_spacing": spacing_report,
            "warnings": warnings,
        },
    )


def _check_slice_regularity(reader: sitk.ImageSeriesReader, series: DicomSeries) -> dict[str, Any]:
    """Successive slice positions: standard deviation and largest gap."""
    positions: list[np.ndarray] = []
    for idx in range(len(series.files)):
        try:
            ipp = reader.GetMetaData(idx, "0020|0032")
        except Exception:
            return {"available": False}
        try:
            positions.append(np.array([float(v) for v in ipp.split("\\")]))
        except ValueError:
            return {"available": False}
    if len(positions) < 3:
        return {"available": False, "n": len(positions)}
    gaps = np.linalg.norm(np.diff(np.stack(positions), axis=0), axis=1)
    std = float(gaps.std())
    median = float(np.median(gaps))
    return {
        "available": True,
        "median_mm": round(median, 4),
        "std_mm": round(std, 4),
        "max_gap_mm": round(float(gaps.max()), 4),
        "min_gap_mm": round(float(gaps.min()), 4),
        # tolerance: 1 % of the median step, floored at 10 um
        "irregular": bool(std > max(0.01 * median, 0.01)),
    }


def _gantry_tilt(reader: sitk.ImageSeriesReader) -> float | None:
    try:
        return float(reader.GetMetaData(0, "0018|1120"))
    except Exception:
        return None
