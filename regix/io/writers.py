"""Writing results: NIfTI, points, derived DICOM, DICOM registration object.

The point that matters for real-world use: a registration is only worth something
if its output can go back into the imaging information system. We therefore
produce three distinct things, not just a NIfTI:

1. the resampled volume (NIfTI, and optionally a derived DICOM series);
2. the transform (elastix format + ITK .tfm + Insight Transform File .txt);
3. a DICOM Spatial Registration Object (rigid/affine) expressing the transform
   between two Frames of Reference -- this is what treatment planning systems and
   most fusion workstations consume.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from regix import __version__
from regix.env import DICOM_UID_ROOT_ENV
from regix.logging_utils import get_logger

log = get_logger("io.writers")

SPATIAL_REGISTRATION_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.1"
TEST_UID_ROOT = "1.2.826.0.1.3680043.10.1337"
# Backward-compatible public name. New code must call ``resolve_uid_root`` so a site root
# can come from configuration or REGIX_DICOM_UID_ROOT.
REGIX_UID_ROOT = TEST_UID_ROOT


def resolve_uid_root(value: str | None = None) -> str:
    """Return and validate the site-owned DICOM UID root used for generated objects."""
    root = (value or os.getenv(DICOM_UID_ROOT_ENV) or TEST_UID_ROOT).rstrip(".")
    if len(root) > 54:
        raise ValueError("DICOM UID root must be at most 54 characters to leave room for a suffix")
    components = root.split(".")
    if (
        not root
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", root)
        or any(len(component) > 1 and component.startswith("0") for component in components)
        or components[0] not in {"1", "2"}
    ):
        raise ValueError(
            "DICOM UID root must contain numeric dot-separated components without leading zeroes"
        )
    if root == TEST_UID_ROOT:
        log.warning(
            "using Regix's test DICOM UID root; configure output.dicom_uid_root or "
            "REGIX_DICOM_UID_ROOT before sending objects to a production PACS"
        )
    return root


def _new_uid(root: str) -> str:
    from pydicom.uid import generate_uid

    return generate_uid(prefix=root + ".")


def _set_equipment_identity(ds) -> None:
    ds.Manufacturer = "Regix"
    ds.ManufacturerModelName = "Regix registration"
    ds.SoftwareVersions = __version__


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def save_image(
    image: sitk.Image,
    path: str | Path,
    compress: bool = True,
    dtype: int | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.Cast(image, dtype) if dtype is not None else image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(p))
    writer.SetUseCompression(bool(compress))
    writer.Execute(out)
    log.debug("wrote %s (%s)", p.name, out.GetSize())
    return p


# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #
def load_landmarks(path: str | Path) -> np.ndarray:
    """Read landmarks in physical coordinates (mm).

    Accepted formats: one physical ``x y z`` (or ``x,y,z``) line per point, with
    ``#`` comments, or an elastix ``point`` file. Elastix ``index`` files are
    refused because converting indices requires the corresponding reference grid.
    """
    lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if lines and lines[0].lower() == "index":
        raise ValueError(
            f"{path} contains elastix index coordinates; convert them to physical 'point' "
            "coordinates with the reference image before using them as landmarks"
        )
    if lines and lines[0].lower() == "point":
        lines = lines[2:]  # elastix header: keyword then point count
    pts = []
    for ln in lines:
        parts = ln.replace(",", " ").replace(";", " ").split()
        if len(parts) < 3:
            raise ValueError(f"invalid landmark line: {ln!r}")
        pts.append([float(v) for v in parts[:3]])
    if not pts:
        raise ValueError(f"no landmark read from {path}")
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Derived DICOM
# --------------------------------------------------------------------------- #
def write_derived_dicom(
    image: sitk.Image,
    out_dir: str | Path,
    template_files: Sequence[str | Path],
    series_description_suffix: str = "REGIX registered",
    frame_of_reference_uid: str | None = None,
    series_number_offset: int = 9000,
    uid_root: str | None = None,
) -> Path:
    """Write ``image`` as a derived DICOM series.

    ``template_files`` must come from the series the **pixels** originate from
    (the moving image): that series carries the right modality and the right
    intensity semantics. The geometry, however, is that of ``image`` (hence of the
    fixed image frame), and ``frame_of_reference_uid`` should be the fixed image's
    so that workstations display the fusion without re-registering.

    Assumed limitations: no Presentation State, no multi-frame, no reconstruction
    of vendor-specific tags.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for DICOM export (pip install pydicom)") from exc

    if not template_files:
        raise ValueError("template_files is empty: cannot derive a DICOM series")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    template = pydicom.dcmread(str(template_files[0]), stop_before_pixels=True, force=True)
    root = resolve_uid_root(uid_root)
    if image.GetDimension() != 3:
        raise ValueError("write_derived_dicom expects a 3D volume")

    arr = sitk.GetArrayFromImage(image).astype(np.float64)  # (z, y, x)
    n_slices, rows, cols = arr.shape
    spacing = image.GetSpacing()
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    # columns of the direction matrix = the volume's x, y, z axes in patient space
    row_cosines, col_cosines, slice_cosines = direction[:, 0], direction[:, 1], direction[:, 2]

    # Quantisation to signed 16-bit, with explicit slope/intercept.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("empty volume: nothing to write")
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-9:
        slope, intercept = 1.0, vmin
        stored = np.zeros_like(arr, dtype=np.int16)
    else:
        slope = (vmax - vmin) / 32000.0
        intercept = vmin
        stored = np.clip(np.rint((arr - intercept) / slope), 0, 32000).astype(np.int16)

    study_uid = getattr(template, "StudyInstanceUID", None) or _new_uid(root)
    series_uid = _new_uid(root)
    for_uid = frame_of_reference_uid or getattr(template, "FrameOfReferenceUID", None) or _new_uid(root)
    now = _dt.datetime.now()
    source_sop_class = (
        getattr(
            template,
            "SOPClassUID",
            getattr(getattr(template, "file_meta", None), "MediaStorageSOPClassUID", None),
        )
        or SecondaryCaptureImageStorage
    )

    # Explicit whitelist: acquisition and vendor-private geometry from the source no longer
    # describes a resampled image and must not silently survive into the derived series.
    copied_tags = (
        "SpecificCharacterSet",
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "StudyDate",
        "StudyTime",
        "StudyID",
        "AccessionNumber",
        "ReferringPhysicianName",
        "InstitutionName",
        "Modality",
        "RescaleType",
        "Units",
    )

    for k in range(n_slices):
        ds = Dataset()
        for tag in copied_tags:
            if tag in template:
                setattr(ds, tag, getattr(template, tag))
        ds.SOPClassUID = source_sop_class
        ds.SOPInstanceUID = _new_uid(root)
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = for_uid
        ds.SeriesNumber = int(getattr(template, "SeriesNumber", 1) or 1) + series_number_offset
        source_description = getattr(template, "SeriesDescription", "series")
        ds.SeriesDescription = f"{source_description} - {series_description_suffix}"[:64]
        ds.ImageType = ["DERIVED", "SECONDARY", "REGISTERED"]
        ds.InstanceNumber = k + 1
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        ds.SeriesDate = now.strftime("%Y%m%d")
        ds.SeriesTime = now.strftime("%H%M%S")
        ds.InstanceCreationDate = now.strftime("%Y%m%d")
        ds.InstanceCreationTime = now.strftime("%H%M%S")
        ds.DerivationDescription = "Resampled with Regix (research software, not a medical device)"
        derivation_code = Dataset()
        derivation_code.CodeValue = "113085"
        derivation_code.CodingSchemeDesignator = "DCM"
        derivation_code.CodeMeaning = "Spatial resampling"
        ds.DerivationCodeSequence = [derivation_code]
        _set_equipment_identity(ds)

        ds.Rows, ds.Columns = int(rows), int(cols)
        ds.PixelSpacing = [f"{spacing[1]:.6f}", f"{spacing[0]:.6f}"]  # [row, column] = [y, x]
        ds.SliceThickness = f"{spacing[2]:.6f}"
        ds.SpacingBetweenSlices = f"{spacing[2]:.6f}"
        pos = origin + slice_cosines * (k * spacing[2])
        ds.ImagePositionPatient = [f"{v:.6f}" for v in pos]
        ds.ImageOrientationPatient = [f"{v:.9f}" for v in list(row_cosines) + list(col_cosines)]
        ds.SliceLocation = f"{float(np.dot(pos, slice_cosines)):.6f}"

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = f"{slope:.9g}"
        ds.RescaleIntercept = f"{intercept:.9g}"
        ds.PixelData = stored[k].tobytes()
        # Under explicit VR, PixelData has an ambiguous VR ('OB or OW') that pydicom
        # refuses to write: 16 bits per sample implies OW.
        ds["PixelData"].VR = "OW"

        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.ImplementationClassUID = root + ".1"
        ds.preamble = b"\0" * 128
        ds.remove_private_tags()
        ds.save_as(str(out / f"regix_{k + 1:05d}.dcm"), enforce_file_format=True)

    log.info("derived DICOM series: %d slices in %s", n_slices, out)
    return out


def write_spatial_registration_dicom(
    path: str | Path,
    matrix_moving_to_fixed: np.ndarray,
    fixed_reference_files: Sequence[str | Path],
    moving_reference_files: Sequence[str | Path],
    transformation_type: str = "RIGID",
    label: str = "REGIX",
    uid_root: str | None = None,
) -> Path:
    """Write a DICOM Spatial Registration Object (rigid or affine).

    ``matrix_moving_to_fixed`` is a 4x4 homogeneous matrix such that
    ``p_fixed = M @ p_moving`` in patient coordinates (mm) -- the DICOM convention
    for FrameOfReferenceTransformationMatrix. Note: elastix provides the inverse
    transform (fixed -> moving), so its inverse must be passed here.
    ``regix.registration.transforms`` takes care of that.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for the SRO export") from exc

    M = np.asarray(matrix_moving_to_fixed, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"expected a 4x4 homogeneous matrix, got {M.shape}")
    if transformation_type not in ("RIGID", "RIGID_SCALE", "AFFINE"):
        raise ValueError("transformation_type must be RIGID, RIGID_SCALE or AFFINE")
    if not fixed_reference_files or not moving_reference_files:
        raise ValueError("both fixed and moving reference series must contain at least one file")

    root = resolve_uid_root(uid_root)

    fixed_ds = pydicom.dcmread(str(fixed_reference_files[0]), stop_before_pixels=True, force=True)
    moving_ds = pydicom.dcmread(str(moving_reference_files[0]), stop_before_pixels=True, force=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SPATIAL_REGISTRATION_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = _new_uid(root)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = root + ".1"

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    now = _dt.datetime.now()
    ds.SOPClassUID = SPATIAL_REGISTRATION_SOP_CLASS
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "REG"
    ds.SeriesInstanceUID = _new_uid(root)
    ds.SeriesNumber = 9900
    ds.SeriesDescription = f"{label} spatial registration"
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S")
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.ContentLabel = label[:16].upper().replace(" ", "_")
    ds.ContentDescription = "Registration computed by Regix (research software)"
    ds.ContentCreatorName = "REGIX"
    _set_equipment_identity(ds)

    # Patient/study identity taken from the fixed image (the SRO lives in its study).
    # These are type-2 attributes: they must exist even when the source value is unknown.
    for tag in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "StudyID",
        "AccessionNumber",
        "ReferringPhysicianName",
    ):
        setattr(ds, tag, getattr(fixed_ds, tag, ""))
    ds.StudyInstanceUID = getattr(fixed_ds, "StudyInstanceUID", None) or _new_uid(root)
    ds.StudyDate = getattr(fixed_ds, "StudyDate", "")
    ds.StudyTime = getattr(fixed_ds, "StudyTime", "")
    ds.FrameOfReferenceUID = getattr(fixed_ds, "FrameOfReferenceUID", None) or _new_uid(root)

    def _references(files: Iterable[str | Path]) -> list[Dataset]:
        refs = []
        for filename in files:
            source = pydicom.dcmread(str(filename), stop_before_pixels=True, force=True)
            ref = Dataset()
            ref.ReferencedSOPClassUID = getattr(source, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.7")
            ref.ReferencedSOPInstanceUID = source.SOPInstanceUID
            refs.append(ref)
        return refs

    def _referenced_series(files: Sequence[str | Path]) -> list[Dataset]:
        grouped: dict[str, list[Dataset]] = {}
        for filename in files:
            source = pydicom.dcmread(str(filename), stop_before_pixels=True, force=True)
            uid = str(getattr(source, "SeriesInstanceUID", ""))
            ref = Dataset()
            ref.ReferencedSOPClassUID = getattr(source, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.7")
            ref.ReferencedSOPInstanceUID = source.SOPInstanceUID
            grouped.setdefault(uid, []).append(ref)
        series_items = []
        for uid, refs in grouped.items():
            item = Dataset()
            item.SeriesInstanceUID = uid or _new_uid(root)
            item.ReferencedInstanceSequence = refs
            series_items.append(item)
        return series_items

    ds.ReferencedSeriesSequence = _referenced_series(
        list(fixed_reference_files) + list(moving_reference_files)
    )

    def _registration_item(reference_ds, files: Iterable[str | Path], matrix: np.ndarray | None) -> Dataset:
        item = Dataset()
        item.FrameOfReferenceUID = getattr(reference_ds, "FrameOfReferenceUID", None) or _new_uid(root)
        refs = _references(files)
        item.ReferencedImageSequence = refs

        matrix_item = Dataset()
        matrix_item.FrameOfReferenceTransformationMatrixType = (
            "RIGID" if matrix is None else transformation_type
        )
        flat = (np.eye(4) if matrix is None else matrix).reshape(-1)
        matrix_item.FrameOfReferenceTransformationMatrix = [f"{v:.10g}" for v in flat]
        reg = Dataset()
        code = Dataset()
        code.CodeValue = "125024"
        code.CodingSchemeDesignator = "DCM"
        code.CodeMeaning = "Image Content-based Alignment"
        reg.RegistrationTypeCodeSequence = [code]
        reg.MatrixSequence = [matrix_item]
        item.MatrixRegistrationSequence = [reg]
        return item

    # Item 1: the fixed image, identity transform (destination frame).
    # Item 2: the moving image, with the matrix that brings it into the fixed frame.
    ds.RegistrationSequence = [
        _registration_item(fixed_ds, fixed_reference_files, None),
        _registration_item(moving_ds, moving_reference_files, M),
    ]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ds.remove_private_tags()
    ds.save_as(str(p), enforce_file_format=True)
    log.info("DICOM Spatial Registration Object written: %s", p.name)
    return p
