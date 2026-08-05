"""DICOM input/output tests on a synthetic series.

Why these tests matter more than they look: every "real world" claim in the README
rests on this path -- series discovery, geometry, irregular slices, derived series,
registration object. It is also the code that never runs on a phantom NIfTI, so
without these tests it would ship unexercised.

The series is built with pydicom rather than mocked, so the SimpleITK/GDCM reader
is genuinely exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

pydicom = pytest.importorskip("pydicom", reason="pydicom is required for the DICOM path")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402


def write_ct_series(
    directory,
    n_slices: int = 12,
    rows: int = 32,
    columns: int = 32,
    pixel_spacing: tuple[float, float] = (0.75, 0.75),
    slice_spacing: float = 2.5,
    origin: tuple[float, float, float] = (-12.0, -20.0, 40.0),
    irregular_gap_at: int | None = None,
    series_description: str = "TEST CT",
    patient_id: str = "PAT-0001",
):
    """Write a minimal but valid CT series and return (paths, expected geometry).

    ``irregular_gap_at`` inserts an anomalous inter-slice gap at that index, which is
    what a real acquisition looks like after a breathing artefact or a partial
    reconstruction.
    """
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    rng = np.random.default_rng(0)

    paths = []
    positions = []
    z = origin[2]
    for k in range(n_slices):
        if irregular_gap_at is not None and k == irregular_gap_at:
            z += slice_spacing * 1.6  # the anomalous gap
        elif k > 0:
            z += slice_spacing
        positions.append(z)

        # A recognisable pattern: a bright disc on a dark background.
        yy, xx = np.mgrid[0:rows, 0:columns]
        disc = ((yy - rows / 2) ** 2 + (xx - columns / 2) ** 2) < (rows / 3) ** 2
        arr = np.full((rows, columns), -1000, dtype=np.int16)
        arr[disc] = 40 + int(20 * np.sin(k))
        arr = arr + rng.integers(-3, 4, size=arr.shape, dtype=np.int16)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.preamble = b"\0" * 128
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.SeriesDescription = series_description
        ds.PatientID = patient_id
        ds.PatientName = "TEST^PHANTOM"
        ds.StudyDate = "20260101"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = rows, columns
        ds.PixelSpacing = [f"{pixel_spacing[0]}", f"{pixel_spacing[1]}"]
        ds.SliceThickness = f"{slice_spacing}"
        ds.ImagePositionPatient = [f"{origin[0]}", f"{origin[1]}", f"{z}"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = "1"
        ds.RescaleIntercept = "0"
        ds.PixelData = arr.tobytes()
        ds["PixelData"].VR = "OW"

        path = directory / f"slice_{k:03d}.dcm"
        ds.save_as(str(path), enforce_file_format=True)
        paths.append(path)

    return paths, {
        "series_uid": series_uid,
        "frame_of_reference_uid": frame_uid,
        "size": (columns, rows, n_slices),
        "spacing": (pixel_spacing[1], pixel_spacing[0], slice_spacing),
        "positions": positions,
        "patient_id": patient_id,
    }


# --------------------------------------------------------------------------- #
def test_series_discovery_and_geometry(tmp_path):
    from regix.io.dicom import list_series, load_series

    _, expected = write_ct_series(tmp_path / "ct")
    series = list_series(tmp_path / "ct")
    assert len(series) == 1
    assert series[0].modality == "CT"
    assert series[0].n_files == 12
    assert series[0].description == "TEST CT"
    assert series[0].series_uid == expected["series_uid"]

    volume = load_series(series[0])
    assert volume.size == expected["size"]
    assert np.allclose(volume.spacing, expected["spacing"], atol=1e-6)
    # HU must survive the read: air stays around -1000.
    arr = volume.array()
    assert arr.min() < -900
    assert arr.max() > 0
    assert volume.meta["n_slices"] == 12


def test_patient_identifier_is_pseudonymised(tmp_path):
    """No identifier in clear text: this is a privacy requirement, not a nicety."""
    from regix.io.dicom import list_series, load_series

    write_ct_series(tmp_path / "ct", patient_id="DUPONT-12345")
    series = list_series(tmp_path / "ct")

    volume = load_series(series[0], pseudonymize_ids=True, salt="unit-test")
    assert "DUPONT" not in volume.subject_id
    assert volume.subject_id != "DUPONT-12345"
    assert len(volume.subject_id) == 10

    summary = series[0].summary(pseudonymize_ids=True, salt="unit-test")
    assert summary["subject"] == volume.subject_id

    # The raw value is only exposed when explicitly requested.
    clear = load_series(series[0], pseudonymize_ids=False)
    assert clear.subject_id == "DUPONT-12345"


def test_irregular_slice_spacing_is_detected(tmp_path):
    """A breathing artefact must be reported, never silently averaged."""
    from regix.io.dicom import list_series, load_series

    write_ct_series(tmp_path / "regular")
    write_ct_series(tmp_path / "irregular", irregular_gap_at=5)

    regular = load_series(list_series(tmp_path / "regular")[0])
    assert regular.meta["slice_spacing"]["irregular"] is False
    assert not regular.meta["warnings"]

    irregular = load_series(list_series(tmp_path / "irregular")[0])
    report = irregular.meta["slice_spacing"]
    assert report["irregular"] is True
    assert report["max_gap_mm"] > report["median_mm"]
    assert any("spacing" in w for w in irregular.meta["warnings"])


def test_two_series_in_one_folder_are_separated(tmp_path):
    """Localizers and derived series live next to the real one: pick the largest."""
    from regix.io.dicom import list_series
    from regix.io.volume import load_volume

    write_ct_series(tmp_path / "study" / "a", n_slices=4, series_description="LOCALIZER")
    write_ct_series(tmp_path / "study" / "b", n_slices=16, series_description="AXIAL")

    series = list_series(tmp_path / "study")
    assert len(series) == 2
    assert [s.n_files for s in series] == [16, 4]  # sorted by decreasing slice count

    volume = load_volume(tmp_path / "study")
    assert volume.size[2] == 16, "the largest series must be selected"


def test_derived_dicom_series_round_trip(tmp_path):
    """The exported series must be re-readable, with the right geometry and new UIDs."""
    from regix.io.dicom import list_series, load_series
    from regix.io.writers import write_derived_dicom

    paths, expected = write_ct_series(tmp_path / "source")
    source = load_series(list_series(tmp_path / "source")[0])

    out = write_derived_dicom(
        source.image,
        tmp_path / "derived",
        [str(p) for p in paths],
        series_description_suffix="REGIX registered",
    )
    derived_series = list_series(out)
    assert len(derived_series) == 1
    assert derived_series[0].n_files == 12
    assert derived_series[0].series_uid != expected["series_uid"], "a new SeriesInstanceUID is required"

    derived = load_series(derived_series[0])
    assert derived.size == source.size
    assert np.allclose(derived.spacing, source.spacing, atol=1e-4)
    assert np.allclose(derived.origin, source.origin, atol=1e-3)

    # The intensities go through a slope/intercept quantisation: check the HU are
    # recovered rather than the raw stored values.
    a = source.array().astype(float)
    b = derived.array().astype(float)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999
    assert abs(a.min() - b.min()) < 5.0

    ds = pydicom.dcmread(str(sorted(out.glob("*.dcm"))[0]))
    assert list(ds.ImageType[:2]) == ["DERIVED", "SECONDARY"]
    assert "not a medical device" in ds.DerivationDescription


def test_spatial_registration_object_is_valid(tmp_path):
    """The DICOM REG object is what planning systems consume: check its structure."""
    from regix.io.writers import SPATIAL_REGISTRATION_SOP_CLASS, write_spatial_registration_dicom
    from regix.registration.transforms import matrix_moving_to_fixed
    from tests.conftest import known_rigid, make_phantom

    fixed_paths, _ = write_ct_series(tmp_path / "fixed")
    moving_paths, _ = write_ct_series(tmp_path / "moving")

    image, _ = make_phantom("CT")
    transform = known_rigid(image)
    matrix = matrix_moving_to_fixed(transform)

    path = write_spatial_registration_dicom(
        tmp_path / "reg.dcm",
        matrix,
        [str(p) for p in fixed_paths],
        [str(p) for p in moving_paths],
        transformation_type="RIGID",
    )

    ds = pydicom.dcmread(str(path))
    assert str(ds.SOPClassUID) == SPATIAL_REGISTRATION_SOP_CLASS
    assert ds.Modality == "REG"
    assert len(ds.RegistrationSequence) == 2

    # Item 0 = the fixed image, identity. Item 1 = the moving image with the matrix.
    identity = np.array(
        [float(v) for v in ds.RegistrationSequence[0].MatrixRegistrationSequence[0]
         .MatrixSequence[0].FrameOfReferenceTransformationMatrix]
    ).reshape(4, 4)
    assert np.allclose(identity, np.eye(4))

    written = np.array(
        [float(v) for v in ds.RegistrationSequence[1].MatrixRegistrationSequence[0]
         .MatrixSequence[0].FrameOfReferenceTransformationMatrix]
    ).reshape(4, 4)
    assert np.allclose(written, matrix, atol=1e-6)
    assert (
        ds.RegistrationSequence[1].MatrixRegistrationSequence[0].MatrixSequence[0]
        .FrameOfReferenceTransformationMatrixType == "RIGID"
    )


def test_registration_from_dicom_series_end_to_end(tmp_path):
    """A full registration with DICOM in and DICOM out, including the SRO."""
    from regix.config import load_preset
    from regix.pipeline import RegistrationPipeline

    write_ct_series(tmp_path / "fixed", n_slices=16, rows=48, columns=48)
    # The moving series is shifted by one slice plus an in-plane offset.
    write_ct_series(
        tmp_path / "moving", n_slices=16, rows=48, columns=48,
        origin=(-9.0, -18.0, 42.5),
    )

    cfg = load_preset("base").with_overrides(
        fixed_modality="CT",
        moving_modality="CT",
        preprocess={"working_spacing_mm": 1.5},
        output={"dir": str(tmp_path / "out"), "overwrite": True, "write_dicom": True},
        runtime={"log_level": "WARNING"},
        qc={"report_html": False},
    )
    result = RegistrationPipeline(cfg).run(tmp_path / "fixed", tmp_path / "moving", tmp_path / "out")

    assert result.status in ("PASS", "WARN")
    # DICOM-specific outputs: the registration object and the derived series.
    assert "dicom_sro" in result.outputs, "a DICOM series pair must yield an SRO"
    assert "dicom_series" in result.outputs
    assert result.outputs["dicom_sro"].exists()
    assert len(list(result.outputs["dicom_series"].glob("*.dcm"))) == 16

    # The registered volume stays on the original grid of the fixed image.
    fixed_image = sitk.ReadImage(str(sorted((tmp_path / "fixed").glob("*.dcm"))[0]))
    assert result.registered_image.GetSize()[:2] == fixed_image.GetSize()[:2]
