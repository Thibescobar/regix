"""Numerical phantoms, so the test suite needs no patient data.

The phantom mimics what matters for a registration: distinct organ structures, a
different contrast between "CT" and "MR" (to exercise the multimodal path), a
non-trivial geometry (anisotropic spacing, non-zero origin) and an exact reference
mask.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk

ORGAN_LABELS = {1: "liver", 2: "spleen", 3: "kidney_left"}


def _ellipsoid(shape, centre, radii) -> np.ndarray:
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    return (
        ((zz - centre[0]) / radii[0]) ** 2
        + ((yy - centre[1]) / radii[1]) ** 2
        + ((xx - centre[2]) / radii[2]) ** 2
    ) <= 1.0


def oblique_direction(rotation_deg: tuple[float, float, float] = (15.0, 0.0, 0.0)) -> tuple[float, ...]:
    """Direction cosines of a tilted grid, as the 9 row-major values ITK expects.

    Built through ``sitk.Euler3DTransform`` rather than by hand so that the rotation
    convention is the same one the rest of Regix uses -- a phantom whose obliquity was
    defined by a different convention would test the wrong thing.
    """
    rotation = sitk.Euler3DTransform()
    rotation.SetComputeZYX(False)
    rotation.SetRotation(*[float(np.radians(a)) for a in rotation_deg])
    return tuple(float(v) for v in rotation.GetMatrix())


#: Grid geometries the suite should exercise, not just the comfortable one.
#:
#: Regix claims that direction cosines are honoured everywhere and that getting them
#: wrong is "fatal on oblique acquisitions" (README), yet every phantom until now used an
#: identity direction and a mild 1.25 anisotropy -- so the claim was never exercised
#: end to end. Parametrising a test over this mapping is what turns the claim into a
#: measurement. ``identity`` reproduces the historical phantom exactly, so a test that
#: only uses it keeps its previous behaviour.
GEOMETRIES: dict[str, dict] = {
    "identity": {},
    "oblique": {"direction": oblique_direction((15.0, 8.0, 0.0))},
    "thick_slices": {"spacing": (1.0, 1.0, 5.0)},
}


def make_phantom(
    modality: str = "CT",
    shape: tuple[int, int, int] = (64, 80, 80),
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.5),
    origin: tuple[float, float, float] = (-40.0, -60.0, 30.0),
    noise: float = 0.0,
    seed: int = 0,
    direction: tuple[float, ...] | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """Return (image, label map) on the same grid.

    ``spacing`` follows the ITK convention (x, y, z); ``shape`` follows the numpy
    convention (z, y, x). The organ contrast is inverted for 'MR' to simulate a
    realistic multimodal pair: same anatomy, intensities with no affine relation.

    ``direction`` is the 9 row-major direction cosines; ``None`` leaves the identity,
    which is what every caller got before the parameter existed. Pass
    ``oblique_direction(...)`` -- or spread ``GEOMETRIES[name]`` -- to build a tilted
    grid: the anatomy is unchanged, only the mapping from voxel indices to patient
    coordinates is, which is exactly the part a registration pipeline gets wrong
    silently.
    """
    rng = np.random.default_rng(seed)
    body = _ellipsoid(
        shape,
        (shape[0] / 2, shape[1] / 2, shape[2] / 2),
        (shape[0] * 0.42, shape[1] * 0.40, shape[2] * 0.44),
    )
    liver = _ellipsoid(
        shape,
        (shape[0] * 0.45, shape[1] * 0.42, shape[2] * 0.36),
        (shape[0] * 0.20, shape[1] * 0.17, shape[2] * 0.15),
    )
    spleen = _ellipsoid(
        shape,
        (shape[0] * 0.50, shape[1] * 0.45, shape[2] * 0.66),
        (shape[0] * 0.11, shape[1] * 0.10, shape[2] * 0.08),
    )
    kidney = _ellipsoid(
        shape,
        (shape[0] * 0.62, shape[1] * 0.62, shape[2] * 0.60),
        (shape[0] * 0.09, shape[1] * 0.08, shape[2] * 0.07),
    )

    if modality.upper() == "CT":
        arr = np.full(shape, -1000.0, dtype=np.float32)
        arr[body] = 30.0
        arr[liver] = 90.0
        arr[spleen] = 55.0
        arr[kidney] = 130.0
    else:  # 'MR': inverted contrast and an arbitrary dynamic range
        arr = np.zeros(shape, dtype=np.float32)
        arr[body] = 420.0
        arr[liver] = 120.0
        arr[spleen] = 610.0
        arr[kidney] = 250.0
    if noise > 0:
        arr = arr + rng.normal(0.0, noise, size=shape).astype(np.float32)

    labels = np.zeros(shape, dtype=np.uint16)
    labels[liver] = 1
    labels[spleen] = 2
    labels[kidney] = 3

    image = sitk.GetImageFromArray(arr)
    labelmap = sitk.GetImageFromArray(labels)
    for img in (image, labelmap):
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        if direction is not None:
            img.SetDirection([float(v) for v in direction])
    return image, labelmap


def known_rigid(
    reference: sitk.Image,
    rotation_deg: tuple[float, float, float] = (4.0, -3.0, 6.0),
    translation_mm: tuple[float, float, float] = (7.0, -5.0, 4.0),
) -> sitk.Euler3DTransform:
    """Ground-truth transform, centred on the volume."""
    size = np.asarray(reference.GetSize(), dtype=float)
    centre = reference.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2).tolist())
    t = sitk.Euler3DTransform()
    t.SetCenter(centre)
    t.SetComputeZYX(False)
    t.SetParameters([*np.radians(rotation_deg), *translation_mm])
    return t


def warp(image: sitk.Image, transform: sitk.Transform, is_label: bool = False) -> sitk.Image:
    """Apply the inverse of ``transform`` to build a "moving" image.

    Convention: Regix will look for T such that moving(T(x)) ~ fixed(x). Building the
    moving image with T^-1 means we know the expected answer exactly.
    """
    return sitk.Resample(
        image,
        image,
        transform.GetInverse(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        float(sitk.GetArrayViewFromImage(image).min()) if not is_label else 0.0,
        image.GetPixelID(),
    )


@pytest.fixture
def ct_phantom():
    return make_phantom("CT")


@pytest.fixture
def mr_phantom():
    return make_phantom("MR")


@pytest.fixture
def rigid_pair(tmp_path):
    """(paths to fixed, moving, fixed labels, moving labels, ground-truth transform)."""
    image, labels = make_phantom("CT", noise=3.0)
    truth = known_rigid(image)
    moving_image = warp(image, truth)
    moving_labels = warp(labels, truth, is_label=True)

    paths = {}
    for name, img, dtype in (
        ("fixed", image, sitk.sitkFloat32),
        ("moving", moving_image, sitk.sitkFloat32),
        ("fixed_labels", labels, sitk.sitkUInt16),
        ("moving_labels", moving_labels, sitk.sitkUInt16),
    ):
        path = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(sitk.Cast(img, dtype), str(path), True)
        paths[name] = path

    # Nomenclature next to the label maps: this is what most segmentation tools
    # produce, and Regix must find it on its own.
    for name in ("fixed_labels", "moving_labels"):
        (tmp_path / f"{name}.labels.json").write_text(
            json.dumps({str(k): v for k, v in ORGAN_LABELS.items()}), encoding="utf-8"
        )
    return paths, truth
