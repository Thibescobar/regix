"""Registration internals: initialization strategies and transform application.

These two modules carry the decisions that are hardest to notice when they go wrong.
An initialization that silently falls back to the grid centre still produces a
plausible-looking result; a transform applied through the wrong path still produces an
image. So they are tested against explicit expectations rather than "it ran".
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from regix.io.volume import Volume
from tests.conftest import ORGAN_LABELS, known_rigid, make_phantom, warp


def _volume(image: sitk.Image, modality: str = "CT") -> Volume:
    return Volume(image=image, modality=modality)


def _segmentation(labelmap: sitk.Image):
    from regix.organs.segmenter import OrganSegmentation

    return OrganSegmentation(
        labelmap=sitk.Cast(labelmap, sitk.sitkUInt16),
        label_names=dict(ORGAN_LABELS),
        backend="test",
    )


# --------------------------------------------------------------------------- #
# Initialization strategies
# --------------------------------------------------------------------------- #
def test_identity_and_geometry_initialization():
    from regix.preprocess.geometry import image_center_physical
    from regix.registration.initialize import geometry_init, identity_init

    fixed_img, _ = make_phantom("CT")
    # A moving volume on a deliberately shifted grid.
    moving_img, _ = make_phantom("CT", origin=(-10.0, -30.0, 55.0))

    assert np.allclose(identity_init().TransformPoint((7.0, -3.0, 11.0)), (7.0, -3.0, 11.0))

    t = geometry_init(_volume(fixed_img), _volume(moving_img))
    c_fixed = image_center_physical(fixed_img)
    c_moving = image_center_physical(moving_img)
    # The fixed centre must land exactly on the moving centre.
    assert np.allclose(t.TransformPoint(c_fixed.tolist()), c_moving, atol=1e-6)


def test_moments_initialization_uses_the_masks():
    from regix.registration.initialize import moments_init

    fixed_img, fixed_labels = make_phantom("CT")
    truth = known_rigid(fixed_img, (0.0, 0.0, 0.0), (8.0, -6.0, 4.0))
    moving_img = warp(fixed_img, truth)
    moving_labels = warp(fixed_labels, truth, is_label=True)

    liver_fixed = sitk.Cast(sitk.Equal(fixed_labels, 1), sitk.sitkUInt8)
    liver_moving = sitk.Cast(sitk.Equal(moving_labels, 1), sitk.sitkUInt8)

    t = moments_init(_volume(fixed_img), _volume(moving_img), liver_fixed, liver_moving)
    # A pure translation was applied, so aligning the centres of mass must recover it.
    offset = np.asarray(t.TransformPoint((0.0, 0.0, 0.0))) - np.asarray(
        sitk.Euler3DTransform().TransformPoint((0.0, 0.0, 0.0))
    )
    assert np.linalg.norm(offset - np.array([8.0, -6.0, 4.0])) < 2.5


def test_organ_centroid_initialization_recovers_a_translation():
    from regix.registration.initialize import organ_centroid_init

    fixed_img, fixed_labels = make_phantom("CT")
    truth = known_rigid(fixed_img, (0.0, 0.0, 0.0), (9.0, -7.0, 5.0))
    moving_labels = warp(fixed_labels, truth, is_label=True)

    t, info = organ_centroid_init(
        _segmentation(fixed_labels), _segmentation(moving_labels), ["liver"]
    )
    assert info["organs_used"] == ["liver"]
    assert info["offset_mm"] > 5.0
    probe = (0.0, 0.0, 60.0)
    assert np.linalg.norm(
        np.asarray(t.TransformPoint(probe)) - np.asarray(truth.TransformPoint(list(probe)))
    ) < 2.5


def test_organ_centroid_initialization_needs_a_common_organ():
    from regix.registration.initialize import organ_centroid_init

    _, labels = make_phantom("CT")
    empty = sitk.Image(labels.GetSize(), sitk.sitkUInt16)
    empty.CopyInformation(labels)

    with pytest.raises(ValueError, match="no organ in common"):
        organ_centroid_init(_segmentation(labels), _segmentation(empty), ["liver"])


def test_organ_moments_initialization_never_mirrors():
    """Eigenvectors have an arbitrary sign; a mirror flip would be anatomically absurd."""
    from regix.registration.initialize import organ_moments_init
    from regix.registration.transforms import decompose_affine, to_matrix_4x4

    fixed_img, fixed_labels = make_phantom("CT")
    truth = known_rigid(fixed_img, (5.0, -4.0, 8.0), (6.0, -3.0, 4.0))
    moving_labels = warp(fixed_labels, truth, is_label=True)

    t, info = organ_moments_init(
        _segmentation(fixed_labels), _segmentation(moving_labels), ["liver"]
    )
    assert info["organ_used"] == "liver"
    assert 0.7 <= info["scale"] <= 1.4, "the scale guard rail must hold"

    decomposition = decompose_affine(to_matrix_4x4(t))
    assert decomposition["determinant"] > 0, "a negative determinant means a mirror flip"


def test_probe_rotation_composition_is_linear_and_reversible():
    from regix.preprocess.geometry import image_center_physical
    from regix.registration.initialize import geometry_init, with_extra_rotation
    from regix.registration.transforms import linear_matrix_from_transform

    fixed_img, _ = make_phantom("CT")
    moving_img, _ = make_phantom("CT", origin=(-30.0, -50.0, 40.0))
    centre = image_center_physical(fixed_img)
    base = geometry_init(_volume(fixed_img), _volume(moving_img))

    # A zero rotation must return the base transform untouched, not a wrapper.
    assert with_extra_rotation(base, centre, (0, 0, 0)) is base

    rotated = with_extra_rotation(base, centre, (0, 0, 30))
    assert linear_matrix_from_transform(rotated) is not None, "must stay linear"
    # The rotation is about the fixed centre, so that point only moves by the base part.
    assert np.allclose(
        rotated.TransformPoint(centre.tolist()), base.TransformPoint(centre.tolist()), atol=1e-6
    )


def test_multistart_enumerates_candidates_and_scores_them():
    from regix.config import InitConfig, InitMode
    from regix.registration.initialize import build_candidates, choose_initialization

    fixed_img, fixed_labels = make_phantom("CT", noise=2.0)
    truth = known_rigid(fixed_img, (0.0, 0.0, 0.0), (6.0, -4.0, 5.0))
    moving_img = warp(fixed_img, truth)
    moving_labels = warp(fixed_labels, truth, is_label=True)

    config = InitConfig(
        mode=InitMode.MULTISTART,
        candidates=[InitMode.IDENTITY, InitMode.GEOMETRY, InitMode.ORGAN_CENTROID],
        multistart_rotations_deg=[(0, 0, 0), (0, 0, 25)],
    )
    candidates = build_candidates(
        _volume(fixed_img),
        _volume(moving_img),
        config,
        _segmentation(fixed_labels),
        _segmentation(moving_labels),
        ["liver"],
    )
    assert len(candidates) == 6, [c.name for c in candidates]
    assert any("rot(0,0,25)" in c.name for c in candidates)

    chosen, report = choose_initialization(
        _volume(fixed_img),
        _volume(moving_img),
        config,
        _segmentation(fixed_labels),
        _segmentation(moving_labels),
        ["liver"],
    )
    scores = [c["score"] for c in report["candidates"]]
    assert scores == sorted(scores, reverse=True), "candidates must be ranked"
    assert chosen.score == scores[0]
    # A 25 deg probe rotation is wrong here: it must not win.
    assert "rot(0,0,25)" not in report["chosen"]
    assert chosen.summary()["name"] == report["chosen"]


def test_flip_check_doubles_the_candidate_list():
    from regix.config import InitConfig, InitMode
    from regix.registration.initialize import build_candidates

    fixed_img, _ = make_phantom("CT")
    moving_img, _ = make_phantom("CT", origin=(-20.0, -40.0, 45.0))
    config = InitConfig(
        mode=InitMode.MULTISTART,
        candidates=[InitMode.GEOMETRY],
        multistart_rotations_deg=[(0, 0, 0)],
        flip_check=True,
    )
    candidates = build_candidates(_volume(fixed_img), _volume(moving_img), config)
    assert len(candidates) == 2
    assert any("flip180z" in c.name for c in candidates)


def test_unusable_candidate_is_dropped_not_fatal(caplog):
    """A missing segmentation must degrade to the grid centre, with a warning."""
    from regix.config import InitConfig, InitMode
    from regix.registration.initialize import build_candidates

    fixed_img, _ = make_phantom("CT")
    moving_img, _ = make_phantom("CT")
    config = InitConfig(mode=InitMode.ORGAN_CENTROID)

    candidates = build_candidates(_volume(fixed_img), _volume(moving_img), config)
    assert len(candidates) == 1
    assert candidates[0].name == "geometry", "must fall back, not crash"


def test_initialization_from_a_file(tmp_path):
    from regix.config import InitConfig, InitMode
    from regix.registration.initialize import build_candidates
    from regix.registration.transforms import save_transform

    fixed_img, _ = make_phantom("CT")
    truth = known_rigid(fixed_img)
    path = save_transform(truth, tmp_path / "start.tfm")

    config = InitConfig(mode=InitMode.FILE, transform_file=path)
    candidates = build_candidates(_volume(fixed_img), _volume(fixed_img), config)
    assert len(candidates) == 1 and candidates[0].name == "file"
    probe = [11.0, -22.0, 33.0]
    assert np.allclose(
        candidates[0].transform.TransformPoint(probe), truth.TransformPoint(probe), atol=1e-6
    )


@pytest.mark.parametrize("metric", ["ncc", "nmi", "auto"])
def test_candidate_scoring_prefers_the_true_alignment(metric):
    from regix.registration.initialize import score_candidate

    fixed_img, _ = make_phantom("CT", noise=2.0, seed=4)
    truth = known_rigid(fixed_img, (0.0, 0.0, 0.0), (0.0, 0.0, 10.0))
    moving_img = warp(fixed_img, truth)

    aligned = score_candidate(fixed_img, moving_img, truth, metric=metric)
    misaligned = score_candidate(fixed_img, moving_img, sitk.Euler3DTransform(), metric=metric)
    assert aligned > misaligned, f"{metric}: {aligned} vs {misaligned}"


# --------------------------------------------------------------------------- #
# Transform application
# --------------------------------------------------------------------------- #
def test_sitk_applied_transform_resamples_and_reports():
    from regix.registration.warp import SitkAppliedTransform

    fixed_img, labels = make_phantom("CT")
    truth = known_rigid(fixed_img)
    moving_img = warp(fixed_img, truth)
    moving_labels = warp(labels, truth, is_label=True)

    applied = SitkAppliedTransform(truth, label="ground_truth")
    assert applied.kind == "sitk"
    assert applied.describe() == {"kind": "sitk", "label": "ground_truth"}
    assert applied.as_sitk_transform() is truth

    registered = applied.resample(moving_img, fixed_img)
    assert registered.GetSize() == fixed_img.GetSize()
    # Applying the true transform must bring the moving image back onto the fixed one.
    a = sitk.GetArrayFromImage(fixed_img).ravel()
    b = sitk.GetArrayFromImage(registered).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.95

    warped_labels = applied.resample(moving_labels, fixed_img, is_label=True, default_value=0)
    values = set(np.unique(sitk.GetArrayViewFromImage(warped_labels)).tolist())
    assert values <= {0, 1, 2, 3}, f"nearest neighbour must invent nothing: {values}"


def test_applied_transform_point_transport_and_field_agree():
    """The analytical path and the dense-field path must give the same answer."""
    from regix.registration.warp import SitkAppliedTransform, transform_points_via_field

    fixed_img, _ = make_phantom("CT")
    truth = known_rigid(fixed_img)
    applied = SitkAppliedTransform(truth)

    points = np.array([[0.0, 0.0, 60.0], [15.0, -25.0, 70.0], [-20.0, 10.0, 50.0]])
    analytical = applied.transform_points(points)
    assert analytical.shape == (3, 3)

    field = applied.displacement_field(fixed_img)
    assert field is not None
    through_field = transform_points_via_field(field, points)
    assert np.allclose(analytical, through_field, atol=1e-3)


def test_inverse_transport_is_exact_for_a_linear_transform():
    from regix.registration.warp import SitkAppliedTransform, warp_landmarks_moving_to_fixed

    fixed_img, _ = make_phantom("CT")
    truth = known_rigid(fixed_img)
    applied = SitkAppliedTransform(truth)

    fixed_points = np.array([[5.0, -5.0, 55.0], [22.0, 8.0, 65.0]])
    moving_points = applied.transform_points(fixed_points)
    back = warp_landmarks_moving_to_fixed(applied, moving_points, fixed_img)
    assert back is not None
    assert np.allclose(back, fixed_points, atol=1e-4)


def test_inverse_is_refused_for_a_dense_transform():
    """No approximate inverse: an invisible few-millimetre error is worse than nothing."""
    from regix.registration.warp import SitkAppliedTransform, warp_landmarks_moving_to_fixed

    fixed_img, _ = make_phantom("CT")
    shape = sitk.GetArrayViewFromImage(fixed_img).shape
    arr = np.zeros(shape + (3,), dtype=np.float64)
    arr[..., 2] = 5.0 * np.sin(np.linspace(0.0, 4.0, shape[0]))[:, None, None]
    field = sitk.GetImageFromArray(arr, isVector=True)
    field.CopyInformation(fixed_img)

    applied = SitkAppliedTransform(sitk.DisplacementFieldTransform(field))
    assert warp_landmarks_moving_to_fixed(applied, np.zeros((2, 3)), fixed_img) is None


def test_elastix_applied_transform_uses_transformix(tmp_path):
    """The fallback path: a transform carried by an elastix parameter file."""
    from regix.registration.transforms import transform_to_elastix_initial
    from regix.registration.warp import ElastixAppliedTransform

    fixed_img, _ = make_phantom("CT", shape=(32, 40, 40))
    truth = known_rigid(fixed_img, (0.0, 0.0, 0.0), (5.0, -4.0, 6.0))
    moving_img = warp(fixed_img, truth)

    parameter_file = transform_to_elastix_initial(truth, fixed_img, tmp_path / "t.txt")
    applied = ElastixAppliedTransform(parameter_file, work_dir=tmp_path / "tx", linear_transform=truth)
    assert applied.kind == "elastix"
    assert applied.describe()["linear_available"] is True

    # The background must be passed explicitly: the parameter file carries
    # DefaultPixelValue 0, whereas the phantom's air is at -1000 HU. Leaving it out
    # fills the out-of-field band with soft-tissue-like zeros.
    registered = applied.resample(moving_img, fixed_img, default_value=-1000.0)
    assert registered.GetSize() == fixed_img.GetSize()
    a = sitk.GetArrayFromImage(fixed_img).ravel()
    b = sitk.GetArrayFromImage(registered).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.9

    # With the linear transform supplied, points transport analytically.
    points = np.array([[0.0, 0.0, 60.0]])
    assert np.allclose(applied.transform_points(points), truth.TransformPoint(points[0].tolist()))


def test_elastix_applied_transform_requires_an_existing_file(tmp_path):
    from regix.registration.warp import ElastixAppliedTransform

    with pytest.raises(FileNotFoundError):
        ElastixAppliedTransform(tmp_path / "missing.txt")
