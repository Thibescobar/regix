"""Unit tests: configuration, elastix parameters, geometry, transforms, metrics."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import SimpleITK as sitk

from tests.conftest import known_rigid, make_phantom


# --------------------------------------------------------------------------- #
# Configuration and presets
# --------------------------------------------------------------------------- #
def test_the_regulatory_disclaimer_has_a_single_wording():
    """One sentence, reused everywhere: five paraphrases would eventually contradict.

    The manifest, the HTML report and the API all have to carry the same claim, because
    they are the three artefacts that leave the machine.
    """
    from regix import DISCLAIMER
    from regix.logging_utils import RunManifest

    for fragment in ("not a medical device", "qualified operator", "regulatory authority"):
        assert fragment in DISCLAIMER

    manifest = RunManifest(run_id="test", output_dir=pathlib.Path("."))
    assert manifest.to_dict()["disclaimer"] == DISCLAIMER

    licence = pathlib.Path(__file__).resolve().parent.parent / "LICENSE"
    normalised = " ".join(licence.read_text(encoding="utf-8").split())
    assert " ".join(DISCLAIMER.replace("Regix is research software.", "").split()) in normalised


def test_every_preset_is_valid():
    from regix.config import available_presets, load_preset

    names = available_presets()
    assert "base" in names and len(names) >= 6
    for name in names:
        cfg = load_preset(name)
        assert cfg.stages, f"{name} has no stage"
        assert cfg.name == name


def test_preset_inheritance_replaces_stages():
    from regix.config import load_preset

    base = load_preset("base")
    child = load_preset("ct_cbct_igrt")
    assert len(base.stages) == 2
    assert len(child.stages) == 1           # the child replaces, it does not concatenate
    assert child.preprocess.fixed.window == "ct_bone"
    assert child.preprocess.orientation == base.preprocess.orientation  # inherited


def test_config_rejects_incoherent_combinations():
    from pydantic import ValidationError

    from regix.config import DeformableEngine, RegistrationConfig, StageConfig, TransformType

    with pytest.raises(ValidationError):
        RegistrationConfig(
            stages=[StageConfig(type=TransformType.BSPLINE)],
            deformable_engine=DeformableEngine.CONVEXADAM,
        )
    with pytest.raises(ValidationError):
        RegistrationConfig(stages=[])
    with pytest.raises(ValidationError):  # organ_centroid without a segmentation
        RegistrationConfig(init={"mode": "organ_centroid"})


def test_deep_override_preserves_the_rest():
    from regix.config import load_preset

    cfg = load_preset("ct_mr_abdomen")
    modified = cfg.with_overrides(preprocess={"working_spacing_mm": 1.0})
    assert modified.preprocess.working_spacing_mm == 1.0
    assert modified.preprocess.fixed.window == cfg.preprocess.fixed.window
    assert len(modified.stages) == len(cfg.stages)


# --------------------------------------------------------------------------- #
# elastix parameters
# --------------------------------------------------------------------------- #
def test_automatic_metric_choice():
    from regix.config import Metric, StageConfig, TransformType
    from regix.registration.params import ParamContext, resolve_metric

    stage = StageConfig(type=TransformType.RIGID)
    monomodal = ParamContext(fixed_modality="CT", moving_modality="CT")
    multimodal = ParamContext(fixed_modality="CT", moving_modality="MR")
    with_features = ParamContext(
        fixed_modality="CT", moving_modality="MR", features_available=True, n_channels=4
    )
    assert resolve_metric(stage, monomodal) is Metric.NCC
    assert resolve_metric(stage, multimodal) is Metric.MI
    assert resolve_metric(stage, with_features) is Metric.FEATURES_NCC


def test_multi_channel_yields_one_entry_per_metric():
    from regix.config import Metric, StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(type=TransformType.AFFINE, metric=Metric.FEATURES_NCC)
    ctx = ParamContext(n_channels=4, features_available=True)
    pmap = build_parameter_map(stage, ctx)
    assert pmap["Registration"] == ("MultiMetricMultiResolutionRegistration",)
    for key in ("Metric", "FixedImagePyramid", "MovingImagePyramid", "Interpolator", "ImageSampler"):
        assert len(pmap[key]) == 4, key
    weights = sum(float(pmap[f"Metric{i}Weight"][0]) for i in range(4))
    assert weights == pytest.approx(1.0)


def test_bspline_adds_the_bending_energy_penalty():
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(type=TransformType.BSPLINE, n_resolutions=3, final_grid_spacing_mm=15.0)
    pmap = build_parameter_map(stage, ParamContext())
    assert "TransformBendingEnergyPenalty" in pmap["Metric"]
    assert pmap["FinalGridSpacingInPhysicalUnits"] == ("15.0000",) * 3
    assert pmap["GridSpacingSchedule"] == ("4.0000", "2.0000", "1.0000")
    assert pmap["Registration"] == ("MultiMetricMultiResolutionRegistration",)


def test_direction_cosines_are_always_enabled():
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    for transform in TransformType:
        pmap = build_parameter_map(StageConfig(type=transform), ParamContext())
        assert pmap["UseDirectionCosines"] == ("true",)
        assert pmap["HowToCombineTransforms"] == ("Compose",)


def test_parameter_file_round_trip(tmp_path):
    from regix.config import StageConfig, TransformType
    from regix.registration.params import (
        ParamContext,
        build_parameter_map,
        read_parameter_file,
        write_parameter_file,
    )

    pmap = build_parameter_map(StageConfig(type=TransformType.RIGID), ParamContext())
    path = write_parameter_file(pmap, tmp_path / "p.txt")
    reloaded = read_parameter_file(path)
    assert reloaded["Transform"] == ("EulerTransform",)
    assert reloaded["NumberOfResolutions"] == pmap["NumberOfResolutions"]
    assert reloaded["Metric"] == pmap["Metric"]


def test_valid_sample_ratio_is_permissive():
    """Regression: the elastix default (0.25) fails on disjoint fields of view.

    Observed symptom: 'Too many samples map outside moving image buffer'. Partially
    disjoint fields of view are Regix's main use case (whole-body CT against
    abdominal MR), so this parameter must stay low.
    """
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    pmap = build_parameter_map(StageConfig(type=TransformType.RIGID), ParamContext())
    assert "RequiredRatioOfValidSamples" in pmap
    assert float(pmap["RequiredRatioOfValidSamples"][0]) <= 0.1

    strict = build_parameter_map(
        StageConfig(type=TransformType.RIGID, required_ratio_valid_samples=0.5), ParamContext()
    )
    assert float(strict["RequiredRatioOfValidSamples"][0]) == 0.5


def test_user_override_has_the_final_say():
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(type=TransformType.RIGID, extra={"MaximumStepLength": "9.5"})
    pmap = build_parameter_map(stage, ParamContext())
    assert pmap["MaximumStepLength"] == ("9.5",)


def test_elastix_engine_is_available():
    """Guard rail: SimpleITK no longer bundles elastix, itk-elastix must be present."""
    from regix.registration.itk_bridge import engine_available

    ok, detail = engine_available()
    assert ok, detail


def test_image_count_required_by_elastix():
    """A penalty is a metric without an image: the image count must follow."""
    from regix.config import Metric, StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map, required_image_count

    ctx_mono = ParamContext(n_channels=1)
    ctx_multi = ParamContext(n_channels=4, features_available=True)

    rigid = build_parameter_map(StageConfig(type=TransformType.RIGID), ctx_mono)
    assert required_image_count(rigid) == 1

    features = build_parameter_map(
        StageConfig(type=TransformType.AFFINE, metric=Metric.FEATURES_NCC), ctx_multi
    )
    assert required_image_count(features) == 4

    # 4 channels + bending-energy penalty = 5 metrics -> 5 images expected
    bspline = build_parameter_map(
        StageConfig(type=TransformType.BSPLINE, metric=Metric.FEATURES_NCC), ctx_multi
    )
    assert len(bspline["Metric"]) == 5
    assert required_image_count(bspline) == 5
    assert len(bspline["FixedImagePyramid"]) == 5

    # 1 channel + penalty: elastix accepts a single image for all metrics
    bspline_mono = build_parameter_map(StageConfig(type=TransformType.BSPLINE), ctx_mono)
    assert len(bspline_mono["Metric"]) == 2
    assert required_image_count(bspline_mono) == 1


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_resampling_preserves_the_physical_extent(ct_phantom):
    from regix.preprocess.geometry import resample_to_spacing

    image, _ = ct_phantom
    resampled = resample_to_spacing(image, 3.0)
    before = np.asarray(image.GetSize()) * np.asarray(image.GetSpacing())
    after = np.asarray(resampled.GetSize()) * np.asarray(resampled.GetSpacing())
    assert np.allclose(before, after, atol=3.0)
    assert np.allclose(resampled.GetOrigin(), image.GetOrigin())


def test_mask_centre_of_mass_is_its_barycentre(ct_phantom):
    from regix.preprocess.geometry import center_of_mass_physical

    image, labels = ct_phantom
    liver = sitk.Cast(sitk.Equal(labels, 1), sitk.sitkUInt8)
    com = center_of_mass_physical(image, liver)
    arr = sitk.GetArrayViewFromImage(liver)
    idx = np.argwhere(arr > 0).mean(axis=0)
    expected = labels.TransformContinuousIndexToPhysicalPoint(
        [float(idx[2]), float(idx[1]), float(idx[0])]
    )
    assert np.allclose(com, expected, atol=1e-6)


def test_principal_axes_are_orthonormal(ct_phantom):
    from regix.preprocess.geometry import principal_axes

    _, labels = ct_phantom
    liver = sitk.Cast(sitk.Equal(labels, 1), sitk.sitkUInt8)
    _, axes, lengths = principal_axes(liver)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-6)
    assert np.all(np.diff(lengths) <= 1e-9)  # sorted in decreasing order


def test_dilation_in_mm_respects_anisotropy(ct_phantom):
    from regix.preprocess.geometry import dilate_mask_mm

    _, labels = ct_phantom
    liver = sitk.Cast(sitk.Equal(labels, 1), sitk.sitkUInt8)
    dilated = dilate_mask_mm(liver, 6.0)
    before = int(sitk.GetArrayViewFromImage(liver).sum())
    after = int(sitk.GetArrayViewFromImage(dilated).sum())
    assert after > before


def test_padding_to_multiples_of_16_and_back():
    from regix.preprocess.geometry import pad_to_multiple, unpad

    arr = np.random.default_rng(0).random((5, 30, 47)).astype(np.float32)
    padded, pads = pad_to_multiple(arr, 16)
    assert all(dim % 16 == 0 for dim in padded.shape)
    assert np.allclose(unpad(padded, pads), arr)
    stacked = np.stack([arr] * 3)
    padded_stack, pads_stack = pad_to_multiple(stacked, 16)
    assert padded_stack.shape[0] == 3
    assert np.allclose(unpad(padded_stack, pads_stack), stacked)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_matrix_4x4_reproduces_the_transform(ct_phantom):
    from regix.registration.transforms import to_matrix_4x4

    image, _ = ct_phantom
    truth = known_rigid(image)
    M = to_matrix_4x4(truth)
    point = np.array([12.0, -33.0, 47.0])
    expected = np.asarray(truth.TransformPoint(point.tolist()))
    assert np.allclose(M[:3, :3] @ point + M[:3, 3], expected, atol=1e-8)


def test_linearisation_of_a_composition(ct_phantom):
    from regix.preprocess.geometry import image_center_physical
    from regix.registration.initialize import with_extra_rotation
    from regix.registration.transforms import linear_matrix_from_transform

    image, _ = ct_phantom
    centre = image_center_physical(image)
    composite = with_extra_rotation(known_rigid(image), centre, (0, 0, 15))
    M = linear_matrix_from_transform(composite)
    assert M is not None
    point = np.array([5.0, 9.0, -21.0])
    assert np.allclose(
        M[:3, :3] @ point + M[:3, 3], np.asarray(composite.TransformPoint(point.tolist())), atol=1e-6
    )


def test_linearisation_refuses_a_dense_field(ct_phantom):
    from regix.registration.transforms import linear_matrix_from_transform

    image, _ = ct_phantom
    arr = np.zeros(sitk.GetArrayViewFromImage(image).shape + (3,), dtype=np.float64)
    arr[..., 0] = np.linspace(0, 12, arr.shape[2])[None, None, :] ** 1.5  # non-linear
    field = sitk.GetImageFromArray(arr, isVector=True)
    field.CopyInformation(image)
    assert linear_matrix_from_transform(sitk.DisplacementFieldTransform(field)) is None


def test_linear_flattening_is_exact_and_single(tmp_path, ct_phantom):
    """A linear chain must be written as ONE affine, losslessly.

    A multi-level CompositeTransform is unreadable in a visualisation station; the
    flattening is mathematically exact, so there is no reason to impose the former
    on the user.
    """
    from regix.preprocess.geometry import image_center_physical
    from regix.registration.initialize import with_extra_rotation
    from regix.registration.transforms import compose, flatten_linear, save_transform

    image, _ = ct_phantom
    centre = image_center_physical(image)
    chain = compose(
        [
            with_extra_rotation(known_rigid(image), centre, (0, 0, 12)),
            sitk.TranslationTransform(3, (3.0, -7.0, 2.0)),
        ]
    )
    flat = flatten_linear(chain)
    assert flat is not None
    assert isinstance(flat, sitk.AffineTransform)
    for point in ([0.0, 0.0, 0.0], [61.0, -23.0, 44.0], [-88.0, 12.0, -5.0]):
        assert np.allclose(flat.TransformPoint(point), chain.TransformPoint(point), atol=1e-6)

    # Insight Transform File with a single transform: what 3D Slicer reads.
    path = save_transform(flat, tmp_path / "final_transform.txt")
    body = path.read_text(encoding="utf-8")
    assert "#Insight Transform File V1.0" in body
    assert body.count("#Transform ") == 1
    assert "AffineTransform_double_3_3" in body
    assert "CompositeTransform" not in body
    assert np.allclose(
        sitk.ReadTransform(str(path)).TransformPoint((10.0, 20.0, 30.0)),
        chain.TransformPoint([10.0, 20.0, 30.0]),
        atol=1e-6,
    )


def test_flattening_refuses_a_dense_transform(ct_phantom):
    """Flattening a dense field would produce a wrong transform: it must refuse."""
    from regix.registration.transforms import flatten_linear

    image, _ = ct_phantom
    shape = sitk.GetArrayViewFromImage(image).shape
    arr = np.zeros(shape + (3,), dtype=np.float64)
    arr[..., 2] = np.linspace(0.0, 9.0, shape[0])[:, None, None] ** 2
    field = sitk.GetImageFromArray(arr, isVector=True)
    field.CopyInformation(image)
    assert flatten_linear(sitk.DisplacementFieldTransform(field)) is None


def test_affine_decomposition_recovers_the_parameters():
    from regix.registration.transforms import decompose_affine

    angle = np.radians(20.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = R * 1.1
    M[:3, 3] = [10.0, -4.0, 2.0]
    out = decompose_affine(M)
    assert out["translation_norm_mm"] == pytest.approx(np.linalg.norm([10, -4, 2]), abs=1e-3)
    assert out["rotation_norm_deg"] == pytest.approx(20.0, abs=0.5)
    assert out["max_scale_deviation"] == pytest.approx(0.1, abs=1e-3)
    assert out["determinant"] > 0


def test_dicom_matrix_is_the_inverse(ct_phantom):
    from regix.registration.transforms import matrix_moving_to_fixed, to_matrix_4x4

    image, _ = ct_phantom
    truth = known_rigid(image)
    forward = to_matrix_4x4(truth)
    backward = matrix_moving_to_fixed(truth)
    assert np.allclose(forward @ backward, np.eye(4), atol=1e-8)


def test_initial_transform_file_is_readable(tmp_path, ct_phantom):
    """Round trip Regix -> elastix file -> Regix, without loss."""
    from regix.registration.params import read_parameter_file
    from regix.registration.transforms import parameter_map_to_transform, transform_to_elastix_initial

    image, _ = ct_phantom
    truth = known_rigid(image)
    path = transform_to_elastix_initial(truth, image, tmp_path / "t0.txt")
    body = path.read_text(encoding="utf-8")
    # elastix 5.x reads the singular key: a typo here would silently break the chain.
    assert '(InitialTransformParameterFileName "NoInitialTransform")' in body
    reloaded = parameter_map_to_transform(read_parameter_file(path))
    point = [3.0, -12.0, 8.0]
    assert np.allclose(reloaded.TransformPoint(point), truth.TransformPoint(point), atol=1e-6)


def test_composition_applies_in_list_order():
    from regix.registration.transforms import compose

    a = sitk.TranslationTransform(3, (10.0, 0.0, 0.0))
    b = sitk.TranslationTransform(3, (0.0, 5.0, 0.0))
    assert np.allclose(compose([a, b]).TransformPoint((0, 0, 0)), (10.0, 5.0, 0.0))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_similarity_metrics_on_identical_images(ct_phantom):
    from regix.qc.metrics import normalized_cross_correlation, normalized_mutual_information

    image, _ = ct_phantom
    assert normalized_cross_correlation(image, image) == pytest.approx(1.0, abs=1e-6)
    assert normalized_mutual_information(image, image) > 1.5


def test_dice_and_surface_distances(ct_phantom):
    from regix.qc.metrics import dice, hausdorff95, mean_surface_distance

    _, labels = ct_phantom
    liver = sitk.Cast(sitk.Equal(labels, 1), sitk.sitkUInt8)
    spleen = sitk.Cast(sitk.Equal(labels, 2), sitk.sitkUInt8)
    assert dice(liver, liver) == pytest.approx(1.0)
    assert dice(liver, spleen) == pytest.approx(0.0)
    assert hausdorff95(liver, liver) == pytest.approx(0.0, abs=1e-6)
    assert mean_surface_distance(liver, liver) == pytest.approx(0.0, abs=1e-6)


def test_jacobian_detects_folding(ct_phantom):
    from regix.qc.metrics import jacobian_statistics

    image, _ = ct_phantom
    shape = sitk.GetArrayViewFromImage(image).shape
    neutral = sitk.GetImageFromArray(np.zeros(shape + (3,), dtype=np.float64), isVector=True)
    neutral.CopyInformation(image)
    stats = jacobian_statistics(neutral)
    assert stats["det_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["folding_voxels"] == 0

    folded = np.zeros(shape + (3,), dtype=np.float64)
    # An x displacement growing faster than the grid -> negative Jacobian
    folded[..., 0] = -3.0 * np.arange(shape[2], dtype=np.float64)[None, None, :] * image.GetSpacing()[0]
    field = sitk.GetImageFromArray(folded, isVector=True)
    field.CopyInformation(image)
    assert jacobian_statistics(field)["folding_fraction"] > 0.5


def test_tre_decreases_with_the_correct_transform(ct_phantom):
    from regix.qc.metrics import target_registration_error

    image, _ = ct_phantom
    truth = known_rigid(image)
    fixed_points = np.array([[0.0, 0.0, 60.0], [10.0, -20.0, 70.0], [-15.0, 5.0, 55.0]])
    moving_points = np.array([truth.TransformPoint(p.tolist()) for p in fixed_points])
    out = target_registration_error(fixed_points, moving_points, truth)
    assert out["tre_mean_mm"] == pytest.approx(0.0, abs=1e-3)
    assert out["tre_before_mean_mm"] > 1.0


# --------------------------------------------------------------------------- #
# QC gates
# --------------------------------------------------------------------------- #
def test_gates_flag_folding_and_divergence():
    from regix.config import QCGates
    from regix.qc.gates import evaluate_gates

    gates = QCGates(min_ncc_gain=0.0, min_dice={"liver": 0.9}, max_folding_fraction=1e-4)
    result = evaluate_gates(
        gates,
        similarity={"ncc_gain": -0.05},
        organ_overlap={"liver": {"dice": 0.5}},
        jacobian={"available": True, "folding_fraction": 0.01, "det_min": -0.3, "det_max": 4.0},
        linear_analysis={"translation_norm_mm": 400.0, "max_scale_deviation": 0.9, "determinant": 1.0},
    )
    assert result.status == "FAIL"
    names = {c.name for c in result.failures}
    assert {"ncc_gain", "dice[liver]", "folding_fraction", "translation_mm", "scale_deviation"} <= names


def test_a_missing_measurement_warns_rather_than_fails():
    from regix.config import QCGates
    from regix.qc.gates import evaluate_gates

    result = evaluate_gates(QCGates(min_dice={"liver": 0.9}), organ_overlap={})
    assert result.status == "WARN"


# --------------------------------------------------------------------------- #
# MIND descriptor (the CPU multimodal path)
# --------------------------------------------------------------------------- #
def test_mind_localises_better_than_intensities_across_modalities():
    """What matters for registration is not absolute correlation but **peak sharpness**.

    We sweep a displacement along z and compare the shape of the similarity curve: a
    useful criterion must collapse as soon as we move away from alignment. Intensity
    correlation between modalities stays high even when misaligned (it is carried by
    the body/air contrast), which gives a flat optimum and an imprecise registration.
    MIND, on the other hand, peaks.
    """
    from regix.features.mind import mind_ssc_features
    from regix.preprocess.intensity import normalize_for_features

    ct, _ = make_phantom("CT", noise=1.0, seed=1)
    mr, _ = make_phantom("MR", noise=8.0, seed=2)
    ct_arr = normalize_for_features(ct, clip=(-450.0, 450.0))
    mr_arr = normalize_for_features(mr)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom else 0.0

    f_ct = mind_ssc_features(ct_arr, spacing=(2.5, 2.0, 2.0))
    f_mr = mind_ssc_features(mr_arr, spacing=(2.5, 2.0, 2.0))
    assert f_ct.shape == (12,) + ct_arr.shape

    intensity: dict[int, float] = {}
    mind: dict[int, float] = {}
    for shift in (-4, 0, 4):
        intensity[shift] = _corr(ct_arr, np.roll(mr_arr, shift, axis=0))
        shifted = np.roll(f_mr, shift, axis=1)
        mind[shift] = float(np.mean([_corr(f_ct[c], shifted[c]) for c in range(12)]))

    # 1. Both criteria must peak at the correct alignment.
    assert mind[0] > mind[-4] and mind[0] > mind[4]
    assert intensity[0] > intensity[-4] and intensity[0] > intensity[4]

    # 2. But MIND must fall off far faster -> a better-defined optimum.
    def _sharpness(curve: dict[int, float]) -> float:
        return (curve[0] - (curve[-4] + curve[4]) / 2) / abs(curve[0])

    assert _sharpness(mind) > 2.0 * _sharpness(intensity), (
        f"MIND sharpness={_sharpness(mind):.3f} vs intensity={_sharpness(intensity):.3f}"
    )


def test_shared_pca_reduces_channels_and_retains_variance():
    from regix.features.reduce import joint_pca_reduce, voxel_normalize

    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 8, 10, 12))
    fixed = np.concatenate([base, base * 0.5 + 0.01 * rng.normal(size=base.shape)], axis=0)
    moving = np.concatenate([base + 0.05, base * 0.5], axis=0)
    reduced_f, reduced_m, info = joint_pca_reduce(fixed, moving, n_components=3)
    assert reduced_f.shape == (3, 8, 10, 12)
    assert reduced_m.shape == (3, 8, 10, 12)
    assert info["explained_variance_ratio"] > 0.9

    normalized = voxel_normalize(fixed, "l2")
    assert np.allclose(np.linalg.norm(normalized, axis=0), 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# QC report
# --------------------------------------------------------------------------- #
def test_jacobian_figure_is_skipped_for_a_linear_transform(ct_phantom):
    """A linear transform has a constant Jacobian: the map carries no information.

    Rendering it produced a uniform grey square that looked like a rendering bug.
    The value belongs in the metrics table instead.
    """
    from regix.qc.report import jacobian_figure
    from regix.registration.convexadam import displacement_field_from_transform

    image, _ = ct_phantom
    field = displacement_field_from_transform(known_rigid(image), image)
    assert jacobian_figure(field) is None

    # A genuinely deformable field, on the other hand, must be rendered.
    shape = sitk.GetArrayViewFromImage(image).shape
    arr = np.zeros(shape + (3,), dtype=np.float64)
    arr[..., 2] = 6.0 * np.sin(np.linspace(0, 3.0, shape[0]))[:, None, None]
    deformable = sitk.GetImageFromArray(arr, isVector=True)
    deformable.CopyInformation(image)
    figure = jacobian_figure(deformable)
    assert figure is not None and figure.startswith("data:image/png;base64,")
