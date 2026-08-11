"""Focused regressions for contracts introduced while closing AUDIT.md."""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from pydantic import ValidationError

from regix.config import QCGates, RegistrationConfig, StageConfig
from regix.features import FeaturePair
from regix.io.volume import Volume, load_volume
from regix.io.writers import load_landmarks
from regix.layout import clean_known_artifacts
from regix.logging_utils import RunManifest, pseudonymization_warning, pseudonymize
from regix.organs.roi import organ_volumes_ml
from regix.organs.segmenter import OrganSegmentation, OrganSegmenter, TotalSegmentatorSegmenter
from regix.pipeline import RunState, param_context_from_state, step_native_masks, step_work_masks
from regix.preprocess.geometry import (
    dilate_mask_mm,
    displacement_field_from_transform,
    mask_bounding_box_mm,
    principal_axes,
    resample_to_spacing,
    same_grid,
)
from regix.qc.gates import FAIL, WARN, evaluate_gates
from regix.qc.report import build_html_report
from regix.registration.itk_bridge import _verification_probes
from regix.registration.params import (
    ParamContext,
    build_parameter_map,
    read_parameter_file,
    write_parameter_file,
)
from regix.registration.transforms import (
    compose,
    load_any_transform,
    save_transform,
    transform_to_elastix_initial,
)


def _volume(image: sitk.Image, modality: str = "CT") -> Volume:
    return Volume(image=image, modality=modality, subject_id="subject")


def _state(tmp_path: Path, fixed: Volume, moving: Volume) -> RunState:
    return RunState(
        fixed=fixed,
        moving=moving,
        targets=(),
        mask_dilate_mm=8.0,
        manifest=RunManifest("audit", tmp_path, redact_paths=False),
        out_dir=tmp_path,
        outputs={},
    )


def _feature_pair(provider: str) -> FeaturePair:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    descriptor = "Anatomix" if provider == "anatomix" else "MIND-SSC"
    return FeaturePair(
        fixed_channels=[image],
        moving_channels=[sitk.Image(image)],
        provider=provider,
        info={"provider": provider, "descriptor": descriptor, "pca": {}},
    )


def _feature_inputs(tmp_path: Path, provider: str = "auto"):
    from regix.pipeline import RegistrationPipeline

    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    fixed = _volume(image, "CT")
    moving = _volume(sitk.Image(image), "MR")
    config = RegistrationConfig(
        features={"enabled": True, "provider": provider},
        deformable_engine="none",
    )
    manifest = RunManifest("features", tmp_path, redact_paths=False)
    return RegistrationPipeline(config), fixed, moving, manifest


def test_feature_provider_auto_uses_anatomix_when_available(tmp_path, monkeypatch):
    from regix.features import anatomix as anatomix_module

    pipeline, fixed, moving, manifest = _feature_inputs(tmp_path)
    monkeypatch.setattr(anatomix_module, "anatomix_available", lambda: (True, "ok"))
    monkeypatch.setattr(
        anatomix_module,
        "extract_feature_pair",
        lambda *_args, **_kwargs: _feature_pair("anatomix"),
    )

    _, _, info = pipeline._extract_features(fixed, moving, None, None, manifest)

    assert info["requested_provider"] == "auto"
    assert info["provider"] == "anatomix"
    assert info["descriptor"] == "Anatomix"
    assert info["fallback"] is None


def test_feature_provider_auto_falls_back_to_mind_when_anatomix_is_unavailable(tmp_path, monkeypatch):
    from regix.features import anatomix as anatomix_module
    from regix.features import mind as mind_module

    pipeline, fixed, moving, manifest = _feature_inputs(tmp_path)
    monkeypatch.setattr(
        anatomix_module,
        "anatomix_available",
        lambda: (False, "the anatomix package is not installed"),
    )
    monkeypatch.setattr(
        mind_module,
        "extract_mind_feature_pair",
        lambda *_args, **_kwargs: _feature_pair("mind"),
    )

    _, _, info = pipeline._extract_features(fixed, moving, None, None, manifest)

    assert info["requested_provider"] == "auto"
    assert info["provider"] == "mind"
    assert info["descriptor"] == "MIND-SSC"
    assert info["fallback"] == "anatomix -> mind"
    assert any("anatomix unavailable" in warning for warning in manifest.warnings)


def test_feature_provider_anatomix_fails_when_unavailable(tmp_path, monkeypatch):
    from regix.features import anatomix as anatomix_module
    from regix.pipeline import RegistrationFailure

    pipeline, fixed, moving, manifest = _feature_inputs(tmp_path, provider="anatomix")
    monkeypatch.setattr(anatomix_module, "anatomix_available", lambda: (False, "missing weights"))

    with pytest.raises(RegistrationFailure, match="features.provider=anatomix.*missing weights"):
        pipeline._extract_features(fixed, moving, None, None, manifest)


def test_feature_provider_mind_does_not_import_anatomix_or_torch(tmp_path, monkeypatch):
    import builtins

    from regix.features import mind as mind_module

    pipeline, fixed, moving, manifest = _feature_inputs(tmp_path, provider="mind")
    monkeypatch.setattr(
        mind_module,
        "extract_mind_feature_pair",
        lambda *_args, **_kwargs: _feature_pair("mind"),
    )
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "anatomix" or name.startswith("torch") or name == "regix.features.anatomix":
            imported.append(name)
            raise AssertionError(f"forbidden import on the MIND path: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _, _, info = pipeline._extract_features(fixed, moving, None, None, manifest)

    assert info["provider"] == "mind"
    assert imported == []


def test_features_disabled_keep_multimodal_registration_on_intensities():
    from regix.pipeline import RegistrationPipeline
    from regix.registration.params import ParamContext, resolve_metric

    config = RegistrationConfig(features={"enabled": False}, deformable_engine="none")
    assert not RegistrationPipeline(config)._features_wanted(multimodal=True)
    context = ParamContext(
        dimension=3,
        n_channels=1,
        working_spacing_mm=2.0,
        has_mask=False,
        fixed_modality="CT",
        moving_modality="MR",
        features_available=False,
        n_voxels=32**3,
        intensity_range=(0.0, 1.0),
    )
    assert resolve_metric(config.stages[0], context).value == "mi"


def test_historical_feature_configuration_defaults_to_auto_provider(tmp_path, monkeypatch):
    from regix.features import anatomix as anatomix_module

    config = RegistrationConfig.model_validate({"features": {"enabled": True}, "deformable_engine": "none"})
    assert config.features.provider == "auto"
    pipeline, fixed, moving, manifest = _feature_inputs(tmp_path)
    pipeline.config = config
    monkeypatch.setattr(anatomix_module, "anatomix_available", lambda: (True, "ok"))
    monkeypatch.setattr(
        anatomix_module,
        "extract_feature_pair",
        lambda *_args, **_kwargs: _feature_pair("anatomix"),
    )

    _, _, info = pipeline._extract_features(fixed, moving, None, None, manifest)
    assert info["provider"] == "anatomix"


def test_a_4d_volume_yields_the_first_3d_time_point(tmp_path):
    array = np.arange(2 * 5 * 6 * 7, dtype=np.float32).reshape(2, 5, 6, 7)
    path = tmp_path / "timeseries.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(array, isVector=False), str(path))

    loaded = load_volume(path, pseudonymize_ids=False)

    assert loaded.image.GetDimension() == 3
    assert np.array_equal(sitk.GetArrayFromImage(loaded.image), array[0])


def test_a_2d_image_is_refused(tmp_path):
    path = tmp_path / "slice.nii.gz"
    sitk.WriteImage(sitk.Image([20, 20], sitk.sitkFloat32), str(path))

    with pytest.raises(ValueError, match="2D"):
        load_volume(path)


def test_duplicate_organ_labels_are_unioned_and_their_volumes_are_summed():
    array = np.zeros((8, 8, 8), dtype=np.uint16)
    array[1:3, 1:3, 1:3] = 1
    array[5:7, 5:7, 5:7] = 2
    labelmap = sitk.GetImageFromArray(array)
    labelmap.SetSpacing((2.0, 2.0, 2.0))
    segmentation = OrganSegmentation(labelmap, {1: "lung_left", 2: "lung_left"}, "test")

    mask = segmentation.mask_for(["lung_left"])

    assert segmentation.labels_of("lung_left") == [1, 2]
    assert segmentation.present_organs() == ["lung_left"]
    assert int(sitk.GetArrayViewFromImage(mask).sum()) == 16
    assert organ_volumes_ml(segmentation)["lung_left"] == pytest.approx(0.13)


def test_native_body_mask_is_built_once_and_resampled_by_role(tmp_path):
    image = sitk.Image([32, 32, 32], sitk.sitkFloat32) + 50.0
    image.SetSpacing((1.0, 1.0, 2.0))
    state = _state(tmp_path, _volume(image), _volume(sitk.Image(image)))

    step_native_masks(state)
    state.fixed_work = state.fixed.with_image(resample_to_spacing(state.fixed.image, 2.0))
    state.moving_work = state.moving.with_image(resample_to_spacing(state.moving.image, 2.0))
    step_work_masks(state)

    assert state.criterion_mask_fixed is not None
    assert state.initialization_mask_fixed is not None
    assert state.initialization_mask_fixed.image is state.criterion_mask_fixed.image
    assert state.work_mask_fixed is not None
    assert state.init_mask_fixed is not None
    assert same_grid(state.work_mask_fixed, state.fixed_work.image)
    assert same_grid(state.init_mask_fixed, state.fixed_work.image)


def test_segmented_initialization_mask_is_not_dilated(tmp_path):
    array = np.full((32, 32, 32), 50.0, dtype=np.float32)
    labels = np.zeros_like(array, dtype=np.uint16)
    labels[12:20, 12:20, 12:20] = 1
    image = sitk.GetImageFromArray(array)
    labelmap = sitk.GetImageFromArray(labels)
    segmentation = OrganSegmentation(labelmap, {1: "liver"}, "test")
    state = _state(tmp_path, _volume(image), _volume(sitk.Image(image)))
    state.targets = ("liver",)
    state.fixed_seg = segmentation
    state.moving_seg = segmentation

    step_native_masks(state)

    assert state.criterion_mask_fixed is not None
    assert state.initialization_mask_fixed is not None
    criterion_count = int(sitk.GetArrayViewFromImage(state.criterion_mask_fixed.image).sum())
    initialization_count = int(sitk.GetArrayViewFromImage(state.initialization_mask_fixed.image).sum())
    assert state.criterion_mask_fixed.dilated_mm == 8.0
    assert state.initialization_mask_fixed.dilated_mm == 0.0
    assert criterion_count > initialization_count


def test_param_context_is_complete_and_comes_from_the_run_state(tmp_path):
    image = sitk.Image([32, 32, 32], sitk.sitkFloat32) + 50.0
    state = _state(tmp_path, _volume(image, "CT"), _volume(sitk.Image(image), "MR"))
    state.fixed_work = state.fixed
    state.moving_work = state.moving
    state.fixed_channels = [image, image]

    context = param_context_from_state(state)

    assert context.n_channels == 2
    assert context.fixed_modality == "CT"
    assert context.moving_modality == "MR"
    assert context.n_voxels == 32**3
    assert context.intensity_range == (50.0, 50.0)


def test_principal_axes_matches_a_physical_point_reference():
    array = np.zeros((9, 10, 11), dtype=np.uint8)
    array[2:8, 3:9, 1:7] = 1
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((1.2, 2.0, 4.0))
    mask.SetOrigin((-10.0, 7.0, 25.0))
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.15, -0.08, 0.04)
    mask.SetDirection(rotation.GetMatrix())

    centroid, axes, lengths = principal_axes(mask)
    indices = np.argwhere(array > 0)[:, ::-1]
    points = np.asarray(
        [mask.TransformIndexToPhysicalPoint(tuple(int(v) for v in index)) for index in indices]
    )
    expected_centroid = points.mean(axis=0)
    expected_covariance = np.cov(points, rowvar=False)
    reconstructed_covariance = axes @ np.diag(lengths**2) @ axes.T

    assert np.allclose(centroid, expected_centroid, atol=1e-10)
    assert np.allclose(reconstructed_covariance, expected_covariance, atol=1e-10)


def test_bounding_box_accepts_a_multilabel_map():
    array = np.zeros((9, 10, 11), dtype=np.uint16)
    array[2:4, 3:5, 1:3] = 3
    array[6:8, 7:9, 8:10] = 7

    start, size = mask_bounding_box_mm(sitk.GetImageFromArray(array))

    assert start == [1, 3, 2]
    assert size == [9, 6, 6]


def test_small_dilation_does_not_jump_across_a_thick_slice():
    array = np.zeros((5, 9, 9), dtype=np.uint8)
    array[2, 4, 4] = 1
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((1.0, 1.0, 5.0))

    dilated = dilate_mask_mm(mask, 2.0)
    result = sitk.GetArrayViewFromImage(dilated)

    assert int(result[1].sum()) == 0
    assert int(result[3].sum()) == 0


def test_dense_field_helper_is_float32():
    reference = sitk.Image([8, 9, 10], sitk.sitkFloat32)
    transform = sitk.TranslationTransform(3, (1.0, 2.0, 3.0))

    field = displacement_field_from_transform(transform, reference)

    assert field.GetPixelID() == sitk.sitkVectorFloat32


def test_transform_conversion_probes_stay_inside_an_oblique_field_of_view():
    reference = sitk.Image([8, 9, 10], sitk.sitkFloat32)
    reference.SetSpacing((1.2, 2.0, 4.0))
    reference.SetOrigin((500.0, -700.0, 1200.0))
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.2, -0.1, 0.05)
    reference.SetDirection(rotation.GetMatrix())

    probes = _verification_probes(reference)

    assert len(probes) == 9
    for point in probes:
        index = reference.TransformPhysicalPointToContinuousIndex(point)
        assert all(
            -1e-6 <= value <= size - 1 + 1e-6 for value, size in zip(index, reference.GetSize(), strict=True)
        )


def test_every_linear_transform_file_written_by_a_run_is_loadable(tmp_path):
    reference = sitk.Image([16, 17, 18], sitk.sitkFloat32)
    transform = sitk.Euler3DTransform()
    transform.SetCenter((4.0, 5.0, 6.0))
    transform.SetComputeZYX(True)
    transform.SetRotation(0.1, -0.2, 0.05)
    transform.SetTranslation((3.0, -2.0, 1.0))
    paths = [
        save_transform(transform, tmp_path / "final.tfm"),
        save_transform(transform, tmp_path / "final.itk.txt"),
        save_transform(transform, tmp_path / "final.h5"),
        transform_to_elastix_initial(transform, reference, tmp_path / "initial.elastix.txt"),
    ]
    probes = ((0.0, 0.0, 0.0), (12.0, -4.0, 27.0), (-8.0, 6.0, 2.0))

    for path in paths:
        loaded = load_any_transform(path)
        for point in probes:
            assert np.allclose(loaded.TransformPoint(point), transform.TransformPoint(point), atol=1e-8), path


def test_compose_of_one_transform_returns_an_independent_copy():
    source = sitk.TranslationTransform(3, (1.0, 2.0, 3.0))

    result = compose([source])

    assert result is not source
    assert result.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx((1.0, 2.0, 3.0))


def test_volume_with_image_does_not_share_metadata():
    source = Volume(sitk.Image([4, 4, 4], sitk.sitkFloat32), meta={"nested": "original"})

    derived = source.with_image(sitk.Image(source.image))
    derived.meta["new"] = True

    assert source.meta == {"nested": "original"}


def test_overwrite_cleanup_preserves_user_files(tmp_path):
    (tmp_path / "transform").mkdir()
    (tmp_path / "transform" / "stale.tfm").write_text("stale", encoding="utf-8")
    (tmp_path / "report.html").write_text("stale", encoding="utf-8")
    note = tmp_path / "review-notes.txt"
    note.write_text("keep me", encoding="utf-8")

    removed = clean_known_artifacts(tmp_path)

    assert Path("transform") in removed
    assert Path("report.html") in removed
    assert note.read_text(encoding="utf-8") == "keep me"


def test_default_pseudonym_key_is_not_the_public_legacy_constant(monkeypatch):
    monkeypatch.delenv("REGIX_PSEUDONYM_SALT", raising=False)
    legacy = hmac.new(b"regix", b"1234", hashlib.sha256).hexdigest()[:32]

    value = pseudonymize("1234")

    assert len(value) == 32
    assert value != legacy
    assert "random process-local HMAC key" in (pseudonymization_warning() or "")


def test_an_empty_configured_pseudonym_key_is_refused(monkeypatch):
    monkeypatch.setenv("REGIX_PSEUDONYM_SALT", "")

    with pytest.raises(ValueError, match="must not be empty"):
        pseudonymize("patient")


def test_a_child_override_can_disable_a_parent_gate_with_null():
    config = RegistrationConfig().with_overrides(qc={"gates": {"max_tre_mm": 5.0}})

    child = config.with_overrides(qc={"gates": {"max_tre_mm": None}})

    assert child.qc.gates.max_tre_mm is None


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (QCGates, {"max_scale_deviation": -0.1}),
        (QCGates, {"min_dice": {"liver": 1.1}}),
        (StageConfig, {"max_iterations": 0}),
        (StageConfig, {"final_grid_spacing_mm": 0}),
    ],
)
def test_numeric_configuration_bounds_reject_impossible_values(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_qc_warns_when_initialization_differs_from_the_request():
    result = evaluate_gates(
        QCGates(),
        initialization={"requested": "organ_centroid", "chosen": "geometry"},
    )

    check = next(item for item in result.checks if item.name == "initialization_as_requested")
    assert check.status == WARN


def test_qc_rejects_a_metric_report_with_an_unknown_schema():
    with pytest.raises(ValueError, match="unexpected similarity schema"):
        evaluate_gates(QCGates(), similarity={"renamed_ncc": 0.5})  # type: ignore[typeddict-unknown-key]


def test_implausible_displacement_is_flagged_against_the_organ_profile():
    result = evaluate_gates(
        QCGates(max_displacement_ratio=2.5),
        displacement={"p95_mm": 80.0},
        expected_motion_mm=20.0,
    )

    check = next(item for item in result.checks if item.name == "displacement_ratio")
    assert check.status == FAIL
    assert check.measured == 4.0


def test_derived_displacement_gate_is_advisory():
    result = evaluate_gates(
        QCGates(max_displacement_ratio=2.5),
        displacement={"p95_mm": 80.0},
        expected_motion_mm=20.0,
        displacement_advisory=True,
    )

    check = next(item for item in result.checks if item.name == "displacement_ratio")
    assert check.status == WARN


def test_a_rigid_organ_profile_rejects_nontrivial_deformation():
    result = evaluate_gates(
        QCGates(max_displacement_ratio=2.5),
        displacement={"p95_mm": 3.0},
        expected_motion_mm=0.0,
        deformable=True,
    )

    check = next(item for item in result.checks if item.name == "rigid_organ_displacement_mm")
    assert check.status == FAIL


def test_elastix_index_landmarks_are_not_misread_as_millimetres(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("index\n1\n10 20 30\n", encoding="utf-8")

    with pytest.raises(ValueError, match="index coordinates"):
        load_landmarks(path)


def test_report_escapes_hostile_content(tmp_path):
    hostile = '<script>alert("patient")</script>'
    path = build_html_report(
        tmp_path / "report.html",
        {
            "title": hostile,
            "subtitle": hostile,
            "qc": {"status": "WARN", "checks": []},
            "warnings": [hostile],
            "configuration": {hostile: hostile},
            "organ_overlap": {hostile: {"dice": 0.5}},
        },
    )
    source = path.read_text(encoding="utf-8")

    assert "<script>" not in source
    assert "&lt;script&gt;" in source


def test_parameter_parser_handles_multiline_quotes_and_duplicate_keys(tmp_path, caplog):
    path = tmp_path / "parameters.txt"
    path.write_text(
        '(Transform "EulerTransform")\n'
        "(ImagePyramidSchedule 8 8 8\n 4 4 4)\n"
        '(ResultImageFormat "a//b") // real comment\n'
        "(MaximumNumberOfIterations 10)\n"
        "(MaximumNumberOfIterations 20)\n",
        encoding="utf-8",
    )

    parsed = read_parameter_file(path)

    assert parsed["ImagePyramidSchedule"] == ("8", "8", "8", "4", "4", "4")
    assert parsed["ResultImageFormat"] == ("a//b",)
    assert parsed["MaximumNumberOfIterations"] == ("20",)
    assert "more than once" in caplog.text


def test_parameter_entry_without_a_value_is_reported_not_crashed(tmp_path):
    path = tmp_path / "empty-value.txt"
    path.write_text(
        '(Transform "EulerTransform")\n'
        '(Metric "AdvancedMattesMutualInformation")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
        "(NumberOfResolutions)\n",
        encoding="utf-8",
    )
    context = ParamContext(
        working_spacing_mm=2.0,
        fixed_modality="CT",
        moving_modality="CT",
        n_voxels=1000,
        intensity_range=(-1000.0, 1000.0),
    )

    with pytest.raises(ValueError, match="NumberOfResolutions"):
        build_parameter_map(StageConfig(type="rigid", parameter_file=path), context)


def test_parameter_writer_quotes_non_elastix_numeric_spellings(tmp_path):
    path = write_parameter_file(
        {"Values": ("nan", "inf", "-inf", "Infinity", "1_000", "1e5", "-3.4", "true", "Compose")},
        tmp_path / "quoted.txt",
    )
    source = path.read_text(encoding="utf-8")

    assert '"nan" "inf" "-inf" "Infinity" "1_000"' in source
    assert '1e5 -3.4 "true" "Compose"' in source


def test_segmentation_cache_key_changes_for_one_voxel_anywhere():
    segmenter = OrganSegmenter()
    first = np.zeros((10, 11, 12), dtype=np.int16)
    second = first.copy()
    second[-1, -1, -1] = 1

    first_key = segmenter._cache_key(_volume(sitk.GetImageFromArray(first)))
    second_key = segmenter._cache_key(_volume(sitk.GetImageFromArray(second)))

    assert first_key != second_key


def test_corrupt_segmentation_cache_is_ignored(tmp_path):
    segmenter = TotalSegmentatorSegmenter(cache_dir=tmp_path)
    volume = _volume(sitk.Image([8, 8, 8], sitk.sitkFloat32))
    cache_path = segmenter._cache_path(volume)
    assert cache_path is not None
    cache_path.write_bytes(b"not an image")
    cache_path.with_suffix("").with_suffix(".labels.txt").write_text("1 liver\n", encoding="utf-8")

    assert segmenter._cached(volume) is None


def test_import_error_inside_totalsegmentator_is_not_mistaken_for_absence(tmp_path, monkeypatch):
    package = types.ModuleType("totalsegmentator")
    python_api = types.ModuleType("totalsegmentator.python_api")

    def broken_call(**kwargs):
        raise ImportError("dependency failed inside inference")

    python_api.totalsegmentator = broken_call
    monkeypatch.setitem(sys.modules, "totalsegmentator", package)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", python_api)

    with pytest.raises(ImportError, match="inside inference"):
        TotalSegmentatorSegmenter()._run(tmp_path / "input.nii.gz", tmp_path / "out")


def test_importing_pipeline_does_not_import_gpu_modules():
    code = (
        "import sys; import regix.pipeline; "
        "names=('torch','matplotlib','monai','totalsegmentator','regix.registration.convexadam'); "
        "loaded=[name for name in names if name in sys.modules]; assert not loaded, loaded"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}

    subprocess.run([sys.executable, "-c", code], check=True, env=env)
