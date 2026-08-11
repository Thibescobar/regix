"""End-to-end phantom tests: does the registration recover a known ground truth?

These are the only tests that prove anything useful. A unit test checking that a
parameter dictionary contains the right key says nothing about whether the
registration is correct; recovering an imposed transform to within a voxel does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk

from tests.conftest import known_rigid, make_phantom, warp

pytestmark = pytest.mark.slow


def _config(tmp_path, **overrides):
    """Test configuration: base plus overrides, merged without key collisions."""
    from regix.config import load_preset

    defaults: dict = {
        "preprocess": {"working_spacing_mm": 2.5},
        "output": {"dir": str(tmp_path / "out"), "overwrite": True},
        "runtime": {"log_level": "WARNING"},
        "qc": {"report_html": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            defaults[key] = {**defaults[key], **value}
        else:
            defaults[key] = value
    return load_preset("base").with_overrides(**defaults)


def _transform_error(estimated, truth, reference: sitk.Image, n: int = 200) -> float:
    """Mean discrepancy in mm between two transforms, measured over volume points."""
    rng = np.random.default_rng(0)
    size = np.asarray(reference.GetSize(), dtype=float)
    indices = rng.uniform(0.15, 0.85, size=(n, 3)) * (size - 1)
    points = [reference.TransformContinuousIndexToPhysicalPoint(list(idx)) for idx in indices]
    errors = []
    for p in points:
        a = np.asarray(estimated.TransformPoint(p))
        b = np.asarray(truth.TransformPoint(list(p)))
        errors.append(float(np.linalg.norm(a - b)))
    return float(np.mean(errors))


def _small_embedded_pair():
    """Fast deterministic pair for embedded/standalone lifecycle tests."""
    from regix.io.volume import Volume

    fixed_image, _ = make_phantom("CT", shape=(28, 36, 36), noise=0.5)
    truth = known_rigid(
        fixed_image,
        rotation_deg=(1.0, -1.0, 1.0),
        translation_mm=(2.0, -1.0, 2.0),
    )
    moving_image = warp(fixed_image, truth)
    return Volume(fixed_image, modality="CT"), Volume(moving_image, modality="CT")


def _embedded_config(tmp_path):
    return _config(
        tmp_path,
        preprocess={"working_spacing_mm": 3.5},
        features={"enabled": False},
        stages=[
            {
                "type": "rigid",
                "n_resolutions": 1,
                "max_iterations": 20,
                "n_spatial_samples": 512,
            }
        ],
        qc={"enabled": True, "report_html": False, "jacobian": False},
        runtime={"log_level": "ERROR", "threads": 1},
    )


def test_compute_is_numerically_equivalent_and_leaves_no_artifacts(tmp_path):
    """The embedded lifecycle must only change I/O, never the registration result."""
    from regix.pipeline import RegistrationPipeline

    fixed, moving = _small_embedded_pair()
    cfg = _embedded_config(tmp_path)
    original_config = cfg.model_dump(mode="python")

    standalone = RegistrationPipeline(cfg).run(fixed, moving, tmp_path / "standalone")
    workspace = tmp_path / "compute-work"
    workspace.mkdir()
    embedded = RegistrationPipeline(cfg).compute(fixed, moving, work_dir=workspace)

    assert list(workspace.iterdir()) == []
    assert embedded.outputs == {}
    assert embedded.manifest_path is None
    assert embedded.registered_image is not None
    assert embedded.applied_transform.capabilities >= {"points", "field", "sitk"}
    assert "disk-resample" not in embedded.applied_transform.capabilities
    assert all("transform_parameter_file" not in stage for stage in embedded.stages)
    assert cfg.model_dump(mode="python") == original_config

    standalone_transform = standalone.applied_transform.as_sitk_transform()
    embedded_transform = embedded.applied_transform.as_sitk_transform()
    assert standalone_transform is not None and embedded_transform is not None
    probes = np.asarray([[0.0, 0.0, 0.0], [10.0, -5.0, 17.0], [-20.0, 8.0, 3.0]])
    expected = np.asarray([standalone_transform.TransformPoint(p) for p in probes])
    actual = embedded.applied_transform.transform_points(probes)
    assert actual is not None
    assert np.allclose(actual, expected, atol=1e-8)
    assert np.allclose(
        sitk.GetArrayViewFromImage(embedded.registered_image),
        sitk.GetArrayViewFromImage(standalone.registered_image),
        atol=1e-6,
    )
    assert embedded.metrics["similarity"] == standalone.metrics["similarity"]

    # The transform is exercised after TemporaryDirectory has already been removed.
    replay = embedded.applied_transform.resample(moving.image, fixed.image)
    assert replay.GetSize() == fixed.image.GetSize()


def test_compute_transform_only_skips_restitution_and_qc(tmp_path):
    from regix import compute
    from regix.pipeline import RegistrationPipeline

    fixed, moving = _small_embedded_pair()
    cfg = _embedded_config(tmp_path)
    workspace = tmp_path / "compute-work"
    workspace.mkdir()

    result = compute(
        fixed,
        moving,
        cfg,
        registered_image=False,
        qc=False,
        work_dir=workspace,
    )

    assert result.registered_image is None
    assert result.status == "NOT_EVALUATED"
    assert result.qc == {}
    assert "similarity" not in result.metrics
    assert list(workspace.iterdir()) == []

    with pytest.raises(ValueError, match="qc=True requires registered_image=True"):
        RegistrationPipeline(cfg).compute(fixed, moving, registered_image=False, qc=True)


def test_compute_cleans_its_workspace_after_failure(tmp_path, monkeypatch):
    from regix.pipeline import RegistrationPipeline

    fixed, moving = _small_embedded_pair()
    cfg = _embedded_config(tmp_path)
    workspace = tmp_path / "compute-work"
    workspace.mkdir()

    def fail_after_write(_self, _fixed, _moving, out_dir, _manifest, _outputs, _policy):
        (out_dir / "partial-artifact").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(RegistrationPipeline, "_run_inner", fail_after_write)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        RegistrationPipeline(cfg).compute(fixed, moving, qc=False, work_dir=workspace)

    assert list(workspace.iterdir()) == []


def test_detaching_a_file_chain_keeps_its_nonlinear_part(tmp_path, monkeypatch):
    """A convertible linear prefix must never eclipse a file-backed B-spline tail."""
    from regix.pipeline import _detach_transform
    from regix.registration.warp import ElastixAppliedTransform

    parameter_file = tmp_path / "TransformParameters.0.txt"
    parameter_file.write_text('(Transform "BSplineTransform")\n', encoding="utf-8")
    linear_prefix = sitk.TranslationTransform(3, (100.0, 0.0, 0.0))
    applied = ElastixAppliedTransform(
        parameter_file,
        work_dir=tmp_path,
        linear_transform=linear_prefix,
    )
    reference = sitk.Image([4, 4, 4], sitk.sitkFloat32)
    field_array = np.zeros((4, 4, 4, 3), dtype=np.float64)
    field_array[..., 0] = 2.0
    complete_field = sitk.GetImageFromArray(field_array, isVector=True)
    complete_field.CopyInformation(reference)

    def full_chain(*_args, **kwargs):
        assert kwargs["compute_deformation_field"] is True
        return reference, complete_field

    monkeypatch.setattr("regix.pipeline.apply_transform", full_chain)
    detached = _detach_transform(applied, reference)

    assert np.allclose(detached.transform_points(np.asarray([[0.0, 0.0, 0.0]])), [[2.0, 0.0, 0.0]])


def test_rigid_recovers_the_ground_truth(tmp_path, rigid_pair):
    """Rigid + affine on a CT-CT pair: expected error well below one voxel."""
    from regix.pipeline import RegistrationPipeline

    paths, truth = rigid_pair
    cfg = _config(tmp_path)
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    assert result.registered_image is not None
    transform = result.applied_transform.as_sitk_transform()
    assert transform is not None, "the final transform must be usable from SimpleITK"

    fixed = sitk.ReadImage(str(paths["fixed"]))
    error = _transform_error(transform, truth, fixed)
    assert error < 1.5, f"registration error of {error:.2f} mm (2 mm voxel)"

    similarity = result.metrics["similarity"]
    assert similarity["ncc_after"] > similarity["ncc_before"]
    assert result.status in ("PASS", "WARN")
    # The output must be on the original grid of the fixed image.
    assert result.registered_image.GetSize() == fixed.GetSize()
    assert np.allclose(result.registered_image.GetSpacing(), fixed.GetSpacing())


def test_rigid_recovers_the_ground_truth_on_an_oblique_grid(tmp_path):
    from regix.pipeline import RegistrationPipeline
    from tests.conftest import oblique_direction

    fixed, _ = make_phantom(
        "CT",
        shape=(40, 48, 48),
        spacing=(1.2, 1.2, 4.0),
        direction=oblique_direction((15.0, 8.0, 3.0)),
        noise=2.0,
    )
    truth = known_rigid(
        fixed,
        rotation_deg=(2.0, -1.0, 2.0),
        translation_mm=(3.0, -2.0, 4.0),
    )
    moving = warp(fixed, truth)
    fixed_path = tmp_path / "fixed-oblique.nii.gz"
    moving_path = tmp_path / "moving-oblique.nii.gz"
    sitk.WriteImage(fixed, str(fixed_path), True)
    sitk.WriteImage(moving, str(moving_path), True)
    cfg = _config(tmp_path, preprocess={"working_spacing_mm": 2.5})

    result = RegistrationPipeline(cfg).run(fixed_path, moving_path, tmp_path / "out")

    estimated = result.applied_transform.as_sitk_transform()
    assert estimated is not None
    assert _transform_error(estimated, truth, fixed) < 2.0


#: A zoo-shaped rigid parameter file: recursive pyramids, StandardGradientDescent with
#: its SP_* schedule, a Random sampler, no UseDirectionCosines at all (its elastix
#: default is false), HowToCombineTransforms "Add" and WriteResultImage "true". None of
#: that is what Regix generates, which is the point.
_ZOO_RIGID = """\
// zoo-style rigid parameter file
(Registration "MultiResolutionRegistration")
(Transform "EulerTransform")
(Metric "AdvancedNormalizedCorrelation")
(Optimizer "StandardGradientDescent")
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ImageSampler "Random")
(NumberOfResolutions 2)
(MaximumNumberOfIterations 200)
(NumberOfSpatialSamples 2048)
(SP_a 1000.0)
(SP_alpha 0.602)
(SP_A 50.0)
(AutomaticScalesEstimation "true")
(DefaultPixelValue 0)
(WriteResultImage "true")
(HowToCombineTransforms "Add")
"""


def test_a_zoo_parameter_file_drives_a_real_registration(tmp_path, rigid_pair):
    """elastix must accept what Regix hands it, not just pass our own validation.

    The unit tests check the map that is built; only a run proves that a stage described
    entirely by a third-party file actually registers.
    """
    from regix.pipeline import RegistrationPipeline

    paths, _ = rigid_pair
    zoo = tmp_path / "Par0000.rigid.txt"
    zoo.write_text(_ZOO_RIGID, encoding="utf-8")

    cfg = _config(tmp_path, stages=[{"type": "rigid", "parameter_file": str(zoo)}])
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    similarity = result.metrics["similarity"]
    assert similarity["ncc_after"] > similarity["ncc_before"]

    # The stage is reported from the file, so the manifest describes the actual run.
    stage = result.stages[0]
    assert stage["transform"] == "EulerTransform"
    assert stage["resolutions"] == 2 and stage["iterations"] == 200
    assert stage["parameter_file"] == str(zoo)

    # The effective map on disk keeps the file's tuning and carries the enforced keys.
    effective = (tmp_path / "out" / "elastix" / "stage00_rigid" / "parameters.txt").read_text(
        encoding="utf-8"
    )
    assert '(Optimizer "StandardGradientDescent")' in effective
    assert '(FixedImagePyramid "FixedRecursiveImagePyramid")' in effective
    assert "(SP_a 1000.0)" in effective
    assert '(UseDirectionCosines "true")' in effective  # absent from the file
    assert '(HowToCombineTransforms "Compose")' in effective  # the file said "Add"
    assert '(WriteResultImage "false")' in effective  # the file said "true"


def test_the_quantisation_warning_fires_through_the_full_pipeline(tmp_path, rigid_pair):
    """The engine must enrich ParamContext with the real post-preprocessing range."""
    from regix.pipeline import RegistrationPipeline

    paths, _ = rigid_pair
    zoo = tmp_path / "quantised.rigid.txt"
    source = _ZOO_RIGID.replace(
        '(MovingImagePyramid "MovingRecursiveImagePyramid")',
        '(MovingImagePyramid "MovingRecursiveImagePyramid")\n'
        '(FixedInternalImagePixelType "short")\n'
        '(MovingInternalImagePixelType "short")',
    ).replace("(MaximumNumberOfIterations 200)", "(MaximumNumberOfIterations 1)")
    zoo.write_text(source, encoding="utf-8")
    cfg = _config(
        tmp_path,
        stages=[{"type": "rigid", "parameter_file": str(zoo)}],
        preprocess={
            "working_spacing_mm": 3.0,
            "fixed": {"normalize": "minmax", "percentile_clip": None},
            "moving": {"normalize": "minmax", "percentile_clip": None},
        },
        qc={"enabled": False, "report_html": False},
    )
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    assert "almost no distinct values" in (tmp_path / "out" / "regix.log").read_text(encoding="utf-8")
    assert result.status == "NOT_EVALUATED"


def test_a_real_zoo_file_registers_correctly(tmp_path, rigid_pair):
    """The end-to-end guard on the internal-pixel-type enforcement.

    `Parameters.Par0008.affine.txt` declares (FixedInternalImagePixelType "short"),
    which is right for HU images read from disk and catastrophic for the normalised
    [0, 1] floats Regix hands elastix: every voxel rounds to 0 or 1 and Mattes MI drops
    to 6.7e-16, so the optimiser does nothing while the run still reports WARN. Only a
    real registration catches that, because the parameter map is perfectly valid either
    way -- which is exactly why this test asserts the recovered transform and not the
    contents of a dictionary.
    """
    import pathlib

    from regix.pipeline import RegistrationPipeline

    zoo = pathlib.Path(__file__).parent / "data" / "Parameters.Par0008.affine.txt"
    paths, truth = rigid_pair
    cfg = _config(
        tmp_path,
        stages=[{"type": "affine", "parameter_file": str(zoo)}],
        deformable_engine="none",
    )
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    transform = result.applied_transform.as_sitk_transform()
    assert transform is not None
    error = _transform_error(transform, truth, sitk.ReadImage(str(paths["fixed"])))
    assert error < 1.5, f"zoo-driven affine off by {error:.2f} mm (2 mm voxel)"

    # A degenerate criterion is the symptom of the quantisation bug: MI on a binarised
    # image is ~0, and the stage reports it without failing.
    criterion = result.stages[0]["final_metric"]
    assert criterion is not None and abs(criterion) > 1e-3, (
        f"criterion {criterion} is degenerate: the intensities were probably quantised"
    )
    similarity = result.metrics["similarity"]
    assert similarity["ncc_after"] > similarity["ncc_before"]


def test_output_keeps_native_intensities(tmp_path, rigid_pair):
    """The delivered volume keeps its HU: it is not normalised by the preprocessing."""
    from regix.pipeline import RegistrationPipeline

    paths, _ = rigid_pair
    result = RegistrationPipeline(_config(tmp_path)).run(paths["fixed"], paths["moving"], tmp_path / "out")
    arr = sitk.GetArrayViewFromImage(result.registered_image)
    assert arr.min() < -500, "air HU must survive (preprocessing is not applied to the output)"
    assert arr.max() > 50


def test_missing_nomenclature_guesses_nothing(tmp_path, rigid_pair):
    """Without a nomenclature the labels stay neutral: no fabricated 'liver'."""
    from regix.io.volume import load_volume
    from regix.organs.segmenter import ExternalSegmenter

    paths, _ = rigid_pair
    # Copy the label map without its names file.
    orphan = tmp_path / "orphan.nii.gz"
    sitk.WriteImage(sitk.ReadImage(str(paths["fixed_labels"])), str(orphan), True)

    seg = ExternalSegmenter(labelmap=orphan).segment(load_volume(paths["fixed"]))
    assert seg.present_organs() == ["label_1", "label_2", "label_3"]
    assert seg.label_of("liver") is None, "no nomenclature must ever be guessed"

    # With an explicit nomenclature, the real names appear.
    named = ExternalSegmenter(
        labelmap=orphan, label_names={1: "liver", 2: "spleen", 3: "kidney_left"}
    ).segment(load_volume(paths["fixed"]))
    assert named.label_of("liver") == 1


def test_per_organ_dice_and_centroid_initialization(tmp_path, rigid_pair):
    """External masks: organ-centroid initialization and per-organ QC.

    The nomenclature is found on its own through the sidecar '.labels.json'.
    """
    from regix.pipeline import RegistrationPipeline

    paths, truth = rigid_pair
    cfg = _config(
        tmp_path,
        organs={
            "backend": "external",
            "fixed_labelmap": str(paths["fixed_labels"]),
            "moving_labelmap": str(paths["moving_labels"]),
            "targets": ["liver"],
            "mask_dilate_mm": 10.0,
            "qc_labels": ["liver", "spleen"],
        },
        init={"mode": "organ_centroid"},
    )
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    assert result.initialization["chosen"] == "organ_centroid"
    overlap = result.metrics["organ_overlap"]
    assert "liver" in overlap, f"organs evaluated: {list(overlap)}"
    assert overlap["liver"]["dice"] > 0.9, f"liver Dice = {overlap['liver']['dice']}"

    fixed = sitk.ReadImage(str(paths["fixed"]))
    assert _transform_error(result.applied_transform.as_sitk_transform(), truth, fixed) < 1.5


def test_multistart_picks_the_right_start(tmp_path, rigid_pair):
    from regix.pipeline import RegistrationPipeline

    paths, truth = rigid_pair
    cfg = _config(
        tmp_path,
        init={
            "mode": "multistart",
            "candidates": ["identity", "geometry", "moments"],
            "multistart_rotations_deg": [[0, 0, 0], [0, 0, 20]],
        },
    )
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")
    assert result.initialization["n_candidates"] == 6
    assert result.initialization["candidates"][0]["score"] >= result.initialization["candidates"][-1]["score"]
    fixed = sitk.ReadImage(str(paths["fixed"]))
    assert _transform_error(result.applied_transform.as_sitk_transform(), truth, fixed) < 1.5


def test_bspline_deformable_without_folding(tmp_path):
    """B-spline: the field must improve similarity without folding."""
    from regix.config import StageConfig, TransformType, load_preset
    from regix.pipeline import RegistrationPipeline

    # Deformed phantom: cranio-caudal compression, not just a translation.
    fixed_img, _ = make_phantom("CT", noise=2.0, seed=3)
    arr = sitk.GetArrayFromImage(fixed_img)
    squeezed = sitk.GetImageFromArray(np.concatenate([arr[:1], arr[:-1]], axis=0) * 0.5 + arr * 0.5)
    squeezed.CopyInformation(fixed_img)
    moving_img = warp(squeezed, known_rigid(fixed_img, (0, 0, 0), (3.0, -2.0, 5.0)))

    fixed_path = tmp_path / "f.nii.gz"
    moving_path = tmp_path / "m.nii.gz"
    sitk.WriteImage(fixed_img, str(fixed_path), True)
    sitk.WriteImage(moving_img, str(moving_path), True)

    cfg = load_preset("base").with_overrides(
        preprocess={"working_spacing_mm": 2.5},
        output={"dir": str(tmp_path / "out"), "overwrite": True, "write_deformation_field": True},
        runtime={"log_level": "WARNING"},
        qc={"report_html": False, "jacobian": True},
        deformable_engine="elastix",
    )
    cfg = cfg.model_copy(
        update={
            "stages": [
                StageConfig(type=TransformType.RIGID, n_resolutions=3, max_iterations=250),
                StageConfig(
                    type=TransformType.BSPLINE,
                    n_resolutions=3,
                    max_iterations=250,
                    final_grid_spacing_mm=20.0,
                ),
            ]
        }
    )
    result = RegistrationPipeline(cfg).run(fixed_path, moving_path, tmp_path / "out")

    jac = result.metrics["jacobian"]
    assert jac.get("available"), "the Jacobian must be computable for a deformable registration"
    assert jac["folding_fraction"] < 1e-3, f"folded field: {jac['folding_fraction']}"
    similarity = result.metrics["similarity"]
    assert similarity["ncc_after"] > similarity["ncc_before"]
    assert "deformation_field" in result.outputs


def test_multimodal_with_the_mind_descriptor(tmp_path):
    """CT against an inverted-contrast 'MR': features must rescue the registration."""
    from regix.config import load_preset
    from regix.pipeline import RegistrationPipeline

    ct, _ = make_phantom("CT", noise=2.0, seed=5)
    mr_source, _ = make_phantom("MR", noise=10.0, seed=6)
    truth = known_rigid(ct, (0.0, 0.0, 3.0), (5.0, -4.0, 6.0))
    mr = warp(mr_source, truth)

    fixed_path = tmp_path / "ct.nii.gz"
    moving_path = tmp_path / "mr.nii.gz"
    sitk.WriteImage(ct, str(fixed_path), True)
    sitk.WriteImage(mr, str(moving_path), True)

    cfg = load_preset("base").with_overrides(
        fixed_modality="CT",
        moving_modality="MR",
        preprocess={"working_spacing_mm": 2.5},
        features={"enabled": True, "provider": "mind", "n_components": 3},
        output={"dir": str(tmp_path / "out"), "overwrite": True},
        runtime={"log_level": "WARNING"},
        qc={"report_html": False},
    )
    result = RegistrationPipeline(cfg).run(fixed_path, moving_path, tmp_path / "out")

    feature_info = result.metrics["features"]
    assert feature_info["requested_provider"] == "mind"
    assert feature_info["provider"] == "mind"
    assert feature_info["descriptor"] == "MIND-SSC"
    assert feature_info["fallback"] is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["features"]["provider"] == "mind"
    assert manifest["metrics"]["features"]["descriptor"] == "MIND-SSC"
    similarity = result.metrics["similarity"]
    assert similarity["nmi_after"] >= similarity["nmi_before"] - 1e-3
    error = _transform_error(result.applied_transform.as_sitk_transform(), truth, ct)
    assert error < 3.0, f"multimodal error of {error:.2f} mm"


def test_html_report_and_manifest_are_produced(tmp_path, rigid_pair):
    from regix.pipeline import RegistrationPipeline

    paths, _ = rigid_pair
    cfg = _config(tmp_path, qc={"report_html": True, "n_slices": 2})
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")

    report = result.outputs["report"]
    text = report.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "data:image/webp;base64," in text, "figures must be embedded"
    assert "not a medical device" in text, "the disclaimer must stay visible"
    assert result.manifest_path is not None and result.manifest_path.exists()

    import json

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in ("PASS", "WARN", "FAIL")
    assert [s["name"] for s in manifest["steps"]][:2] == ["loading", "preprocessing"]
    assert manifest["environment"]["simpleitk"]
    # Parameter files must be archived so the run can be replayed.
    transform_dir = result.outputs["transform_dir"]
    assert list(transform_dir.glob("*parameters.txt"))
    assert list(transform_dir.glob("*.elastix.txt"))


def test_slicer_readable_transforms(tmp_path, rigid_pair):
    """One ITK .txt per stage plus a final one, flattened, reloadable as they are."""
    from regix.pipeline import RegistrationPipeline

    paths, truth = rigid_pair
    result = RegistrationPipeline(_config(tmp_path)).run(paths["fixed"], paths["moving"], tmp_path / "out")
    transform_dir = result.outputs["transform_dir"]
    assert (transform_dir / "stage00_rigid.itk.txt").exists()
    assert (transform_dir / "stage01_affine.itk.txt").exists()
    final = result.outputs["transform_slicer"]

    body = final.read_text(encoding="utf-8")
    assert body.startswith("#Insight Transform File V1.0")
    assert body.count("#Transform ") == 1, "a single transform, not a composition"

    reloaded = sitk.ReadTransform(str(final))
    fixed = sitk.ReadImage(str(paths["fixed"]))
    assert _transform_error(reloaded, truth, fixed) < 1.5
    # The reloaded .txt must match the internal transform.
    internal = result.applied_transform.as_sitk_transform()
    for point in ([0.0, 0.0, 60.0], [20.0, -30.0, 75.0]):
        assert np.allclose(reloaded.TransformPoint(point), internal.TransformPoint(point), atol=1e-4)


def test_qc_gate_flags_an_impossible_registration(tmp_path):
    """Two volumes with nothing in common: QC must say so, not deliver silently."""
    from regix.pipeline import RegistrationPipeline

    fixed_img, _ = make_phantom("CT", noise=1.0, seed=11)
    rng = np.random.default_rng(12)
    noise = sitk.GetImageFromArray(
        rng.normal(0.0, 200.0, size=sitk.GetArrayViewFromImage(fixed_img).shape).astype(np.float32)
    )
    noise.CopyInformation(fixed_img)

    fixed_path = tmp_path / "f.nii.gz"
    moving_path = tmp_path / "m.nii.gz"
    sitk.WriteImage(fixed_img, str(fixed_path), True)
    sitk.WriteImage(noise, str(moving_path), True)

    cfg = _config(tmp_path, qc={"report_html": False, "gates": {"min_ncc_gain": 0.05}})
    result = RegistrationPipeline(cfg).run(fixed_path, moving_path, tmp_path / "out")
    assert result.status == "FAIL"
    # Which gate catches it is not the contract -- that it is caught, and named, is.
    # (It used to be ncc_gain; on native intensities the scale blows up first.)
    failures = [c["name"] for c in result.qc["checks"] if c["status"] == "FAIL"]
    assert failures, f"no gate fired on unrelated volumes: {result.qc['checks']}"


def test_roi_cropping_speeds_up_without_changing_the_output_grid(tmp_path, rigid_pair):
    from regix.pipeline import RegistrationPipeline

    paths, truth = rigid_pair
    cfg = _config(
        tmp_path,
        organs={
            "backend": "external",
            "fixed_labelmap": str(paths["fixed_labels"]),
            "moving_labelmap": str(paths["moving_labels"]),
            "targets": ["liver"],
            "roi_crop": True,
            "roi_margin_mm": 15.0,
        },
        init={"mode": "organ_centroid"},
    )
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], tmp_path / "out")
    fixed = sitk.ReadImage(str(paths["fixed"]))
    # Despite the internal cropping, the output stays on the full fixed grid.
    assert result.registered_image.GetSize() == fixed.GetSize()
    assert _transform_error(result.applied_transform.as_sitk_transform(), truth, fixed) < 2.0
