"""Unit tests: configuration, elastix parameters, geometry, transforms, metrics."""

from __future__ import annotations

import logging
import pathlib
from contextlib import contextmanager

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from tests.conftest import known_rigid, make_phantom


@contextmanager
def captured_logs(logger_name: str, level: int = logging.WARNING):
    """Collect records from a Regix logger, whatever the global logging state.

    Not `caplog`: that attaches to the root logger, and `setup_logging` sets
    ``propagate = False`` on the "regix" logger, so any test that has already run a
    pipeline leaves `caplog` blind. Attaching to the logger itself is order-independent
    -- which matters, because the failure mode is a test that passes in isolation and
    fails in the suite.
    """
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=level)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(previous, level) if previous else level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


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
    assert len(child.stages) == 1  # the child replaces, it does not concatenate
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


# --------------------------------------------------------------------------- #
# Externally supplied elastix parameter files (the "parameter zoo" format)
# --------------------------------------------------------------------------- #
#: Shape of a published zoo file: it tunes the optimisation, omits
#: UseDirectionCosines entirely (whose elastix default is false), and asks for the
#: result image Regix never reads.
_ZOO_FILE = """\
// Example parameter file, zoo style.
(Registration "MultiResolutionRegistration")
(Transform "BSplineTransform")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "StandardGradientDescent")
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(Interpolator "BSplineInterpolator")
(ImageSampler "Random")
(NumberOfResolutions 3)
(MaximumNumberOfIterations 250)
(NumberOfSpatialSamples 2048)
(FinalGridSpacingInPhysicalUnits 16.0 16.0 16.0)
(GridSpacingSchedule 4.0 2.0 1.0)
(NumberOfHistogramBins 64)
(WriteResultImage "true")
(HowToCombineTransforms "Add")
"""


def _zoo_stage(tmp_path, **kwargs):
    from regix.config import StageConfig, TransformType

    path = tmp_path / "Par0001.bspline.txt"
    path.write_text(_ZOO_FILE, encoding="utf-8")
    kwargs.setdefault("type", TransformType.BSPLINE)
    return StageConfig(parameter_file=path, **kwargs)


def test_a_zoo_parameter_file_is_used_verbatim(tmp_path):
    """The point of accepting these files is that they are honoured, not reinterpreted."""
    from regix.registration.params import ParamContext, build_parameter_map

    pmap = build_parameter_map(_zoo_stage(tmp_path), ParamContext())

    # Everything the file tunes survives, including values Regix would never generate.
    assert pmap["Optimizer"] == ("StandardGradientDescent",)
    assert pmap["ImageSampler"] == ("Random",)
    assert pmap["FixedImagePyramid"] == ("FixedRecursiveImagePyramid",)
    assert pmap["NumberOfResolutions"] == ("3",)
    assert pmap["MaximumNumberOfIterations"] == ("250",)
    assert pmap["NumberOfHistogramBins"] == ("64",)
    assert pmap["GridSpacingSchedule"] == ("4.0", "2.0", "1.0")
    # ... and no bending-energy penalty was silently bolted on.
    assert pmap["Metric"] == ("AdvancedMattesMutualInformation",)


def test_a_zoo_file_cannot_break_the_geometry_or_the_transform_chain(tmp_path):
    """The four keys of ENFORCED_WITH_PARAMETER_FILE are re-imposed.

    Each one is a silent corruption rather than a tuning choice: a zoo file that omits
    UseDirectionCosines gets the elastix default (false), which misregisters every
    oblique acquisition without a single warning, and "Add" would make the engine's
    composition arithmetic wrong.
    """
    from regix.registration.params import (
        ENFORCED_WITH_PARAMETER_FILE,
        ParamContext,
        build_parameter_map,
    )

    pmap = build_parameter_map(_zoo_stage(tmp_path), ParamContext())
    for key, value in ENFORCED_WITH_PARAMETER_FILE.items():
        assert pmap[key] == value, key
    assert pmap["HowToCombineTransforms"] == ("Compose",)  # the file said "Add"
    assert pmap["WriteResultImage"] == ("false",)  # the file said "true"


#: A real published zoo file, kept verbatim as a fixture. Generated files cannot stand
#: in for it: this one carries `short` internal pixel types, an under-specified pyramid
#: schedule and `AutomaticTransformInitialization true`, none of which Regix ever emits.
_REAL_ZOO_FILE = pathlib.Path(__file__).parent / "data" / "Parameters.Par0008.affine.txt"


def test_a_real_zoo_file_keeps_its_own_internal_pixel_type():
    """Compliance: `short` is the file's call, and on native intensities it is correct.

    This used to be overridden to float, because Regix min-max normalised its inputs
    into [0, 1] where an integer type rounds every voxel to 0 or 1 (Mattes MI down to
    6.7e-16 on a CT-CT phantom, the optimiser moving nothing, the run still reporting
    WARN). The right fix was upstream: Regix no longer rescales, so a Hounsfield unit
    reaches elastix as a Hounsfield unit and the file's declaration holds.
    """
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    raw = _REAL_ZOO_FILE.read_text(encoding="utf-8")
    assert '(FixedInternalImagePixelType "short")' in raw, "fixture no longer covers the case"

    stage = StageConfig(type=TransformType.AFFINE, parameter_file=_REAL_ZOO_FILE)
    pmap = build_parameter_map(stage, ParamContext(intensity_range=(-1024.0, 1641.0)))
    assert pmap["FixedInternalImagePixelType"] == ("short",)
    assert pmap["MovingInternalImagePixelType"] == ("short",)


def test_an_integer_pixel_type_on_rescaled_data_is_reported():
    """The detection that replaced the override: honour the file, but say when it cannot work.

    A user can still re-create the old catastrophe by asking for `normalize: minmax`
    explicitly. Then `short` has two distinct values to work with and the criterion is
    meaningless -- silently, because elastix reports success.
    """
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(type=TransformType.AFFINE, parameter_file=_REAL_ZOO_FILE)

    with captured_logs("regix.registration.params") as records:
        build_parameter_map(stage, ParamContext(intensity_range=(0.0, 1.0)))
    assert any("quantis" in r.getMessage() or "distinct values" in r.getMessage() for r in records), (
        "an integer pixel type on [0, 1] data must be reported"
    )

    # ... and stays quiet on native intensities, which is the normal case.
    with captured_logs("regix.registration.params") as records:
        build_parameter_map(stage, ParamContext(intensity_range=(-1024.0, 3071.0)))
    assert not any("distinct values" in r.getMessage() for r in records)


def test_a_real_zoo_file_keeps_its_own_tuning():
    """Everything that is a genuine tuning choice survives, and the rest is reported."""
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(type=TransformType.AFFINE, parameter_file=_REAL_ZOO_FILE)
    with captured_logs("regix.registration.params") as records:
        pmap = build_parameter_map(stage, ParamContext(dimension=3))

    # The file's own choices, none of which Regix would generate.
    assert pmap["Optimizer"] == ("StandardGradientDescent",)
    assert pmap["ImageSampler"] == ("RandomSparseMask",)
    assert pmap["FixedImagePyramid"] == ("FixedRecursiveImagePyramid",)
    assert pmap["Metric"] == ("AdvancedMattesMutualInformation",)
    assert pmap["SP_a"] == ("500.0",) and pmap["SP_alpha"] == ("0.602",)
    assert pmap["NumberOfSpatialSamples"] == ("5000",)
    assert pmap["FinalBSplineInterpolationOrder"] == ("0",)
    assert pmap["ErodeMask"] == ("false",)

    # UseDirectionCosines is absent from the file; its elastix default is false.
    assert pmap["UseDirectionCosines"] == ("true",)

    messages = " | ".join(r.getMessage() for r in records)
    assert "AutomaticTransformInitialization" in messages
    # 5 values for 4 resolutions in 3D: elastix silently falls back to its default
    # schedule, so the file's intended pyramid is not the one that runs.
    assert "ImagePyramidSchedule" in messages


def test_extra_accepts_a_numeric_list():
    """YAML parses `[8, 8, 8]` as ints; refusing them was a pointless papercut."""
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = StageConfig(
        type=TransformType.RIGID,
        n_resolutions=2,  # 2 resolutions x 3 dimensions = the 6 values below
        extra={"ImagePyramidSchedule": [8, 8, 8, 1, 1, 1]},
    )
    pmap = build_parameter_map(stage, ParamContext(dimension=3))
    assert pmap["ImagePyramidSchedule"] == ("8", "8", "8", "1", "1", "1")


def test_extra_still_overrides_a_parameter_file(tmp_path):
    from regix.registration.params import ParamContext, build_parameter_map

    stage = _zoo_stage(tmp_path, extra={"MaximumNumberOfIterations": 42})
    pmap = build_parameter_map(stage, ParamContext())
    assert pmap["MaximumNumberOfIterations"] == ("42",)


def test_a_parameter_file_must_agree_with_the_declared_stage_type(tmp_path):
    """Refusal, not a warning: `type` drives how the stage result is interpreted.

    The engine decides from ``stage.type`` whether to parse the stage output as a linear
    transform. A B-spline file declared as affine would have Regix read a deformation
    field as a 4x4 matrix -- and report a plausible one.
    """
    from regix.config import TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    stage = _zoo_stage(tmp_path, type=TransformType.AFFINE)
    with pytest.raises(ValueError, match="BSplineTransform"):
        build_parameter_map(stage, ParamContext())


def test_a_parameter_file_of_the_wrong_dimension_is_refused(tmp_path):
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    path = tmp_path / "twod.txt"
    path.write_text(
        '(Transform "EulerTransform")\n(Metric "AdvancedNormalizedCorrelation")\n'
        "(NumberOfResolutions 3)\n(FixedImageDimension 2)\n",
        encoding="utf-8",
    )
    stage = StageConfig(type=TransformType.RIGID, parameter_file=path)
    with pytest.raises(ValueError, match="2"):
        build_parameter_map(stage, ParamContext(dimension=3))


def test_a_file_that_is_not_an_elastix_parameter_file_is_refused(tmp_path):
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map

    path = tmp_path / "notes.txt"
    path.write_text("just some notes about the case\n", encoding="utf-8")
    stage = StageConfig(type=TransformType.RIGID, parameter_file=path)
    with pytest.raises(ValueError, match="no elastix parameter"):
        build_parameter_map(stage, ParamContext())


def test_a_parameter_file_stage_is_described_from_the_file(tmp_path):
    """The manifest must report what elastix received, not what the config asked for."""
    from regix.registration.params import ParamContext, build_parameter_map, describe_stage

    stage = _zoo_stage(tmp_path)
    ctx = ParamContext()
    pmap = build_parameter_map(stage, ctx)
    described = describe_stage(stage, ctx, pmap)

    assert described["transform"] == "BSplineTransform"
    assert described["metric"] == "mi"
    assert described["metric_elastix"] == "AdvancedMattesMutualInformation"
    assert described["resolutions"] == 3
    assert described["iterations"] == 250
    assert described["samples"] == 2048
    assert described["grid_mm"] == 16.0
    assert described["parameter_file"] == str(stage.parameter_file)


def test_describe_stage_reports_the_effective_value_not_the_requested_one():
    """Same guarantee on the generated path, where `extra` is the source of drift."""
    from regix.config import StageConfig, TransformType
    from regix.registration.params import ParamContext, build_parameter_map, describe_stage

    stage = StageConfig(type=TransformType.RIGID, n_resolutions=4, extra={"NumberOfResolutions": 2})
    ctx = ParamContext()
    described = describe_stage(stage, ctx, build_parameter_map(stage, ctx))
    assert described["resolutions"] == 2, "the manifest reported the request, not the run"


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
# Intensity preparation: resolving the modality defaults
# --------------------------------------------------------------------------- #
def test_an_unspecified_ct_is_left_on_its_native_scale():
    """A CT reaches elastix as Hounsfield units: no window, no rescaling.

    This is the interoperability invariant. The former default was the anatomix paper's
    (-450, 450) window plus a min-max rescale, which silently broke every hand-written
    elastix parameter file -- those assume the acquisition scale. anatomix still gets
    exactly that preparation, applied inside the feature path on its own inputs.
    """
    from regix.config import ImagePrep
    from regix.preprocess.intensity import resolve_prep

    resolved = resolve_prep(ImagePrep(), "CT")  # what base.yaml produces
    assert resolved.window is None
    assert resolved.clip is None
    assert resolved.percentile_clip is None
    assert resolved.normalize == "none"


def test_no_bundled_preset_rescales_the_intensities():
    """Guard on the invariant: a preset may clip (scale-preserving), never rescale."""
    from regix.config import available_presets, load_preset

    for name in available_presets():
        cfg = load_preset(name)
        for side in ("fixed", "moving"):
            prep = getattr(cfg.preprocess, side)
            assert prep.normalize == "none", (
                f"{name}.{side} rescales to '{prep.normalize}': hand-written elastix "
                "parameter files assume the acquisition scale"
            )


def test_the_anatomix_preparation_stays_inside_the_feature_path():
    """anatomix's window and [0, 1] normalisation are its own, not the pipeline's."""
    from regix.features.anatomix import clip_for_modality
    from regix.preprocess.intensity import (
        DEFAULT_PREP_BY_MODALITY,
        HU_WINDOWS,
        normalize_for_features,
    )

    # It still applies them, on its side.
    assert clip_for_modality("CT") == HU_WINDOWS["ct_registration"] == (-450.0, 450.0)
    arr = normalize_for_features(
        sitk.GetImageFromArray(np.array([[[-1000.0, 0.0, 3000.0]]], dtype=np.float32)),
        clip=(-450.0, 450.0),
    )
    assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0

    # And the general defaults do not.
    for modality in ("CT", "CBCT"):
        assert DEFAULT_PREP_BY_MODALITY[modality]["window"] is None
        assert "normalize" not in DEFAULT_PREP_BY_MODALITY[modality]


def test_an_explicit_percentile_clip_on_a_ct_is_honoured():
    """Regression: the detection used to be `percentile_clip == (0.5, 99.5)` -- the default.

    An explicit request for robust percentiles was therefore indistinguishable from
    silence, and got silently replaced by the HU window. That is a legitimate request on
    a CBCT whose HU scale is offset, where a fixed window clips the anatomy away.
    """
    from regix.config import ImagePrep
    from regix.preprocess.intensity import resolve_prep

    prep = ImagePrep(percentile_clip=(0.5, 99.5))
    resolved = resolve_prep(prep, "CT")
    assert resolved.percentile_clip == (0.5, 99.5)
    assert resolved.window is None


def test_the_auto_sentinel_survives_a_configuration_round_trip():
    """The reason the sentinel is a value and not `model_fields_set`.

    `with_overrides` rebuilds the configuration through model_dump/model_validate -- the
    CLI, the API and the pipeline test helpers all go through it -- which marks every
    field as explicitly set. A value-based sentinel is the only one that survives.
    """
    from regix.config import RegistrationConfig, load_preset

    cfg = load_preset("base")
    assert cfg.preprocess.fixed.percentile_clip == "auto"

    merged = cfg.with_overrides(preprocess={"working_spacing_mm": 1.0})
    assert merged.preprocess.fixed.percentile_clip == "auto"

    reloaded = RegistrationConfig.model_validate(yaml.safe_load(merged.to_yaml()))
    assert reloaded.preprocess.fixed.percentile_clip == "auto"


def test_resolve_prep_leaves_an_explicit_choice_alone_and_is_idempotent():
    from regix.config import ImagePrep
    from regix.preprocess.intensity import resolve_prep

    # window + percentile_clip: null, the shape every CT preset uses
    explicit = ImagePrep(window="ct_liver", percentile_clip=None)
    assert resolve_prep(explicit, "CT") is explicit

    once = resolve_prep(ImagePrep(normalize="minmax"), "MR")
    assert resolve_prep(once, "CT") == once, "resolution must not cascade on a second call"


def test_an_auto_prep_is_never_applied_unresolved(ct_phantom):
    """Guard rail: applying "auto" needs the modality, and guessing it is what broke."""
    from regix.config import ImagePrep
    from regix.io.volume import Volume
    from regix.preprocess.intensity import resolve_clip_bounds

    image, _ = ct_phantom
    with pytest.raises(ValueError, match="auto"):
        resolve_clip_bounds(Volume(image=image, modality="CT"), ImagePrep())


def test_n4_runs_before_clipping_and_percentiles_follow_it():
    """Order guard: N4 -> clipping, with the percentiles measured after the correction.

    A bias field is defined on the acquired signal, so estimating it after a window has
    truncated that signal estimates it on a distorted version. The observable
    consequence of getting this wrong is the clip bounds: with N4 first they are the
    percentiles of the *corrected* volume, which is what actually gets clipped.
    """
    from regix.config import ImagePrep
    from regix.io.volume import Volume
    from regix.preprocess.intensity import apply_intensity_prep, resolve_clip_bounds

    image, _ = make_phantom("MR", noise=4.0, seed=5)
    # A strong multiplicative gradient along z: exactly what N4 is meant to remove.
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
    ramp = np.linspace(0.45, 1.9, arr.shape[0], dtype=np.float32)[:, None, None]
    biased = sitk.GetImageFromArray(arr * ramp)
    biased.CopyInformation(image)
    volume = Volume(image=biased, modality="MR")

    prep = ImagePrep(percentile_clip=(0.5, 99.5), n4_bias_correction=True, normalize="none")
    out = apply_intensity_prep(volume, prep)
    assert out.meta["intensity_prep"]["n4"] is True

    applied = out.meta["intensity_prep"]["clip"]
    uncorrected = resolve_clip_bounds(volume, prep)  # percentiles of the biased volume
    assert uncorrected is not None
    # The bounds actually used must be the corrected volume's, not the biased one's.
    assert applied != [round(uncorrected[0], 4), round(uncorrected[1], 4)], (
        "clip bounds were measured before N4: they do not describe the clipped data"
    )


def test_no_bundled_preset_enables_n4_by_default():
    """N4 stays opt-in: it is worth far less to a registration than to a segmentation."""
    from regix.config import available_presets, load_preset

    for name in available_presets():
        cfg = load_preset(name)
        for side in ("fixed", "moving"):
            prep = getattr(cfg.preprocess, side)
            assert not prep.n4_bias_correction, f"{name}.{side} enables N4"


def test_every_preset_resolves_to_the_clipping_it_declares():
    """No bundled preset may have its explicit intent overwritten by the defaults."""
    from regix.config import available_presets, load_preset
    from regix.preprocess.intensity import resolve_prep

    for name in available_presets():
        cfg = load_preset(name)
        for side, modality in (("fixed", cfg.fixed_modality), ("moving", cfg.moving_modality)):
            prep = getattr(cfg.preprocess, side)
            resolved = resolve_prep(prep, modality)
            if prep.percentile_clip != "auto":
                assert resolved is prep, f"{name}.{side}: declared clipping was overridden"
            else:
                assert resolved.percentile_clip != "auto", f"{name}.{side}: sentinel survived"


def test_the_effective_config_reports_the_clipping_that_was_applied(tmp_path):
    """`config_effective.yaml` must not disagree with the manifest it sits next to.

    Observed on a real run: the manifest recorded clip [-450, 450] while the effective
    configuration still advertised percentile_clip [0.5, 99.5].
    """
    import json

    from regix.config import RegistrationConfig, load_preset
    from regix.io.volume import Volume
    from regix.pipeline import RegistrationPipeline

    # An MR pair: its modality default *is* a clipping (robust percentiles), so there is
    # something resolved to compare. A CT now resolves to "no clipping at all", which
    # would make the assertion vacuous.
    fixed, _ = make_phantom("MR", seed=1)
    moving, _ = make_phantom("MR", seed=2)
    cfg = load_preset("base").with_overrides(
        fixed_modality="MR",
        moving_modality="MR",
        preprocess={"working_spacing_mm": 4.0},
        stages=[{"type": "rigid", "n_resolutions": 1, "max_iterations": 8}],
        qc={"enabled": False, "report_html": False},
        output={"dir": str(tmp_path / "out"), "overwrite": True},
        runtime={"log_level": "WARNING"},
    )
    result = RegistrationPipeline(cfg).run(
        Volume(image=fixed, modality="MR"), Volume(image=moving, modality="MR"), tmp_path / "out"
    )

    saved = RegistrationConfig.model_validate(
        yaml.safe_load(pathlib.Path(result.outputs["config"]).read_text(encoding="utf-8"))
    )
    assert saved.preprocess.fixed.percentile_clip == (0.5, 99.5), "sentinel left unresolved"
    assert saved.preprocess.fixed.window is None

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    applied = next(s for s in manifest["steps"] if s["name"] == "preprocessing")
    # The manifest records the bounds that were actually applied; the effective config
    # records the percentiles they came from. The two must describe the same run.
    assert "clip" in applied["fixed"], "no clipping was applied for an MR volume"
    assert manifest["config"]["preprocess"]["fixed"]["percentile_clip"] == [0.5, 99.5]
    assert manifest["config"]["preprocess"]["fixed"]["normalize"] == "none"


def test_the_configuration_object_is_not_mutated_by_a_run():
    """A pipeline reused for a second pair must not inherit the first pair's modality."""
    from regix.config import load_preset
    from regix.preprocess.intensity import resolve_prep

    cfg = load_preset("base")
    before = cfg.preprocess.fixed.model_dump()
    resolve_prep(cfg.preprocess.fixed, "CT", "fixed")
    assert cfg.preprocess.fixed.model_dump() == before


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
    expected = labels.TransformContinuousIndexToPhysicalPoint([float(idx[2]), float(idx[1]), float(idx[0])])
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


def test_a_degenerate_stage_criterion_fails_the_gates():
    """The gate that makes the whole "silent failure" class visible.

    A stage can run, report success and optimise nothing: the transform stays plausible,
    the similarity gain is ~0 (which passes the gain gate, since `gain < threshold` is
    false for 0 < 0) and only the criterion betrays it. 6.7e-16 was the real value
    measured when an internal pixel type quantised the intensities away.
    """
    from regix.config import QCGates
    from regix.qc.gates import evaluate_gates

    degenerate = evaluate_gates(
        QCGates(min_ncc_gain=0.0),
        similarity={"ncc_gain": 0.0},
        stages=[{"stage": "affine", "final_metric": 6.66134e-16}],
    )
    assert degenerate.status == "FAIL"
    assert any(c.name == "final_metric[affine]" and c.status == "FAIL" for c in degenerate.checks)

    healthy = evaluate_gates(
        QCGates(min_ncc_gain=0.0),
        similarity={"ncc_gain": 0.42},
        stages=[{"stage": "affine", "final_metric": -0.372217}],
    )
    assert healthy.status == "PASS"


def test_a_zero_similarity_gain_is_not_reported_as_a_clean_pass():
    """`gain < threshold` cannot separate "did nothing" from "improved" at threshold 0."""
    from regix.config import QCGates
    from regix.qc.gates import evaluate_gates

    result = evaluate_gates(QCGates(min_ncc_gain=0.0), similarity={"ncc_gain": 0.0})
    check = next(c for c in result.checks if c.name == "ncc_gain")
    assert check.status == "WARN", "a strictly zero gain used to pass silently"


def test_the_overlay_figure_moves_in_all_three_planes(ct_phantom, monkeypatch):
    """Regression: only the axial index varied, so coronal and sagittal repeated.

    Asserted at the call site, not on the helper: the bug was that `overlay_figure`
    passed `centre[1]` and `centre[2]` unchanged for every slice, which no test of the
    index generator could have caught. Three identical panels read as three checks.
    """
    from regix.qc import report

    image, _ = ct_phantom
    requested: list[tuple[int, int, int]] = []
    original = report._extract_planes

    def _spy(arr, index):
        requested.append(tuple(index))
        return original(arr, index)

    monkeypatch.setattr(report, "_extract_planes", _spy)
    report.overlay_figure(image, image, image, n_slices=3)

    assert requested, "the figure rendered no plane"
    for axis, name in enumerate(("axial (z)", "coronal (y)", "sagittal (x)")):
        distinct = {index[axis] for index in requested}
        assert len(distinct) >= 3, f"{name} index never moved: {sorted(distinct)}"


def test_slice_positions_stay_inside_the_volume():
    from regix.qc.report import _slice_positions_around

    assert _slice_positions_around(centre=5, size=11, n=1) == [5]
    # A centre near the edge must clamp, not wrap or go negative.
    for index in _slice_positions_around(centre=0, size=40, n=5):
        assert 0 <= index < 40
    for index in _slice_positions_around(centre=39, size=40, n=5):
        assert 0 <= index < 40
    # ... and a tiny volume must not crash.
    assert all(0 <= i < 2 for i in _slice_positions_around(centre=1, size=2, n=3))


def test_a_missing_measurement_warns_rather_than_fails():
    from regix.config import QCGates
    from regix.qc.gates import evaluate_gates

    result = evaluate_gates(QCGates(min_dice={"liver": 0.9}), organ_overlap={})
    assert result.status == "WARN"


# --------------------------------------------------------------------------- #
# MIND descriptor (the CPU multimodal path)
# --------------------------------------------------------------------------- #
def test_box_mean_matches_a_separable_reference():
    """Guards the removal of scipy from the dependency list.

    ``_box_mean`` replaced ``scipy.ndimage.uniform_filter(..., mode="nearest")`` with
    ITK's MeanImageFilter. The reference here is an independent separable box mean
    written from the definition -- deliberately not scipy, so the test keeps working in
    an environment where scipy is not installed, which is the whole point of the change.
    Note that ``sitk.BoxMean`` is *not* a valid substitute: it normalises by the
    in-bounds voxel count instead of replicating the edge voxel (~20 % disagreement).
    """
    from regix.features.mind import _box_mean

    def reference(volume: np.ndarray, radius: int) -> np.ndarray:
        out = volume.astype(np.float64)
        size = 2 * radius + 1
        for axis in range(volume.ndim):
            padded = np.pad(
                out,
                [(radius, radius) if ax == axis else (0, 0) for ax in range(volume.ndim)],
                mode="edge",
            )
            stacked = np.stack(
                [np.take(padded, np.arange(k, k + out.shape[axis]), axis=axis) for k in range(size)]
            )
            out = stacked.mean(axis=0)
        return out

    rng = np.random.default_rng(7)
    for shape in [(9, 11, 13), (4, 4, 4), (3, 20, 17)]:
        for radius in (1, 2, 3):
            volume = (rng.random(shape) * 3.0).astype(np.float32)
            got = _box_mean(volume, radius)
            expected = reference(volume, radius)
            scale = float(np.abs(expected).max())
            assert got.shape == volume.shape
            assert np.abs(got - expected).max() / scale < 1e-5, (shape, radius)


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
