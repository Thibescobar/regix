"""Construction of elastix parameter files.

Structural choices, and why:

* ``UseDirectionCosines true``: without it elastix ignores the DICOM direction
  cosines and an oblique acquisition is wrong from the start. This is the first
  trap of any home-grown registration chain;
* ``AutomaticScalesEstimation`` + ``AdaptiveStochasticGradientDescent``: the
  pairing recommended by the elastix authors, which avoids hand-tuning the
  relative step of rotation against translation;
* ``MultiMetricMultiResolutionRegistration`` as soon as there is more than one
  channel: metric *i* compares channel *i* of the fixed image against channel
  *i* of the moving image. That mechanism is what allows anatomix features to be
  used as the substrate, and therefore CT against MR with a plain cross
  correlation;
* ``TransformBendingEnergyPenalty`` added as an extra metric on B-spline stages:
  without it a free deformation produces folded fields that score beautifully
  and make no anatomical sense.

An elastix rule established experimentally (not guessed) constrains the whole
construction: **the number of images must be 1 or equal to the number of
metrics**, and the "per image" components (pyramids, interpolators, samplers)
must have as many entries as there are *metrics*. Since a penalty is a metric
without an image, a multi-channel B-spline with N channels declares N+1 metrics
and therefore requires N+1 images: ``required_image_count()`` returns that
count, and the engine duplicates channel 0 to reach it. Without this, elastix
fails with ``FixedSmoothingPyramid: Input Primary is required but not set``.

Every generated file is written to disk: they can be replayed as-is with the
elastix binary, which is indispensable when investigating a case six months
later.

The reverse direction is supported too. ``StageConfig.parameter_file`` takes a
hand-written elastix parameter file -- typically one from the elastix parameter
zoo, or a file a site has already validated -- and uses it verbatim instead of
building a map. That is the point: those files *are* the interchange format of
the elastix world, so accepting them is what makes a Regix stage comparable with
what everyone else publishes. Four keys are nevertheless re-imposed, with a
warning, because a zoo file that disagrees with them does not merely tune the
optimisation, it silently invalidates the geometry or the transform chain around
it -- see ``ENFORCED_WITH_PARAMETER_FILE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regix.config import Metric, StageConfig, TransformType
from regix.logging_utils import get_logger

log = get_logger("registration.params")

ParameterMap = dict[str, tuple[str, ...]]

_TRANSFORM_NAMES = {
    TransformType.TRANSLATION: "TranslationTransform",
    TransformType.RIGID: "EulerTransform",
    TransformType.SIMILARITY: "SimilarityTransform",
    TransformType.AFFINE: "AffineTransform",
    TransformType.BSPLINE: "BSplineTransform",
}

_METRIC_NAMES = {
    Metric.MI: "AdvancedMattesMutualInformation",
    Metric.NCC: "AdvancedNormalizedCorrelation",
    Metric.MSE: "AdvancedMeanSquares",
    Metric.FEATURES_NCC: "AdvancedNormalizedCorrelation",
    Metric.FEATURES_MSE: "AdvancedMeanSquares",
}

#: Regularisation metrics: they have no associated image.
PENALTY_METRICS = frozenset(
    {
        "TransformBendingEnergyPenalty",
        "DistancePreservingRigidityPenalty",
        "TransformRigidityPenalty",
        "VarianceOverLastDimensionMetric",
    }
)

#: Modalities treated as equivalent when choosing a metric.
_MODALITY_FAMILIES = {"CT": "CT", "CBCT": "CT", "MR": "MR", "PT": "PT", "NM": "PT", "US": "US"}

#: Keys Regix re-imposes on an externally supplied parameter file, and why. Everything
#: else in such a file is honoured as written; these four are not tuning knobs, they are
#: assumptions the surrounding pipeline is built on, and a file that contradicts one of
#: them produces a plausible-looking result that is wrong:
#:
#: * ``UseDirectionCosines``: false makes elastix ignore the DICOM direction cosines, so
#:   an oblique acquisition is misregistered from the first iteration. Many zoo files
#:   predate this parameter and simply omit it -- and its elastix default is false;
#: * ``HowToCombineTransforms``: the whole chain (``-t0`` files, ``compose()``, the
#:   4x4 export) is written for Compose semantics. "Add" would make the composition
#:   arithmetic in the engine wrong without any error;
#: * ``AutomaticTransformInitialization``: Regix computes and records its own
#:   initialisation; letting elastix add a second one makes the reported transform
#:   disagree with the one that was applied;
#: * ``WriteResultImage``: Regix never reads elastix's resampled output -- it resamples
#:   the *native* moving intensities onto the original fixed grid itself. Leaving this
#:   true only costs a resample and a write per stage.
#:
#: Note what is deliberately **not** in this list: ``FixedInternalImagePixelType``.
#: ``"short"`` is the natural choice for images at their acquisition scale, and it used
#: to be catastrophic here only because Regix min-max normalised its inputs into
#: [0, 1], where an integer type rounds every voxel to 0 or 1 (measured with
#: Par0008.affine.txt: Mattes MI down to 6.7e-16, the optimiser doing nothing, and the
#: run still reporting WARN). The fix was to stop normalising -- Regix now hands elastix
#: native intensities, so the file's own choice is honoured. ``_warn_on_quantisation``
#: below still checks the combination, because a user can re-create it by asking for
#: ``normalize: minmax`` explicitly.
ENFORCED_WITH_PARAMETER_FILE: dict[str, tuple[str, ...]] = {
    "UseDirectionCosines": ("true",),
    "HowToCombineTransforms": ("Compose",),
    "AutomaticTransformInitialization": ("false",),
    "WriteResultImage": ("false",),
}

#: elastix internal pixel types that cannot represent a fractional intensity.
_INTEGER_PIXEL_TYPES = frozenset({"short", "unsigned short", "char", "unsigned char", "int", "long"})


@dataclass
class ParamContext:
    """What the engine knows at the time the parameters are built."""

    dimension: int = 3
    n_channels: int = 1
    working_spacing_mm: float = 2.0
    has_mask: bool = False
    fixed_modality: str = "UNKNOWN"
    moving_modality: str = "UNKNOWN"
    features_available: bool = False
    n_voxels: int | None = None
    #: (min, max) of the fixed image as handed to elastix. Only used to detect a
    #: parameter file whose internal pixel type would quantise it away.
    intensity_range: tuple[float, float] | None = None


def same_modality(fixed: str | None, moving: str | None) -> bool:
    f = _MODALITY_FAMILIES.get((fixed or "").upper(), (fixed or "").upper())
    m = _MODALITY_FAMILIES.get((moving or "").upper(), (moving or "").upper())
    return bool(f) and f == m


def resolve_metric(stage: StageConfig, ctx: ParamContext) -> Metric:
    """Resolve ``metric: auto`` into a concrete metric.

    Rule: features when available (the problem becomes monomodal, including for
    the rigid stage), otherwise mutual information for multimodal pairs,
    otherwise cross correlation for monomodal pairs.
    """
    if stage.metric is not Metric.AUTO:
        if stage.metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE) and not ctx.features_available:
            log.warning(
                "stage %s requests a feature metric but features are unavailable: falling back to %s",
                stage.display_name,
                "NCC" if same_modality(ctx.fixed_modality, ctx.moving_modality) else "MI",
            )
            return Metric.NCC if same_modality(ctx.fixed_modality, ctx.moving_modality) else Metric.MI
        return stage.metric

    if ctx.features_available and ctx.n_channels > 1:
        return Metric.FEATURES_NCC
    if same_modality(ctx.fixed_modality, ctx.moving_modality):
        return Metric.NCC
    return Metric.MI


# --------------------------------------------------------------------------- #
def build_parameter_map(stage: StageConfig, ctx: ParamContext) -> ParameterMap:
    """Build the elastix parameter map for one stage.

    With ``stage.parameter_file`` set, the file is read and used as the map instead
    of being generated; see ``_from_parameter_file``.
    """
    if stage.parameter_file is not None:
        return _from_parameter_file(stage, ctx)

    metric = resolve_metric(stage, ctx)
    n_img = max(1, ctx.n_channels if metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE) else 1)
    transform = _TRANSFORM_NAMES[stage.type]
    metric_name = _METRIC_NAMES[metric]

    metrics: list[str] = [metric_name] * n_img
    weights: dict[str, tuple[str, ...]] = {}
    for i in range(n_img):
        weights[f"Metric{i}Weight"] = (f"{1.0 / n_img:.6f}",)

    # Penalties: extra metrics with no image attached.
    penalty_index = n_img
    if stage.type is TransformType.BSPLINE and stage.bending_energy_weight > 0:
        metrics.append("TransformBendingEnergyPenalty")
        weights[f"Metric{penalty_index}Weight"] = (f"{stage.bending_energy_weight:.6f}",)
        penalty_index += 1
    if stage.type is TransformType.BSPLINE and stage.rigidity_penalty_weight > 0:
        metrics.append("DistancePreservingRigidityPenalty")
        weights[f"Metric{penalty_index}Weight"] = (f"{stage.rigidity_penalty_weight:.6f}",)
        penalty_index += 1

    n_metrics = len(metrics)
    needs_multi_metric = n_metrics > 1
    registration = (
        "MultiMetricMultiResolutionRegistration" if needs_multi_metric else "MultiResolutionRegistration"
    )
    # "Per image" components are counted in METRICS (see module docstring).
    per_metric = n_metrics

    pmap: ParameterMap = {
        # --- components ----------------------------------------------------- #
        "Registration": (registration,),
        "Transform": (transform,),
        "Metric": tuple(metrics),
        "Optimizer": (stage.optimizer,),
        "Interpolator": ("BSplineInterpolator",) * per_metric,
        "ResampleInterpolator": ("FinalBSplineInterpolator",),
        "Resampler": ("DefaultResampler",),
        "FixedImagePyramid": ("FixedSmoothingImagePyramid",) * per_metric,
        "MovingImagePyramid": ("MovingSmoothingImagePyramid",) * per_metric,
        # --- geometry ------------------------------------------------------- #
        "FixedInternalImagePixelType": ("float",),
        "MovingInternalImagePixelType": ("float",),
        "FixedImageDimension": (str(ctx.dimension),),
        "MovingImageDimension": (str(ctx.dimension),),
        "UseDirectionCosines": ("true",),
        "HowToCombineTransforms": ("Compose",),
        # our initialization is explicit: we do not want a second one
        "AutomaticTransformInitialization": ("false",),
        # --- multi-resolution ----------------------------------------------- #
        "NumberOfResolutions": (str(stage.n_resolutions),),
        "MaximumNumberOfIterations": (str(stage.max_iterations),),
        "AutomaticParameterEstimation": ("true",),
        "AutomaticScalesEstimation": ("true" if stage.automatic_scales else "false",),
        # --- sampling -------------------------------------------------------- #
        "ImageSampler": (stage.sampler,) * per_metric,
        "NumberOfSpatialSamples": (str(stage.n_spatial_samples),),
        "NewSamplesEveryIteration": ("true",),
        "CheckNumberOfSamples": ("true",),
        "MaximumNumberOfSamplingAttempts": ("8",),
        # Without this, elastix refuses to start as soon as 25 % of the samples fall
        # outside the moving image or mask -- which is the norm when fields of view
        # differ (whole-body CT against abdominal MR). See StageConfig.
        "RequiredRatioOfValidSamples": (f"{stage.required_ratio_valid_samples:.4f}",),
        "BSplineInterpolationOrder": (str(stage.interpolator_order),),
        "FinalBSplineInterpolationOrder": (str(stage.final_bspline_order),),
        # --- outputs --------------------------------------------------------- #
        "DefaultPixelValue": ("0",),
        "WriteResultImage": ("false",),
        "ResultImagePixelType": ("float",),
        "WriteTransformParametersEachIteration": ("false",),
        "WriteTransformParametersEachResolution": ("false",),
        "ShowExactMetricValue": ("false",),
        "ErodeMask": ("true" if stage.erode_mask else "false",),
    }
    pmap.update(weights)

    # --- metric specifics --------------------------------------------------- #
    if metric is Metric.MI:
        pmap["NumberOfHistogramBins"] = ("32",) * per_metric
        pmap["NumberOfFixedHistogramBins"] = ("32",) * per_metric
        pmap["NumberOfMovingHistogramBins"] = ("32",) * per_metric
    if metric in (Metric.NCC, Metric.FEATURES_NCC):
        pmap["SubtractMean"] = ("true",) * per_metric

    # --- transform specifics ------------------------------------------------ #
    if stage.type is TransformType.BSPLINE:
        spacing = float(stage.final_grid_spacing_mm)
        pmap["FinalGridSpacingInPhysicalUnits"] = (f"{spacing:.4f}",) * ctx.dimension
        schedule = stage.grid_spacing_schedule or [
            float(2 ** (stage.n_resolutions - 1 - r)) for r in range(stage.n_resolutions)
        ]
        pmap["GridSpacingSchedule"] = tuple(f"{v:.4f}" for v in schedule)
    else:
        # Rigid/affine: a bounded maximum step avoids absurd jumps in the first
        # iterations when the initialization is poor.
        pmap["MaximumStepLength"] = (f"{max(1.0, 2.0 * ctx.working_spacing_mm):.3f}",)

    if stage.use_masks and ctx.has_mask:
        # On a narrow mask, a uniform sampler misses its target.
        if stage.sampler == "RandomCoordinate":
            pmap["ImageSampler"] = ("RandomSparseMask",) * per_metric
        pmap["UseRandomSampleRegion"] = ("false",)

    # --- user overrides (final say) ----------------------------------------- #
    for key, value in stage.extra.items():
        pmap[key] = _as_tuple(value)

    _validate(pmap, dimension=ctx.dimension)
    return pmap


# --------------------------------------------------------------------------- #
def _from_parameter_file(stage: StageConfig, ctx: ParamContext) -> ParameterMap:
    """Use a hand-written elastix parameter file as the map for one stage.

    Honoured verbatim except for ``ENFORCED_WITH_PARAMETER_FILE``, then
    ``stage.extra`` on top, then the same validation as a generated map. Two
    consistency checks are refusals rather than warnings, because both fail
    *silently* otherwise:

    * ``stage.type`` must match the file's ``(Transform ...)``. Regix reads
      ``stage.type`` downstream to decide whether a stage produced a linear
      transform it can decompose and export (engine.py); a file declaring
      BSplineTransform under ``type: affine`` would have Regix parse a B-spline
      result as an affine matrix;
    * the file's dimension must match the pair being registered, otherwise elastix
      aborts with a message that says nothing about the parameter file.
    """
    path = Path(stage.parameter_file)  # type: ignore[arg-type]
    pmap = read_parameter_file(path)
    if not pmap:
        raise ValueError(
            f"{path} contains no elastix parameter. Expected lines of the form "
            '(Key "value") or (Key 3.0) -- is this really an elastix parameter file?'
        )
    for key in ("Transform", "Metric"):
        if key not in pmap:
            raise ValueError(f"{path} declares no ({key} ...): not a usable elastix parameter file")

    expected_transform = _TRANSFORM_NAMES[stage.type]
    if pmap["Transform"][0] != expected_transform:
        matching = [
            name for name, elastix_name in _TRANSFORM_NAMES.items() if elastix_name == pmap["Transform"][0]
        ]
        raise ValueError(
            f'{path.name} declares (Transform "{pmap["Transform"][0]}") but the stage says '
            f"type: {stage.type.value} (= {expected_transform}). Declare "
            + (f"type: {matching[0].value}" if matching else "a matching stage type")
            + " so that Regix interprets the stage result correctly."
        )

    for key, value in ENFORCED_WITH_PARAMETER_FILE.items():
        current = tuple(pmap.get(key, ()))
        if current and current != value:
            log.warning(
                "%s: (%s %s) overridden to %s -- Regix requires it, see ENFORCED_WITH_PARAMETER_FILE",
                path.name,
                key,
                " ".join(current),
                " ".join(value),
            )
        pmap[key] = value

    # A single-metric file cannot consume feature channels: elastix pairs metric i with
    # image i, so only channel 0 would reach the criterion. Silent otherwise.
    if ctx.features_available and ctx.n_channels > 1:
        n_image_metrics = sum(1 for m in pmap["Metric"] if m not in PENALTY_METRICS)
        if n_image_metrics < ctx.n_channels:
            log.warning(
                "%s declares %d image metric(s) for %d feature channels: elastix will only see "
                "channel 0. Use MultiMetricMultiResolutionRegistration with one metric per "
                "channel in the file, or drop parameter_file for this stage.",
                path.name,
                n_image_metrics,
                ctx.n_channels,
            )

    for key, value in stage.extra.items():
        pmap[key] = _as_tuple(value)

    _validate(pmap, dimension=ctx.dimension)
    _warn_on_quantisation(pmap, ctx, path.name)
    log.info("stage %s: parameters read from %s (%d keys)", stage.display_name, path.name, len(pmap))
    return pmap


def _warn_on_quantisation(pmap: ParameterMap, ctx: ParamContext, name: str) -> None:
    """Catch an integer internal pixel type on data too narrow to survive it.

    Regix keeps native intensities, so ``(FixedInternalImagePixelType "short")`` is
    normally fine -- a Hounsfield unit is already an integer. The combination only
    breaks when the images have been rescaled (``normalize: minmax`` asked for
    explicitly), because rounding [0, 1] to integers leaves two distinct values and the
    metric loses all information. That failure is silent: elastix reports success and
    the criterion is ~0. One line here is cheaper than diagnosing it later.
    """
    if ctx.intensity_range is None:
        return
    declared = {
        key: pmap[key][0]
        for key in ("FixedInternalImagePixelType", "MovingInternalImagePixelType")
        if key in pmap and pmap[key]
    }
    integer_types = {k: v for k, v in declared.items() if v.lower() in _INTEGER_PIXEL_TYPES}
    if not integer_types:
        return
    lo, hi = ctx.intensity_range
    if (hi - lo) >= 50.0:  # comfortably more distinct integers than any metric needs
        return
    log.warning(
        "%s declares %s but the images span only [%.4g, %.4g]: rounding to integers "
        "leaves almost no distinct values and the criterion will be meaningless. Set "
        "preprocess.<side>.normalize=none (the default) so the acquisition scale "
        "reaches elastix, or override the pixel type through the stage's 'extra'.",
        name,
        " and ".join(f"({k} {v})" for k, v in sorted(integer_types.items())),
        lo,
        hi,
    )


def required_image_count(pmap: ParameterMap) -> int:
    """Number of images the engine must hand to elastix for this map.

    elastix constraint: 1, or exactly the number of metrics. Since a penalty is a
    metric without an image, this returns ``n_metrics`` as soon as there are
    several image channels -- the engine then duplicates channel 0.
    """
    metrics = list(pmap.get("Metric", ()))
    if not metrics:
        return 1
    n_image_metrics = sum(1 for m in metrics if m not in PENALTY_METRICS)
    return 1 if n_image_metrics <= 1 else len(metrics)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _validate(pmap: ParameterMap, dimension: int | None = None) -> None:
    """Checks that avoid cryptic elastix messages.

    Keys absent from ``pmap`` are treated as "elastix will use its default", which is
    what an externally supplied parameter file relies on. A generated map always sets
    all of them, so nothing is weakened on that path -- but the three keys the checks
    themselves need are required either way.
    """
    for key in ("Transform", "Metric", "NumberOfResolutions"):
        if key not in pmap:
            raise ValueError(f"the parameter map declares no ({key} ...)")

    n_metrics = len(pmap["Metric"])
    for key in ("FixedImagePyramid", "MovingImagePyramid", "Interpolator", "ImageSampler"):
        values = pmap.get(key)
        if values is not None and len(values) != n_metrics:
            raise ValueError(
                f"{key} has {len(values)} entries for {n_metrics} metrics: elastix requires "
                "one entry per metric"
            )
    # Absent Registration means the elastix default, which is single-metric.
    registration = pmap.get("Registration", ("MultiResolutionRegistration",))[0]
    if n_metrics > 1 and registration != "MultiMetricMultiResolutionRegistration":
        raise ValueError(
            f"{n_metrics} metrics require MultiMetricMultiResolutionRegistration, not {registration}"
        )
    n_res = int(pmap["NumberOfResolutions"][0])
    if "GridSpacingSchedule" in pmap and len(pmap["GridSpacingSchedule"]) not in (n_res, n_res * 3):
        raise ValueError(
            f"GridSpacingSchedule has {len(pmap['GridSpacingSchedule'])} values for {n_res} resolutions"
        )
    # A warning, not a refusal: elastix has a documented fallback here ("the pyramid
    # schedule is not fully specified! A default pyramid schedule is used") and the
    # result stays correct. But it says so in elastix.log, which nobody reads, and it
    # means the file's intended pyramid is not the one that ran -- worth one line.
    schedule = pmap.get("ImagePyramidSchedule")
    if schedule is not None and dimension is not None and len(schedule) != n_res * dimension:
        log.warning(
            "ImagePyramidSchedule has %d values for %d resolutions in %dD (expected %d): "
            "elastix will ignore it and use its default schedule",
            len(schedule),
            n_res,
            dimension,
            n_res * dimension,
        )
    if dimension is not None:
        for key in ("FixedImageDimension", "MovingImageDimension"):
            declared = pmap.get(key)
            if declared and int(declared[0]) != dimension:
                raise ValueError(f"{key} is {declared[0]} but the volumes are {dimension}D")


# --------------------------------------------------------------------------- #
def to_itk_parameter_map(pmap: ParameterMap) -> dict[str, list[str]]:
    """Convert to a dictionary consumable by ``itk.ParameterObject.AddParameterMap``."""
    return {key: [str(v) for v in values] for key, values in pmap.items()}


def to_itk_parameter_object(pmap: ParameterMap):
    """Build a single-map ``itk.ParameterObject``."""
    import itk

    obj = itk.ParameterObject.New()
    obj.AddParameterMap(to_itk_parameter_map(pmap))
    return obj


def write_parameter_file(pmap: ParameterMap, path: str | Path) -> Path:
    """Write an elastix ``.txt`` parameter file, replayable from the command line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "// Parameter file generated by Regix.",
        "// Replay with: elastix -f fixed.nii.gz -m moving.nii.gz -out . -p " + p.name,
        "",
    ]
    for key in sorted(pmap):
        values = " ".join(_quote(v) for v in pmap[key])
        lines.append(f"({key} {values})")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _quote(value: str) -> str:
    text = str(value)
    if text.lower() in ("true", "false"):
        return f'"{text.lower()}"'
    try:
        float(text)
        return text
    except ValueError:
        return f'"{text}"'


def read_parameter_file(path: str | Path) -> ParameterMap:
    """Read back an elastix parameter file (supports external overrides)."""
    pmap: ParameterMap = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("//")[0].strip()
        if not line.startswith("(") or not line.endswith(")"):
            continue
        body = line[1:-1].strip()
        if not body:
            continue
        parts = _split_values(body)
        if len(parts) < 1:
            continue
        pmap[parts[0]] = tuple(parts[1:])
    return pmap


def _split_values(body: str) -> list[str]:
    out: list[str] = []
    token = ""
    in_quotes = False
    for ch in body:
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if ch.isspace() and not in_quotes:
            if token:
                out.append(token)
                token = ""
            continue
        token += ch
    if token:
        out.append(token)
    return out


def describe_stage(stage: StageConfig, ctx: ParamContext, pmap: ParameterMap | None = None) -> dict[str, Any]:
    """Readable summary of a stage, for the logs and the QC report.

    Pass the built ``pmap`` whenever it is available: the summary then reports what
    elastix was actually handed rather than what the configuration asked for. The two
    differ whenever ``stage.extra`` or ``stage.parameter_file`` is in play, and a
    manifest that reports the request instead of the effective value is worse than no
    manifest at all.
    """
    if pmap is None:
        metric = resolve_metric(stage, ctx)
        with_features = metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE)
        return {
            "stage": stage.display_name,
            "transform": _TRANSFORM_NAMES[stage.type],
            "metric": metric.value,
            "metric_elastix": _METRIC_NAMES[metric],
            "channels": ctx.n_channels if with_features else 1,
            "resolutions": stage.n_resolutions,
            "iterations": stage.max_iterations,
            "samples": stage.n_spatial_samples,
            "masked": bool(stage.use_masks and ctx.has_mask),
            "grid_mm": stage.final_grid_spacing_mm if stage.type is TransformType.BSPLINE else None,
            "parameter_file": None,
        }

    image_metrics = [m for m in pmap["Metric"] if m not in PENALTY_METRICS]
    elastix_metric = image_metrics[0] if image_metrics else pmap["Metric"][0]
    grid = pmap.get("FinalGridSpacingInPhysicalUnits")
    return {
        "stage": stage.display_name,
        "transform": pmap["Transform"][0],
        "metric": _metric_label(elastix_metric, len(image_metrics)),
        "metric_elastix": elastix_metric,
        "channels": len(image_metrics) if len(image_metrics) > 1 else 1,
        "resolutions": int(pmap["NumberOfResolutions"][0]),
        "iterations": int(pmap.get("MaximumNumberOfIterations", ("0",))[0]),
        "samples": int(pmap.get("NumberOfSpatialSamples", ("0",))[0]),
        "masked": bool(stage.use_masks and ctx.has_mask),
        "grid_mm": float(grid[0]) if grid else None,
        "parameter_file": str(stage.parameter_file) if stage.parameter_file else None,
    }


def _metric_label(elastix_metric: str, n_image_metrics: int) -> str:
    """Regix-level name of an elastix metric, for the logs and the report.

    ``AdvancedNormalizedCorrelation`` covers both ``ncc`` and ``features_ncc`` -- the
    number of image metrics is what distinguishes them.
    """
    multichannel = n_image_metrics > 1
    if elastix_metric == "AdvancedMattesMutualInformation":
        return Metric.MI.value
    if elastix_metric == "AdvancedNormalizedCorrelation":
        return (Metric.FEATURES_NCC if multichannel else Metric.NCC).value
    if elastix_metric == "AdvancedMeanSquares":
        return (Metric.FEATURES_MSE if multichannel else Metric.MSE).value
    return elastix_metric
