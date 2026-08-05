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
    """Build the elastix parameter map for one stage."""
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

    _validate(pmap)
    return pmap


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


def _validate(pmap: ParameterMap) -> None:
    """Checks that avoid cryptic elastix messages."""
    n_metrics = len(pmap["Metric"])
    for key in ("FixedImagePyramid", "MovingImagePyramid", "Interpolator", "ImageSampler"):
        if len(pmap[key]) != n_metrics:
            raise ValueError(
                f"{key} has {len(pmap[key])} entries for {n_metrics} metrics: elastix requires "
                "one entry per metric"
            )
    if n_metrics > 1 and pmap["Registration"][0] != "MultiMetricMultiResolutionRegistration":
        raise ValueError("several metrics require MultiMetricMultiResolutionRegistration")
    n_res = int(pmap["NumberOfResolutions"][0])
    if "GridSpacingSchedule" in pmap and len(pmap["GridSpacingSchedule"]) not in (n_res, n_res * 3):
        raise ValueError(
            f"GridSpacingSchedule has {len(pmap['GridSpacingSchedule'])} values for {n_res} resolutions"
        )


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


def describe_stage(stage: StageConfig, ctx: ParamContext) -> dict[str, Any]:
    """Readable summary of a stage, for the logs and the QC report."""
    metric = resolve_metric(stage, ctx)
    return {
        "stage": stage.display_name,
        "transform": _TRANSFORM_NAMES[stage.type],
        "metric": metric.value,
        "metric_elastix": _METRIC_NAMES[metric],
        "channels": ctx.n_channels if metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE) else 1,
        "resolutions": stage.n_resolutions,
        "iterations": stage.max_iterations,
        "samples": stage.n_spatial_samples,
        "masked": bool(stage.use_masks and ctx.has_mask),
        "grid_mm": stage.final_grid_spacing_mm if stage.type is TransformType.BSPLINE else None,
    }
