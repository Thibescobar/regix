"""Registration engine (elastix through the ``itk-elastix`` binding).

Architectural choice: **one elastix invocation per stage**, chained through the
initial-transform file (``-t0``). Everything could be chained in a single
multi-parameter invocation, but that would lose three things needed in
production:

* the ability to change images between stages (rigid on intensities with mutual
  information, then B-spline on feature channels);
* a per-stage duration, log and criterion, hence a usable diagnosis when a
  single stage misbehaves;
* restart from the middle: each ``TransformParameters.0.txt`` is self-contained
  and replayable with the elastix binary.

Two behaviours were established experimentally rather than assumed:

1. file-based chaining also works for a B-spline stage with a bending-energy
   penalty, which ``SetExternalInitialTransform`` does **not** allow (the
   external-transform adapter does not implement the spatial Jacobian the
   penalty needs);
2. at the end, ``GetCombinationTransform()`` holds the entire chain, and
   converting it to a ``sitk.Transform`` through HDF5 is exact. Everything
   downstream (resampling, Jacobian, point transport, inversion) is therefore
   done in SimpleITK, without going back through transformix.
"""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.config import StageConfig, TransformType
from regix.io.volume import Volume
from regix.logging_utils import get_logger
from regix.registration.itk_bridge import (
    image_types,
    itk_to_sitk,
    itk_transform_to_sitk,
    require_itk,
    sitk_to_itk,
)
from regix.registration.params import (
    ParamContext,
    build_parameter_map,
    describe_stage,
    required_image_count,
    to_itk_parameter_object,
    write_parameter_file,
)
from regix.registration.transforms import (
    decompose_affine,
    parameter_map_to_transform,
    to_matrix_4x4,
)

log = get_logger("registration.engine")

_FINAL_METRIC_RE = re.compile(r"Final metric value\s*=\s*([-\d.eE+]+)")
_DESCRIPTION_RE = re.compile(r"^Description: (.*)$", re.MULTILINE)


class RegistrationFailure(RuntimeError):
    """elastix failed, or produced an unusable transform."""


@dataclass
class StageResult:
    name: str
    transform_parameter_file: Path
    description: dict[str, Any]
    seconds: float
    final_metric: float | None = None
    transform: sitk.Transform | None = None
    linear_analysis: dict[str, Any] | None = None
    log_file: Path | None = None
    n_images: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "seconds": round(self.seconds, 2),
            "final_metric": self.final_metric,
            "transform_parameter_file": str(self.transform_parameter_file),
            **self.description,
            **({"linear": self.linear_analysis} if self.linear_analysis else {}),
        }


@dataclass
class RegistrationOutcome:
    """Raw engine result: the transform chain and its traces."""

    stages: list[StageResult] = field(default_factory=list)
    final_parameter_file: Path | None = None
    final_transform: sitk.Transform | None = None
    final_linear_transform: sitk.Transform | None = None
    output_dir: Path | None = None
    initial_transform_file: Path | None = None

    @property
    def is_deformable(self) -> bool:
        return any(s.description.get("transform") == "BSplineTransform" for s in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "final_parameter_file": str(self.final_parameter_file) if self.final_parameter_file else None,
            "deformable": self.is_deformable,
            "final_transform": type(self.final_transform).__name__ if self.final_transform else None,
        }


# --------------------------------------------------------------------------- #
class ElastixEngine:
    """Thin wrapper around ``itk.ElastixRegistrationMethod``."""

    def __init__(
        self,
        work_dir: str | Path,
        keep_intermediate: bool = False,
        verbose: bool = False,
        write_inputs: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.keep_intermediate = keep_intermediate
        self.verbose = verbose
        self.write_inputs = write_inputs
        require_itk()  # fail early, with an actionable message

    # ------------------------------------------------------------------ #
    def run(
        self,
        fixed: Volume,
        moving: Volume,
        stages: Sequence[StageConfig],
        context: ParamContext,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
        fixed_channels: Sequence[sitk.Image] | None = None,
        moving_channels: Sequence[sitk.Image] | None = None,
        initial_transform_file: str | Path | None = None,
        initial_transform: sitk.Transform | None = None,
    ) -> RegistrationOutcome:
        """Run the stage sequence and return the transform chain."""
        outcome = RegistrationOutcome(
            output_dir=self.work_dir,
            initial_transform_file=Path(initial_transform_file) if initial_transform_file else None,
        )
        current_t0 = Path(initial_transform_file) if initial_transform_file else None
        last_registration = None

        # Convert once: the images do not change from one stage to the next.
        itk_cache = _ItkInputs(fixed, moving, fixed_mask, moving_mask, fixed_channels, moving_channels)

        for index, stage in enumerate(stages):
            stage_dir = self.work_dir / f"stage{index:02d}_{stage.display_name}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            use_features = stage.metric.value.startswith("features") or (
                stage.metric.value == "auto" and context.features_available and context.n_channels > 1
            )
            channels_available = bool(fixed_channels) and bool(moving_channels)
            with_features = bool(use_features and channels_available)
            stage_ctx = replace(
                context,
                n_channels=context.n_channels if with_features else 1,
                has_mask=fixed_mask is not None,
                features_available=context.features_available and channels_available,
            )
            pmap = build_parameter_map(stage, stage_ctx)
            n_images = required_image_count(pmap)
            replay = None
            if self.write_inputs:
                replay = itk_cache.write_replay_bundle(
                    stage_dir=stage_dir,
                    with_features=with_features,
                    n_images=n_images,
                    use_masks=stage.use_masks,
                    initial_transform_file=current_t0,
                )
            write_parameter_file(pmap, stage_dir / "parameters.txt", replay_command=replay)
            description = describe_stage(stage, stage_ctx, pmap)
            log.info(
                "stage %d/%d: %s | %s | %d channel(s) -> %d elastix image(s) | %d resolutions | masked=%s",
                index + 1,
                len(stages),
                description["stage"],
                description["metric"],
                description["channels"],
                n_images,
                description["resolutions"],
                "yes" if description["masked"] else "no",
            )

            t0 = time.perf_counter()
            registration, result_file = self._invoke(
                pmap=pmap,
                stage_dir=stage_dir,
                inputs=itk_cache,
                with_features=with_features,
                n_images=n_images,
                use_masks=stage.use_masks,
                initial_transform_file=current_t0,
            )
            elapsed = time.perf_counter() - t0
            last_registration = registration

            log_file = stage_dir / "elastix.log"
            metric_value = _parse_final_metric(log_file)
            stage_transform = None
            linear = None
            if stage.type is not TransformType.BSPLINE:
                stage_transform = parameter_map_to_transform(_read_parameter_file(result_file))
                if stage_transform is not None:
                    matrix = to_matrix_4x4(stage_transform)
                    if matrix is not None:
                        linear = decompose_affine(matrix)

            if not self.keep_intermediate:
                for trace in stage_dir.glob("IterationInfo.*"):
                    trace.unlink(missing_ok=True)
                log_file.unlink(missing_ok=True)

            outcome.stages.append(
                StageResult(
                    name=description["stage"],
                    transform_parameter_file=result_file,
                    description=description,
                    seconds=elapsed,
                    final_metric=metric_value,
                    transform=stage_transform,
                    linear_analysis=linear,
                    log_file=log_file if log_file.exists() else None,
                    n_images=n_images,
                )
            )
            if linear:
                log.info(
                    "  -> translation %.1f mm, rotation %.2f deg, scale deviation %.3f, criterion %s",
                    linear["translation_norm_mm"],
                    linear["rotation_norm_deg"],
                    linear["max_scale_deviation"],
                    f"{metric_value:.5f}" if metric_value is not None else "n/a",
                )
            current_t0 = result_file

        outcome.final_parameter_file = current_t0

        # --- full final transform (chain included) -------------------------- #
        if last_registration is not None:
            outcome.final_transform = self._extract_final_transform(last_registration, fixed.image)

        # --- global linear transform, for export and analysis ---------------- #
        # The order follows the elastix HowToCombineTransforms=Compose semantics:
        # T_total(x) = T_last( ... T_first( T_initial(x) ) ).
        if not outcome.is_deformable:
            stage_transforms = [s.transform for s in outcome.stages]
            if stage_transforms and all(t is not None for t in stage_transforms):
                from regix.registration.transforms import compose

                chain = ([initial_transform] if initial_transform is not None else []) + list(
                    stage_transforms
                )
                outcome.final_linear_transform = compose(chain)
            if outcome.final_transform is None:
                outcome.final_transform = outcome.final_linear_transform
        return outcome

    # ------------------------------------------------------------------ #
    def _extract_final_transform(
        self,
        registration,
        reference: sitk.Image | None = None,
    ) -> sitk.Transform | None:
        """Retrieve the combination transform and convert it to a ``sitk.Transform``."""
        _, _, method = image_types()
        try:
            combination = registration.GetCombinationTransform()
            itk_transform = method.ConvertToItkTransform(combination)
        except Exception as exc:
            log.warning("ITK combination transform unavailable (%s): falling back to transformix", exc)
            return None
        try:
            return itk_transform_to_sitk(
                itk_transform,
                work_dir=self.work_dir / "final",
                reference=reference,
            )
        except Exception as exc:
            log.warning("conversion to SimpleITK failed (%s): falling back to transformix", exc)
            return None

    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        pmap: dict[str, tuple[str, ...]],
        stage_dir: Path,
        inputs: _ItkInputs,
        with_features: bool,
        n_images: int,
        use_masks: bool,
        initial_transform_file: Path | None,
    ):
        require_itk()
        _, _, method = image_types()

        registration = method.New()
        fixed_list, moving_list = inputs.image_lists(with_features, n_images)
        registration.SetFixedImage(fixed_list[0])
        registration.SetMovingImage(moving_list[0])
        for image in fixed_list[1:]:
            registration.AddFixedImage(image)
        for image in moving_list[1:]:
            registration.AddMovingImage(image)

        if use_masks:
            # One mask per image: elastix pairs mask i with metric i.
            if inputs.fixed_mask is not None:
                registration.SetFixedMask(inputs.fixed_mask)
                for _ in range(len(fixed_list) - 1):
                    registration.AddFixedMask(inputs.fixed_mask)
            if inputs.moving_mask is not None:
                registration.SetMovingMask(inputs.moving_mask)
                for _ in range(len(moving_list) - 1):
                    registration.AddMovingMask(inputs.moving_mask)

        if initial_transform_file is not None:
            registration.SetInitialTransformParameterFileName(str(initial_transform_file))

        registration.SetParameterObject(to_itk_parameter_object(pmap))
        registration.SetOutputDirectory(str(stage_dir))
        registration.SetLogToConsole(bool(self.verbose))
        registration.SetLogToFile(True)
        registration.SetLogFileName("elastix.log")

        try:
            registration.UpdateLargestPossibleRegion()
        except Exception as exc:
            raise RegistrationFailure(
                f"elastix failed at stage '{stage_dir.name}'.\n"
                f"{_diagnose(stage_dir / 'elastix.log', len(fixed_list), len(pmap['Metric']))}\n"
                f"ITK error: {exc}"
            ) from exc

        result = stage_dir / "TransformParameters.0.txt"
        if not result.exists():
            raise RegistrationFailure(f"elastix did not produce {result}. Log: {stage_dir / 'elastix.log'}")
        return registration, result


# --------------------------------------------------------------------------- #
class _ItkInputs:
    """One-off conversion of the inputs to ITK."""

    def __init__(
        self,
        fixed: Volume,
        moving: Volume,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        fixed_channels: Sequence[sitk.Image] | None,
        moving_channels: Sequence[sitk.Image] | None,
    ):
        self.fixed_intensity_sitk = fixed.image
        self.moving_intensity_sitk = moving.image
        self.fixed_mask_sitk = fixed_mask
        self.moving_mask_sitk = moving_mask
        self.fixed_features_sitk = list(fixed_channels or [])
        self.moving_features_sitk = list(moving_channels or [])
        self.fixed_intensity = sitk_to_itk(fixed.image)
        self.moving_intensity = sitk_to_itk(moving.image)
        self.fixed_mask = None
        self.moving_mask = None
        if fixed_mask is not None:
            _check_mask(fixed_mask, "fixed")
            self.fixed_mask = sitk_to_itk(fixed_mask, as_mask=True)
        if moving_mask is not None:
            _check_mask(moving_mask, "moving")
            self.moving_mask = sitk_to_itk(moving_mask, as_mask=True)
        self.fixed_features = [sitk_to_itk(c) for c in (fixed_channels or [])]
        self.moving_features = [sitk_to_itk(c) for c in (moving_channels or [])]

    def image_lists(self, with_features: bool, n_images: int):
        """Image lists handed to elastix, padded to the required count.

        The padding duplicates channel 0: penalty metrics then receive an image
        they never use, which satisfies the "number of images = number of
        metrics" constraint without changing the criterion.
        """
        if with_features and self.fixed_features and self.moving_features:
            fixed = list(self.fixed_features)
            moving = list(self.moving_features)
        else:
            fixed = [self.fixed_intensity]
            moving = [self.moving_intensity]
        if n_images > len(fixed):
            padding = n_images - len(fixed)
            fixed += [fixed[0]] * padding
            moving += [moving[0]] * padding
        elif n_images < len(fixed):  # pragma: no cover - defensive
            fixed, moving = fixed[:n_images], moving[:n_images]
        return fixed, moving

    def write_replay_bundle(
        self,
        stage_dir: Path,
        with_features: bool,
        n_images: int,
        use_masks: bool,
        initial_transform_file: Path | None,
    ) -> str:
        """Persist exactly what this stage receives and return its executable command."""
        input_dir = stage_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        fixed = self.fixed_features_sitk if with_features else [self.fixed_intensity_sitk]
        moving = self.moving_features_sitk if with_features else [self.moving_intensity_sitk]
        if n_images > len(fixed):
            fixed = fixed + [fixed[0]] * (n_images - len(fixed))
            moving = moving + [moving[0]] * (n_images - len(moving))
        else:
            fixed, moving = fixed[:n_images], moving[:n_images]

        fixed_paths: list[Path] = []
        moving_paths: list[Path] = []
        for index, (fixed_image, moving_image) in enumerate(zip(fixed, moving, strict=True)):
            fixed_path = input_dir / f"fixed_{index:02d}.nii.gz"
            moving_path = input_dir / f"moving_{index:02d}.nii.gz"
            if not fixed_path.exists():
                sitk.WriteImage(sitk.Cast(fixed_image, sitk.sitkFloat32), str(fixed_path), True)
            if not moving_path.exists():
                sitk.WriteImage(sitk.Cast(moving_image, sitk.sitkFloat32), str(moving_path), True)
            fixed_paths.append(fixed_path.relative_to(stage_dir))
            moving_paths.append(moving_path.relative_to(stage_dir))

        command = ["elastix"]
        if n_images == 1:
            command += ["-f", str(fixed_paths[0]), "-m", str(moving_paths[0])]
        else:
            for index, (fixed_path, moving_path) in enumerate(zip(fixed_paths, moving_paths, strict=True)):
                command += [f"-f{index}", str(fixed_path), f"-m{index}", str(moving_path)]
        if use_masks and self.fixed_mask_sitk is not None:
            fixed_mask = input_dir / "fixed_mask.nii.gz"
            sitk.WriteImage(sitk.Cast(self.fixed_mask_sitk, sitk.sitkUInt8), str(fixed_mask), True)
            command += ["-fMask", str(fixed_mask.relative_to(stage_dir))]
        if use_masks and self.moving_mask_sitk is not None:
            moving_mask = input_dir / "moving_mask.nii.gz"
            sitk.WriteImage(sitk.Cast(self.moving_mask_sitk, sitk.sitkUInt8), str(moving_mask), True)
            command += ["-mMask", str(moving_mask.relative_to(stage_dir))]
        if initial_transform_file is not None:
            command += ["-t0", str(Path(initial_transform_file).resolve())]
        command += ["-out", ".", "-p", "parameters.txt"]
        return shlex.join(command)


def _check_mask(mask: sitk.Image, side: str, min_voxels: int = 512) -> None:
    count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
    if count == 0:
        raise RegistrationFailure(f"the {side} mask is empty")
    if count < min_voxels:
        log.warning(
            "%s mask is very small (%d voxels): increase organs.mask_dilate_mm or reduce n_spatial_samples",
            side,
            count,
        )


def _read_parameter_file(path: Path):
    """Read back an elastix transform file as a dictionary."""
    from regix.registration.params import read_parameter_file

    return read_parameter_file(path)


def _parse_final_metric(log_file: Path) -> float | None:
    """Last 'Final metric value' in the elastix log (final resolution)."""
    if not log_file.exists():
        return None
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return None
    matches = _FINAL_METRIC_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:  # pragma: no cover
        return None


def _diagnose(log_file: Path, n_images: int, n_metrics: int) -> str:
    """Translate common elastix errors into a concrete action."""
    if not log_file.exists():
        return f"(no log at {log_file})"
    text = log_file.read_text(encoding="utf-8", errors="replace")
    descriptions = _DESCRIPTION_RE.findall(text)
    lowered = text.lower()
    hints: list[str] = []
    if "map outside moving image buffer" in lowered:
        hints.append(
            "too many samples fall outside the moving image or mask: the fields of view "
            "overlap poorly, or the initialization is too far off. Lower "
            "stages[i].required_ratio_valid_samples, widen organs.mask_dilate_mm, "
            "or use init.mode=organ_centroid / multistart"
        )
    elif "too many samples" in lowered or "not enough samples" in lowered:
        hints.append(
            "sampling impossible inside the mask: reduce n_spatial_samples, "
            "increase organs.mask_dilate_mm, or switch to sampler=Full"
        )
    if "input primary is required" in lowered or "should equal 1 or equal" in lowered:
        hints.append(
            f"multi-metric mismatch: {n_images} image(s) for {n_metrics} metric(s). "
            "elastix requires 1 image, or as many images as metrics"
        )
    if "not implemented for advancedtransformadapter" in lowered:
        hints.append(
            "a penalty (bending / rigidity) does not work with an external initial "
            "transform: chaining must go through a -t0 file"
        )
    if "singular" in lowered or "error in metric" in lowered:
        hints.append(
            "criterion became invalid: non-finite intensities, mask outside the common "
            "field of view, or insufficient overlap. Check preprocessing and initialization"
        )
    parts = []
    if hints:
        parts.append("Diagnosis:\n" + "\n".join(f"  - {h}" for h in hints))
    if descriptions:
        parts.append("elastix message: " + descriptions[-1].strip())
    parts.append("Full log: " + str(log_file))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Applying transforms (transformix) -- fallback path
# --------------------------------------------------------------------------- #
def set_output_grid(parameter_object, reference: sitk.Image, index: int = 0):
    """Force transformix to produce its result on the grid of ``reference``.

    This is what allows optimising at 2 mm on a cropped ROI and restoring the
    volume on the original fixed grid: the transform is defined in physical
    coordinates, only the sampling grid changes.
    """
    parameter_object.SetParameter(index, "Size", [str(int(s)) for s in reference.GetSize()])
    parameter_object.SetParameter(index, "Index", ["0"] * reference.GetDimension())
    parameter_object.SetParameter(index, "Spacing", [f"{v:.10f}" for v in reference.GetSpacing()])
    parameter_object.SetParameter(index, "Origin", [f"{v:.10f}" for v in reference.GetOrigin()])
    parameter_object.SetParameter(index, "Direction", [f"{v:.10f}" for v in reference.GetDirection()])
    return parameter_object


def apply_transform(
    parameter_file: str | Path,
    moving: sitk.Image,
    reference: sitk.Image | None = None,
    is_label: bool = False,
    default_value: float | None = None,
    work_dir: str | Path | None = None,
    compute_deformation_field: bool = False,
    verbose: bool = False,
) -> tuple[sitk.Image, sitk.Image | None]:
    """Apply an elastix transform chain through transformix.

    Returns (resampled image, displacement field or None). Only used when the
    conversion to ``sitk.Transform`` failed: the normal path goes through
    ``RegistrationOutcome.final_transform``.
    """
    itk = require_itk()
    image_f, _, _ = image_types()

    parameters = itk.ParameterObject.New()
    parameters.ReadParameterFile(str(parameter_file))
    if reference is not None:
        set_output_grid(parameters, reference)
    if is_label:
        # Nearest neighbour is mandatory: an interpolated label is an invented label.
        parameters.SetParameter(0, "FinalBSplineInterpolationOrder", "0")
    if default_value is not None:
        parameters.SetParameter(0, "DefaultPixelValue", f"{float(default_value):.6f}")

    filt = itk.TransformixFilter[image_f].New()
    filt.SetMovingImage(sitk_to_itk(moving))
    filt.SetTransformParameterObject(parameters)
    filt.SetLogToConsole(bool(verbose))
    if compute_deformation_field:
        filt.SetComputeDeformationField(True)
    directory = Path(work_dir) if work_dir is not None else Path(str(Path(parameter_file).parent))
    directory.mkdir(parents=True, exist_ok=True)
    filt.SetOutputDirectory(str(directory))

    filt.UpdateLargestPossibleRegion()
    result = itk_to_sitk(filt.GetOutput())
    field = None
    if compute_deformation_field:
        try:
            field = itk_to_sitk(filt.GetOutputDeformationField(), is_vector=True)
        except Exception as exc:  # pragma: no cover
            log.warning("displacement field unavailable: %s", exc)

    if is_label:
        result = sitk.Cast(sitk.Round(result), moving.GetPixelID())
    return result, field
