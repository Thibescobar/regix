"""Full orchestration of a registration.

The sequence, and the reason for each step:

  1. loading (DICOM or NIfTI) and geometric inventory;
  2. organ segmentation -- before any preprocessing, on native intensities, because
     segmentation networks expect Hounsfield units;
  3. field-of-view overlap: a diagnosis, not a correction;
  4. preprocessing (orientation, optional clipping, working resolution): for the
     optimisation only, and scale-preserving -- what elastix receives keeps the
     acquisition scale, see ``regix.preprocess.intensity``;
  5. per-organ ROI (optional): do not pay for a whole-body registration when only
     the pancreas matters;
  6. modality-invariant features (optional): makes a multimodal pair tractable as
     a monomodal one, from the rigid stage onwards;
  7. initialization, optionally multi-start scored independently;
  8. elastix stages (rigid -> affine -> B-spline), chained through files;
  9. alternative feature-based deformable stage (GPU) when requested;
 10. restitution on the original grid of the fixed image, with the native
     intensities of the moving image;
 11. QC independent of the optimised criterion, gates, report, manifest.

One rule runs through the whole file: **the output image is reconstructed from the
original moving volume, not from the preprocessed one**. Otherwise the clinician
receives a clipped, twice-resampled image at the working resolution -- and has lost
the Hounsfield units.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.config import (
    DeformableEngine,
    ImagePrep,
    Metric,
    OrganBackend,
    RegistrationConfig,
    TransformType,
)
from regix.io.volume import Volume, load_volume
from regix.io.writers import (
    load_landmarks,
    save_image,
    write_derived_dicom,
    write_spatial_registration_dicom,
)
from regix.logging_utils import RunManifest, get_logger, setup_logging
from regix.organs.labels import merged_profile, resolve_targets
from regix.organs.roi import combined_mask, organ_volumes_ml, plan_roi, roi_overlap_report
from regix.organs.segmenter import OrganSegmentation, build_segmenter
from regix.preprocess.geometry import reorient, resample_like, resample_to_spacing
from regix.preprocess.intensity import apply_intensity_prep, resolve_prep
from regix.qc.gates import evaluate_gates
from regix.qc.metrics import (
    displacement_statistics,
    jacobian_statistics,
    organ_overlap_report,
    similarity_report,
    target_registration_error,
)
from regix.qc.report import (
    build_html_report,
    checkerboard_figure,
    contour_figure,
    jacobian_figure,
    overlay_figure,
)
from regix.registration.engine import ElastixEngine, RegistrationFailure
from regix.registration.initialize import choose_initialization
from regix.registration.params import ParamContext, same_modality
from regix.registration.transforms import (
    compose,
    decompose_affine,
    flatten_linear,
    matrix_moving_to_fixed,
    save_transform,
    to_matrix_4x4,
    transform_to_elastix_initial,
)
from regix.registration.warp import (
    AppliedTransform,
    ElastixAppliedTransform,
    SitkAppliedTransform,
    transform_points_via_field,
)

log = get_logger("pipeline")


@dataclass
class RegistrationResult:
    """Everything a run produces."""

    status: str
    applied_transform: AppliedTransform
    registered_image: sitk.Image | None
    outputs: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    initialization: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    manifest_path: Path | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def summary(self) -> str:
        lines = [f"Registration {self.status} in {self.seconds:.1f} s"]
        sim = self.metrics.get("similarity", {})
        if sim:
            lines.append(
                "  similarity: NCC {ncc_before} -> {ncc_after} | NMI {nmi_before} -> {nmi_after}".format(
                    ncc_before=sim.get("ncc_before"),
                    ncc_after=sim.get("ncc_after"),
                    nmi_before=sim.get("nmi_before"),
                    nmi_after=sim.get("nmi_after"),
                )
            )
        for organ, values in (self.metrics.get("organ_overlap") or {}).items():
            lines.append(f"  Dice {organ}: {values.get('dice')} (HD95 {values.get('hd95_mm')} mm)")
        tre = self.metrics.get("landmarks") or {}
        if tre:
            lines.append(
                f"  TRE: {tre.get('tre_before_mean_mm')} mm -> {tre.get('tre_mean_mm')} mm "
                f"({tre.get('n_landmarks')} landmarks)"
            )
        for check in self.qc.get("checks", []):
            if check["status"] != "PASS":
                lines.append(f"  [{check['status']}] {check['name']}: {check['message']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
class RegistrationPipeline:
    """Run the configuration on a pair of volumes."""

    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.targets = resolve_targets(config.organs.targets)

    # ------------------------------------------------------------------ #
    def run(
        self,
        fixed: str | Path | Volume,
        moving: str | Path | Volume,
        output_dir: str | Path | None = None,
    ) -> RegistrationResult:
        cfg = self.config
        started = time.perf_counter()
        out_dir = Path(output_dir) if output_dir is not None else Path(cfg.output.dir)
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        if out_dir.exists() and any(out_dir.iterdir()) and not cfg.output.overwrite:
            raise FileExistsError(
                f"{out_dir} is not empty. Use output.overwrite=true or a different directory."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(cfg.runtime.log_level, log_file=out_dir / "regix.log")
        if cfg.runtime.threads:
            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(cfg.runtime.threads))

        manifest = RunManifest(run_id=run_id, output_dir=out_dir)
        manifest.config = cfg.model_dump(mode="json")
        outputs: dict[str, Path] = {}
        log.info("=== Regix %s | preset '%s' ===", run_id, cfg.name)

        try:
            result = self._run_inner(fixed, moving, out_dir, manifest, outputs)
        except Exception as exc:
            manifest.status = "ERROR"
            manifest.warn(f"failure: {type(exc).__name__}: {exc}")
            manifest.save()
            log.exception("the registration failed")
            raise
        result.seconds = time.perf_counter() - started
        manifest.status = result.status
        manifest.metrics = result.metrics
        manifest.metrics["qc"] = result.qc
        result.manifest_path = manifest.save()
        result.warnings = list(manifest.warnings)
        log.info("=== finished: %s (%.1f s) ===", result.status, result.seconds)
        return result

    # ------------------------------------------------------------------ #
    def _run_inner(
        self,
        fixed_input: str | Path | Volume,
        moving_input: str | Path | Volume,
        out_dir: Path,
        manifest: RunManifest,
        outputs: dict[str, Path],
    ) -> RegistrationResult:
        cfg = self.config

        # -- 1. loading ------------------------------------------------------ #
        with manifest.step("loading") as info:
            fixed = self._load(fixed_input, cfg.fixed_modality, "fixed")
            moving = self._load(moving_input, cfg.moving_modality, "moving")
            info["fixed"] = fixed.describe()
            info["moving"] = moving.describe()
            manifest.inputs = {"fixed": info["fixed"], "moving": info["moving"]}
        for volume, side in ((fixed, "fixed"), (moving, "moving")):
            for warning in volume.meta.get("warnings", []):
                manifest.warn(f"[{side}] {warning}")
        multimodal = not same_modality(fixed.modality, moving.modality)
        log.info(
            "pair %s -> %s (%s)",
            moving.modality,
            fixed.modality,
            "multimodal" if multimodal else "monomodal",
        )

        # -- 2. segmentation (on native intensities) -------------------------- #
        fixed_seg: OrganSegmentation | None = None
        moving_seg: OrganSegmentation | None = None
        if cfg.organs.backend is not OrganBackend.NONE:
            with manifest.step("segmentation") as info:
                fixed_seg, moving_seg = self._segment(fixed, moving, out_dir, manifest)
                info["backend"] = cfg.organs.backend.value
                if fixed_seg:
                    info["fixed_organs"] = fixed_seg.present_organs()
                    info["fixed_volumes_ml"] = organ_volumes_ml(fixed_seg)
                if moving_seg:
                    info["moving_organs"] = moving_seg.present_organs()
                    info["moving_volumes_ml"] = organ_volumes_ml(moving_seg)

        # -- 3. overlap diagnosis -------------------------------------------- #
        qc_mask_fixed = combined_mask(
            fixed, fixed_seg, self.targets, cfg.organs.mask_dilate_mm, fallback_body_mask=True
        )
        qc_mask_moving = combined_mask(
            moving, moving_seg, self.targets, cfg.organs.mask_dilate_mm, fallback_body_mask=True
        )
        overlap = roi_overlap_report(fixed, moving, qc_mask_fixed, qc_mask_moving)
        manifest.metrics["fov"] = overlap
        if min(overlap["fov_overlap_fraction_fixed"], overlap["fov_overlap_fraction_moving"]) < 0.25:
            manifest.warn(
                "field-of-view overlap below 25 %: organ-based initialization is strongly recommended"
            )

        # -- 4. preprocessing ------------------------------------------------ #
        # The modality defaults are resolved once, here, and the resolved preparations
        # are what gets written to config_effective.yaml and to the manifest below. The
        # configuration object itself is left untouched: a pipeline reused for a second
        # pair must not inherit the first pair's modality.
        prep_fixed = resolve_prep(cfg.preprocess.fixed, fixed.modality, "fixed")
        prep_moving = resolve_prep(cfg.preprocess.moving, moving.modality, "moving")
        effective = cfg.model_copy(
            update={
                "preprocess": cfg.preprocess.model_copy(update={"fixed": prep_fixed, "moving": prep_moving})
            }
        )
        manifest.config = effective.model_dump(mode="json")
        with manifest.step("preprocessing") as info:
            fixed_work = self._prepare(fixed, prep_fixed)
            moving_work = self._prepare(moving, prep_moving)
            info["fixed"] = {
                "size": list(fixed_work.size),
                "spacing": [round(v, 3) for v in fixed_work.spacing],
                **fixed_work.meta.get("intensity_prep", {}),
            }
            info["moving"] = {
                "size": list(moving_work.size),
                "spacing": [round(v, 3) for v in moving_work.spacing],
                **moving_work.meta.get("intensity_prep", {}),
            }

        # -- 5. per-organ ROI ------------------------------------------------ #
        if cfg.organs.roi_crop and self.targets:
            with manifest.step("roi") as info:
                roi = plan_roi(
                    fixed_work, moving_work, fixed_seg, moving_seg, self.targets, cfg.organs.roi_margin_mm
                )
                fixed_work, moving_work = roi.fixed, roi.moving
                info.update(roi.info)

        # -- 6. masks at the working resolution ------------------------------ #
        work_mask_fixed = self._work_mask(fixed_work, fixed_seg)
        work_mask_moving = self._work_mask(moving_work, moving_seg)

        # -- 7. features ----------------------------------------------------- #
        fixed_channels: list[sitk.Image] | None = None
        moving_channels: list[sitk.Image] | None = None
        feature_info: dict[str, Any] = {}
        if self._features_wanted(multimodal):
            with manifest.step("features") as info:
                fixed_channels, moving_channels, feature_info = self._extract_features(
                    fixed_work, moving_work, work_mask_fixed, work_mask_moving, manifest
                )
                info.update(feature_info)
            if fixed_channels and cfg.output.write_features:
                for idx, channel in enumerate(fixed_channels):
                    save_image(channel, out_dir / "features" / f"fixed_c{idx:02d}.nii.gz")
                for idx, channel in enumerate(moving_channels or []):
                    save_image(channel, out_dir / "features" / f"moving_c{idx:02d}.nii.gz")
                outputs["features"] = out_dir / "features"

        # -- 8. initialization ----------------------------------------------- #
        with manifest.step("initialization") as info:
            candidate, init_report = choose_initialization(
                fixed_work,
                moving_work,
                cfg.init,
                fixed_seg,
                moving_seg,
                self.targets,
                work_mask_fixed,
                work_mask_moving,
            )
            info.update({k: v for k, v in init_report.items() if k != "candidates"})
            initial_file = None
            if not _is_identity(candidate.transform):
                initial_file = transform_to_elastix_initial(
                    candidate.transform, fixed_work.image, out_dir / "elastix" / "initial_transform.txt"
                )
            if init_report.get("ambiguous"):
                manifest.warn("several equivalent initializations: verify the result visually")

        # -- 9. elastix stages ----------------------------------------------- #
        engine = ElastixEngine(
            out_dir / "elastix",
            keep_intermediate=cfg.runtime.keep_intermediate,
            verbose=cfg.runtime.log_level == "DEBUG",
        )
        context = ParamContext(
            dimension=3,
            n_channels=len(fixed_channels) if fixed_channels else 1,
            working_spacing_mm=float(np.mean(fixed_work.spacing)),
            has_mask=work_mask_fixed is not None,
            fixed_modality=fixed.modality,
            moving_modality=moving.modality,
            features_available=bool(fixed_channels),
            n_voxels=fixed_work.n_voxels,
            intensity_range=_intensity_range(fixed_work.image),
        )
        stages = self._resolve_stages()
        with manifest.step("elastix") as info:
            outcome = engine.run(
                fixed_work,
                moving_work,
                stages,
                context,
                fixed_mask=work_mask_fixed,
                moving_mask=work_mask_moving,
                fixed_channels=fixed_channels,
                moving_channels=moving_channels,
                initial_transform_file=initial_file,
                initial_transform=candidate.transform,
            )
            info["stages"] = [s.to_dict() for s in outcome.stages]

        # Normal path: the combination transform converted to a sitk.Transform. It also
        # covers the B-spline case, which removes any dependency on transformix for
        # resampling, the Jacobian and point transport.
        if outcome.final_transform is not None:
            applied: AppliedTransform = SitkAppliedTransform(outcome.final_transform, label="elastix_chain")
        else:
            manifest.warn(
                "combination transform not convertible: outputs go through transformix "
                "(no point transport and no analytical inversion)"
            )
            applied = ElastixAppliedTransform(
                outcome.final_parameter_file,
                work_dir=out_dir / "transformix",
                linear_transform=outcome.final_linear_transform,
            )

        # -- 10. alternative deformable stage -------------------------------- #
        deformable_info: dict[str, Any] = {}
        if cfg.deformable_engine is DeformableEngine.CONVEXADAM:
            with manifest.step("deformable_convexadam") as info:
                applied, deformable_info = self._convexadam(
                    fixed_work, moving_work, applied, outcome, work_mask_fixed, manifest
                )
                info.update(deformable_info)

        # -- 11. restitution on the original grid ---------------------------- #
        with manifest.step("restitution") as info:
            registered = applied.resample(
                moving.image,  # native intensities, never the preprocessed volume
                fixed.image,
                is_label=False,
                default_value=_background_of(moving.image),
            )
            info["output_size"] = list(registered.GetSize())
            info["output_spacing"] = [round(v, 3) for v in registered.GetSpacing()]
            if cfg.output.write_resampled:
                outputs["registered"] = save_image(
                    registered,
                    out_dir / f"moving_registered.nii{'.gz' if cfg.output.compress else ''}",
                    compress=cfg.output.compress,
                )

        # -- 12. QC ---------------------------------------------------------- #
        metrics: dict[str, Any] = {"fov": overlap, "features": feature_info, "deformable": deformable_info}
        qc_result: dict[str, Any] = {}
        figures: dict[str, str] = {}
        if cfg.qc.enabled:
            with manifest.step("qc") as info:
                metrics, qc_result, figures = self._quality_control(
                    fixed=fixed,
                    moving=moving,
                    registered=registered,
                    applied=applied,
                    outcome=outcome,
                    initial_transform=candidate.transform,
                    fixed_seg=fixed_seg,
                    moving_seg=moving_seg,
                    qc_mask=qc_mask_fixed,
                    multimodal=multimodal,
                    base_metrics=metrics,
                    out_dir=out_dir,
                    outputs=outputs,
                    manifest=manifest,
                )
                info["status"] = qc_result.get("status")

        # -- 13. transforms and exports -------------------------------------- #
        with manifest.step("exports") as info:
            self._export_transforms(applied, outcome, fixed, moving, out_dir, outputs, candidate.transform)
            if cfg.output.write_dicom:
                self._export_dicom(registered, fixed, moving, applied, out_dir, outputs, manifest)
            # `effective`, not `self.config`: the file has to describe the run that
            # happened, resolved modality defaults included.
            outputs["config"] = effective.save(out_dir / "config_effective.yaml")
            info["files"] = {k: str(v) for k, v in outputs.items()}

        # -- 14. report ------------------------------------------------------ #
        if cfg.qc.enabled and cfg.qc.report_html:
            context_html = {
                "title": f"Registration {moving.modality} -> {fixed.modality}",
                "subtitle": (
                    f"subject {fixed.subject_id} · preset {cfg.name} · "
                    f"targets: {', '.join(self.targets) or 'whole body'}"
                ),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "qc": qc_result,
                "figures": figures,
                "similarity": metrics.get("similarity"),
                "organ_overlap": metrics.get("organ_overlap"),
                "landmarks": metrics.get("landmarks"),
                "jacobian": metrics.get("jacobian"),
                "stages": [{**s.to_dict(), "final_metric": s.final_metric} for s in outcome.stages],
                "initialization": init_report,
                "inputs": {
                    "fixed": _describe_grid(fixed),
                    "moving": _describe_grid(moving),
                    "subject": fixed.subject_id,
                    "field of view overlap": overlap.get("fov_overlap_fraction_fixed"),
                },
                "configuration": {
                    "preset": cfg.name,
                    "stages": " -> ".join(s.display_name for s in stages),
                    "metrics": " / ".join(
                        {s.get("metric", "?") for s in [x.to_dict() for x in outcome.stages]}
                    ),
                    "features": feature_info.get("provider", "none"),
                    "deformable": cfg.deformable_engine.value,
                    "segmentation": cfg.organs.backend.value,
                    "initialization": init_report.get("chosen"),
                    "working resolution (mm)": cfg.preprocess.working_spacing_mm,
                },
                "environment": manifest.environment,
                "warnings": manifest.warnings,
            }
            outputs["report"] = build_html_report(out_dir / "report.html", context_html)

        return RegistrationResult(
            status=qc_result.get("status", "WARN") if cfg.qc.enabled else "WARN",
            applied_transform=applied,
            registered_image=registered,
            outputs=outputs,
            metrics=metrics,
            qc=qc_result,
            stages=[s.to_dict() for s in outcome.stages],
            initialization=init_report,
        )

    # ================================================================== #
    # Elementary steps
    # ================================================================== #
    def _load(self, source: str | Path | Volume, modality: str | None, side: str) -> Volume:
        if isinstance(source, Volume):
            volume = source
            if modality:
                volume.modality = modality.upper()
        else:
            volume = load_volume(source, modality=modality, pseudonymize_ids=self.config.runtime.pseudonymize)
        log.info("%s volume: %r", side, volume)
        if volume.modality == "UNKNOWN":
            log.warning(
                "unknown %s modality: set %s_modality, otherwise the metric and windowing "
                "choices will be approximate",
                side,
                side,
            )
        return volume

    def _segment(
        self, fixed: Volume, moving: Volume, out_dir: Path, manifest: RunManifest
    ) -> tuple[OrganSegmentation | None, OrganSegmentation | None]:
        cfg = self.config.organs
        cache = out_dir / "cache"
        results: list[OrganSegmentation | None] = []
        for volume, side in ((fixed, "fixed"), (moving, "moving")):
            segmenter = build_segmenter(cfg, side=side, cache_dir=cache)
            if segmenter is None:
                manifest.warn(f"no segmenter available for the {side} image")
                results.append(None)
                continue
            try:
                seg = segmenter.segment(volume)
            except Exception as exc:
                manifest.warn(f"{side} segmentation failed ({type(exc).__name__}: {exc})")
                log.warning("%s segmentation failed: %s", side, exc)
                results.append(None)
                continue
            present = seg.present_organs()
            log.info("%s segmentation: %d organs (%s)", side, len(present), ", ".join(present[:8]))
            missing = [o for o in self.targets if o not in present]
            if missing:
                manifest.warn(f"target organs missing from the {side} image: {missing}")
            if cfg.save_masks:
                save_image(seg.labelmap, out_dir / "masks" / f"{side}_labelmap.nii.gz", dtype=sitk.sitkUInt16)
            results.append(seg)
        return results[0], results[1]

    def _prepare(self, volume: Volume, prep: ImagePrep) -> Volume:
        """Apply an already-resolved preparation (see ``resolve_prep``) to one volume."""
        cfg = self.config.preprocess
        work = volume
        if cfg.orientation:
            work = work.with_image(reorient(work.image, cfg.orientation))
        work = apply_intensity_prep(work, prep)
        if cfg.working_spacing_mm is not None:
            work = work.with_image(
                resample_to_spacing(work.image, cfg.working_spacing_mm, interpolator="linear")
            )
        return work

    def _work_mask(self, work: Volume, seg: OrganSegmentation | None) -> sitk.Image | None:
        cfg = self.config.organs
        mask = combined_mask(
            work,
            seg.resampled_to(work.image) if seg is not None else None,
            self.targets,
            cfg.mask_dilate_mm,
            fallback_body_mask=True,
        )
        if mask is None:
            return None
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
        if count < 512:
            log.warning("working mask too small (%d voxels): registering without a mask", count)
            return None
        return mask

    def _features_wanted(self, multimodal: bool) -> bool:
        setting = self.config.features.enabled
        if setting is False:
            return False
        needs_features = (
            any(s.metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE) for s in self.config.stages)
            or self.config.deformable_engine is DeformableEngine.CONVEXADAM
        )
        if setting is True:
            return True
        # auto mode
        return bool(multimodal or needs_features)

    def _extract_features(
        self,
        fixed_work: Volume,
        moving_work: Volume,
        mask_fixed: sitk.Image | None,
        mask_moving: sitk.Image | None,
        manifest: RunManifest,
    ) -> tuple[list[sitk.Image] | None, list[sitk.Image] | None, dict[str, Any]]:
        from regix.features.anatomix import anatomix_available, extract_feature_pair

        cfg = self.config.features
        ok, reason = anatomix_available()
        provider = "auto"
        if not ok:
            if self.config.deformable_engine is DeformableEngine.CONVEXADAM:
                raise RegistrationFailure(
                    f"deformable_engine=convexadam requires torch + anatomix ({reason})"
                )
            manifest.warn(f"anatomix unavailable ({reason}): using the MIND-SSC descriptor instead")
            provider = "mind"
        try:
            pair = extract_feature_pair(
                fixed_work,
                moving_work,
                cfg,
                fixed_mask=mask_fixed,
                moving_mask=mask_moving,
                provider=provider,
                seed=self.config.runtime.seed,
            )
        except Exception as exc:
            manifest.warn(f"feature extraction failed ({type(exc).__name__}: {exc})")
            log.warning("feature extraction failed: %s -- falling back to intensities", exc)
            return None, None, {"provider": "none", "error": str(exc)}
        log.info(
            "features %s: %d channels (explained variance %s)",
            pair.provider,
            pair.n_channels,
            pair.info.get("pca", {}).get("explained_variance_ratio", "n/a"),
        )
        return pair.fixed_channels, pair.moving_channels, pair.info

    def _resolve_stages(self):
        """Adapt the stages to the target organ profile (B-spline grid, deformability)."""
        cfg = self.config
        stages = [s.model_copy(deep=True) for s in cfg.stages]
        if not self.targets:
            return stages
        profile = merged_profile(self.targets)
        for stage in stages:
            if stage.type is TransformType.BSPLINE:
                if not profile.deformable:
                    log.warning(
                        "the target organs (%s) are considered rigid: the B-spline grid is "
                        "widened to 40 mm to limit non-physical deformations",
                        profile.name,
                    )
                    stage.final_grid_spacing_mm = max(stage.final_grid_spacing_mm, 40.0)
                    stage.bending_energy_weight = max(stage.bending_energy_weight, 5.0)
                elif stage.final_grid_spacing_mm > profile.bspline_grid_mm:
                    log.info(
                        "B-spline grid tightened to %.0f mm (profile %s)",
                        profile.bspline_grid_mm,
                        profile.name,
                    )
                    stage.final_grid_spacing_mm = profile.bspline_grid_mm
        return stages

    def _convexadam(
        self,
        fixed_work: Volume,
        moving_work: Volume,
        applied_linear: AppliedTransform,
        outcome,
        mask_fixed: sitk.Image | None,
        manifest: RunManifest,
    ) -> tuple[AppliedTransform, dict[str, Any]]:
        """Deformable residual optimised on features, after the linear part."""
        from regix.features.anatomix import extract_feature_pair
        from regix.registration.convexadam import adam_instance_optimization

        linear = outcome.final_linear_transform or outcome.final_transform
        if linear is None:
            raise RegistrationFailure(
                "the feature-based deformable stage requires a convertible linear part: "
                "remove any B-spline stage from stages"
            )
        warped_moving = resample_like(
            moving_work.image, fixed_work.image, transform=linear, interpolator="linear"
        )
        pair = extract_feature_pair(
            fixed_work,
            moving_work.with_image(warped_moving),
            self.config.features,
            fixed_mask=mask_fixed,
            moving_mask=mask_fixed,  # same grid after linear registration
            provider="anatomix",
            seed=self.config.runtime.seed,
        )
        f_feat = np.stack([sitk.GetArrayFromImage(c) for c in pair.fixed_channels], axis=0)
        m_feat = np.stack([sitk.GetArrayFromImage(c) for c in pair.moving_channels], axis=0)
        mask_arr = sitk.GetArrayViewFromImage(mask_fixed).astype(np.uint8) if mask_fixed is not None else None
        deformable = adam_instance_optimization(
            f_feat,
            m_feat,
            reference=fixed_work.image,
            fixed_mask=mask_arr,
            device=self.config.features.device if self.config.features.device != "auto" else "auto",
        )
        # T_total(x) = T_linear(D(x)): D first, then the linear part.
        total = compose([deformable.transform, linear])
        manifest.warn(
            "deformable registration by instance optimisation: the result is not bit-exact "
            "reproducible (GPU, Adam). Keep the displacement field for traceability."
        )
        return SitkAppliedTransform(total, label="linear+adam_features"), {
            **deformable.info,
            "features": pair.info,
        }

    # ------------------------------------------------------------------ #
    def _quality_control(
        self,
        fixed: Volume,
        moving: Volume,
        registered: sitk.Image,
        applied: AppliedTransform,
        outcome,
        initial_transform: sitk.Transform | None,
        fixed_seg: OrganSegmentation | None,
        moving_seg: OrganSegmentation | None,
        qc_mask: sitk.Image | None,
        multimodal: bool,
        base_metrics: dict[str, Any],
        out_dir: Path,
        outputs: dict[str, Path],
        manifest: RunManifest,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        cfg = self.config
        metrics = dict(base_metrics)

        # -- initial state: the moving volume as stored, on the fixed grid ---- #
        moving_before = resample_like(
            moving.image,
            fixed.image,
            transform=None,
            interpolator="linear",
            default_value=_background_of(moving.image),
        )
        metrics["similarity"] = similarity_report(fixed.image, moving_before, registered, qc_mask)

        # -- per-organ overlap ------------------------------------------------ #
        organ_overlap: dict[str, dict[str, float]] = {}
        warped_labelmap = None
        if fixed_seg is not None and moving_seg is not None:
            try:
                warped_labelmap = applied.resample(
                    moving_seg.labelmap, fixed.image, is_label=True, default_value=0
                )
                names = {
                    label: name
                    for label, name in fixed_seg.label_names.items()
                    if moving_seg.label_of(name) == label
                }
                if not names:
                    manifest.warn(
                        "different segmentation nomenclatures between fixed and moving: Dice not computed"
                    )
                organ_overlap = organ_overlap_report(
                    fixed_seg.labelmap,
                    warped_labelmap,
                    names,
                    organs=cfg.organs.qc_labels or self.targets or None,
                )
                if cfg.organs.save_masks and warped_labelmap is not None:
                    outputs["warped_labelmap"] = save_image(
                        warped_labelmap,
                        out_dir / "masks" / "moving_labelmap_registered.nii.gz",
                        dtype=sitk.sitkUInt16,
                    )
            except Exception as exc:
                manifest.warn(f"per-organ Dice unavailable ({type(exc).__name__}: {exc})")
        metrics["organ_overlap"] = organ_overlap

        # -- displacement field, Jacobian ------------------------------------- #
        field = None
        jacobian: dict[str, Any] = {}
        if cfg.qc.jacobian or cfg.output.write_deformation_field or cfg.output.write_jacobian:
            field = applied.displacement_field(fixed.image)
            if field is not None:
                jacobian = jacobian_statistics(field, qc_mask)
                metrics["displacement"] = displacement_statistics(field, qc_mask)
                if cfg.output.write_deformation_field:
                    outputs["deformation_field"] = save_image(
                        field, out_dir / "deformation_field.nii.gz", dtype=sitk.sitkVectorFloat32
                    )
                if cfg.output.write_jacobian:
                    outputs["jacobian"] = save_image(
                        sitk.DisplacementFieldJacobianDeterminant(sitk.Cast(field, sitk.sitkVectorFloat64)),
                        out_dir / "jacobian.nii.gz",
                    )
            else:
                manifest.warn("displacement field unavailable: Jacobian not evaluated")
        metrics["jacobian"] = jacobian

        # -- landmarks -------------------------------------------------------- #
        landmarks: dict[str, Any] = {}
        if cfg.qc.landmarks_fixed and cfg.qc.landmarks_moving:
            try:
                pts_fixed = load_landmarks(cfg.qc.landmarks_fixed)
                pts_moving = load_landmarks(cfg.qc.landmarks_moving)
                mapper = applied.transform_points(pts_fixed)
                if mapper is None and field is not None:
                    mapper = transform_points_via_field(field, pts_fixed)
                if mapper is None:
                    manifest.warn("TRE not computable: the transform cannot be evaluated at a point")
                else:
                    landmarks = target_registration_error(pts_fixed, pts_moving, lambda _p: mapper)
                    log.info(
                        "mean TRE: %.2f mm -> %.2f mm (%d landmarks)",
                        landmarks["tre_before_mean_mm"],
                        landmarks["tre_mean_mm"],
                        landmarks["n_landmarks"],
                    )
            except Exception as exc:
                manifest.warn(f"landmarks unusable ({type(exc).__name__}: {exc})")
        metrics["landmarks"] = landmarks

        # -- global linear analysis -------------------------------------------- #
        # We compose the initialization AND all linear stages: that is the only way to
        # detect divergence, because an aberrant initialization makes the optimised part
        # small and therefore falsely reassuring. Any B-spline stage is ignored here --
        # it is judged by the Jacobian.
        linear_analysis = None
        linear_chain = [t for t in [initial_transform] if t is not None]
        linear_chain += [s.transform for s in outcome.stages if s.transform is not None]
        if linear_chain:
            matrix = to_matrix_4x4(compose(linear_chain))
            if matrix is not None:
                linear_analysis = decompose_affine(matrix)
                linear_analysis["includes_initialization"] = True
        if linear_analysis is None and outcome.stages:
            linear_analysis = next(
                (s.linear_analysis for s in reversed(outcome.stages) if s.linear_analysis), None
            )
        metrics["linear"] = linear_analysis

        # -- gates -------------------------------------------------------------- #
        gates = cfg.qc.gates
        if gates.min_ncc_gain is None and gates.min_nmi_gain is None:
            # Automatic choice: NMI for multimodal pairs, NCC otherwise.
            gates = gates.model_copy(update={"min_nmi_gain": 0.0} if multimodal else {"min_ncc_gain": 0.0})
        qc = evaluate_gates(
            gates,
            similarity=metrics.get("similarity"),
            organ_overlap=organ_overlap,
            jacobian=jacobian,
            linear_analysis=linear_analysis,
            landmarks=landmarks,
            deformable=outcome.is_deformable or applied.kind == "sitk",
            stages=[s.to_dict() | {"final_metric": s.final_metric} for s in outcome.stages],
        ).to_dict()

        if qc["status"] == "FAIL" and cfg.runtime.fail_fast:
            raise RegistrationFailure(
                "QC gates not met: " + "; ".join(c["name"] for c in qc["checks"] if c["status"] == "FAIL")
            )

        # -- figures ------------------------------------------------------------ #
        figures: dict[str, str] = {}
        if cfg.qc.report_html:
            try:
                figures["overlay"] = overlay_figure(
                    fixed.image, moving_before, registered, qc_mask, n_slices=cfg.qc.n_slices
                )
                figures["checkerboard"] = checkerboard_figure(fixed.image, registered, qc_mask)
                if fixed_seg is not None and warped_labelmap is not None:
                    contours = contour_figure(fixed.image, fixed_seg.labelmap, warped_labelmap, qc_mask)
                    if contours:
                        figures["contours"] = contours
                if field is not None and jacobian.get("available"):
                    # jacobian_figure returns None for a linear transform: det(J) is
                    # constant, so the map would be a flat colour that looks like a bug.
                    jac_fig = jacobian_figure(field, qc_mask, stats=jacobian)
                    if jac_fig:
                        figures["jacobian"] = jac_fig
            except Exception as exc:
                manifest.warn(f"QC figures not generated ({type(exc).__name__}: {exc})")

        return metrics, qc, figures

    # ------------------------------------------------------------------ #
    def _export_transforms(
        self,
        applied: AppliedTransform,
        outcome,
        fixed: Volume,
        moving: Volume,
        out_dir: Path,
        outputs: dict[str, Path],
        initial_transform: sitk.Transform | None = None,
    ) -> None:
        if not self.config.output.write_transform:
            return
        transform_dir = out_dir / "transform"
        transform_dir.mkdir(parents=True, exist_ok=True)

        # elastix chain: these files are the replayable reference.
        for index, stage in enumerate(outcome.stages):
            shutil.copy2(
                stage.transform_parameter_file,
                transform_dir / f"stage{index:02d}_{stage.name}_TransformParameters.txt",
            )
            params = stage.transform_parameter_file.parent / "parameters.txt"
            if params.exists():
                shutil.copy2(params, transform_dir / f"stage{index:02d}_{stage.name}_parameters.txt")
        outputs["transform_dir"] = transform_dir

        sitk_transform = applied.as_sitk_transform()
        if sitk_transform is not None:
            outputs["transform_itk"] = save_transform(sitk_transform, transform_dir / "final_transform.tfm")

        # --- Insight Transform File format (.txt): readable by 3D Slicer ------ #
        # One cumulative file per stage, which is how it is actually read: load
        # stage00_rigid.txt to judge the rigid stage alone, then stage01_affine.txt
        # to see what the affine added.
        if self.config.output.write_slicer_transform:
            written: list[str] = []
            cumulative = [t for t in [initial_transform] if t is not None]
            for index, stage in enumerate(outcome.stages):
                if stage.transform is None:
                    # Non-linear stage: an ITK .txt file cannot carry it on its own.
                    continue
                cumulative.append(stage.transform)
                try:
                    written.append(
                        self._write_slicer_transform(
                            compose(list(cumulative)),
                            transform_dir / f"stage{index:02d}_{stage.name}.txt",
                        ).name
                    )
                except Exception as exc:  # pragma: no cover
                    log.warning("could not export stage %s as .txt: %s", stage.name, exc)
            if sitk_transform is not None:
                try:
                    final_txt = self._write_slicer_transform(
                        sitk_transform, transform_dir / "final_transform.txt"
                    )
                    outputs["transform_slicer"] = final_txt
                    written.append(final_txt.name)
                except Exception as exc:
                    # A composite dense field is not representable as an ITK .txt file.
                    log.info(
                        "final_transform.txt not written (%s): use final_transform.tfm "
                        "or the displacement field",
                        exc,
                    )
            if written:
                log.info("Slicer-readable transforms: %s", ", ".join(written))

        if sitk_transform is not None:
            matrix = matrix_moving_to_fixed(sitk_transform)
            if matrix is not None:
                np.savetxt(
                    transform_dir / "moving_to_fixed_matrix.txt",
                    matrix,
                    fmt="%.10f",
                    header="4x4 homogeneous matrix: p_fixed = M @ p_moving (mm, patient frame)",
                )
                outputs["transform_matrix"] = transform_dir / "moving_to_fixed_matrix.txt"

                # DICOM registration object, when both inputs are DICOM series.
                if _is_dicom_dir(fixed.source) and _is_dicom_dir(moving.source):
                    try:
                        from regix.io.dicom import list_series

                        f_series = list_series(fixed.source)[0]
                        m_series = list_series(moving.source)[0]
                        outputs["dicom_sro"] = write_spatial_registration_dicom(
                            transform_dir / "spatial_registration.dcm",
                            matrix,
                            f_series.files,
                            m_series.files,
                            transformation_type=(
                                "RIGID" if abs(np.linalg.det(matrix[:3, :3]) - 1.0) < 1e-3 else "AFFINE"
                            ),
                        )
                    except Exception as exc:
                        log.warning("DICOM registration object not written: %s", exc)

    @staticmethod
    def _write_slicer_transform(transform: sitk.Transform, path: Path) -> Path:
        """Write an Insight Transform File, flattened to a single affine when possible.

        A linear chain is reduced to one ``AffineTransform``: numerically identical,
        but directly usable in a visualisation station instead of a multi-level
        CompositeTransform.
        """
        flattened = flatten_linear(transform)
        return save_transform(flattened if flattened is not None else transform, path)

    def _export_dicom(
        self,
        registered: sitk.Image,
        fixed: Volume,
        moving: Volume,
        applied: AppliedTransform,
        out_dir: Path,
        outputs: dict[str, Path],
        manifest: RunManifest,
    ) -> None:
        if not _is_dicom_dir(moving.source):
            manifest.warn("DICOM export requested but the moving image is not a DICOM series: skipped")
            return
        try:
            from regix.io.dicom import list_series

            m_series = list_series(moving.source)[0]
            for_uid = None
            if _is_dicom_dir(fixed.source):
                import pydicom

                f_series = list_series(fixed.source)[0]
                ds = pydicom.dcmread(str(f_series.files[0]), stop_before_pixels=True, force=True)
                for_uid = getattr(ds, "FrameOfReferenceUID", None)
            outputs["dicom_series"] = write_derived_dicom(
                registered,
                out_dir / "dicom_registered",
                m_series.files,
                frame_of_reference_uid=for_uid,
            )
        except Exception as exc:
            manifest.warn(f"DICOM export failed ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
def _describe_grid(volume: Volume) -> str:
    """One-line geometry summary, as shown in the report."""
    spacing = tuple(round(v, 2) for v in volume.spacing)
    return f"{volume.modality} {volume.size} @ {spacing} mm"


def _is_identity(transform: sitk.Transform, tolerance: float = 1e-6) -> bool:
    probes = [(0.0, 0.0, 0.0), (100.0, -50.0, 37.0), (-13.0, 71.0, -29.0)]
    for p in probes:
        if float(np.linalg.norm(np.asarray(transform.TransformPoint(list(p))) - np.asarray(p))) > tolerance:
            return False
    return True


def _background_of(image: sitk.Image) -> float:
    f = sitk.MinimumMaximumImageFilter()
    f.Execute(sitk.Cast(image, sitk.sitkFloat32))
    return float(f.GetMinimum())


def _intensity_range(image: sitk.Image) -> tuple[float, float]:
    """(min, max) of the working image, for the parameter-file quantisation check."""
    f = sitk.MinimumMaximumImageFilter()
    f.Execute(sitk.Cast(image, sitk.sitkFloat32))
    return float(f.GetMinimum()), float(f.GetMaximum())


def _is_dicom_dir(path: Path | None) -> bool:
    return path is not None and Path(path).is_dir()


# --------------------------------------------------------------------------- #
def register(
    fixed: str | Path | Volume,
    moving: str | Path | Volume,
    config: RegistrationConfig | None = None,
    output_dir: str | Path | None = None,
) -> RegistrationResult:
    """Functional shortcut: ``regix.pipeline.register(fixed, moving, config)``."""
    return RegistrationPipeline(config or RegistrationConfig()).run(fixed, moving, output_dir)
