"""Registration initialization: half the work.

A registration optimiser is local. On a whole-body CT against an abdominal MR it
is not the criterion that fails, it is the starting point. Hence several
strategies, from the most neutral to the most informed:

* ``identity``       : both volumes already share a frame (same study, same
                       Frame of Reference);
* ``geometry``       : align the grid centres. Sensible default;
* ``moments``        : align the intensity centres of mass. Good when the fields
                       of view are comparable, bad when one covers the whole body
                       and the other a single organ;
* ``organ_centroid`` : align the centroids of a segmented organ. This is the
                       right answer to differing fields of view;
* ``organ_moments``  : also align the principal axes and the scale of the organ.
                       Useful for inter-patient work or large posture differences;
* ``multistart``     : evaluate several candidates (crossed with probe rotations)
                       and keep the best according to a metric computed
                       independently of the optimiser.

All produced transforms follow the elastix convention: they map the **fixed**
frame to the **moving** frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.config import InitConfig, InitMode
from regix.io.volume import Volume
from regix.logging_utils import get_logger
from regix.organs.roi import organ_centroids
from regix.organs.segmenter import OrganSegmentation
from regix.preprocess.geometry import (
    center_of_mass_physical,
    image_center_physical,
    principal_axes,
    resample_like,
    resample_to_spacing,
)
from regix.qc.metrics import normalized_cross_correlation, normalized_mutual_information
from regix.registration.transforms import decompose_affine, to_matrix_4x4

log = get_logger("registration.init")


@dataclass
class InitCandidate:
    """A candidate starting point, with its provenance and score."""

    name: str
    transform: sitk.Transform
    score: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        matrix = to_matrix_4x4(self.transform)
        out: dict[str, Any] = {"name": self.name, "score": self.score, **self.info}
        if matrix is not None:
            decomposition = decompose_affine(matrix)
            out["translation_mm"] = decomposition["translation_norm_mm"]
            out["rotation_deg"] = decomposition["rotation_norm_deg"]
        return out


# --------------------------------------------------------------------------- #
# Elementary constructions
# --------------------------------------------------------------------------- #
def _euler(
    center: np.ndarray, translation: np.ndarray, rotation_rad: Sequence[float] = (0, 0, 0)
) -> sitk.Euler3DTransform:
    t = sitk.Euler3DTransform()
    t.SetCenter([float(v) for v in center])
    t.SetComputeZYX(False)
    t.SetParameters(
        [float(rotation_rad[0]), float(rotation_rad[1]), float(rotation_rad[2])]
        + [float(v) for v in translation]
    )
    return t


def identity_init() -> sitk.Transform:
    return sitk.Euler3DTransform()


def geometry_init(fixed: Volume, moving: Volume) -> sitk.Euler3DTransform:
    """Align the centres of the two grids."""
    c_f = image_center_physical(fixed.image)
    c_m = image_center_physical(moving.image)
    return _euler(c_f, c_m - c_f)


def moments_init(
    fixed: Volume,
    moving: Volume,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
) -> sitk.Euler3DTransform:
    """Align the intensity centres of mass (inside the masks when provided)."""
    c_f = center_of_mass_physical(fixed.image, fixed_mask)
    c_m = center_of_mass_physical(moving.image, moving_mask)
    return _euler(c_f, c_m - c_f)


def organ_centroid_init(
    fixed_seg: OrganSegmentation,
    moving_seg: OrganSegmentation,
    targets: Sequence[str],
) -> tuple[sitk.Euler3DTransform, dict[str, Any]]:
    """Align the centroids of the common organs (mean over the organs found)."""
    f_cent = organ_centroids(fixed_seg, targets)
    m_cent = organ_centroids(moving_seg, targets)
    common = [o for o in f_cent if o in m_cent]
    if not common:
        raise ValueError(
            f"no organ in common among {list(targets)} "
            f"(fixed: {sorted(f_cent)}, moving: {sorted(m_cent)})"
        )
    c_f = np.mean([f_cent[o] for o in common], axis=0)
    c_m = np.mean([m_cent[o] for o in common], axis=0)
    info = {
        "organs_used": common,
        "fixed_centroid": [round(float(v), 2) for v in c_f],
        "moving_centroid": [round(float(v), 2) for v in c_m],
        "offset_mm": round(float(np.linalg.norm(c_m - c_f)), 2),
    }
    log.info("centroid initialization on %s: %.1f mm offset", common, info["offset_mm"])
    return _euler(c_f, c_m - c_f), info


def organ_moments_init(
    fixed_seg: OrganSegmentation,
    moving_seg: OrganSegmentation,
    targets: Sequence[str],
    with_scale: bool = True,
) -> tuple[sitk.AffineTransform, dict[str, Any]]:
    """Align the centroid, principal axes and scale of an organ.

    Eigenvectors have an arbitrary sign: we reorient them to maximise agreement
    with the fixed ones, then force a positive determinant -- otherwise the
    initialization introduces a mirror flip, which is anatomically absurd and
    makes everything downstream diverge.
    """
    wanted = [o for o in targets] or fixed_seg.organs
    chosen = None
    for organ in wanted:
        if fixed_seg.label_of(organ) is not None and moving_seg.label_of(organ) is not None:
            chosen = organ
            break
    if chosen is None:
        raise ValueError(f"no organ in common among {list(wanted)} for moment alignment")

    c_f, V_f, len_f = principal_axes(fixed_seg.mask_for([chosen]))
    c_m, V_m, len_m = principal_axes(moving_seg.mask_for([chosen]))

    for axis in range(3):
        if float(np.dot(V_f[:, axis], V_m[:, axis])) < 0:
            V_m[:, axis] *= -1.0
    R = V_m @ V_f.T
    if np.linalg.det(R) < 0:
        V_m[:, 2] *= -1.0
        R = V_m @ V_f.T

    scale = 1.0
    if with_scale:
        ratio = np.prod(np.maximum(len_m, 1e-6)) / np.prod(np.maximum(len_f, 1e-6))
        scale = float(np.clip(ratio ** (1.0 / 3.0), 0.7, 1.4))  # guard: no absurd factor
        R = R * scale

    t = sitk.AffineTransform(3)
    t.SetCenter([float(v) for v in c_f])
    t.SetMatrix([float(v) for v in R.reshape(-1)])
    t.SetTranslation([float(v) for v in (c_m - c_f)])
    info = {
        "organ_used": chosen,
        "scale": round(scale, 4),
        "offset_mm": round(float(np.linalg.norm(c_m - c_f)), 2),
        "fixed_axis_lengths_mm": [round(float(v), 2) for v in len_f],
        "moving_axis_lengths_mm": [round(float(v), 2) for v in len_m],
    }
    log.info(
        "moment initialization on %s: %.1f mm offset, scale %.3f",
        chosen,
        info["offset_mm"],
        scale,
    )
    return t, info


def with_extra_rotation(
    base: sitk.Transform, center: np.ndarray, rotation_deg: Sequence[float]
) -> sitk.Transform:
    """Compose a probe rotation (about ``center``) with a base transform.

    Order matters, and getting it wrong is not obvious. The rotation is applied
    **first**, in the fixed frame, about ``center`` (the fixed image centre); the base
    alignment follows. Composing the other way round would rotate in the *moving*
    frame about a *fixed*-frame point: a 15 deg probe rotation combined with a 200 mm
    base translation would then displace points by ~50 mm instead of the intended
    amplitude, and the multi-start candidates would be scattered far wider than asked.
    """
    if all(abs(a) < 1e-9 for a in rotation_deg):
        return base
    rot = _euler(center, np.zeros(3), [np.radians(a) for a in rotation_deg])
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(base)  # applied second (CompositeTransform convention)
    composite.AddTransform(rot)   # applied first, in the fixed frame
    return composite


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #
def score_candidate(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    mask: sitk.Image | None = None,
    metric: str = "auto",
) -> float:
    """Score a starting point, computed on coarse versions of the images.

    Deliberately independent of elastix: we want to rank the starting points, not
    reproduce the criterion that will be optimised afterwards.
    """
    warped = resample_like(moving, fixed, transform=transform, interpolator="linear")
    if metric == "ncc":
        return normalized_cross_correlation(fixed, warped, mask)
    if metric == "nmi":
        return normalized_mutual_information(fixed, warped, mask, bins=48)
    ncc = normalized_cross_correlation(fixed, warped, mask)
    nmi = normalized_mutual_information(fixed, warped, mask, bins=48)
    # NMI recentred on 0 so it is comparable with an NCC.
    parts = [v for v in (ncc, (nmi - 1.0) if np.isfinite(nmi) else np.nan) if np.isfinite(v)]
    return float(np.mean(parts)) if parts else float("-inf")


def _coarse(image: sitk.Image, spacing_mm: float, is_mask: bool = False) -> sitk.Image:
    return resample_to_spacing(image, spacing_mm, is_mask=is_mask)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_candidates(
    fixed: Volume,
    moving: Volume,
    config: InitConfig,
    fixed_segmentation: OrganSegmentation | None = None,
    moving_segmentation: OrganSegmentation | None = None,
    targets: Sequence[str] = (),
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
) -> list[InitCandidate]:
    """Build the list of starting points to evaluate."""
    modes = list(config.candidates) if config.mode is InitMode.MULTISTART else [config.mode]
    center = image_center_physical(fixed.image)
    candidates: list[InitCandidate] = []

    for mode in modes:
        try:
            if mode is InitMode.IDENTITY:
                candidates.append(InitCandidate("identity", identity_init()))
            elif mode is InitMode.GEOMETRY:
                candidates.append(InitCandidate("geometry", geometry_init(fixed, moving)))
            elif mode is InitMode.MOMENTS:
                candidates.append(
                    InitCandidate("moments", moments_init(fixed, moving, fixed_mask, moving_mask))
                )
            elif mode is InitMode.ORGAN_CENTROID:
                if fixed_segmentation is None or moving_segmentation is None:
                    raise ValueError("segmentations are required for organ_centroid")
                t, info = organ_centroid_init(fixed_segmentation, moving_segmentation, targets)
                candidates.append(InitCandidate("organ_centroid", t, info=info))
            elif mode is InitMode.ORGAN_MOMENTS:
                if fixed_segmentation is None or moving_segmentation is None:
                    raise ValueError("segmentations are required for organ_moments")
                t, info = organ_moments_init(fixed_segmentation, moving_segmentation, targets)
                candidates.append(InitCandidate("organ_moments", t, info=info))
            elif mode is InitMode.FILE:
                if config.transform_file is None:
                    raise ValueError("init.mode=file without transform_file")
                t = sitk.ReadTransform(str(config.transform_file))
                candidates.append(InitCandidate("file", t, info={"path": str(config.transform_file)}))
            else:  # pragma: no cover
                raise ValueError(f"unhandled initialization mode: {mode}")
        except Exception as exc:
            log.warning("candidate '%s' discarded: %s", mode.value, exc)

    if not candidates:
        log.warning("no candidate could be built: falling back to grid-centre alignment")
        candidates.append(InitCandidate("geometry", geometry_init(fixed, moving)))

    if config.mode is InitMode.MULTISTART:
        rotated: list[InitCandidate] = []
        for cand in candidates:
            for angles in config.multistart_rotations_deg:
                if all(abs(a) < 1e-9 for a in angles):
                    rotated.append(cand)
                    continue
                label = f"{cand.name}+rot({','.join(str(int(a)) for a in angles)})"
                rotated.append(
                    InitCandidate(
                        label,
                        with_extra_rotation(cand.transform, center, angles),
                        info={**cand.info, "extra_rotation_deg": list(angles)},
                    )
                )
        if config.flip_check:
            for cand in list(rotated):
                label = f"{cand.name}+flip180z"
                rotated.append(
                    InitCandidate(
                        label,
                        with_extra_rotation(cand.transform, center, (0, 0, 180)),
                        info={**cand.info, "flip": "180 deg about z"},
                    )
                )
        candidates = rotated

    return candidates


def choose_initialization(
    fixed: Volume,
    moving: Volume,
    config: InitConfig,
    fixed_segmentation: OrganSegmentation | None = None,
    moving_segmentation: OrganSegmentation | None = None,
    targets: Sequence[str] = (),
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    scoring_spacing_mm: float = 6.0,
) -> tuple[InitCandidate, dict[str, Any]]:
    """Select the best starting point.

    Ranking is done on heavily downsampled images (6 mm by default): that is
    enough to tell a viable start from an absurd one, and it costs a few hundred
    milliseconds per candidate.
    """
    candidates = build_candidates(
        fixed, moving, config, fixed_segmentation, moving_segmentation, targets, fixed_mask, moving_mask
    )
    report: dict[str, Any] = {"mode": config.mode.value, "n_candidates": len(candidates)}

    if len(candidates) == 1:
        chosen = candidates[0]
        report["candidates"] = [chosen.summary()]
        report["chosen"] = chosen.name
        return chosen, report

    coarse_fixed = _coarse(fixed.image, scoring_spacing_mm)
    coarse_moving = _coarse(moving.image, scoring_spacing_mm)
    coarse_mask = (
        resample_like(fixed_mask, coarse_fixed, is_mask=True) if fixed_mask is not None else None
    )

    for cand in candidates:
        try:
            cand.score = score_candidate(coarse_fixed, coarse_moving, cand.transform, coarse_mask)
        except Exception as exc:
            log.warning("could not score '%s': %s", cand.name, exc)
            cand.score = float("-inf")
        log.debug("candidate %-32s score=%.4f", cand.name, cand.score)

    ranked = sorted(
        candidates, key=lambda c: (c.score if c.score is not None else float("-inf")), reverse=True
    )
    chosen = ranked[0]
    report["candidates"] = [c.summary() for c in ranked]
    report["chosen"] = chosen.name
    report["score_spread"] = (
        round(float(ranked[0].score - ranked[-1].score), 4)
        if ranked[0].score is not None and ranked[-1].score is not None and np.isfinite(ranked[-1].score)
        else None
    )
    log.info(
        "initialization selected: %s (score %.4f out of %d candidates)",
        chosen.name,
        chosen.score if chosen.score is not None else float("nan"),
        len(candidates),
    )
    if len(ranked) > 1 and ranked[0].score is not None and ranked[1].score is not None:
        if abs(ranked[0].score - ranked[1].score) < 0.01:
            log.warning(
                "the two best starting points are tied (%.4f vs %.4f): the result is "
                "sensitive to initialization, verify visually",
                ranked[0].score,
                ranked[1].score,
            )
            report["ambiguous"] = True
    return chosen, report
