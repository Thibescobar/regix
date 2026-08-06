"""Intensity preparation, per modality.

The rules below come from practice, not aesthetics:

* CT: Hounsfield units are quantitative -> use fixed window bounds, never
  percentiles, otherwise two CTs of the same patient are no longer comparable.
* MR: arbitrary intensities -> robust percentiles (0.5 / 99.5) then min-max;
  N4 optionally, when the B1 field is visible.
* PET / NM: huge dynamic range with a long tail -> upper percentiles only.
* Clipping is applied **before** any normalisation, and the bounds used are kept
  in the metadata for traceability.

Order of operations, and why it is that order: N4 -> clipping -> denoising ->
normalisation. N4 comes **first**, on the native intensities, because that is where
a bias field is defined: it is a smooth multiplicative factor on the acquired
signal, and estimating it after an intensity window has already truncated the
signal means estimating it on a distorted version of the thing being modelled.
Percentile bounds are therefore measured on the *corrected* image -- measuring them
before N4 and applying them after would shift them by whatever the correction did.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from regix.config import ImagePrep
from regix.io.volume import Volume
from regix.logging_utils import get_logger

log = get_logger("preprocess.intensity")

#: Common HU windows (width/centre converted to bounds).
HU_WINDOWS: dict[str, tuple[float, float]] = {
    "ct_full": (-1024.0, 3071.0),
    "ct_soft": (-160.0, 240.0),
    "ct_abdomen": (-150.0, 250.0),
    "ct_liver": (-30.0, 180.0),
    "ct_lung": (-1000.0, 200.0),
    "ct_mediastinum": (-125.0, 225.0),
    "ct_bone": (-200.0, 1000.0),
    "ct_brain": (0.0, 80.0),
    "ct_registration": (-450.0, 450.0),  # CT bounds used in the anatomix paper
}

#: Suggested default preparation per modality. Every entry states ``percentile_clip``
#: explicitly, so that resolving ``"auto"`` can never yield ``"auto"`` again.
DEFAULT_PREP_BY_MODALITY: dict[str, dict] = {
    "CT": {"window": "ct_registration", "percentile_clip": None, "normalize": "minmax"},
    "CBCT": {"window": "ct_registration", "percentile_clip": None, "normalize": "minmax"},
    "MR": {"percentile_clip": (0.5, 99.5), "normalize": "minmax", "n4_bias_correction": False},
    "PT": {"percentile_clip": (0.0, 99.5), "normalize": "minmax"},
    "NM": {"percentile_clip": (0.0, 99.5), "normalize": "minmax"},
    "US": {"percentile_clip": (1.0, 99.0), "normalize": "minmax"},
}

#: Fallback for a modality that is not in the table (typically UNKNOWN): robust
#: percentiles, which assume nothing about the intensity scale.
_FALLBACK_PREP: dict = {"percentile_clip": (0.5, 99.5), "normalize": "minmax"}


def default_prep_for(modality: str | None) -> ImagePrep:
    """Sensible preparation for a given modality."""
    return ImagePrep(**DEFAULT_PREP_BY_MODALITY.get((modality or "").upper(), _FALLBACK_PREP))


def resolve_prep(prep: ImagePrep, modality: str | None, side: str = "") -> ImagePrep:
    """Turn ``percentile_clip="auto"`` into the concrete clipping for this modality.

    Why a sentinel value rather than "the field still holds its default": the two are
    not the same question, and the difference is what used to make this wrong. The
    detection was ``percentile_clip == (0.5, 99.5)``, the default itself, so an explicit
    request for robust percentiles was indistinguishable from silence and got replaced
    by the HU window. On a CT those must not lead to the same place -- Hounsfield units
    are quantitative, so silence has to become a fixed window, while an explicit pair is
    exactly what a CBCT with an offset HU scale needs.

    Pydantic's ``model_fields_set`` looks like the answer and is not: ``with_overrides``
    round-trips the configuration through ``model_dump`` / ``model_validate`` (the CLI,
    the API and the test helpers all go through it), which marks every field as set. An
    explicit ``"auto"`` survives that round-trip, survives ``to_yaml``, and reads
    correctly in ``config_effective.yaml``.

    Idempotent: the returned preparation no longer holds the sentinel.
    """
    if prep.percentile_clip != "auto":
        return prep
    if prep.clip is not None or prep.window is not None:
        # An explicit window or explicit bounds win over percentiles anyway
        # (see resolve_clip_bounds): just discharge the sentinel.
        return prep.model_copy(update={"percentile_clip": None})

    suggested = default_prep_for(modality)
    resolved = prep.model_copy(
        update={"window": suggested.window, "percentile_clip": suggested.percentile_clip}
    )
    log.debug(
        "%s: clipping left to auto, applying the %s default -> %s",
        side or "volume",
        (modality or "unknown").upper(),
        resolved.window or f"percentiles {resolved.percentile_clip}",
    )
    return resolved


def resolve_clip_bounds(
    volume: Volume, prep: ImagePrep, source: sitk.Image | None = None
) -> tuple[float, float] | None:
    """Effective clipping bounds, in priority order clip > window > percentiles.

    ``source`` is the image the percentiles are measured on, defaulting to
    ``volume.image``. ``apply_intensity_prep`` passes the N4-corrected image: bias
    correction changes the intensity distribution, so percentiles taken before it and
    applied after it are not the percentiles of the data being clipped.
    """
    if prep.percentile_clip == "auto":
        # Reaching here means resolve_prep() was skipped; the modality is needed to
        # decide, and guessing it silently is how the CT window went missing before.
        raise ValueError(
            "percentile_clip is still 'auto': call resolve_prep(prep, modality) before "
            "applying an intensity preparation"
        )
    if prep.clip is not None:
        return float(prep.clip[0]), float(prep.clip[1])
    if prep.window is not None:
        key = prep.window.lower()
        if key not in HU_WINDOWS:
            raise ValueError(f"unknown window '{prep.window}'. Available: {sorted(HU_WINDOWS)}")
        if volume.modality not in ("CT", "CBCT", "UNKNOWN"):
            log.warning(
                "HU window '%s' applied to a %s volume: the intensities are not Hounsfield units",
                prep.window,
                volume.modality,
            )
        return HU_WINDOWS[key]
    if prep.percentile_clip is not None:
        measured = volume.image if source is None else source
        arr = sitk.GetArrayFromImage(sitk.Cast(measured, sitk.sitkFloat32))
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, list(prep.percentile_clip))
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            return None
        return float(lo), float(hi)
    return None


def apply_intensity_prep(volume: Volume, prep: ImagePrep) -> Volume:
    """Apply N4 -> clipping -> smoothing -> normalisation. Returns a new ``Volume``.

    See the module docstring for why N4 comes first.
    """
    image = sitk.Cast(volume.image, sitk.sitkFloat32)
    applied: dict[str, object] = {}

    # NaN/Inf: elastix registrations diverge silently on them.
    arr = sitk.GetArrayViewFromImage(image)
    n_bad = int(np.count_nonzero(~np.isfinite(arr)))
    if n_bad:
        log.warning("%d non-finite voxels replaced with the finite minimum", n_bad)
        clean = np.nan_to_num(
            sitk.GetArrayFromImage(image),
            nan=float(np.nanmin(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0,
            posinf=float(np.nanmax(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0,
            neginf=float(np.nanmin(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0,
        )
        fixed = sitk.GetImageFromArray(clean)
        fixed.CopyInformation(image)
        image = fixed
        applied["nonfinite_voxels"] = n_bad

    # N4 first, on the native intensities: a bias field is defined on the acquired
    # signal, not on a windowed version of it.
    if prep.n4_bias_correction:
        image = n4_bias_correction(image)
        applied["n4"] = True

    # ... and the percentiles are therefore measured on the corrected image.
    bounds = resolve_clip_bounds(volume, prep, source=image)
    if bounds is not None:
        image = sitk.Clamp(image, sitk.sitkFloat32, bounds[0], bounds[1])
        applied["clip"] = [round(bounds[0], 4), round(bounds[1], 4)]

    if prep.denoise_sigma_mm:
        image = sitk.SmoothingRecursiveGaussian(image, float(prep.denoise_sigma_mm))
        applied["denoise_sigma_mm"] = prep.denoise_sigma_mm

    if prep.normalize == "minmax":
        image = _minmax(image)
        applied["normalize"] = "minmax"
    elif prep.normalize == "zscore":
        image = _zscore(image)
        applied["normalize"] = "zscore"

    out = volume.with_image(image)
    out.meta = {**volume.meta, "intensity_prep": applied}
    return out


def _minmax(image: sitk.Image) -> sitk.Image:
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    lo, hi = stats.GetMinimum(), stats.GetMaximum()
    if hi - lo < 1e-8:
        log.warning("volume with zero dynamic range: min-max normalisation skipped")
        return image
    return sitk.ShiftScale(image, shift=-lo, scale=1.0 / (hi - lo))


def _zscore(image: sitk.Image) -> sitk.Image:
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    sigma = stats.GetSigma()
    if sigma < 1e-8:
        return image
    return sitk.ShiftScale(image, shift=-stats.GetMean(), scale=1.0 / sigma)


def n4_bias_correction(
    image: sitk.Image,
    shrink_factor: int = 4,
    iterations: tuple[int, ...] = (50, 40, 30),
    mask: sitk.Image | None = None,
) -> sitk.Image:
    """N4 bias field correction (MR). Estimated at low resolution, applied at full resolution."""
    img = sitk.Cast(image, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    small = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    small_mask = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(img)
    corrected = sitk.Divide(img, sitk.Exp(log_bias))
    log.debug("N4 applied (shrink=%d, iterations=%s)", shrink_factor, iterations)
    return corrected


def normalize_for_features(image: sitk.Image, clip: tuple[float, float] | None = None) -> np.ndarray:
    """Normalisation expected by anatomix: optional clipping then min-max into [0, 1].

    Returns a float32 numpy array in reversed ITK order (z, y, x).
    """
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        arr = np.clip(arr, clip[0], clip[1])
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
