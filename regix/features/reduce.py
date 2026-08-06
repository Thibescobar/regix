"""Normalisation and dimensionality reduction of feature maps.

Why a PCA with a *shared* basis? elastix compares channel *i* of the fixed image
against channel *i* of the moving image. If each volume were projected into its
own basis the channels would no longer be homologous and the criterion would
become noise. We therefore estimate a single basis on a sample of both volumes,
then project both into it.

Why reduce at all? The cost of elastix is linear in the number of channels. Out
of anatomix's 16 channels, 4 components typically retain more than 90 % of the
variance: a 4x speed-up for a negligible loss of accuracy.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("features.reduce")


def voxel_normalize(features: np.ndarray, mode: str = "l2", eps: float = 1e-6) -> np.ndarray:
    """Normalise each voxel across channels (required by the anatomix-dev variants)."""
    if mode == "none":
        return features
    f = features.astype(np.float32, copy=False)
    if mode == "l2":
        norm = np.sqrt((f * f).sum(axis=0, keepdims=True)) + eps
        return f / norm
    if mode == "zscore":
        mean = f.mean(axis=0, keepdims=True)
        std = f.std(axis=0, keepdims=True) + eps
        return (f - mean) / std
    raise ValueError(f"unknown normalisation mode: {mode}")


def joint_pca_reduce(
    fixed_features: np.ndarray,
    moving_features: np.ndarray,
    n_components: int = 4,
    max_voxels: int = 200_000,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project both feature sets into a shared PCA basis.

    The arrays are ``(C, Z, Y, X)``; the masks, when provided, restrict the basis
    estimation to the useful voxels (otherwise the empty space around the patient
    dominates the variance entirely).
    """
    if fixed_features.ndim != 4 or moving_features.ndim != 4:
        raise ValueError("features must be shaped (C, Z, Y, X)")
    c_fixed, c_moving = fixed_features.shape[0], moving_features.shape[0]
    if c_fixed != c_moving:
        raise ValueError(f"inconsistent channel count: {c_fixed} vs {c_moving}")
    n_components = int(min(n_components, c_fixed))
    if n_components >= c_fixed:
        log.debug("PCA unnecessary (%d components for %d channels)", n_components, c_fixed)
        return fixed_features, moving_features, {"applied": False, "n_components": c_fixed}

    rng = np.random.default_rng(seed)
    samples = [
        _sample_voxels(fixed_features, fixed_mask, max_voxels // 2, rng),
        _sample_voxels(moving_features, moving_mask, max_voxels // 2, rng),
    ]
    data = np.concatenate(samples, axis=0)  # (N, C)
    mean = data.mean(axis=0, keepdims=True)
    centred = data - mean
    # Economy SVD: C is small (<= 32), the cost is dominated by N.
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:n_components]  # (k, C)
    variance = singular**2
    explained = float(variance[:n_components].sum() / max(variance.sum(), 1e-12))

    def _project(features: np.ndarray) -> np.ndarray:
        flat = features.reshape(features.shape[0], -1).T  # (V, C)
        proj = (flat - mean) @ basis.T  # (V, k)
        return proj.T.reshape((n_components,) + features.shape[1:]).astype(np.float32)

    info = {
        "applied": True,
        "n_components": n_components,
        "input_channels": c_fixed,
        "explained_variance_ratio": round(explained, 4),
        "sample_voxels": int(data.shape[0]),
    }
    log.info(
        "shared PCA: %d -> %d channels (%.1f %% of the variance retained)",
        c_fixed,
        n_components,
        100 * explained,
    )
    if explained < 0.7:
        log.warning(
            "only %.0f %% of the feature variance is retained: the channels are weakly "
            "correlated with each other (typical of MIND-SSC). Increase "
            "features.n_components (6 to 8) if the registration lacks accuracy.",
            100 * explained,
        )
    return _project(fixed_features), _project(moving_features), info


def _sample_voxels(
    features: np.ndarray, mask: np.ndarray | None, n: int, rng: np.random.Generator
) -> np.ndarray:
    flat = features.reshape(features.shape[0], -1).T
    if mask is not None:
        keep = np.flatnonzero(mask.reshape(-1) > 0)
        if keep.size >= max(1024, n // 10):
            flat = flat[keep]
        else:
            log.debug("mask too small (%d voxels): sampling over the whole volume", keep.size)
    if flat.shape[0] > n:
        flat = flat[rng.choice(flat.shape[0], size=n, replace=False)]
    return flat.astype(np.float64, copy=False)


def features_to_sitk(features: np.ndarray, reference: sitk.Image, scale: float = 1.0) -> list[sitk.Image]:
    """Convert ``(C, Z, Y, X)`` into a list of scalar images sharing the geometry of ``reference``.

    This is the form in which elastix consumes the channels
    (``AddFixedImage`` / ``AddMovingImage`` plus a multi-metric setup).
    """
    if features.shape[1:] != tuple(sitk.GetArrayViewFromImage(reference).shape):
        raise ValueError(
            f"features {features.shape[1:]} incompatible with the reference "
            f"{sitk.GetArrayViewFromImage(reference).shape}"
        )
    channels: list[sitk.Image] = []
    for c in range(features.shape[0]):
        img = sitk.GetImageFromArray(np.ascontiguousarray(features[c] * scale, dtype=np.float32))
        img.CopyInformation(reference)
        channels.append(img)
    return channels
