"""MIND-SSC: an analytical multimodal descriptor, no network and no GPU.

Reference: Heinrich et al., *Towards Realtime Multimodal Fusion for Image-Guided
Interventions Using Self-Similarities*, MICCAI 2013.

The idea: describe each voxel not by its intensity (which means nothing across
modalities) but by the *self-similarity pattern* of its neighbourhood, which
depends on the anatomical structure rather than on contrast. A CT and an MR of
the same liver have similar MIND descriptors; their intensities have nothing in
common.

This is the fallback when torch/anatomix are not installed or when there is no
GPU. Less powerful than anatomix, but deterministic, with no weights to version,
and available everywhere.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("features.mind")

# 6-neighbourhood (in voxels, before dilation).
_NEIGHBOURS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=int
)
# 12 pairs of non-collinear neighbours: the 12 channels of the SSC descriptor.
_PAIRS = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
]


def mind_ssc_features(
    volume: np.ndarray,
    radius: int = 2,
    dilation: int = 2,
    spacing: tuple[float, float, float] | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the 12 MIND-SSC channels of a 3D volume.

    ``volume`` is ``(Z, Y, X)`` and must already be normalised (typically into
    [0, 1]). ``spacing`` (z, y, x) lets the dilation adapt to voxel anisotropy,
    without which the descriptor describes a squashed anatomy.

    Returns ``(12, Z, Y, X)`` float32, each voxel normalised by its maximum across
    channels (as in the reference implementation).
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {volume.shape}")
    vol = np.ascontiguousarray(volume, dtype=np.float32)

    if spacing is not None:
        smallest = float(min(spacing))
        steps = tuple(max(1, int(round(dilation * smallest / s))) for s in spacing)
    else:
        steps = (dilation, dilation, dilation)
    log.debug("MIND-SSC: radius=%d, per-axis step=%s", radius, steps)

    shifted = np.stack([_shift(vol, offset * np.asarray(steps)) for offset in _NEIGHBOURS], axis=0)

    distances = np.empty((len(_PAIRS),) + vol.shape, dtype=np.float32)
    for k, (a, b) in enumerate(_PAIRS):
        diff = shifted[a] - shifted[b]
        distances[k] = _box_mean(diff * diff, radius)

    variance = distances.mean(axis=0, keepdims=True)
    # Bound the variance: avoids exp(-large) = 0 everywhere inside air.
    median = float(np.median(variance[variance > 0])) if np.any(variance > 0) else 1.0
    variance = np.clip(variance, 1e-3 * median, 1e3 * median) + eps

    mind = np.exp(-distances / variance)
    mind /= mind.max(axis=0, keepdims=True) + eps
    return mind.astype(np.float32)


def _box_mean(volume: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a ``(2*radius+1)**3`` box, edge-clamped at the borders.

    Stands in for ``scipy.ndimage.uniform_filter(..., mode="nearest")``, which was
    the single reason scipy was a dependency of Regix at all. ITK's
    ``MeanImageFilter`` computes the same box mean with the same boundary handling:
    its ZeroFluxNeumann condition replicates the edge voxel, which is exactly what
    ``mode="nearest"`` means. Verified to float32 epsilon by
    ``tests/test_units.py::test_box_mean_matches_a_separable_reference``.

    It costs roughly 3x the scipy call (~120 ms against ~38 ms on a 180x180x166
    volume) because ``MeanImageFilter`` walks the full 5x5x5 neighbourhood instead
    of being separable. That is ~2 s per registration, on the MIND fallback path
    only, and it buys one fewer dependency to pin. Two alternatives were measured
    and rejected: ``sitk.BoxMean`` normalises by the in-bounds voxel count instead
    of replicating the edge (20 % disagreement, and no faster), and a separable
    cumsum in numpy is bit-exact but 2.5x slower still.
    """
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume, dtype=np.float32))
    return sitk.GetArrayFromImage(sitk.Mean(image, [int(radius)] * volume.ndim))


def _shift(volume: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Integer translation with edge extension (no circular wrap-around)."""
    out = volume
    for axis, delta in enumerate(int(d) for d in offset):
        if delta == 0:
            continue
        out = np.roll(out, delta, axis=axis)
        # replace the wrapped region with the first/last valid slice
        idx: list[slice | int] = [slice(None)] * out.ndim
        if delta > 0:
            idx[axis] = slice(0, delta)
            edge = [slice(None)] * out.ndim
            edge[axis] = slice(delta, delta + 1)
        else:
            idx[axis] = slice(delta, None)
            edge = [slice(None)] * out.ndim
            edge[axis] = slice(delta - 1, delta)
        out = out.copy()
        out[tuple(idx)] = out[tuple(edge)]
    return out
