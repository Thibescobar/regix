"""Deformable registration by instance optimisation on features (GPU).

Inspired by ConvexAdam (Siebert et al.) and by the registration path of the
anatomix repository: instead of optimising a multimodal intensity criterion, we
optimise a plain sum of squares on **modality-invariant features**, with a
diffusion regulariser. There is nothing to train: the optimisation runs directly
on the displacement field of the case at hand.

Difference from the original ConvexAdam, assumed and documented: their first
stage is a discrete convex optimisation over a sampled displacement space; here
we keep only the instance-optimisation stage (Adam on a control grid),
initialised by the affine transform already obtained with elastix. In practice
that initialisation plays the role of the discrete stage for the displacement
magnitudes we care about. To reproduce their method exactly, use their script:

    python run_convex_adam_with_network_feats.py --fixed f.nii.gz --moving m.nii.gz \\
        --hf_variant anatomix --exp_name demo --fixed_minclip -450 --fixed_maxclip 450

When to prefer this over the elastix B-spline: large multimodal displacements,
with a GPU available. When not to: when bit-exact reproducibility is required, or
when there is no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("registration.convexadam")


@dataclass
class DeformableResult:
    """Displacement field (mm, world coordinates) on the fixed image grid."""

    transform: sitk.DisplacementFieldTransform
    displacement_field: sitk.Image
    info: dict[str, Any] = field(default_factory=dict)


def adam_instance_optimization(
    fixed_features: np.ndarray,
    moving_features: np.ndarray,
    reference: sitk.Image,
    grid_spacing_voxels: int = 4,
    iterations: int = 100,
    lambda_diffusion: float = 1.5,
    learning_rate: float = 0.05,
    fixed_mask: np.ndarray | None = None,
    device: str = "auto",
) -> DeformableResult:
    """Optimise a dense displacement field that aligns the features.

    ``fixed_features`` and ``moving_features`` are ``(C, Z, Y, X)`` on the **same**
    grid (that of ``reference``): the moving volume must therefore have been
    resampled beforehand with the linear transform found by elastix. Only a
    deformable residual is optimised, which makes the optimisation stable.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the ConvexAdam-style deformable stage requires torch. Use "
            "deformable_engine=elastix to stay on CPU."
        ) from exc

    if fixed_features.shape != moving_features.shape:
        raise ValueError(f"features of different shapes: {fixed_features.shape} vs {moving_features.shape}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("instance optimisation on CPU: expect several minutes")

    shape = fixed_features.shape[1:]  # (Z, Y, X)
    fixed_t = torch.from_numpy(np.ascontiguousarray(fixed_features)).to(device)[None]
    moving_t = torch.from_numpy(np.ascontiguousarray(moving_features)).to(device)[None]

    grid_shape = tuple(max(2, int(round(s / grid_spacing_voxels))) for s in shape)
    # Parameters: displacement in normalised [-1, 1] units, (x, y, z) order.
    control = torch.zeros((1, 3) + grid_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([control], lr=learning_rate)

    identity = F.affine_grid(
        torch.eye(3, 4, device=device)[None], (1, 1) + tuple(shape), align_corners=True
    )  # (1, Z, Y, X, 3), last dim = (x, y, z)

    weight = None
    if fixed_mask is not None and fixed_mask.shape == shape:
        weight = torch.from_numpy((fixed_mask > 0).astype(np.float32)).to(device)[None, None]
        if float(weight.sum()) < 64:
            log.warning("mask nearly empty for instance optimisation: ignored")
            weight = None

    history: list[float] = []
    for step in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        disp = F.interpolate(control, size=tuple(shape), mode="trilinear", align_corners=True)
        grid = identity + disp.permute(0, 2, 3, 4, 1)
        warped = F.grid_sample(moving_t, grid, mode="bilinear", padding_mode="border", align_corners=True)

        residual = (warped - fixed_t) ** 2
        similarity = (
            (residual * weight).sum() / weight.sum().clamp(min=1.0) / fixed_t.shape[1]
            if weight is not None
            else residual.mean()
        )

        # Diffusion regulariser: penalises the gradients of the field.
        smooth = (
            (control[:, :, 1:] - control[:, :, :-1]).pow(2).mean()
            + (control[:, :, :, 1:] - control[:, :, :, :-1]).pow(2).mean()
            + (control[:, :, :, :, 1:] - control[:, :, :, :, :-1]).pow(2).mean()
        )
        loss = similarity + lambda_diffusion * smooth
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if step % 20 == 0:
            log.debug(
                "iteration %3d: loss=%.6f (similarity=%.6f)", step, history[-1], float(similarity.detach())
            )

    with torch.no_grad():
        disp = F.interpolate(control, size=tuple(shape), mode="trilinear", align_corners=True)
        disp_np = disp[0].detach().to("cpu").numpy()  # (3, Z, Y, X) in normalised (x, y, z) units

    field = _normalized_to_world_field(disp_np, reference)
    info = {
        "iterations": int(iterations),
        "grid_shape": list(grid_shape),
        "grid_spacing_voxels": grid_spacing_voxels,
        "lambda_diffusion": lambda_diffusion,
        "loss_initial": round(history[0], 6) if history else None,
        "loss_final": round(history[-1], 6) if history else None,
        "device": device,
        "engine": "regix-adam-instance-optimization",
    }
    if history and history[-1] > history[0]:
        log.warning("loss increased: lower learning_rate or raise lambda_diffusion")
    log.info(
        "instance optimisation finished: loss %.6f -> %.6f (%d iterations, %s)",
        history[0] if history else float("nan"),
        history[-1] if history else float("nan"),
        iterations,
        device,
    )

    transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    return DeformableResult(transform=transform, displacement_field=field, info=info)


def _normalized_to_world_field(disp_norm: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Convert a field in ``grid_sample`` units to a field in world millimetres.

    Three successive conversions, each a classic source of error:
    1. normalised [-1, 1] units -> voxels: factor (size - 1) / 2 per axis;
    2. torch axes (x, y, z) aligned with the reversed numpy axes (z, y, x);
    3. voxels -> world mm: multiply by the spacing **and** by the image direction
       cosines (otherwise an oblique volume comes out wrong).
    """
    _, nz, ny, nx = disp_norm.shape
    # torch: last dimension = fastest axis = x (numpy axis 2)
    dx_vox = disp_norm[0] * (nx - 1) / 2.0
    dy_vox = disp_norm[1] * (ny - 1) / 2.0
    dz_vox = disp_norm[2] * (nz - 1) / 2.0

    spacing = np.asarray(reference.GetSpacing(), dtype=np.float64)  # (x, y, z)
    direction = np.asarray(reference.GetDirection(), dtype=np.float64).reshape(3, 3)
    vox = np.stack([dx_vox, dy_vox, dz_vox], axis=-1)  # (Z, Y, X, 3) in voxels
    physical = vox * spacing[None, None, None, :]
    world = physical @ direction.T  # columns = image axes

    field = sitk.GetImageFromArray(np.ascontiguousarray(world, dtype=np.float64), isVector=True)
    field.SetSpacing(reference.GetSpacing())
    field.SetOrigin(reference.GetOrigin())
    field.SetDirection(reference.GetDirection())
    return field


def displacement_field_from_transform(transform: sitk.Transform, reference: sitk.Image) -> sitk.Image:
    """Materialise any transform as a displacement field on ``reference``.

    Used for QC (Jacobian, magnitude) even when the transform is purely linear.
    """
    f = sitk.TransformToDisplacementFieldFilter()
    f.SetReferenceImage(reference)
    f.SetOutputPixelType(sitk.sitkVectorFloat64)
    return f.Execute(transform)
