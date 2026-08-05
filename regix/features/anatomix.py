"""anatomix feature extraction (https://github.com/neel-dey/anatomix, MIT).

anatomix is a 3D U-Net pre-trained on synthetic data with randomised contrast.
The practical consequence: its 16 output channels encode anatomy independently of
the modality. Two volumes of different modalities become comparable through a
plain cross correlation, which turns multimodal registration into a monomodal
problem -- **from the rigid stage onwards**, not just for the deformable one.

This module re-implements nothing: it loads the official model (HuggingFace or a
local checkpoint) and only handles what the upstream repository leaves to the
caller -- normalisation, sliding-window inference, padding to multiples of 16,
voxel-wise channel normalisation, and a clean fallback when nothing is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.config import FeatureConfig
from regix.features.mind import mind_ssc_features
from regix.features.reduce import features_to_sitk, joint_pca_reduce, voxel_normalize
from regix.io.volume import Volume
from regix.logging_utils import get_logger
from regix.preprocess.geometry import pad_to_multiple, unpad
from regix.preprocess.intensity import HU_WINDOWS, normalize_for_features

log = get_logger("features.anatomix")

_DOWNSAMPLING_MULTIPLE = 16  # U-Net with 4 levels: 2^4

_INSTALL_HINT = (
    "anatomix unavailable. Install with: pip install torch monai "
    "'anatomix @ git+https://github.com/neel-dey/anatomix.git' "
    "(or set features.enabled=false / metric=mi, or use the CPU MIND descriptor)."
)


def anatomix_available() -> tuple[bool, str]:
    """(available, reason) -- loads no weights and does not touch the GPU."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    try:
        import anatomix  # noqa: F401
    except ImportError:
        return False, "the anatomix package is not installed"
    return True, "ok"


def resolve_device(requested: str = "auto", allow_cpu: bool = False) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if not allow_cpu:
        raise RuntimeError(
            "no GPU detected. anatomix inference on CPU takes several minutes per volume: "
            "set features.allow_cpu=true deliberately, or stay with mutual information "
            "(metric=mi) / the MIND descriptor."
        )
    log.warning("anatomix inference on CPU: expect several minutes per volume")
    return "cpu"


# --------------------------------------------------------------------------- #
@dataclass
class FeaturePair:
    """Result of extraction on a fixed/moving pair."""

    fixed_channels: list[sitk.Image]
    moving_channels: list[sitk.Image]
    provider: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.fixed_channels)


class AnatomixExtractor:
    """Wrapper around the anatomix model. The model is loaded once."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._model = None
        self._device: str | None = None

    # -- loading ---------------------------------------------------------- #
    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self.config.device, self.config.allow_cpu)
        return self._device

    def load(self):
        if self._model is not None:
            return self._model
        ok, reason = anatomix_available()
        if not ok:
            raise RuntimeError(f"{_INSTALL_HINT} (cause: {reason})")

        import torch

        ckpt = self.config.checkpoint
        if ckpt is not None:
            ckpt = Path(ckpt)
            if not ckpt.exists():
                raise FileNotFoundError(f"anatomix checkpoint not found: {ckpt}")
            from anatomix.model.network import Unet

            if self.config.variant != "anatomix":
                raise ValueError(
                    "a local checkpoint is only supported for the 'anatomix' variant "
                    "(16 channels); for the dev variants use hf_variant"
                )
            model = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16)
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            source = str(ckpt)
        else:
            from anatomix.model.load_from_hf import load_from_hf

            model = load_from_hf(self.config.variant, repo_id=self.config.hf_repo_id, map_location="cpu")
            source = f"hf:{self.config.hf_repo_id}:{self.config.variant}"

        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("anatomix loaded (%s, %.1f M parameters, device=%s)", source, n_params / 1e6, self.device)
        self._model = model
        self._source = source
        return model

    # -- inference -------------------------------------------------------- #
    def extract(self, image: sitk.Image, clip: tuple[float, float] | None = None) -> np.ndarray:
        """Return ``(C, Z, Y, X)`` float32 on the grid of ``image``."""
        import torch

        model = self.load()
        arr = normalize_for_features(image, clip)  # (Z, Y, X) in [0, 1]
        padded, pads = pad_to_multiple(arr, _DOWNSAMPLING_MULTIPLE)

        n_vox = int(np.prod(padded.shape))
        est_gb = n_vox * 16 * 4 / 1e9
        if est_gb > 4.0:
            log.warning(
                "features estimated at %.1f GB (%s voxels): increase working_spacing_mm "
                "or enable organs.roi_crop",
                est_gb,
                padded.shape,
            )

        tensor = torch.from_numpy(padded[None, None]).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self._forward(model, tensor)
        features = out[0].detach().to("cpu", dtype=torch.float32).numpy()
        del tensor, out
        try:  # pragma: no cover
            torch.cuda.empty_cache()
        except Exception:
            pass

        features = unpad(features, pads)
        features = voxel_normalize(features, self.config.voxel_normalize)
        return np.ascontiguousarray(features, dtype=np.float32)

    def _forward(self, model, tensor):
        """Sliding window when the volume is larger than the patch size."""
        import torch

        patch = tuple(int(p) for p in self.config.patch_size)
        shape = tuple(tensor.shape[2:])
        if all(s <= p for s, p in zip(shape, patch, strict=False)):
            return model(tensor)
        try:
            from monai.inferers import sliding_window_inference
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"volume {shape} is larger than the window {patch}: monai is required "
                "for sliding-window inference (pip install monai)"
            ) from exc
        return sliding_window_inference(
            inputs=tensor,
            roi_size=patch,
            sw_batch_size=int(self.config.sw_batch_size),
            predictor=model,
            overlap=float(self.config.sw_overlap),
            mode="gaussian",
            sigma_scale=0.25,
            padding_mode="replicate",
            device=torch.device("cpu") if self.device != "cpu" else None,
            sw_device=torch.device(self.device),
            progress=False,
        )


# --------------------------------------------------------------------------- #
def clip_for_modality(modality: str | None) -> tuple[float, float] | None:
    """Recommended clipping bounds before feature extraction (CT: the paper's bounds)."""
    mod = (modality or "").upper()
    if mod in ("CT", "CBCT"):
        return HU_WINDOWS["ct_registration"]
    return None


def extract_feature_pair(
    fixed: Volume,
    moving: Volume,
    config: FeatureConfig | None = None,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    provider: str = "auto",
    seed: int = 0,
) -> FeaturePair:
    """Extract, normalise and reduce the features of both volumes.

    ``provider``: ``anatomix``, ``mind``, or ``auto`` (anatomix when available,
    otherwise MIND-SSC on CPU).
    """
    cfg = config or FeatureConfig()
    chosen = provider
    if chosen == "auto":
        ok, reason = anatomix_available()
        chosen = "anatomix" if ok else "mind"
        if not ok:
            log.warning("falling back to the CPU MIND-SSC descriptor (%s)", reason)

    info: dict[str, Any] = {"provider": chosen}

    if chosen == "anatomix":
        extractor = AnatomixExtractor(cfg)
        f_feat = extractor.extract(fixed.image, clip_for_modality(fixed.modality))
        m_feat = extractor.extract(moving.image, clip_for_modality(moving.modality))
        info.update(
            variant=cfg.variant,
            device=extractor.device,
            source=getattr(extractor, "_source", None),
            raw_channels=int(f_feat.shape[0]),
        )
    elif chosen == "mind":
        f_arr = normalize_for_features(fixed.image, clip_for_modality(fixed.modality))
        m_arr = normalize_for_features(moving.image, clip_for_modality(moving.modality))
        f_spacing = tuple(float(s) for s in reversed(fixed.spacing))    # ITK (x,y,z) -> numpy (z,y,x)
        m_spacing = tuple(float(s) for s in reversed(moving.spacing))
        f_feat = mind_ssc_features(f_arr, spacing=f_spacing)
        m_feat = mind_ssc_features(m_arr, spacing=m_spacing)
        info.update(descriptor="MIND-SSC", raw_channels=int(f_feat.shape[0]))
    else:
        raise ValueError(f"unknown feature provider: {provider}")

    f_mask_arr = _mask_array(fixed_mask, f_feat.shape[1:])
    m_mask_arr = _mask_array(moving_mask, m_feat.shape[1:])
    f_red, m_red, pca_info = joint_pca_reduce(
        f_feat,
        m_feat,
        n_components=cfg.n_components,
        max_voxels=cfg.pca_max_voxels,
        fixed_mask=f_mask_arr,
        moving_mask=m_mask_arr,
        seed=seed,
    )
    info["pca"] = pca_info

    return FeaturePair(
        fixed_channels=features_to_sitk(f_red, fixed.image),
        moving_channels=features_to_sitk(m_red, moving.image),
        provider=chosen,
        info=info,
    )


def _mask_array(mask: sitk.Image | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = sitk.GetArrayViewFromImage(mask)
    if arr.shape != tuple(shape):
        log.debug("mask %s incompatible with features %s: ignored for the PCA", arr.shape, shape)
        return None
    return (arr > 0).astype(np.uint8)
