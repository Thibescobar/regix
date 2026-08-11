"""Modality-invariant descriptors.

Two providers, one contract: ``(C, Z, Y, X) float32`` on the working grid, with
channels normalised voxel by voxel.

* ``anatomix`` (GPU, MIT pre-trained network): 16 learned channels, robust across
  CT/MR/PET, the best choice when a GPU is available;
* ``mind`` (CPU, analytical): the MIND-SSC self-similarity descriptor, with no
  torch dependency. This is the honest fallback on a workstation without a GPU,
  and a safety net when anatomix is used outside its training domain.

In both cases the channels are then reduced by a PCA with a **shared** basis
across the two volumes (``reduce.py``): without a shared basis, comparing the
fixed and moving channels is meaningless.

Provider implementations remain lazy, so importing this package never imports torch,
Anatomix or probes a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import SimpleITK as sitk


@dataclass
class FeaturePair:
    """Descriptor channels and traceability information for one image pair."""

    fixed_channels: list[sitk.Image]
    moving_channels: list[sitk.Image]
    provider: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.fixed_channels)
