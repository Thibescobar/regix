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
"""

from regix.features.anatomix import (
    AnatomixExtractor,
    FeaturePair,
    anatomix_available,
    clip_for_modality,
    extract_feature_pair,
    resolve_device,
)
from regix.features.mind import mind_ssc_features
from regix.features.reduce import features_to_sitk, joint_pca_reduce, voxel_normalize

__all__ = [
    "AnatomixExtractor",
    "FeaturePair",
    "anatomix_available",
    "clip_for_modality",
    "extract_feature_pair",
    "resolve_device",
    "mind_ssc_features",
    "joint_pca_reduce",
    "features_to_sitk",
    "voxel_normalize",
]
