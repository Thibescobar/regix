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

Import from the modules directly: ``from regix.features.anatomix import
extract_feature_pair``. Nothing is re-exported here, so that importing the package
never drags torch in.
"""
