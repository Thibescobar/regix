"""Regix - multimodal and multi-organ registration for medical imaging.

Three building blocks, deliberately independent:

* ``regix.registration``: the elastix engine (rigid / affine / B-spline), masks,
  multi-resolution, multi-metric multi-channel registration. This is the core and
  it works on its own, on CPU.
* ``regix.features``: anatomix features (pre-trained network, MIT) that make two
  different modalities comparable voxel by voxel. Multimodal registration then
  becomes a monomodal problem (cross correlation over feature channels instead of
  mutual information over intensities).
* ``regix.organs``: organ segmentation (SuPreM, TotalSegmentator, or user-supplied
  masks) feeding initialization, criterion masking, per-organ cropping and quality
  control.

See ``DISCLAIMER`` below for the regulatory statement, which is the single wording
reused by the run manifest, the HTML report, the API and the LICENSE.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: The single source of truth for the regulatory statement. The run manifest, the HTML
#: report, the API description and the LICENSE all carry this exact sentence: one
#: wording, so it cannot drift into five slightly different claims.
DISCLAIMER = (
    "Regix is research software. It is not a medical device and has not been cleared "
    "or approved by any regulatory authority. It must not be used to make clinical "
    "decisions without review by a qualified operator and validation on the data of "
    "the deploying site."
)

__all__ = [
    "__version__",
    "DISCLAIMER",
    "RegistrationConfig",
    "load_preset",
    "available_presets",
    "Volume",
    "RegistrationPipeline",
    "RegistrationResult",
    "register",
]


def __getattr__(name: str):  # lazy imports: `import regix` stays instant
    if name in ("RegistrationConfig", "load_preset", "available_presets"):
        from regix import config

        return getattr(config, name)
    if name == "Volume":
        from regix.io.volume import Volume

        return Volume
    if name in ("RegistrationPipeline", "RegistrationResult", "register"):
        from regix import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'regix' has no attribute {name!r}")
