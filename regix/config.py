"""Declarative configuration for Regix (pydantic v2) plus the bundled YAML presets.

Principle: *everything* that changes the numerical result lives in this object, it
is serialised into the run manifest, and any elastix parameter remains overridable
through ``StageConfig.extra`` -- because in the field you always end up having to
touch a parameter no API anticipated.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from regix.env import DICOM_UID_ROOT_ENV

PRESET_DIR = Path(__file__).parent / "presets"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Metric(str, Enum):
    AUTO = "auto"  # chosen from the modalities and feature availability
    MI = "mi"  # AdvancedMattesMutualInformation (classic multimodal)
    NCC = "ncc"  # AdvancedNormalizedCorrelation (monomodal)
    MSE = "mse"  # AdvancedMeanSquares
    FEATURES_NCC = "features_ncc"  # multi-channel NCC on anatomix features
    FEATURES_MSE = "features_mse"  # multi-channel SSD on anatomix features


class TransformType(str, Enum):
    TRANSLATION = "translation"
    RIGID = "rigid"
    SIMILARITY = "similarity"
    AFFINE = "affine"
    BSPLINE = "bspline"


class InitMode(str, Enum):
    IDENTITY = "identity"
    GEOMETRY = "geometry"  # geometric centres of the grids
    MOMENTS = "moments"  # intensity centres of mass
    ORGAN_CENTROID = "organ_centroid"  # centres of mass of organ masks
    ORGAN_MOMENTS = "organ_moments"  # plus scale/inertia alignment of the organ
    MULTISTART = "multistart"  # try several candidates, keep the best
    FILE = "file"  # initial transform supplied


class OrganBackend(str, Enum):
    NONE = "none"
    EXTERNAL = "external"  # pre-computed NIfTI masks (the most common clinical case)
    TOTALSEGMENTATOR = "totalsegmentator"


class DeformableEngine(str, Enum):
    NONE = "none"
    ELASTIX = "elastix"  # B-spline, CPU, deterministic, readable parameters
    CONVEXADAM = "convexadam"  # anatomix + instance optimisation, GPU, fast, strong multimodal


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class ImagePrep(BaseModel):
    """Preparation of a single volume. The default values touch nothing."""

    model_config = ConfigDict(extra="forbid")

    window: str | None = Field(
        default=None,
        description="Named HU window for CT: ct_abdomen, ct_liver, ct_lung, ct_bone, ct_soft, ct_full.",
    )
    clip: tuple[float, float] | None = Field(
        default=None, description="Explicit intensity bounds (take priority over window)."
    )
    percentile_clip: Literal["auto"] | tuple[float, float] | None = Field(
        default="auto",
        description=(
            "Robustness percentiles, applied when clip/window are absent. Three states, "
            "deliberately distinct: 'auto' (default) lets the modality decide -- a fixed HU "
            "window for CT/CBCT, (0.5, 99.5) for MR, (0.0, 99.5) for PET/NM; an explicit "
            "pair is always honoured, including on a CT, which is the right answer on a "
            "CBCT whose HU scale is offset and where a fixed window would clip the anatomy "
            "away; null means no percentile clipping at all. A plain default value could "
            "not express the difference between the first two, and silently turned an "
            "explicit request into the modality window."
        ),
    )
    normalize: Literal["minmax", "zscore", "none"] = Field(
        default="none",
        description=(
            "Intensity rescaling of the image handed to elastix. 'none' by default, and "
            "that default matters: rescaling breaks every published elastix parameter "
            "file, which assumes the acquisition scale (see regix.preprocess.intensity). "
            "The [0, 1] normalisation that anatomix and MIND need is applied inside the "
            "feature path, on their own inputs. Set this only if you know the parameters "
            "of your stages were tuned on rescaled data."
        ),
    )
    n4_bias_correction: bool = Field(
        default=False,
        description=(
            "N4 bias field correction (MR). Off by default, and worth less to a "
            "registration than to a segmentation: MI is a histogram statistic and barely "
            "notices a smooth bias, and the anatomix / MIND descriptors are built to be "
            "contrast-invariant -- MIND compares a ~10 mm neighbourhood with itself, over "
            "which a bias field is very nearly constant, so it cancels. Turn it on for "
            "intensity-based NCC on MR with a visible gradient (surface coil), where "
            "SubtractMean only removes a *global* intensity change, not a spatially "
            "varying one. Runs on the native intensities before any clipping, so the cost "
            "follows the acquisition grid, not preprocess.working_spacing_mm."
        ),
    )
    denoise_sigma_mm: float | None = Field(default=None, gt=0.0, le=50.0)

    @field_validator("clip", "percentile_clip")
    @classmethod
    def _ordered(_cls, v, info):
        if isinstance(v, str):  # the "auto" sentinel carries no bounds to order
            return v
        if v is not None and not v[0] < v[1]:
            raise ValueError(f"unordered bounds: {v}")
        if info.field_name == "percentile_clip" and v is not None and (v[0] < 0 or v[1] > 100):
            raise ValueError("percentile bounds must lie in [0, 100]")
        return v


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed: ImagePrep = Field(default_factory=ImagePrep)
    moving: ImagePrep = Field(default_factory=ImagePrep)
    working_spacing_mm: float | tuple[float, float, float] | None = Field(
        default=2.0,
        description=(
            "Isotropic working resolution for the optimisation. The output is always "
            "reconstructed on the original grid of the fixed image. None = no resampling."
        ),
    )
    orientation: str | None = Field(
        default="LPS", description="Canonical reorientation before processing (None = leave as is)."
    )

    @field_validator("working_spacing_mm")
    @classmethod
    def _positive_spacing(_cls, value):
        if value is None:
            return value
        values = (value,) if isinstance(value, (int, float)) else value
        if any(not 0 < float(component) <= 50 for component in values):
            raise ValueError("working spacing must lie in (0, 50] mm")
        return value


# --------------------------------------------------------------------------- #
# anatomix features
# --------------------------------------------------------------------------- #
class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | Literal["auto"] = Field(
        default="auto",
        description=(
            "auto = request features when modalities differ or a feature/deformable stage "
            "needs them. Providers fall back anatomix -> CPU MIND-SSC -> intensities; "
            "--allow-cpu-features only permits anatomix itself to run on CPU."
        ),
    )
    variant: Literal["anatomix", "anatomix-dev", "anatomix-dev-vit"] = "anatomix"
    hf_repo_id: str = "neel-dey/anatomix"
    checkpoint: Path | None = Field(default=None, description="Local .pth weights (take priority over HF).")
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    patch_size: tuple[int, int, int] = Field(default=(128, 128, 128))
    sw_overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    sw_batch_size: int = Field(default=1, ge=1, le=64)
    voxel_normalize: Literal["l2", "zscore", "none"] = "l2"
    n_components: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "PCA reduction (basis shared between fixed and moving) of the number of "
            "channels passed to elastix."
        ),
    )
    pca_max_voxels: int = Field(default=200_000, ge=1_000, le=100_000_000)
    allow_cpu: bool = Field(default=False, description="Allow feature inference on CPU (slow).")

    @field_validator("patch_size")
    @classmethod
    def _positive_patch_size(_cls, value):
        if any(not 8 <= component <= 1024 for component in value):
            raise ValueError("every feature patch dimension must lie in [8, 1024] voxels")
        return value


# --------------------------------------------------------------------------- #
# Organs
# --------------------------------------------------------------------------- #
class OrganConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: OrganBackend = OrganBackend.NONE
    targets: list[str] = Field(
        default_factory=list,
        description="Organs of interest (Regix names: liver, spleen, kidney_left, ...). Empty = whole body.",
    )
    fixed_mask: Path | None = None
    moving_mask: Path | None = None
    fixed_labelmap: Path | None = Field(default=None, description="Pre-computed multi-organ label map.")
    moving_labelmap: Path | None = None
    label_names: dict[int, str] | None = Field(
        default=None,
        description=(
            "Label -> organ-name mapping for the supplied maps, e.g. {1: liver, 2: spleen}. "
            "Failing that, Regix looks for a sidecar file '<map>.labels.json' or '.labels.txt'; "
            "with neither, labels are named label_N and targeting organs by name will not "
            "work. Regix NEVER guesses a nomenclature: a wrong organ name would produce a "
            "wrong mask with no visible sign."
        ),
    )
    device: Literal["auto", "cuda", "cpu"] = "auto"
    ts_task: Literal["auto", "total", "total_mr"] = Field(
        default="auto",
        description="TotalSegmentator task. auto selects total_mr for MR and total for CT.",
    )
    ts_fast: bool = Field(default=True, description="Use TotalSegmentator's lower-resolution fast mode.")
    ts_timeout_seconds: int = Field(
        default=1800,
        ge=1,
        le=86_400,
        description="Timeout for the TotalSegmentator CLI fallback.",
    )
    mask_dilate_mm: float | None = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description=(
            "Dilation of the criterion mask. None uses the merged organ profile "
            "(or 8 mm for whole-body registration)."
        ),
    )
    roi_crop: bool = Field(
        default=False, description="Crop to the organ bounding box (local registration, much faster)."
    )
    roi_margin_mm: float | None = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="ROI margin. None uses the merged organ profile (20 mm for whole body).",
    )
    qc_labels: list[str] = Field(
        default_factory=list, description="Organs used for the QC Dice (default: targets)."
    )
    save_masks: bool = True


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
class InitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: InitMode = InitMode.GEOMETRY
    candidates: list[InitMode] = Field(
        default_factory=lambda: [InitMode.GEOMETRY, InitMode.MOMENTS, InitMode.ORGAN_CENTROID],
        description="Used only when mode == multistart.",
    )
    multistart_rotations_deg: list[tuple[float, float, float]] = Field(
        default_factory=lambda: [(0, 0, 0), (0, 0, 15), (0, 0, -15), (10, 0, 0), (0, 10, 0)],
        description="Probe rotations combined with each candidate (ZYX Euler angles, degrees).",
    )
    transform_file: Path | None = Field(
        default=None,
        description="file mode: elastix parameter .txt or ITK .tfm/.itk.txt/.h5.",
    )
    flip_check: bool = Field(
        default=False,
        description="Also test head-feet symmetries: useful when the DICOM orientation is doubtful.",
    )

    @field_validator("multistart_rotations_deg")
    @classmethod
    def _bounded_rotations(_cls, value):
        if len(value) > 128 or any(abs(float(angle)) > 360 for rotation in value for angle in rotation):
            raise ValueError("multistart rotations must contain at most 128 angles within [-360, 360]")
        return value


# --------------------------------------------------------------------------- #
# Registration stages
# --------------------------------------------------------------------------- #
class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TransformType
    metric: Metric = Metric.AUTO
    n_resolutions: int = Field(default=4, ge=1, le=8)
    max_iterations: int = Field(default=512, ge=1, le=100_000)
    n_spatial_samples: int = Field(default=4096, ge=64, le=100_000_000)
    sampler: Literal["RandomCoordinate", "RandomSparseMask", "Random", "Full", "Grid"] = "RandomCoordinate"
    interpolator_order: int = Field(default=1, ge=0, le=3)
    final_bspline_order: int = Field(default=3, ge=0, le=5)
    required_ratio_valid_samples: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum fraction of samples that must land inside the moving image and mask. "
            "The elastix default (0.25) makes the stage fail as soon as the fields of view "
            "differ or the masks are narrow -- that is, in Regix's main use case. Lowered to "
            "0.05: the optimisation starts, and QC judges the result."
        ),
    )
    use_masks: bool = True
    erode_mask: bool = False
    # B-spline only
    final_grid_spacing_mm: float = Field(default=20.0, gt=0.0, le=1000.0)
    grid_spacing_schedule: list[float] | None = Field(
        default=None, description="Grid factors per resolution (default: 2^(n-1-r))."
    )
    bending_energy_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1_000_000.0,
        description="Bending-energy penalty weight: keeps the deformation physically plausible.",
    )
    rigidity_penalty_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1_000_000.0,
        description="DistancePreservingRigidityPenalty weight (bone / implants).",
    )
    # Misc
    automatic_scales: bool = True
    optimizer: Literal["AdaptiveStochasticGradientDescent", "StandardGradientDescent", "QuasiNewtonLBFGS"] = (
        "AdaptiveStochasticGradientDescent"
    )
    parameter_file: Path | None = Field(
        default=None,
        description=(
            "Use a hand-written elastix parameter file for this stage (e.g. one from the "
            "elastix parameter zoo) instead of letting Regix build the map. The file is "
            "taken verbatim, then 'extra' is applied on top; the fields above that describe "
            "components (metric, n_resolutions, max_iterations, sampler, optimizer, "
            "final_grid_spacing_mm...) are ignored because the file already sets them. "
            "'type' must still match the file's (Transform ...) -- Regix uses it downstream "
            "to decide whether the stage result is a linear transform. Four keys are "
            "re-imposed even if the file disagrees, and a warning says so: see "
            "regix.registration.params.ENFORCED_WITH_PARAMETER_FILE."
        ),
    )
    extra: dict[str, list[str | bool | float | int] | str | bool | float | int] = Field(
        default_factory=dict,
        description=(
            "Raw elastix parameter overrides, e.g. {'MaximumStepLength': '2.0'} or "
            "{'ImagePyramidSchedule': [8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1, 1]}. Values are "
            "stringified on the way out, so numbers need no quoting -- which matters "
            "because YAML parses a numeric list as ints. Booleans are accepted explicitly "
            "and emitted as elastix lowercase true/false tokens."
        ),
    )
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.type.value

    @field_validator("grid_spacing_schedule")
    @classmethod
    def _positive_grid_schedule(_cls, value):
        if value is not None and (len(value) > 32 or any(not 0 < float(item) <= 10_000 for item in value)):
            raise ValueError("grid spacing schedule values must lie in (0, 10000]")
        return value


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
class QCGates(BaseModel):
    """Acceptance thresholds. A registration that fails them is marked FAIL, not deleted."""

    model_config = ConfigDict(extra="forbid")

    min_ncc_gain: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Minimum NCC gain (after - before) inside the QC mask. Relevant for monomodal "
            "pairs only; leaving it None lets the pipeline choose NCC or NMI from the "
            "modalities."
        ),
    )
    min_nmi_gain: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Minimum normalised mutual information gain. The criterion suited to multimodal pairs.",
    )
    min_dice: dict[str, float] = Field(
        default_factory=dict, description="Minimum Dice per organ, e.g. {'liver': 0.85}."
    )
    max_folding_fraction: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Max fraction of voxels with Jacobian <= 0 (field folding).",
    )
    min_abs_final_metric: float | None = Field(
        default=1e-6,
        ge=0.0,
        le=1_000_000.0,
        description=(
            "Floor on |final metric| of every stage. Not a quality threshold -- a floor: "
            "any real criterion is above 1e-3, so this only catches a *degenerate* one, "
            "where elastix ran, reported success and optimised nothing. That happens when "
            "the images and the stage parameters disagree (an internal pixel type that "
            "quantises the intensities away, for instance), and it is otherwise silent: "
            "the transform stays plausible and the similarity gain is ~0, which passes "
            "the gain gate. Set to null to disable."
        ),
    )
    max_tre_mm: float | None = Field(default=None, ge=0.0, le=10_000.0)
    max_translation_mm: float | None = Field(
        default=150.0,
        ge=0.0,
        le=100_000.0,
        description="Guard rail: beyond this it is divergence, not registration.",
    )
    max_scale_deviation: float | None = Field(
        default=0.35,
        ge=0.0,
        le=100.0,
        description="Maximum deviation of the affine scales from 1.0.",
    )
    max_displacement_ratio: float | None = Field(
        default=None,
        gt=0.0,
        le=1000.0,
        description=(
            "Maximum p95 displacement divided by the organ profile's expected physiological motion. "
            "None disables this site-specific plausibility gate."
        ),
    )

    @field_validator("min_dice")
    @classmethod
    def _dice_bounds(_cls, value):
        invalid = {name: threshold for name, threshold in value.items() if not 0 <= threshold <= 1}
        if invalid:
            raise ValueError(f"Dice thresholds must lie in [0, 1]: {invalid}")
        return value


class QCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    report_html: bool = True
    landmarks_fixed: Path | None = Field(
        default=None, description="Points (mm, LPS), one 'x y z' line per point."
    )
    landmarks_moving: Path | None = None
    gates: QCGates = Field(default_factory=QCGates)
    n_slices: int = Field(default=3, ge=1, le=12, description="Slices per plane in the report.")
    jacobian: bool = True
    max_similarity_samples: int = Field(
        default=2_000_000,
        ge=10_000,
        le=100_000_000,
        description="Deterministic cap for paired intensity samples used by QC.",
    )
    figure_format: Literal["png", "jpeg", "webp"] = "webp"
    figure_dpi: int = Field(default=90, ge=50, le=300)
    embed_figures: bool = Field(
        default=True,
        description="Embed report figures as data URIs; false writes compact sidecar images.",
    )


# --------------------------------------------------------------------------- #
# Outputs / runtime
# --------------------------------------------------------------------------- #
class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("regix_out")
    write_resampled: bool = True
    write_transform: bool = True
    write_slicer_transform: bool = Field(
        default=True,
        description=(
            "Write the transforms in Insight Transform File format (.txt), directly loadable "
            "in 3D Slicer (Data > Add data) and in any ITK pipeline. One cumulative file per "
            "stage (stage00_rigid.itk.txt, stage01_affine.itk.txt, ...) plus "
            "final_transform.itk.txt."
        ),
    )
    write_deformation_field: bool = False
    write_jacobian: bool = False
    write_features: bool = False
    write_dicom: bool = Field(
        default=False, description="Also write the registered volume as a derived DICOM series."
    )
    dicom_uid_root: str | None = Field(
        default_factory=lambda: os.getenv(DICOM_UID_ROOT_ENV),
        description="Registered site UID root. REGIX_DICOM_UID_ROOT is used when omitted.",
    )
    write_elastix_inputs: bool = Field(
        default=True,
        description="Persist the exact fixed/moving images handed to elastix for replay.",
    )
    compress: bool = True
    overwrite: bool = False

    @field_validator("dicom_uid_root")
    @classmethod
    def _valid_uid_root(_cls, value):
        if value is None:
            return value
        root = value.rstrip(".")
        components = root.split(".")
        if (
            len(root) > 54
            or not root
            or any(not component.isdigit() for component in components)
            or components[0] not in {"1", "2"}
            or any(len(component) > 1 and component.startswith("0") for component in components)
        ):
            raise ValueError(
                "DICOM UID root must be <=54 characters and contain numeric components without leading zeroes"
            )
        return root


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threads: int | None = Field(default=None, ge=1, le=1024, description="None = let ITK decide.")
    seed: int = Field(default=20250101, ge=0, le=4_294_967_295)
    keep_intermediate: bool = False
    log_level: LogLevel = LogLevel.INFO
    pseudonymize: bool = True
    fail_fast: bool = Field(default=False, description="True = raise if a QC gate fails.")


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class RegistrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    description: str = ""
    fixed_modality: str | None = Field(default=None, description="CT, MR, PT, US... None = read from DICOM.")
    moving_modality: str | None = None
    fixed_series_uid: str | None = Field(
        default=None,
        description="Exact DICOM SeriesInstanceUID to load when the fixed directory is ambiguous.",
    )
    moving_series_uid: str | None = Field(
        default=None,
        description="Exact DICOM SeriesInstanceUID to load when the moving directory is ambiguous.",
    )
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    organs: OrganConfig = Field(default_factory=OrganConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(type=TransformType.RIGID),
            StageConfig(type=TransformType.AFFINE),
        ]
    )
    deformable_engine: DeformableEngine = DeformableEngine.ELASTIX
    qc: QCConfig = Field(default_factory=QCConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _coherence(self) -> RegistrationConfig:
        if not self.stages:
            raise ValueError("at least one registration stage is required")
        has_bspline = any(s.type is TransformType.BSPLINE for s in self.stages)
        if has_bspline and self.deformable_engine is DeformableEngine.CONVEXADAM:
            raise ValueError("a B-spline stage and deformable_engine=convexadam are redundant: choose one")
        if has_bspline and self.deformable_engine is DeformableEngine.NONE:
            raise ValueError("a bspline stage is defined but deformable_engine=none")
        for s in self.stages:
            if s.metric in (Metric.FEATURES_NCC, Metric.FEATURES_MSE) and self.features.enabled is False:
                raise ValueError(
                    f"stage {s.display_name} requests a feature metric but features.enabled=false"
                )
        # Note: backend=external without paths is deliberately accepted. A preset is a
        # reusable template; masks are patient-specific and arrive through the CLI or
        # the API. Their absence is reported at run time, where the pipeline falls back
        # cleanly to a body mask.
        if self.init.mode in (InitMode.ORGAN_CENTROID, InitMode.ORGAN_MOMENTS) and (
            self.organs.backend is OrganBackend.NONE
        ):
            raise ValueError(f"init.mode={self.init.mode.value} requires an organ backend")
        if self.init.mode is InitMode.FILE and self.init.transform_file is None:
            raise ValueError("init.mode=file requires init.transform_file")
        if self.qc.landmarks_fixed and not self.qc.landmarks_moving:
            raise ValueError("landmarks_fixed provided without landmarks_moving")
        return self

    # ------------------------------------------------------------------ #
    def with_overrides(self, **overrides: Any) -> RegistrationConfig:
        """Non-destructive deep merge (used by the CLI)."""
        data = self.model_dump(mode="python")
        _deep_update(data, overrides)
        return RegistrationConfig.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=False), sort_keys=False, allow_unicode=True
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset(name_or_path: str | Path) -> RegistrationConfig:
    """Load a bundled preset (by name) or a user YAML file (by path).

    A YAML file can inherit from a preset through the ``extends`` key.
    """
    candidate = Path(name_or_path)
    looks_like_path = candidate.suffix.lower() in (".yaml", ".yml") or any(
        separator in str(name_or_path) for separator in ("/", "\\")
    )
    if looks_like_path and candidate.exists():
        path = candidate
    elif looks_like_path:
        raise FileNotFoundError(f"configuration file not found: {candidate}")
    else:
        path = PRESET_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            matches = get_close_matches(str(name_or_path), available_presets(), n=1)
            suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
            raise FileNotFoundError(
                f"unknown preset '{name_or_path}'.{suggestion} Available: {', '.join(available_presets())}"
            )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_from_raw(raw, origin=path)


def _build_from_raw(raw: dict, origin: Path, _seen: Sequence[str] = ()) -> RegistrationConfig:
    raw = deepcopy(raw)
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return RegistrationConfig.model_validate(raw)
    if parent_name in _seen:
        raise ValueError(f"circular preset inheritance: {parent_name}")
    parent_path = PRESET_DIR / f"{parent_name}.yaml"
    if not parent_path.exists():
        parent_path = (origin.parent / f"{parent_name}.yaml").resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"parent preset '{parent_name}' not found")
    parent_raw = yaml.safe_load(parent_path.read_text(encoding="utf-8")) or {}
    merged = _build_from_raw(parent_raw, parent_path, tuple(_seen) + (parent_name,)).model_dump(mode="python")
    _deep_update(merged, raw)
    return RegistrationConfig.model_validate(merged)


def preset_source(name: str) -> str:
    """Return a bundled preset verbatim, preserving its rationale comments."""
    path = PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown preset '{name}'. Available: {', '.join(available_presets())}")
    return path.read_text(encoding="utf-8")
