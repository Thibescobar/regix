"""Organ nomenclature and per-organ registration profiles.

Two things live here:

1. the **translation** between the nomenclatures met in the wild (TotalSegmentator,
   exported radiotherapy structures, hand-named mask files) and a single canonical
   Regix name;
2. a **per-organ registration profile**: relevant HU window, mask margin,
   expected deformability, recommended stages. That is what makes the
   registration organ-aware instead of applying one setting to the whole body --
   a liver moves 20 mm with breathing, a femur does not deform at all.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Synonyms found in mask file names and label sidecars -> canonical name.
#: Regix reads organ names from file names (``ExternalSegmenter._from_directory``),
#: so this table is what makes a directory of third-party masks usable without
#: renaming anything.
ORGAN_ALIASES: dict[str, str] = {
    # Side written as a prefix rather than a suffix
    "left_kidney": "kidney_left",
    "right_kidney": "kidney_right",
    "left_lung": "lung_left",
    "right_lung": "lung_right",
    "left_head_of_femur": "femur_left",
    "right_head_of_femur": "femur_right",
    # Anatomical synonyms
    "gall_bladder": "gallbladder",
    "postcava": "inferior_vena_cava",
    "intestine": "small_bowel",
    "bladder": "urinary_bladder",
    # TotalSegmentator v2
    "adrenal_gland_left": "adrenal_gland_left",
    "adrenal_gland_right": "adrenal_gland_right",
    "lung_upper_lobe_left": "lung_left",
    "lung_lower_lobe_left": "lung_left",
    "lung_upper_lobe_right": "lung_right",
    "lung_middle_lobe_right": "lung_right",
    "lung_lower_lobe_right": "lung_right",
    "portal_vein_and_splenic_vein": "portal_vein_and_splenic_vein",
    "inferior_vena_cava": "inferior_vena_cava",
    "femur_left": "femur_left",
    "femur_right": "femur_right",
    "hip_left": "hip_left",
    "hip_right": "hip_right",
    "sacrum": "sacrum",
    "brain": "brain",
    "heart": "heart",
    "vertebrae_l1": "vertebrae_l1",
}


def canonical_organ_name(name: str) -> str:
    """Normalise an organ name (case, spaces, synonyms, file extension)."""
    key = name.strip().lower()
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    key = key.replace(" ", "_").replace("-", "_")
    return ORGAN_ALIASES.get(key, key)


# --------------------------------------------------------------------------- #
# Registration profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrganProfile:
    """Recommended settings for registering a given organ."""

    name: str
    region: str                                  # abdomen | thorax | pelvis | head | musculoskeletal
    deformable: bool                             # is a non-rigid stage relevant?
    bspline_grid_mm: float = 20.0                # recommended B-spline grid spacing
    hu_window: str | None = None                 # suitable CT window
    mask_dilate_mm: float = 8.0                  # criterion mask margin
    roi_margin_mm: float = 25.0
    typical_motion_mm: float = 10.0              # expected physiological amplitude
    notes: str = ""

    def recommended_stage_types(self) -> list[str]:
        return ["rigid", "affine", "bspline"] if self.deformable else ["rigid", "affine"]


def _p(name, region, deformable, **kw) -> OrganProfile:
    return OrganProfile(name=name, region=region, deformable=deformable, **kw)


ORGAN_PROFILES: dict[str, OrganProfile] = {
    # --- abdomen: substantial respiratory motion --------------------------- #
    "liver": _p("liver", "abdomen", True, bspline_grid_mm=20.0, hu_window="ct_liver",
                typical_motion_mm=20.0,
                notes="Cranio-caudal respiratory displacement of 10-25 mm; a deformable "
                      "stage is indispensable."),
    "spleen": _p("spleen", "abdomen", True, bspline_grid_mm=20.0, hu_window="ct_abdomen",
                 typical_motion_mm=15.0),
    "pancreas": _p("pancreas", "abdomen", True, bspline_grid_mm=15.0, hu_window="ct_abdomen",
                   mask_dilate_mm=10.0, typical_motion_mm=15.0,
                   notes="Low contrast and a mobile duodenal neighbourhood: mask strictly, "
                         "use a fine grid."),
    "kidney_left": _p("kidney_left", "abdomen", True, bspline_grid_mm=20.0, hu_window="ct_abdomen",
                      typical_motion_mm=10.0),
    "kidney_right": _p("kidney_right", "abdomen", True, bspline_grid_mm=20.0, hu_window="ct_abdomen",
                       typical_motion_mm=10.0),
    "stomach": _p("stomach", "abdomen", True, bspline_grid_mm=15.0, hu_window="ct_abdomen",
                  typical_motion_mm=25.0,
                  notes="Highly variable filling: registration will not compensate a change "
                        "in gastric volume."),
    "duodenum": _p("duodenum", "abdomen", True, bspline_grid_mm=12.0, hu_window="ct_abdomen",
                   typical_motion_mm=20.0),
    "gallbladder": _p("gallbladder", "abdomen", True, bspline_grid_mm=15.0, hu_window="ct_abdomen"),
    "esophagus": _p("esophagus", "thorax", True, bspline_grid_mm=12.0, hu_window="ct_mediastinum"),
    "colon": _p("colon", "abdomen", True, bspline_grid_mm=15.0, hu_window="ct_abdomen",
                typical_motion_mm=25.0, notes="Peristalsis and gas: Dice-based QC is unreliable."),
    "small_bowel": _p("small_bowel", "abdomen", True, bspline_grid_mm=12.0, hu_window="ct_abdomen",
                      typical_motion_mm=30.0),
    "adrenal_gland_left": _p("adrenal_gland_left", "abdomen", True, bspline_grid_mm=12.0,
                             mask_dilate_mm=12.0, roi_margin_mm=30.0,
                             notes="Sub-centimetre structure: register globally first, then "
                                   "refine on a local ROI."),
    "adrenal_gland_right": _p("adrenal_gland_right", "abdomen", True, bspline_grid_mm=12.0,
                              mask_dilate_mm=12.0, roi_margin_mm=30.0),
    # --- vessels: useful rigid landmarks ---------------------------------- #
    "aorta": _p("aorta", "abdomen", False, hu_window="ct_soft", typical_motion_mm=5.0,
                notes="Excellent initialization landmark: long, well contrasted, barely "
                      "deformable."),
    "inferior_vena_cava": _p("inferior_vena_cava", "abdomen", False, hu_window="ct_soft"),
    "portal_vein_and_splenic_vein": _p("portal_vein_and_splenic_vein", "abdomen", True,
                                       bspline_grid_mm=12.0, hu_window="ct_liver"),
    "hepatic_vessel": _p("hepatic_vessel", "abdomen", True, bspline_grid_mm=10.0, hu_window="ct_liver"),
    "celiac_trunk": _p("celiac_trunk", "abdomen", False, hu_window="ct_soft"),
    # --- thorax ------------------------------------------------------------ #
    "lung_left": _p("lung_left", "thorax", True, bspline_grid_mm=15.0, hu_window="ct_lung",
                    typical_motion_mm=25.0,
                    notes="Large respiratory deformation: plan for 4 resolutions and a low "
                          "bending-energy weight."),
    "lung_right": _p("lung_right", "thorax", True, bspline_grid_mm=15.0, hu_window="ct_lung",
                     typical_motion_mm=25.0),
    "heart": _p("heart", "thorax", True, bspline_grid_mm=15.0, hu_window="ct_mediastinum",
                typical_motion_mm=15.0,
                notes="Cardiac motion is not compensated: register equivalent phases."),
    # --- pelvis ------------------------------------------------------------ #
    "prostate": _p("prostate", "pelvis", True, bspline_grid_mm=12.0, hu_window="ct_soft",
                   mask_dilate_mm=10.0, roi_margin_mm=30.0, typical_motion_mm=8.0,
                   notes="Typical T2 MR -> planning CT case; bladder and rectal filling "
                         "dominate."),
    "urinary_bladder": _p("urinary_bladder", "pelvis", True, bspline_grid_mm=15.0, hu_window="ct_soft",
                          typical_motion_mm=15.0),
    "rectum": _p("rectum", "pelvis", True, bspline_grid_mm=12.0, hu_window="ct_soft",
                 typical_motion_mm=15.0),
    # --- bone / head: rigid ------------------------------------------------ #
    "femur_left": _p("femur_left", "musculoskeletal", False, hu_window="ct_bone",
                     typical_motion_mm=0.0, notes="Rigid by construction: do not allow a B-spline."),
    "femur_right": _p("femur_right", "musculoskeletal", False, hu_window="ct_bone"),
    "hip_left": _p("hip_left", "musculoskeletal", False, hu_window="ct_bone"),
    "hip_right": _p("hip_right", "musculoskeletal", False, hu_window="ct_bone"),
    "sacrum": _p("sacrum", "pelvis", False, hu_window="ct_bone"),
    "brain": _p("brain", "head", False, hu_window="ct_brain", typical_motion_mm=0.0,
                notes="Rigid intra-patient; affine only if a scale change is expected."),
}

#: Groupings usable directly as targets.
ORGAN_GROUPS: dict[str, list[str]] = {
    "abdomen": ["liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach", "gallbladder"],
    "abdomen_solid": ["liver", "spleen", "kidney_left", "kidney_right"],
    "thorax": ["lung_left", "lung_right", "heart", "esophagus"],
    "pelvis": ["prostate", "urinary_bladder", "rectum", "femur_left", "femur_right"],
    "vessels": ["aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "celiac_trunk"],
    "bones": ["femur_left", "femur_right", "hip_left", "hip_right", "sacrum"],
    "liver_region": ["liver", "portal_vein_and_splenic_vein", "hepatic_vessel", "gallbladder"],
}


def organ_profile(name: str) -> OrganProfile:
    """Profile of an organ; a neutral deformable profile if the organ is unknown."""
    key = canonical_organ_name(name)
    if key in ORGAN_PROFILES:
        return ORGAN_PROFILES[key]
    return OrganProfile(name=key, region="unknown", deformable=True, notes="organ not referenced")


def resolve_targets(targets: list[str]) -> list[str]:
    """Expand the groups and normalise the names, preserving order."""
    out: list[str] = []
    for t in targets:
        key = t.strip().lower()
        expanded = ORGAN_GROUPS.get(key, [canonical_organ_name(key)])
        for organ in expanded:
            if organ not in out:
                out.append(organ)
    return out


def merged_profile(targets: list[str]) -> OrganProfile:
    """Combined profile for several organs: the most constraining one wins.

    Deformable if at least one organ is, finest grid, widest margin. This is
    deliberately conservative: an over-fine grid is preferable to a deformation
    that fails to capture the motion of one of the organs.
    """
    resolved = resolve_targets(targets)
    if not resolved:
        return OrganProfile(name="whole_body", region="unknown", deformable=True)
    profiles = [organ_profile(t) for t in resolved]
    return OrganProfile(
        name="+".join(p.name for p in profiles),
        region=profiles[0].region,
        deformable=any(p.deformable for p in profiles),
        bspline_grid_mm=min(p.bspline_grid_mm for p in profiles),
        hu_window=next((p.hu_window for p in profiles if p.hu_window), None),
        mask_dilate_mm=max(p.mask_dilate_mm for p in profiles),
        roi_margin_mm=max(p.roi_margin_mm for p in profiles),
        typical_motion_mm=max(p.typical_motion_mm for p in profiles),
        notes=" | ".join(p.notes for p in profiles if p.notes),
    )
