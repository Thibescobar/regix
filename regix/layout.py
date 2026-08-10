"""Canonical names and safe lifecycle operations for Regix run artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

REPORT = Path("report.html")
MANIFEST = Path("run_manifest.json")
EFFECTIVE_CONFIG = Path("config_effective.yaml")
RUN_LOG = Path("regix.log")
REGISTERED_IMAGE = Path("moving_registered.nii.gz")

# Only paths owned by Regix are removed on overwrite. The output directory itself is
# deliberately never removed because users often keep notes or review files beside a run.
_OWNED_ROOTS = (
    Path("transform"),
    Path("elastix"),
    Path("transformix"),
    Path("masks"),
    Path("features"),
    Path("dicom_registered"),
    Path("report_images"),
    Path("cache"),
    REPORT,
    MANIFEST,
    EFFECTIVE_CONFIG,
    RUN_LOG,
    Path("moving_registered.nii.gz"),
    Path("moving_registered.nii"),
    Path("deformation_field.nii.gz"),
    Path("deformation_field.nii"),
    Path("jacobian.nii.gz"),
    Path("jacobian.nii"),
)


def clean_known_artifacts(output_dir: str | Path) -> list[Path]:
    """Remove only known Regix artifacts and return their relative paths."""
    root = Path(output_dir)
    removed: list[Path] = []
    for relative in _OWNED_ROOTS:
        target = root / relative
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
            removed.append(relative)
        elif target.is_dir():
            shutil.rmtree(target)
            removed.append(relative)
    return removed


def inventory(output_dir: str | Path, *, include_manifest: bool = True) -> list[str]:
    """Return a deterministic relative-file inventory for the run manifest."""
    root = Path(output_dir)
    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    if include_manifest and MANIFEST.as_posix() not in files:
        files.append(MANIFEST.as_posix())
    return sorted(files)


def known_artifact_roots() -> tuple[Path, ...]:
    """Public immutable inventory used by contract tests and documentation."""
    return _OWNED_ROOTS
