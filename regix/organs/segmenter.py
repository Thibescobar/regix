"""Obtaining organ masks.

Two routes, in decreasing order of clinical realism:

1. ``external`` -- the masks already exist (exported radiotherapy contours,
   outputs of a tool validated on site, manual segmentations). This is the most
   frequent case and the only fully traceable one;
2. ``totalsegmentator`` -- broad anatomical coverage, CT (and MR in v2), simple
   installation, and a pip-pinnable version that the run manifest can record.

Regix deliberately supports **one** automatic backend. Here the masks are priors,
never deliverables: they feed the initialisation centroids, the (dilated) elastix
criterion mask, the ROI bounding box and the QC Dice. None of those four uses is
sensitive at the millimetre, so a second, more accurate segmenter would buy no
registration accuracy while doubling the nomenclatures, install paths and failure
modes to validate.

LICENSING WARNING: TotalSegmentator has its own terms. Check them before any
commercial or clinical use -- Regix redistributes no weights.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.config import OrganBackend, OrganConfig
from regix.io.volume import Volume
from regix.logging_utils import get_logger
from regix.organs.labels import canonical_organ_name, resolve_targets
from regix.preprocess.geometry import resample_like

log = get_logger("organs.segmenter")


# --------------------------------------------------------------------------- #
@dataclass
class OrganSegmentation:
    """A label map plus its name dictionary, on the grid of the source volume."""

    labelmap: sitk.Image
    label_names: dict[int, str]
    backend: str
    source: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def organs(self) -> list[str]:
        return [self.label_names[k] for k in sorted(self.label_names)]

    def present_organs(self) -> list[str]:
        """Organs actually present (non-empty label): a model can return nothing."""
        arr = sitk.GetArrayViewFromImage(self.labelmap)
        present = set(np.unique(arr).tolist()) - {0}
        return [self.label_names[k] for k in sorted(self.label_names) if k in present]

    def label_of(self, organ: str) -> int | None:
        key = canonical_organ_name(organ)
        for value, name in self.label_names.items():
            if name == key:
                return int(value)
        return None

    def mask_for(self, organs: Sequence[str] | None = None, missing: str = "warn") -> sitk.Image:
        """Binary mask of the union of the requested organs (None = all labels)."""
        if organs is None:
            return sitk.Cast(sitk.Greater(self.labelmap, 0), sitk.sitkUInt8)
        wanted = resolve_targets(list(organs))
        labels = []
        for organ in wanted:
            lbl = self.label_of(organ)
            if lbl is None:
                message = f"organ '{organ}' missing from the {self.backend} segmentation"
                if missing == "raise":
                    raise KeyError(message)
                log.warning(message)
            else:
                labels.append(lbl)
        if not labels:
            raise ValueError(f"none of the organs {wanted} is available ({self.backend})")
        mask = sitk.Cast(sitk.Equal(self.labelmap, labels[0]), sitk.sitkUInt8)
        for lbl in labels[1:]:
            mask = sitk.Or(mask, sitk.Cast(sitk.Equal(self.labelmap, lbl), sitk.sitkUInt8))
        volume_ml = (
            float(sitk.GetArrayViewFromImage(mask).sum())
            * float(np.prod(self.labelmap.GetSpacing()))
            / 1000.0
        )
        if volume_ml < 0.5:
            log.warning("mask of %s is very small (%.2f mL): check the segmentation", wanted, volume_ml)
        return mask

    def resampled_to(self, reference: sitk.Image) -> OrganSegmentation:
        lm = resample_like(self.labelmap, reference, is_mask=True)
        return OrganSegmentation(lm, dict(self.label_names), self.backend, self.source, dict(self.info))


# --------------------------------------------------------------------------- #
class OrganSegmenter:
    """Common interface."""

    name = "base"

    def segment(self, volume: Volume) -> OrganSegmentation:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- cache ------------------------------------------------------------- #
    def _cache_key(self, volume: Volume, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(f"{self.name}|{extra}|{volume.size}|{volume.spacing}|{volume.origin}".encode())
        arr = sitk.GetArrayViewFromImage(volume.image)
        # cheap fingerprint: a few slices are enough to distinguish two volumes
        step = max(1, arr.shape[0] // 8)
        h.update(np.ascontiguousarray(arr[::step]).tobytes()[: 1 << 20])
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
class ExternalSegmenter(OrganSegmenter):
    """Pre-computed masks: a label map, a binary mask, or a directory of organs."""

    name = "external"

    def __init__(
        self,
        labelmap: str | Path | None = None,
        mask: str | Path | None = None,
        directory: str | Path | None = None,
        label_names: dict[int, str] | None = None,
    ):
        if not any([labelmap, mask, directory]):
            raise ValueError("ExternalSegmenter requires labelmap, mask or directory")
        self.labelmap_path = Path(labelmap) if labelmap else None
        self.mask_path = Path(mask) if mask else None
        self.directory = Path(directory) if directory else None
        self.label_names = label_names

    def segment(self, volume: Volume) -> OrganSegmentation:
        if self.labelmap_path is not None:
            lm = sitk.Cast(sitk.ReadImage(str(self.labelmap_path)), sitk.sitkUInt16)
            present = sorted(set(np.unique(sitk.GetArrayFromImage(lm)).tolist()) - {0})
            names = self.label_names or _sidecar_label_names(self.labelmap_path)
            if names is None:
                # We assume NO nomenclature: mapping 1 -> 'spleen' because some tool
                # numbers its labels that way would produce a wrong organ mask with no
                # visible sign whatsoever. Neutral names plus a warning are safer.
                names = {int(v): f"label_{int(v)}" for v in present}
                log.warning(
                    "label map %s has no nomenclature (%d labels): named label_N. "
                    "Targeting organs by name will not work. Provide organs.label_names, "
                    "or a sidecar file '%s.labels.json'.",
                    self.labelmap_path.name,
                    len(present),
                    self.labelmap_path.name,
                )
            source = str(self.labelmap_path)
        elif self.mask_path is not None:
            binary = sitk.ReadImage(str(self.mask_path))
            lm = sitk.Cast(sitk.Greater(binary, 0), sitk.sitkUInt16)
            names = self.label_names or {1: "target"}
            source = str(self.mask_path)
        else:
            lm, names = self._from_directory(self.directory)  # type: ignore[arg-type]
            source = str(self.directory)

        seg = OrganSegmentation(lm, names, self.name, source)
        if not _same_grid(lm, volume.image):
            log.info("external masks on a different grid: resampling onto the volume")
            seg = seg.resampled_to(volume.image)
        return seg

    @staticmethod
    def _from_directory(directory: Path) -> tuple[sitk.Image, dict[int, str]]:
        """Aggregate a directory of binary masks: the file name is authoritative."""
        files = sorted([p for p in directory.glob("*.nii.gz")] + [p for p in directory.glob("*.nii")])
        if not files:
            raise FileNotFoundError(f"no .nii(.gz) mask in {directory}")
        labelmap: sitk.Image | None = None
        names: dict[int, str] = {}
        for idx, f in enumerate(files, start=1):
            m = sitk.Cast(sitk.Greater(sitk.ReadImage(str(f)), 0), sitk.sitkUInt16)
            if labelmap is None:
                labelmap = sitk.Image(m.GetSize(), sitk.sitkUInt16)
                labelmap.CopyInformation(m)
            elif not _same_grid(m, labelmap):
                m = resample_like(m, labelmap, is_mask=True)
            # first come, first served: an already-placed organ is not overwritten
            free = sitk.Equal(labelmap, 0)
            labelmap = sitk.Add(
                labelmap, sitk.Cast(sitk.And(m, sitk.Cast(free, sitk.sitkUInt16)), sitk.sitkUInt16) * idx
            )
            names[idx] = canonical_organ_name(f.name)
        assert labelmap is not None
        log.info("aggregated %d masks from %s", len(names), directory)
        return labelmap, names


# --------------------------------------------------------------------------- #
class TotalSegmentatorSegmenter(OrganSegmenter):
    """TotalSegmentator (Python API when available, otherwise the CLI)."""

    name = "totalsegmentator"

    def __init__(
        self,
        task: str = "total",
        fast: bool = True,
        roi_subset: Sequence[str] | None = None,
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ):
        self.task = task
        self.fast = fast
        self.roi_subset = list(roi_subset) if roi_subset else None
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def segment(self, volume: Volume) -> OrganSegmentation:
        cache = self._cached(volume)
        if cache is not None:
            return cache
        with tempfile.TemporaryDirectory(prefix="regix_ts_") as tmp:
            tmpdir = Path(tmp)
            ct = tmpdir / "input.nii.gz"
            sitk.WriteImage(volume.image, str(ct), True)
            out = tmpdir / "seg"
            self._run(ct, out)
            lm, names = ExternalSegmenter._from_directory(out)
            seg = OrganSegmentation(
                lm,
                names,
                self.name,
                source=f"totalsegmentator:{self.task}",
                info={"task": self.task, "fast": self.fast},
            )
            if not _same_grid(lm, volume.image):
                seg = seg.resampled_to(volume.image)
            self._store(volume, seg)
            return seg

    def _run(self, ct: Path, out: Path) -> None:
        try:
            from totalsegmentator.python_api import totalsegmentator

            log.info("TotalSegmentator (Python API), task=%s, fast=%s", self.task, self.fast)
            totalsegmentator(
                input=str(ct),
                output=str(out),
                task=self.task,
                fast=self.fast,
                roi_subset=self.roi_subset,
                device=self.device,
                quiet=True,
            )
            return
        except ImportError:
            pass
        exe = shutil.which("TotalSegmentator")
        if exe is None:
            raise RuntimeError(
                "TotalSegmentator not found (neither Python module nor executable). "
                "pip install TotalSegmentator, or use organs.backend=external."
            )
        cmd = [exe, "-i", str(ct), "-o", str(out), "--task", self.task, "--quiet"]
        if self.fast:
            cmd.append("--fast")
        if self.roi_subset:
            cmd += ["--roi_subset", *self.roi_subset]
        log.info("TotalSegmentator (CLI): %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # -- disk cache -------------------------------------------------------- #
    def _cache_path(self, volume: Volume) -> Path | None:
        if self.cache_dir is None:
            return None
        key = self._cache_key(volume, extra=f"{self.task}|{self.fast}|{self.roi_subset}")
        return self.cache_dir / f"ts_{key}.nii.gz"

    def _cached(self, volume: Volume) -> OrganSegmentation | None:
        p = self._cache_path(volume)
        if p is None or not p.exists():
            return None
        names_file = p.with_suffix("").with_suffix(".labels.txt")
        if not names_file.exists():
            return None
        names = {}
        for line in names_file.read_text(encoding="utf-8").splitlines():
            idx, _, name = line.partition(" ")
            names[int(idx)] = name
        log.info("segmentation restored from cache: %s", p.name)
        return OrganSegmentation(sitk.ReadImage(str(p)), names, self.name, source=str(p))

    def _store(self, volume: Volume, seg: OrganSegmentation) -> None:
        p = self._cache_path(volume)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(seg.labelmap, str(p), True)
        p.with_suffix("").with_suffix(".labels.txt").write_text(
            "\n".join(f"{k} {v}" for k, v in sorted(seg.label_names.items())), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
def _sidecar_label_names(labelmap_path: Path) -> dict[int, str] | None:
    """Look for a nomenclature next to the label map.

    Accepted formats, in order: ``<name>.labels.json`` ({"1": "liver"}),
    ``<name>.labels.txt`` (lines ``1 liver``), and a ``labels.json`` in the same
    directory. That is what most segmentation tools produce, and what
    ``regix segment`` itself writes.
    """
    stem = labelmap_path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates = [
        labelmap_path.parent / f"{stem}.labels.json",
        labelmap_path.parent / f"{stem}.labels.txt",
        labelmap_path.parent / "labels.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".json":
                import json

                raw = json.loads(candidate.read_text(encoding="utf-8"))
                names = {int(k): canonical_organ_name(str(v)) for k, v in raw.items()}
            else:
                names = {}
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    index, _, name = line.strip().partition(" ")
                    names[int(index)] = canonical_organ_name(name)
        except (ValueError, OSError) as exc:
            log.warning("nomenclature %s is unreadable: %s", candidate.name, exc)
            continue
        if names:
            log.info("nomenclature read from %s (%d organs)", candidate.name, len(names))
            return names
    return None


# --------------------------------------------------------------------------- #
def build_segmenter(
    config: OrganConfig, side: str = "fixed", cache_dir: str | Path | None = None
) -> OrganSegmenter | None:
    """Instantiate the segmenter described by the configuration, for one side."""
    backend = config.backend
    if backend is OrganBackend.NONE:
        return None
    if backend is OrganBackend.EXTERNAL:
        labelmap = config.fixed_labelmap if side == "fixed" else config.moving_labelmap
        mask = config.fixed_mask if side == "fixed" else config.moving_mask
        if labelmap is None and mask is None:
            return None
        return ExternalSegmenter(labelmap=labelmap, mask=mask, label_names=config.label_names)
    if backend is OrganBackend.TOTALSEGMENTATOR:
        return TotalSegmentatorSegmenter(
            roi_subset=resolve_targets(config.targets) or None,
            device=config.device,
            cache_dir=cache_dir,
        )
    raise ValueError(f"unhandled organ backend: {backend}")


def _same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )
