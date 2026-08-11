#!/usr/bin/env python3
"""Compare standalone, embedded-full and transform-only execution.

Each measurement runs in a fresh process so peak native memory remains comparable.
The workspace is sampled while Regix runs, which captures temporary elastix I/O as
well as the files that remain at the end.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from regix.config import load_preset
from regix.io.volume import Volume
from regix.pipeline import RegistrationPipeline

MODES = ("standalone", "embedded-full", "embedded-transform")


def _ellipsoid(shape, centre, radii):
    zz, yy, xx = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    return (
        ((zz - centre[0]) / radii[0]) ** 2
        + ((yy - centre[1]) / radii[1]) ** 2
        + ((xx - centre[2]) / radii[2]) ** 2
    ) <= 1.0


def _pair() -> tuple[Volume, Volume]:
    shape = (36, 48, 48)
    body = _ellipsoid(shape, (18, 24, 24), (15, 19, 21))
    organ = _ellipsoid(shape, (16, 21, 18), (7, 8, 7))
    array = np.full(shape, -1000.0, dtype=np.float32)
    array[body] = 30.0
    array[organ] = 110.0
    fixed = sitk.GetImageFromArray(array)
    fixed.SetSpacing((2.0, 2.0, 2.5))
    fixed.SetOrigin((-40.0, -60.0, 30.0))

    centre = fixed.TransformContinuousIndexToPhysicalPoint(
        [float((size - 1) / 2) for size in fixed.GetSize()]
    )
    truth = sitk.Euler3DTransform()
    truth.SetCenter(centre)
    truth.SetRotation(*np.radians((2.0, -1.0, 2.0)))
    truth.SetTranslation((3.0, -2.0, 3.0))
    moving = sitk.Resample(
        fixed,
        fixed,
        truth.GetInverse(),
        sitk.sitkLinear,
        -1000.0,
        sitk.sitkFloat32,
    )
    return Volume(fixed, modality="CT"), Volume(moving, modality="CT")


def _config(output_dir: Path):
    return load_preset("base").with_overrides(
        preprocess={"working_spacing_mm": 3.0},
        features={"enabled": False},
        stages=[
            {
                "type": "rigid",
                "n_resolutions": 2,
                "max_iterations": 80,
                "n_spatial_samples": 2048,
            }
        ],
        qc={
            "enabled": True,
            "report_html": False,
            "jacobian": False,
            "max_similarity_samples": 100_000,
        },
        output={"dir": str(output_dir), "overwrite": True, "compress": False},
        runtime={"log_level": "ERROR", "threads": 1},
    )


def _inventory(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
                count += 1
        except FileNotFoundError:
            # Temporary stage files can disappear between the directory walk and stat.
            continue
    return total, count


class _WorkspaceMonitor:
    def __init__(self, root: Path):
        self.root = root
        self.peak_bytes = 0
        self.peak_files = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(0.01):
            size, count = _inventory(self.root)
            self.peak_bytes = max(self.peak_bytes, size)
            self.peak_files = max(self.peak_files, count)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        size, count = _inventory(self.root)
        self.peak_bytes = max(self.peak_bytes, size)
        self.peak_files = max(self.peak_files, count)


def _worker(mode: str, workspace: Path) -> dict[str, object]:
    workspace.mkdir(parents=True, exist_ok=True)
    fixed, moving = _pair()
    pipeline = RegistrationPipeline(_config(workspace / "out"))

    started = time.perf_counter()
    with _WorkspaceMonitor(workspace) as monitor:
        if mode == "standalone":
            result = pipeline.run(fixed, moving, workspace / "out")
        elif mode == "embedded-full":
            result = pipeline.compute(fixed, moving, qc=True, work_dir=workspace)
        else:
            result = pipeline.compute(
                fixed,
                moving,
                registered_image=False,
                qc=False,
                work_dir=workspace,
            )
    seconds = time.perf_counter() - started
    persistent_bytes, persistent_files = _inventory(workspace)
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = peak_rss / (1024 * 1024) if sys.platform == "darwin" else peak_rss / 1024
    return {
        "mode": mode,
        "seconds": round(seconds, 4),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "peak_workspace_bytes": monitor.peak_bytes,
        "peak_workspace_files": monitor.peak_files,
        "persistent_bytes": persistent_bytes,
        "persistent_files": persistent_files,
        "status": result.status,
    }


def _measure(repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="regix-benchmark-") as temporary:
        root = Path(temporary)
        for mode in MODES:
            samples = []
            for repeat in range(repeats):
                workspace = root / f"{mode}-{repeat}"
                process = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--worker",
                        mode,
                        "--workspace",
                        str(workspace),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                samples.append(json.loads(process.stdout.strip().splitlines()[-1]))
            row = {"mode": mode, "repeats": repeats}
            for key in (
                "seconds",
                "peak_rss_mb",
                "peak_workspace_bytes",
                "peak_workspace_files",
                "persistent_bytes",
                "persistent_files",
            ):
                row[key] = round(float(statistics.median(sample[key] for sample in samples)), 4)
            rows.append(row)
    return rows


def _print_table(rows: list[dict[str, object]]) -> None:
    print("mode                 time(s)  RSS(MB)  peak I/O(KB/files)  persistent(KB/files)")
    for row in rows:
        print(
            f"{row['mode']:<21} {row['seconds']:>7.2f}  {row['peak_rss_mb']:>7.1f}  "
            f"{row['peak_workspace_bytes'] / 1024:>10.1f}/{row['peak_workspace_files']:<5.0f}  "
            f"{row['persistent_bytes'] / 1024:>10.1f}/{row['persistent_files']:<5.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.worker:
        if args.workspace is None:
            parser.error("--worker requires --workspace")
        print(json.dumps(_worker(args.worker, args.workspace), sort_keys=True))
        return

    rows = _measure(args.repeats)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
