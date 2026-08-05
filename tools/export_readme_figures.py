"""Regenerate the images embedded in the README.

The README used to redraw its illustrations by hand -- an ASCII table pretending to
be ``regix doctor``, no picture at all of the QC report. Hand-drawn output drifts
away from the real thing without anything turning red. Everything under
``docs/images/`` is therefore produced from an actual run:

* the QC figures are extracted from the ``report.html`` of a completed
  registration (they are already embedded there as base64 PNGs, so no browser and
  no screenshot tool is needed);
* ``doctor.svg`` is the genuine ``regix doctor`` table, rendered by rich itself.

Usage::

    python tools/export_readme_figures.py e2e_out            # after a local run
    python tools/export_readme_figures.py ci_out -o docs/images

CI runs this on the phantom registration it already performs, so a change that
breaks the figures breaks the build.
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

# Runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Figures worth putting in a README, and the file name each one gets.
FIGURES = {
    "Overlay before / after": "qc-overlay.png",
    "Checkerboard": "qc-checkerboard.png",
    "Organ contours": "qc-contours.png",
}

_IMG = re.compile(
    r'<img\s+src="data:image/png;base64,(?P<data>[^"]+)"\s+alt="(?P<alt>[^"]*)"',
    re.IGNORECASE,
)


def _shrink(png: bytes, max_width: int) -> bytes:
    """Downscale a figure so the repository does not carry megabyte screenshots.

    The report keeps the full-resolution version; a README only ever renders these
    at about 900 px anyway.
    """
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(png)).convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.LANCZOS)
    # A greyscale slice under a hot colormap needs nowhere near 24-bit colour, and
    # PNG compresses noisy true-colour medical images very badly (megabytes).
    buffer = BytesIO()
    image.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG).save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def export_report_figures(report: Path, out_dir: Path, max_width: int = 950) -> list[Path]:
    """Extract the base64 figures of a QC report into standalone PNG files."""
    html = report.read_text(encoding="utf-8", errors="replace")
    written: list[Path] = []
    for match in _IMG.finditer(html):
        name = FIGURES.get(match.group("alt"))
        if name is None:
            continue
        target = out_dir / name
        target.write_bytes(_shrink(base64.b64decode(match.group("data")), max_width))
        written.append(target)
    return written


def export_doctor_svg(out_dir: Path) -> Path:
    """Render `regix doctor` to SVG, through the same code path the CLI uses.

    The command writes to the module-level ``console``; swapping in a recording one
    captures the real table rather than a re-implementation of it.
    """
    from rich.console import Console

    from regix import cli

    original = cli.console
    recorder = Console(record=True, width=96, force_terminal=True)
    cli.console = recorder
    try:
        cli.doctor()
    except SystemExit:
        # `doctor` exits non-zero when the engine is missing; the table is already
        # rendered by then, and that state is worth illustrating too.
        pass
    finally:
        cli.console = original

    target = out_dir / "doctor.svg"
    target.write_text(recorder.export_svg(title="regix doctor"), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path("e2e_out"),
        help="Output directory of a completed registration (must contain report.html).",
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("docs/images"))
    parser.add_argument(
        "--skip-doctor", action="store_true", help="Only extract the QC figures."
    )
    parser.add_argument(
        "--max-width", type=int, default=950, help="Downscale figures wider than this."
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = args.run_dir / "report.html"
    if not report.exists():
        parser.error(f"{report} not found: run a registration first (see the README).")

    written = export_report_figures(report, args.out_dir, max_width=args.max_width)
    if not written:
        parser.error(f"no known figure found in {report}: was QC enabled for that run?")
    if not args.skip_doctor:
        written.append(export_doctor_svg(args.out_dir))

    for path in written:
        print(f"{path}  ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
