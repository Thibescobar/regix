"""Quality-control report: a single self-contained HTML file.

Field constraints that dictate its shape:

* **one file**, images embedded as base64: it must be attachable to an email,
  archivable, and openable on a machine with no network;
* **before / after side by side**: that is the only thing a radiologist actually
  looks at. Numbers come second;
* **checkerboard and contours**: two complementary reading modes -- the
  checkerboard exposes discontinuities, contours expose organ boundaries;
* **verdict at the top**, in colour, with the reason. No scrolling required to
  know whether the case is usable.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from regix.logging_utils import get_logger

log = get_logger("qc.report")

_STATUS_COLORS = {"PASS": "#1a7f37", "WARN": "#9a6700", "FAIL": "#b3261e"}


# --------------------------------------------------------------------------- #
# Image rendering
# --------------------------------------------------------------------------- #
def _window(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if hi - lo < 1e-8:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize(arr: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _slice_positions(size: int, n: int) -> list[int]:
    if n <= 1:
        return [size // 2]
    fractions = np.linspace(0.35, 0.65, n)
    return [int(round(f * (size - 1))) for f in fractions]


def _extract_planes(arr: np.ndarray, index: tuple[int, int, int]) -> list[np.ndarray]:
    """(axial, coronal, sagittal) from a (z, y, x) array."""
    z, y, x = index
    return [arr[z, :, :], arr[:, y, :], arr[:, :, x]]


def _aspects(spacing_xyz: Sequence[float]) -> list[float]:
    sx, sy, sz = (float(v) for v in spacing_xyz)
    # axial: rows = y, columns = x; coronal: rows = z; sagittal: rows = z
    return [sy / sx, sz / sx, sz / sy]


def _figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def overlay_figure(
    fixed: sitk.Image,
    moving_before: sitk.Image | None,
    moving_after: sitk.Image,
    mask: sitk.Image | None = None,
    n_slices: int = 3,
    title: str = "",
) -> str:
    """Fixed (grey) / moving (hot) overlay, before and after registration."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    f_arr = sitk.GetArrayFromImage(sitk.Cast(fixed, sitk.sitkFloat32))
    a_arr = sitk.GetArrayFromImage(sitk.Cast(moving_after, sitk.sitkFloat32))
    b_arr = (
        sitk.GetArrayFromImage(sitk.Cast(moving_before, sitk.sitkFloat32))
        if moving_before is not None
        else None
    )

    f_bounds, a_bounds = _window(f_arr), _window(a_arr)
    b_bounds = _window(b_arr) if b_arr is not None else None

    centre = _centre_index(fixed, mask)
    positions = [
        (_slice_positions(f_arr.shape[0], n_slices)[k], centre[1], centre[2]) for k in range(n_slices)
    ]
    aspects = _aspects(fixed.GetSpacing())
    columns = 2 if b_arr is not None else 1
    plane_names = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(
        3, columns * n_slices, figsize=(3.1 * columns * n_slices, 9.0), facecolor="white"
    )
    axes = np.atleast_2d(axes)
    for plane in range(3):
        col = 0
        for slice_idx, index in enumerate(positions):
            for label, arr, bounds in _series(b_arr, b_bounds, a_arr, a_bounds):
                ax = axes[plane, col]
                base = _normalize(_extract_planes(f_arr, index)[plane], f_bounds)
                over = _normalize(_extract_planes(arr, index)[plane], bounds)
                ax.imshow(base, cmap="gray", origin="lower", aspect=aspects[plane], vmin=0, vmax=1)
                ax.imshow(over, cmap="hot", origin="lower", aspect=aspects[plane], alpha=0.45, vmin=0, vmax=1)
                if plane == 0:
                    ax.set_title(f"{label} · slice {slice_idx + 1}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(plane_names[plane], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                col += 1
    fig.suptitle(title or "Overlay: fixed (grey) / moving (hot)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_to_base64(fig)


def _series(b_arr, b_bounds, a_arr, a_bounds):
    if b_arr is not None:
        yield "before", b_arr, b_bounds
    yield "after", a_arr, a_bounds


def checkerboard_figure(
    fixed: sitk.Image, moving_after: sitk.Image, mask: sitk.Image | None = None, tiles: int = 8
) -> str:
    """Checkerboard: discontinuities at tile borders betray residual misalignment."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    f_arr = sitk.GetArrayFromImage(sitk.Cast(fixed, sitk.sitkFloat32))
    a_arr = sitk.GetArrayFromImage(sitk.Cast(moving_after, sitk.sitkFloat32))
    f_norm = _normalize(f_arr, _window(f_arr))
    a_norm = _normalize(a_arr, _window(a_arr))
    index = _centre_index(fixed, mask)
    aspects = _aspects(fixed.GetSpacing())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="white")
    for plane, (ax, name) in enumerate(zip(axes, ["axial", "coronal", "sagittal"], strict=False)):
        f_plane = _extract_planes(f_norm, index)[plane]
        a_plane = _extract_planes(a_norm, index)[plane]
        rows, cols = f_plane.shape
        rr = (np.arange(rows)[:, None] // max(1, rows // tiles)) % 2
        cc = (np.arange(cols)[None, :] // max(1, cols // tiles)) % 2
        pattern = (rr + cc) % 2
        blended = np.where(pattern == 0, f_plane, a_plane)
        ax.imshow(blended, cmap="gray", origin="lower", aspect=aspects[plane], vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Checkerboard: fixed / registered moving", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _figure_to_base64(fig)


def contour_figure(
    fixed: sitk.Image,
    fixed_labelmap: sitk.Image | None,
    warped_labelmap: sitk.Image | None,
    mask: sitk.Image | None = None,
) -> str | None:
    """Organ contours: reference in green, registered in magenta."""
    if fixed_labelmap is None or warped_labelmap is None:
        return None
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    f_arr = sitk.GetArrayFromImage(sitk.Cast(fixed, sitk.sitkFloat32))
    ref = sitk.GetArrayFromImage(fixed_labelmap)
    war = sitk.GetArrayFromImage(warped_labelmap)
    if ref.shape != f_arr.shape or war.shape != f_arr.shape:
        log.debug("skipping contours: incompatible geometries")
        return None
    base = _normalize(f_arr, _window(f_arr))
    index = _centre_index(
        fixed, mask if mask is not None else sitk.Cast(sitk.Greater(fixed_labelmap, 0), sitk.sitkUInt8)
    )
    aspects = _aspects(fixed.GetSpacing())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="white")
    for plane, (ax, name) in enumerate(zip(axes, ["axial", "coronal", "sagittal"], strict=False)):
        ax.imshow(
            _extract_planes(base, index)[plane], cmap="gray", origin="lower",
            aspect=aspects[plane], vmin=0, vmax=1,
        )
        for arr, color in ((ref, "#2ecc71"), (war, "#ff2ec4")):
            plane_labels = _extract_planes(arr, index)[plane]
            if np.any(plane_labels > 0):
                ax.contour(plane_labels > 0, levels=[0.5], colors=[color], linewidths=1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Contours: fixed organs (green) vs registered organs (magenta)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _figure_to_base64(fig)


#: Below this spread of det(J), the transform is effectively linear and the map
#: carries no information (a linear transform has a constant Jacobian).
_JACOBIAN_MIN_SPREAD = 0.02


def jacobian_figure(
    displacement_field: sitk.Image,
    mask: sitk.Image | None = None,
    stats: dict[str, Any] | None = None,
) -> str | None:
    """Map of the Jacobian determinant, on a scale adapted to the actual data.

    Returns None -- deliberately -- when the transform is linear. A linear
    transform has a **constant** Jacobian by construction, so the map is a flat
    uniform colour that tells the reader nothing while looking like a rendering
    bug. The constant value belongs in the table, not in a figure.

    When the transform really is deformable, the colour scale is centred on 1
    and derived from the data percentiles: a fixed [0, 2] scale hides a genuine
    but subtle 0.9-1.1 variation, which is exactly the range worth seeing.
    """
    try:
        jac = sitk.DisplacementFieldJacobianDeterminant(
            sitk.Cast(displacement_field, sitk.sitkVectorFloat64)
        )
    except Exception as exc:  # pragma: no cover
        log.debug("Jacobian map unavailable: %s", exc)
        return None

    arr = sitk.GetArrayFromImage(jac)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    p1, p99 = (float(v) for v in np.percentile(finite, [1, 99]))
    spread = p99 - p1
    if spread < _JACOBIAN_MIN_SPREAD:
        log.info(
            "Jacobian map skipped: det(J) is constant (%.4f). The transform is linear, "
            "so the map would be a flat colour; the value is reported in the metrics table.",
            float(np.median(finite)),
        )
        return None

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    # Scale centred on 1 (volume preserved), widened to the observed variation.
    half = max(0.1, 1.2 * max(abs(p99 - 1.0), abs(1.0 - p1)))
    vmin, vmax = max(0.0, 1.0 - half), 1.0 + half

    index = _centre_index(jac, mask)
    aspects = _aspects(jac.GetSpacing())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="white")
    image = None
    for plane, (ax, name) in enumerate(zip(axes, ["axial", "coronal", "sagittal"], strict=False)):
        image = ax.imshow(
            _extract_planes(arr, index)[plane],
            cmap="coolwarm",
            origin="lower",
            aspect=aspects[plane],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="det(J)")
    folding = int((finite <= 0).sum())
    subtitle = f" — {folding} folded voxels" if folding else " — no folding"
    fig.suptitle(
        f"Jacobian determinant (1 = volume preserved, < 0 = folding){subtitle}", fontsize=11
    )
    return _figure_to_base64(fig)


def _centre_index(image: sitk.Image, mask: sitk.Image | None) -> tuple[int, int, int]:
    """Index of interest in numpy order (z, y, x): mask centroid, else grid centre."""
    size = image.GetSize()
    default = (size[2] // 2, size[1] // 2, size[0] // 2)
    if mask is None:
        return default
    arr = sitk.GetArrayViewFromImage(mask)
    if arr.shape != (size[2], size[1], size[0]) or not np.any(arr > 0):
        return default
    idx = np.argwhere(arr > 0).mean(axis=0)
    return (int(idx[0]), int(idx[1]), int(idx[2]))


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in row) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if np.isfinite(value) else "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "n/a"
    return html.escape(str(value))


def _status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#57606a")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;border-radius:12px;'
        f'font-weight:600;letter-spacing:0.04em">{html.escape(status)}</span>'
    )


def _kv_rows(data: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        rows.append([key, value])
    return rows


def build_html_report(path: str | Path, context: dict[str, Any]) -> Path:
    """Write the report. ``context`` is produced by the pipeline (see ``pipeline.py``)."""
    status = context.get("qc", {}).get("status", "WARN")
    title = context.get("title", "Regix registration report")
    sections: list[str] = []

    # -- verdict --------------------------------------------------------- #
    checks = context.get("qc", {}).get("checks", [])
    verdict_rows = [
        [c["name"], c["status"], c.get("measured"), c.get("threshold"), c.get("message", "")]
        for c in checks
    ]
    sections.append(
        f"<h2>Verdict {_status_badge(status)}</h2>"
        + _table(verdict_rows, ["check", "status", "measured", "threshold", "comment"])
    )

    # -- figures ----------------------------------------------------------- #
    figures = context.get("figures", {})
    for key, caption in (
        ("overlay", "Overlay before / after"),
        ("checkerboard", "Checkerboard"),
        ("contours", "Organ contours"),
        ("jacobian", "Jacobian determinant"),
    ):
        if figures.get(key):
            sections.append(
                f"<h2>{html.escape(caption)}</h2>"
                f'<img src="{figures[key]}" alt="{html.escape(caption)}" style="max-width:100%">'
            )

    # -- metrics ----------------------------------------------------------- #
    similarity = context.get("similarity") or {}
    if similarity:
        sections.append(
            "<h2>Intensity similarity</h2>" + _table(_kv_rows(similarity), ["measure", "value"])
        )

    organs = context.get("organ_overlap") or {}
    if organs:
        rows = [[organ, m.get("dice"), m.get("hd95_mm"), m.get("msd_mm")] for organ, m in organs.items()]
        sections.append(
            "<h2>Per-organ overlap</h2>"
            + _table(rows, ["organ", "Dice", "HD95 (mm)", "mean surface distance (mm)"])
        )

    landmarks = context.get("landmarks") or {}
    if landmarks:
        sections.append(
            "<h2>Landmarks</h2>"
            + _table(
                _kv_rows({k: v for k, v in landmarks.items() if k != "per_landmark_mm"}),
                ["measure", "value"],
            )
        )

    jac = context.get("jacobian") or {}
    if jac.get("available"):
        note = ""
        if not figures.get("jacobian"):
            note = (
                "<p style='font-size:13px;color:#57606a'>No map shown: the Jacobian "
                "determinant is constant, which is expected for a linear transform.</p>"
            )
        sections.append(
            "<h2>Deformation plausibility</h2>" + note + _table(_kv_rows(jac), ["measure", "value"])
        )

    # -- stages ------------------------------------------------------------ #
    stages = context.get("stages") or []
    if stages:
        rows = [
            [
                s.get("stage"),
                s.get("transform"),
                s.get("metric"),
                s.get("channels"),
                s.get("resolutions"),
                s.get("masked"),
                s.get("final_metric"),
                s.get("seconds"),
            ]
            for s in stages
        ]
        headers = [
            "stage", "transform", "metric", "channels", "resolutions", "masked",
            "final metric", "seconds",
        ]
        sections.append("<h2>Registration stages</h2>" + _table(rows, headers))

    init = context.get("initialization") or {}
    if init:
        candidates = init.get("candidates") or []
        if candidates:
            rows = [
                [c.get("name"), c.get("score"), c.get("translation_mm"), c.get("rotation_deg")]
                for c in candidates
            ]
            sections.append(
                f"<h2>Initialization (mode: {html.escape(str(init.get('mode')))}, "
                f"selected: {html.escape(str(init.get('chosen')))})</h2>"
                + _table(rows, ["candidate", "score", "translation (mm)", "rotation (deg)"])
            )

    # -- inputs / configuration -------------------------------------------- #
    for key, caption in (
        ("inputs", "Inputs"),
        ("configuration", "Effective configuration"),
        ("environment", "Environment"),
    ):
        data = context.get(key)
        if data:
            sections.append(f"<h2>{caption}</h2>" + _table(_kv_rows(data), ["key", "value"]))

    warnings = context.get("warnings") or []
    if warnings:
        items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        sections.append(f"<h2>Warnings ({len(warnings)})</h2><ul>{items}</ul>")

    from regix import DISCLAIMER

    document = _HTML_TEMPLATE.format(
        title=html.escape(title),
        subtitle=html.escape(context.get("subtitle", "")),
        generated=html.escape(str(context.get("generated_at", ""))),
        disclaimer=html.escape(DISCLAIMER),
        body="\n".join(sections),
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(document, encoding="utf-8")
    log.info("QC report: %s", p)
    return p


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1180px; padding: 24px; color: #1f2328; background: #fff; }}
  header {{ border-bottom: 2px solid #d0d7de; padding-bottom: 12px; margin-bottom: 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 32px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #d0d7de; }}
  .meta {{ color: #57606a; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 8px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 9px; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fbfcfd; }}
  img {{ border: 1px solid #d0d7de; border-radius: 6px; margin-top: 8px; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #d0d7de;
            color: #57606a; font-size: 12px; }}
  .disclaimer {{ background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px;
                 padding: 10px 14px; font-size: 13px; margin-bottom: 20px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">{subtitle} — generated {generated}</div>
</header>
<div class="disclaimer">
  <strong>&#9888; Not a medical device.</strong> {disclaimer}
  This report does not replace visual verification of the images.
</div>
{body}
<footer>
  Regix — multimodal / multi-organ registration · elastix (engine) · anatomix features (MIT)
  · optional organ segmentation. The parameter files and run manifest shipped alongside
  this report allow the computation to be replayed exactly.
</footer>
</body>
</html>
"""
