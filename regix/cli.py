"""Command-line interface.

    regix doctor                      # what is installed and what is missing
    regix presets                     # ready-to-use scenarios
    regix inspect DICOM_DIR           # series inventory, geometry, pitfalls
    regix register FIXED MOVING -o OUT --preset ct_mr_abdomen --organ liver
    regix batch pairs.csv -o OUT --preset ct_ct_liver_followup
    regix apply transform/final_transform.tfm moving.nii.gz --reference fixed.nii.gz
    regix segment ct.nii.gz -o masks/ --backend totalsegmentator

Any configuration option can be overridden without editing a YAML file:

    regix register f.nii.gz m.nii.gz --set preprocess.working_spacing_mm=1.5 \\
        --set stages.0.max_iterations=1000
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from regix import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Regix - multimodal / multi-organ registration (research software, not a medical device).",
)
console = Console()


# --------------------------------------------------------------------------- #
def _load_config(preset: str | None, config_file: Path | None):
    from regix.config import load_preset

    if config_file is not None:
        return load_preset(config_file)
    if preset is not None:
        return load_preset(preset)
    return load_preset("base")


def _apply_sets(config, assignments: list[str]):
    """Apply ``dotted.path=value`` overrides (values parsed as YAML)."""
    import yaml

    from regix.config import RegistrationConfig

    if not assignments:
        return config
    data = config.model_dump(mode="python")
    for item in assignments:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got: {item!r}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node: Any = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else node.setdefault(part, {})
        last = parts[-1]
        if last.isdigit():
            node[int(last)] = value
        else:
            node[last] = value
    return RegistrationConfig.model_validate(data)


def _echo_result(result) -> None:
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(result.status, "white")
    console.print(f"\n[bold {color}]{result.status}[/] — {result.seconds:.1f} s")
    console.print(result.summary())
    if result.outputs:
        console.print("\n[bold]Outputs[/]")
        for key, path in result.outputs.items():
            console.print(f"  {key:20s} {path}")


# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the Regix version."""
    console.print(f"regix {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: engine, optional dependencies, GPU."""
    from regix.logging_utils import environment_report
    from regix.registration.itk_bridge import engine_available

    report = environment_report()
    table = Table(title="Regix environment", show_lines=False)
    table.add_column("component")
    table.add_column("state")
    table.add_column("consequence if missing")

    has_elastix, engine_detail = engine_available()
    table.add_row(
        "itk-elastix (engine)",
        f"[green]{engine_detail}[/]" if has_elastix else "[red]missing[/]",
        "blocking: this is the registration engine (pip install itk-elastix)",
    )
    table.add_row(
        "SimpleITK",
        f"{report['simpleitk']}",
        "blocking: DICOM I/O, morphology, transforms",
    )
    table.add_row("numpy", f"{report['numpy']}", "blocking")

    torch_ok = report["torch"] is not None
    table.add_row(
        "torch",
        f"[green]{report['torch']}[/]" if torch_ok else "[yellow]missing[/]",
        "no anatomix features and no GPU deformable stage (CPU MIND-SSC fallback)",
    )
    table.add_row(
        "CUDA GPU",
        f"[green]{report['cuda_device']}[/]" if report["cuda_available"] else "[yellow]no[/]",
        "anatomix features are very slow on CPU",
    )
    try:
        import anatomix  # noqa: F401

        anatomix_state = "[green]installed[/]"
    except ImportError:
        anatomix_state = "[yellow]missing[/]"
    table.add_row(
        "anatomix",
        anatomix_state,
        "feature-based multimodal registration unavailable (MI remains usable)",
    )
    table.add_row("monai", f"{report['monai'] or '[yellow]missing[/]'}", "sliding-window inference")
    try:
        import totalsegmentator  # noqa: F401

        ts_state = "[green]installed[/]"
    except ImportError:
        ts_state = "[yellow]missing[/]"
    table.add_row("TotalSegmentator", ts_state, "automatic organ segmentation")
    try:
        import pydicom  # noqa: F401

        dcm_state = "[green]installed[/]"
    except ImportError:
        dcm_state = "[red]missing[/]"
    table.add_row("pydicom", dcm_state, "tag reading and DICOM exports (SRO, derived series)")

    console.print(table)
    console.print(
        "\n[dim]Regix is research software. The anatomix and TotalSegmentator weights "
        "have their own licences: review them before any clinical or commercial use.[/]"
    )
    if not has_elastix:
        raise typer.Exit(code=1)


@app.command()
def presets(
    name: Optional[str] = typer.Argument(None, help="Print the full YAML of one preset."),
) -> None:
    """List the bundled presets (or print one)."""
    from regix.config import available_presets, load_preset

    if name:
        console.print(load_preset(name).to_yaml())
        return
    table = Table(title="Regix presets")
    table.add_column("name")
    table.add_column("pair")
    table.add_column("stages")
    table.add_column("description")
    for preset_name in available_presets():
        cfg = load_preset(preset_name)
        table.add_row(
            preset_name,
            f"{cfg.moving_modality or '?'} -> {cfg.fixed_modality or '?'}",
            " + ".join(s.type.value for s in cfg.stages),
            (cfg.description or "").strip().split("\n")[0][:70],
        )
    console.print(table)


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="Image file or DICOM directory."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Inventory the DICOM series, or describe a volume (geometry, intensities, pitfalls)."""
    from regix.io.dicom import list_series
    from regix.io.volume import load_volume

    if path.is_dir():
        series = list_series(path)
        if not series:
            console.print(f"[red]no DICOM series in {path}[/]")
            raise typer.Exit(code=1)
        summaries = [s.summary() for s in series]
        if as_json:
            console.print_json(json.dumps(summaries, ensure_ascii=False))
            return
        table = Table(title=f"{len(series)} series in {path}")
        for column in ("modality", "slices", "matrix", "thickness", "description", "subject", "UID"):
            table.add_column(column)
        for s in summaries:
            table.add_row(
                s["modality"],
                str(s["n_slices"]),
                f"{s['matrix'][0]}x{s['matrix'][1]}",
                str(s["slice_thickness_mm"]),
                (s["description"] or "")[:32],
                s["subject"],
                s["series_uid"][-14:],
            )
        console.print(table)
        return

    volume = load_volume(path)
    description = volume.describe()
    if as_json:
        console.print_json(json.dumps(description, ensure_ascii=False))
        return
    table = Table(title=str(path))
    table.add_column("property")
    table.add_column("value")
    for key, value in description.items():
        table.add_row(
            key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )
    console.print(table)


@app.command()
def register(
    fixed: Path = typer.Argument(..., help="Fixed (reference) image: file or DICOM series."),
    moving: Path = typer.Argument(..., help="Moving image (to be registered)."),
    output: Path = typer.Option(Path("regix_out"), "-o", "--output", help="Output directory."),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Bundled preset."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML file."),
    organ: list[str] = typer.Option([], "--organ", help="Target organ(s) or group. Repeatable."),
    fixed_modality: Optional[str] = typer.Option(None, "--fixed-modality"),
    moving_modality: Optional[str] = typer.Option(None, "--moving-modality"),
    spacing: Optional[float] = typer.Option(None, "--spacing", help="Working resolution (mm)."),
    rigid_only: bool = typer.Option(False, "--rigid-only", help="Rigid only, no affine, no deformable."),
    deformable: Optional[bool] = typer.Option(
        None, "--deformable/--no-deformable", help="Force or remove the deformable stage."
    ),
    features: Optional[bool] = typer.Option(
        None, "--features/--no-features", help="Force or disable the anatomix features."
    ),
    organ_backend: Optional[str] = typer.Option(
        None, "--organ-backend", help="none | external | totalsegmentator"
    ),
    fixed_mask: Optional[Path] = typer.Option(None, "--fixed-mask", help="Fixed mask / label map."),
    moving_mask: Optional[Path] = typer.Option(None, "--moving-mask", help="Moving mask / label map."),
    labels: Optional[Path] = typer.Option(
        None,
        "--labels",
        help='JSON {"1": "liver", ...} describing the supplied masks. Failing that, Regix '
        "looks for a sidecar '<mask>.labels.json' and never guesses.",
    ),
    roi_crop: Optional[bool] = typer.Option(None, "--roi-crop/--no-roi-crop"),
    init: Optional[str] = typer.Option(
        None, "--init", help="identity | geometry | moments | organ_centroid | organ_moments | multistart"
    ),
    landmarks_fixed: Optional[Path] = typer.Option(None, "--landmarks-fixed"),
    landmarks_moving: Optional[Path] = typer.Option(None, "--landmarks-moving"),
    dicom_out: bool = typer.Option(False, "--dicom-out", help="Also write a derived DICOM series."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite a non-empty output directory."),
    threads: Optional[int] = typer.Option(None, "--threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option([], "--set", help="Override dotted.key=value. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the effective configuration and exit."),
) -> None:
    """Register MOVING onto FIXED and write the resampled image, transform, QC and report."""
    from regix.config import DeformableEngine, OrganBackend, StageConfig, TransformType
    from regix.pipeline import RegistrationPipeline

    cfg = _load_config(preset, config_file)
    overrides: dict[str, Any] = {}
    if fixed_modality:
        overrides["fixed_modality"] = fixed_modality.upper()
    if moving_modality:
        overrides["moving_modality"] = moving_modality.upper()
    if spacing is not None:
        overrides.setdefault("preprocess", {})["working_spacing_mm"] = spacing
    if organ:
        overrides.setdefault("organs", {})["targets"] = list(organ)
    if organ_backend:
        overrides.setdefault("organs", {})["backend"] = organ_backend
    if fixed_mask or moving_mask:
        organs = overrides.setdefault("organs", {})
        organs.setdefault("backend", OrganBackend.EXTERNAL.value)
        if fixed_mask:
            organs["fixed_labelmap"] = str(fixed_mask)
        if moving_mask:
            organs["moving_labelmap"] = str(moving_mask)
    if labels:
        raw = json.loads(labels.read_text(encoding="utf-8"))
        overrides.setdefault("organs", {})["label_names"] = {int(k): str(v) for k, v in raw.items()}
    if roi_crop is not None:
        overrides.setdefault("organs", {})["roi_crop"] = roi_crop
    if features is not None:
        overrides.setdefault("features", {})["enabled"] = features
    if init:
        overrides.setdefault("init", {})["mode"] = init
    if landmarks_fixed:
        overrides.setdefault("qc", {})["landmarks_fixed"] = str(landmarks_fixed)
    if landmarks_moving:
        overrides.setdefault("qc", {})["landmarks_moving"] = str(landmarks_moving)
    if dicom_out:
        overrides.setdefault("output", {})["write_dicom"] = True
    if overwrite:
        overrides.setdefault("output", {})["overwrite"] = True
    if threads:
        overrides.setdefault("runtime", {})["threads"] = threads
    overrides.setdefault("runtime", {})["log_level"] = log_level.upper()

    cfg = cfg.with_overrides(**overrides) if overrides else cfg

    # Structural changes to the stage list: applied after the merge.
    if rigid_only:
        cfg = cfg.model_copy(
            update={
                "stages": [StageConfig(type=TransformType.RIGID)],
                "deformable_engine": DeformableEngine.NONE,
            }
        )
    elif deformable is False:
        kept = [s for s in cfg.stages if s.type is not TransformType.BSPLINE]
        cfg = cfg.model_copy(
            update={
                "stages": kept or [StageConfig(type=TransformType.RIGID)],
                "deformable_engine": DeformableEngine.NONE,
            }
        )
    elif deformable is True and not any(s.type is TransformType.BSPLINE for s in cfg.stages):
        from regix.organs.labels import merged_profile

        grid = merged_profile(list(organ)).bspline_grid_mm if organ else 20.0
        cfg = cfg.model_copy(
            update={
                "stages": list(cfg.stages)
                + [StageConfig(type=TransformType.BSPLINE, final_grid_spacing_mm=grid)],
                "deformable_engine": DeformableEngine.ELASTIX,
            }
        )

    cfg = _apply_sets(cfg, list(set_))
    cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"dir": output})})

    if dry_run:
        console.print(cfg.to_yaml())
        return

    result = RegistrationPipeline(cfg).run(fixed, moving, output)
    _echo_result(result)
    if result.status == "FAIL":
        raise typer.Exit(code=2)


@app.command()
def batch(
    pairs: Path = typer.Argument(..., help="CSV with columns fixed,moving[,name,fixed_mask,moving_mask]."),
    output: Path = typer.Option(Path("regix_batch"), "-o", "--output"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    organ: list[str] = typer.Option([], "--organ"),
    continue_on_error: bool = typer.Option(True, "--continue/--stop-on-error"),
    summary_csv: Optional[Path] = typer.Option(None, "--summary", help="Summary CSV."),
    threads: Optional[int] = typer.Option(None, "--threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option([], "--set", help="Override dotted.key=value. Repeatable."),
) -> None:
    """Register a list of pairs and produce a usable summary."""
    from regix.pipeline import RegistrationPipeline

    rows = list(csv.DictReader(pairs.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        console.print("[red]empty CSV[/]")
        raise typer.Exit(code=1)
    missing = [c for c in ("fixed", "moving") if c not in rows[0]]
    if missing:
        raise typer.BadParameter(f"missing columns in the CSV: {missing}")

    base = _load_config(preset, config_file)
    overrides: dict[str, Any] = {"runtime": {"log_level": log_level.upper()}}
    if organ:
        overrides["organs"] = {"targets": list(organ)}
    if threads:
        overrides["runtime"]["threads"] = threads
    base = _apply_sets(base.with_overrides(**overrides), list(set_))

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        name = row.get("name") or f"case{index:04d}"
        case_dir = output / name
        console.rule(f"[{index}/{len(rows)}] {name}")
        cfg = base
        if row.get("fixed_mask") or row.get("moving_mask"):
            organs: dict[str, Any] = {"backend": "external"}
            if row.get("fixed_mask"):
                organs["fixed_labelmap"] = row["fixed_mask"]
            if row.get("moving_mask"):
                organs["moving_labelmap"] = row["moving_mask"]
            cfg = cfg.with_overrides(organs=organs)
        cfg = cfg.with_overrides(output={"overwrite": True})
        try:
            result = RegistrationPipeline(cfg).run(row["fixed"], row["moving"], case_dir)
            similarity = result.metrics.get("similarity", {})
            records.append(
                {
                    "name": name,
                    "status": result.status,
                    "seconds": round(result.seconds, 1),
                    "ncc_before": similarity.get("ncc_before"),
                    "ncc_after": similarity.get("ncc_after"),
                    "nmi_before": similarity.get("nmi_before"),
                    "nmi_after": similarity.get("nmi_after"),
                    "dice": json.dumps(
                        {k: v.get("dice") for k, v in (result.metrics.get("organ_overlap") or {}).items()}
                    ),
                    "tre_mm": (result.metrics.get("landmarks") or {}).get("tre_mean_mm"),
                    "output": str(case_dir),
                }
            )
            console.print(f"  -> {result.status}")
        except Exception as exc:
            console.print(f"  [red]ERROR[/] {type(exc).__name__}: {exc}")
            records.append({"name": name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            if not continue_on_error:
                raise typer.Exit(code=1) from exc

    target = summary_csv or output / "summary.csv"
    fields = sorted({k for r in records for k in r})
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    console.rule("Summary")
    for status, count in sorted(counts.items()):
        console.print(f"  {status:6s} {count}")
    console.print(f"  summary file: {target}")
    if counts.get("ERROR") or counts.get("FAIL"):
        raise typer.Exit(code=2)


@app.command()
def apply(
    transform: Path = typer.Argument(..., help="TransformParameters.txt (elastix) or .tfm (ITK)."),
    moving: Path = typer.Argument(..., help="Volume to transform."),
    reference: Path = typer.Option(..., "--reference", "-r", help="Volume defining the output grid."),
    output: Path = typer.Option(Path("warped.nii.gz"), "-o", "--output"),
    label: bool = typer.Option(False, "--label", help="Label map: nearest-neighbour interpolation."),
) -> None:
    """Apply an already computed transform (resume, contour propagation)."""
    import SimpleITK as sitk

    from regix.io.volume import load_volume
    from regix.io.writers import save_image
    from regix.registration.warp import ElastixAppliedTransform, SitkAppliedTransform

    moving_volume = load_volume(moving, role="labelmap" if label else "image")
    reference_volume = load_volume(reference)
    if transform.suffix.lower() == ".tfm":
        applied = SitkAppliedTransform(sitk.ReadTransform(str(transform)))
    else:
        applied = ElastixAppliedTransform(transform)
    warped = applied.resample(
        moving_volume.image, reference_volume.image, is_label=label, default_value=0 if label else None
    )
    save_image(warped, output)
    console.print(
        f"[green]written[/] {output}  {warped.GetSize()} @ "
        f"{tuple(round(v, 3) for v in warped.GetSpacing())} mm"
    )


@app.command()
def segment(
    image: Path = typer.Argument(..., help="CT volume (file or DICOM series)."),
    output: Path = typer.Option(Path("regix_masks"), "-o", "--output"),
    organ: list[str] = typer.Option([], "--organ", help="Restrict to the requested organs."),
) -> None:
    """Segment the organs with TotalSegmentator and write a label map plus one mask per organ."""
    import SimpleITK as sitk

    from regix.io.volume import load_volume
    from regix.io.writers import save_image
    from regix.organs.labels import resolve_targets
    from regix.organs.segmenter import TotalSegmentatorSegmenter

    volume = load_volume(image)
    targets = resolve_targets(list(organ))
    segmenter = TotalSegmentatorSegmenter(roi_subset=targets or None)

    seg = segmenter.segment(volume)
    output.mkdir(parents=True, exist_ok=True)
    save_image(seg.labelmap, output / "labelmap.nii.gz", dtype=sitk.sitkUInt16)
    (output / "labels.json").write_text(
        json.dumps(seg.label_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for organ_name in targets or seg.present_organs():
        try:
            save_image(seg.mask_for([organ_name]), output / f"{organ_name}.nii.gz", dtype=sitk.sitkUInt8)
        except ValueError:
            continue
    console.print(f"[green]{len(seg.present_organs())} organs[/] -> {output}")


def main() -> None:  # console entry point
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[yellow]interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
