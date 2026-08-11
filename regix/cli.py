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
import io
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from regix import __version__
from regix.config import InitMode, LogLevel, OrganBackend

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
    help="Regix - multimodal / multi-organ registration (research software, not a medical device).",
)
console = Console()


class SegmentBackend(str, Enum):
    TOTALSEGMENTATOR = "totalsegmentator"


class FeatureProvider(str, Enum):
    AUTO = "auto"
    ANATOMIX = "anatomix"
    MIND = "mind"


@app.callback()
def _global_options(
    show_version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        help="Print the Regix version and exit.",
    ),
) -> None:
    if show_version:
        typer.echo(f"regix {__version__}")
        raise typer.Exit()


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
        if not key or any(not part for part in key.split(".")):
            raise typer.BadParameter(f"--set {item!r}: invalid empty path component")
        try:
            value = yaml.safe_load(raw)
            node: Any = data
            parts = key.split(".")
            for part in parts[:-1]:
                if isinstance(node, list):
                    if not part.isdigit():
                        raise TypeError(f"expected a non-negative list index, got {part!r}")
                    index = int(part)
                    if index >= len(node):
                        raise IndexError(f"index {index} outside 0..{len(node) - 1}")
                    node = node[index]
                elif isinstance(node, dict):
                    if part not in node:
                        choices = ", ".join(sorted(str(k) for k in node)[:12])
                        raise KeyError(f"unknown key {part!r}; available here: {choices}")
                    node = node[part]
                else:
                    raise TypeError(f"cannot descend through {part!r}: value is {type(node).__name__}")
            last = parts[-1]
            if isinstance(node, list):
                if not last.isdigit():
                    raise TypeError(f"expected a non-negative list index, got {last!r}")
                index = int(last)
                if index >= len(node):
                    raise IndexError(f"index {index} outside 0..{len(node) - 1}")
                node[index] = value
            elif isinstance(node, dict):
                node[last] = value
            else:
                raise TypeError(f"cannot assign {last!r}: parent is {type(node).__name__}")
        except typer.BadParameter:
            raise
        except Exception as exc:
            raise typer.BadParameter(f"--set {item!r}: invalid path or value ({exc})") from exc
    try:
        return RegistrationConfig.model_validate(data)
    except Exception as exc:
        raise typer.BadParameter(f"invalid --set override: {exc}") from exc


def _result_payload(result) -> dict[str, Any]:
    return {
        "status": result.status,
        "seconds": round(float(result.seconds), 3),
        "metrics": result.metrics,
        "qc": result.qc,
        "initialization": result.initialization,
        "stages": result.stages,
        "warnings": result.warnings,
        "outputs": {key: str(path) for key, path in result.outputs.items()},
    }


def _echo_result(result, as_json: bool = False) -> None:
    if as_json:
        typer.echo(json.dumps(_result_payload(result), ensure_ascii=False, allow_nan=False))
        return
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(result.status, "white")
    console.print(f"\n[bold {color}]{result.status}[/] — {result.seconds:.1f} s")
    console.print(result.summary())
    if result.outputs:
        console.print("\n[bold]Outputs[/]")
        for key, path in result.outputs.items():
            console.print(f"  {key:20s} {path}")


# --------------------------------------------------------------------------- #
@app.command()
def version(
    as_json: bool = typer.Option(False, "--json", help="Print strict machine-readable JSON."),
) -> None:
    """Print the Regix version."""
    if as_json:
        typer.echo(json.dumps({"regix": __version__}, allow_nan=False))
    else:
        console.print(f"regix {__version__}")


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print strict machine-readable JSON."),
) -> None:
    """Check the environment: engine, optional dependencies, GPU."""
    from regix.logging_utils import environment_report, pseudonymization_warning
    from regix.registration.itk_bridge import engine_available

    report = environment_report()
    has_elastix, engine_detail = engine_available()
    if as_json:
        report["engine_available"] = has_elastix
        report["engine_detail"] = engine_detail
        report["pseudonymization_warning"] = pseudonymization_warning()
        typer.echo(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
        if not has_elastix:
            raise typer.Exit(code=1)
        return

    table = Table(title="Regix environment", show_lines=False)
    table.add_column("component")
    table.add_column("version / capacity")
    table.add_column("state")
    table.add_column("consequence if missing")

    table.add_row(
        "itk-elastix (engine)",
        engine_detail,
        "[green]ready[/]" if has_elastix else "[red]broken/missing[/]",
        "blocking: this is the registration engine (pip install itk-elastix)",
    )
    table.add_row(
        "SimpleITK",
        f"{report['simpleitk']}",
        "[green]installed[/]" if report["simpleitk"] else "[red]missing[/]",
        "blocking: DICOM I/O, morphology, transforms",
    )
    table.add_row(
        "numpy",
        str(report["numpy"]),
        "[green]installed[/]" if report["numpy"] else "[red]missing[/]",
        "blocking",
    )

    optional = (
        ("torch", "torch", "no anatomix/ConvexAdam; CPU MIND-SSC remains available"),
        ("anatomix", "anatomix", "neural features unavailable; MIND-SSC/MI remain available"),
        ("monai", "monai", "neural sliding-window inference unavailable"),
        ("TotalSegmentator", "totalsegmentator", "automatic organ segmentation unavailable"),
    )
    for label, key, consequence in optional:
        version_value = report.get(key)
        table.add_row(
            label,
            str(version_value or "—"),
            "[green]installed[/]" if version_value else "[yellow]optional/missing[/]",
            consequence,
        )
    table.add_row(
        "CUDA GPU",
        str(report.get("cuda_device") or "not probed"),
        "[green]available[/]" if report.get("cuda_available") else "[yellow]not probed/unavailable[/]",
        "feature execution will probe it without making doctor import torch",
    )
    table.add_row(
        "pydicom",
        str(report.get("pydicom") or "—"),
        "[green]installed[/]" if report.get("pydicom") else "[red]missing[/]",
        "tag reading and DICOM exports (SRO, derived series)",
    )
    pseudonym_warning = pseudonymization_warning()
    table.add_row(
        "pseudonymisation",
        str((report.get("configuration") or {}).get("pseudonym_salt_fingerprint", "—")),
        "[yellow]ephemeral/weak key[/]" if pseudonym_warning else "[green]configured[/]",
        pseudonym_warning or "stable HMAC pseudonyms",
    )
    table.add_row(
        "available memory",
        f"{report.get('memory_available_mb')} MB",
        "[green]reported[/]" if report.get("memory_available_mb") is not None else "[yellow]unknown[/]",
        "large volumes may otherwise exhaust RAM",
    )
    table.add_row(
        "free disk",
        f"{report.get('disk_free_mb')} MB",
        "[green]reported[/]" if report.get("disk_free_mb") is not None else "[yellow]unknown[/]",
        "replay inputs and DICOM exports require working space",
    )

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
    resolved: bool = typer.Option(
        False,
        "--resolved",
        help="Print the fully merged configuration instead of the commented source YAML.",
    ),
) -> None:
    """List the bundled presets (or print one)."""
    from regix.config import available_presets, load_preset, preset_source

    if name:
        yaml_text = load_preset(name).to_yaml() if resolved else preset_source(name)
        # YAML is a data stream: Rich markup and line wrapping would corrupt redirected output.
        typer.echo(yaml_text, nl=not yaml_text.endswith("\n"))
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
    fixed_series_uid: Optional[str] = typer.Option(None, "--fixed-series-uid"),
    moving_series_uid: Optional[str] = typer.Option(None, "--moving-series-uid"),
    spacing: Optional[float] = typer.Option(None, "--spacing", help="Working resolution (mm)."),
    rigid_only: bool = typer.Option(False, "--rigid-only", help="Rigid only, no affine, no deformable."),
    deformable: Optional[bool] = typer.Option(
        None, "--deformable/--no-deformable", help="Force or remove the deformable stage."
    ),
    features: Optional[bool] = typer.Option(
        None,
        "--features/--no-features",
        help="Force or disable descriptors according to --feature-provider.",
    ),
    feature_provider: Optional[FeatureProvider] = typer.Option(
        None,
        "--feature-provider",
        help="Descriptor selection: auto | anatomix | mind.",
    ),
    organ_backend: Optional[OrganBackend] = typer.Option(
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
    init: Optional[InitMode] = typer.Option(
        None, "--init", help="identity | geometry | moments | organ_centroid | organ_moments | multistart"
    ),
    landmarks_fixed: Optional[Path] = typer.Option(None, "--landmarks-fixed"),
    landmarks_moving: Optional[Path] = typer.Option(None, "--landmarks-moving"),
    dicom_out: bool = typer.Option(False, "--dicom-out", help="Also write a derived DICOM series."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite a non-empty output directory."),
    threads: Optional[int] = typer.Option(None, "--threads"),
    log_level: LogLevel = typer.Option(LogLevel.INFO, "--log-level"),
    allow_cpu_features: bool = typer.Option(
        False,
        "--allow-cpu-features",
        help="Allow the anatomix feature extractor on CPU (slow).",
    ),
    set_: list[str] = typer.Option([], "--set", help="Override dotted.key=value. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the effective configuration and exit."),
    as_json: bool = typer.Option(False, "--json", help="Emit the final result as strict JSON."),
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
    if fixed_series_uid:
        overrides["fixed_series_uid"] = fixed_series_uid
    if moving_series_uid:
        overrides["moving_series_uid"] = moving_series_uid
    if spacing is not None:
        overrides.setdefault("preprocess", {})["working_spacing_mm"] = spacing
    if organ:
        overrides.setdefault("organs", {})["targets"] = list(organ)
    if organ_backend:
        overrides.setdefault("organs", {})["backend"] = organ_backend.value
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
    if feature_provider is not None:
        overrides.setdefault("features", {})["provider"] = feature_provider.value
    if allow_cpu_features:
        overrides.setdefault("features", {})["allow_cpu"] = True
    if init:
        overrides.setdefault("init", {})["mode"] = init.value
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
    overrides.setdefault("runtime", {})["log_level"] = log_level.value

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
        typer.echo(cfg.to_yaml(), nl=False)
        return

    result = RegistrationPipeline(cfg).run(fixed, moving, output)
    _echo_result(result, as_json=as_json)
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
    log_level: LogLevel = typer.Option(LogLevel.INFO, "--log-level"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace known artifacts of prior cases."),
    set_: list[str] = typer.Option([], "--set", help="Override dotted.key=value. Repeatable."),
    as_json: bool = typer.Option(False, "--json", help="Emit the batch records as strict JSON."),
) -> None:
    """Register a list of pairs and produce a usable summary."""
    from regix.pipeline import RegistrationPipeline

    source = pairs.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(source[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(source), dialect=dialect))
    if not rows:
        console.print("[red]empty CSV[/]")
        raise typer.Exit(code=1)
    missing = [c for c in ("fixed", "moving") if c not in rows[0]]
    if missing:
        raise typer.BadParameter(f"missing columns in the CSV: {missing}")

    base = _load_config(preset, config_file)
    overrides: dict[str, Any] = {"runtime": {"log_level": log_level.value}}
    if organ:
        overrides["organs"] = {"targets": list(organ)}
    if threads:
        overrides["runtime"]["threads"] = threads
    base = _apply_sets(base.with_overrides(**overrides), list(set_))

    raw_names = [row.get("name") or f"case{index:04d}" for index, row in enumerate(rows, start=1)]
    names = [_safe_case_name(name) for name in raw_names]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise typer.BadParameter(f"duplicate case names after validation: {', '.join(duplicates)}")

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (row, name) in enumerate(zip(rows, names, strict=True), start=1):
        case_dir = output / name
        if not as_json:
            console.rule(f"[{index}/{len(rows)}] {name}")
        cfg = base
        series_overrides = {
            key: row[key] for key in ("fixed_series_uid", "moving_series_uid") if row.get(key)
        }
        if series_overrides:
            cfg = cfg.with_overrides(**series_overrides)
        if row.get("fixed_mask") or row.get("moving_mask"):
            organs: dict[str, Any] = {"backend": "external"}
            if row.get("fixed_mask"):
                organs["fixed_labelmap"] = row["fixed_mask"]
            if row.get("moving_mask"):
                organs["moving_labelmap"] = row["moving_mask"]
            cfg = cfg.with_overrides(organs=organs)
        cfg = cfg.with_overrides(output={"overwrite": overwrite})
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
            if not as_json:
                console.print(f"  -> {result.status}")
        except Exception as exc:
            if not as_json:
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
    if as_json:
        typer.echo(json.dumps({"records": records, "counts": counts}, ensure_ascii=False, allow_nan=False))
    else:
        console.rule("Summary")
        for status, count in sorted(counts.items()):
            console.print(f"  {status:6s} {count}")
        console.print(f"  summary file: {target}")
    if counts.get("ERROR") or counts.get("FAIL"):
        raise typer.Exit(code=2)


@app.command()
def apply(
    transform: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="TransformParameters.txt (elastix) or an ITK transform (.tfm/.txt/.h5).",
    ),
    moving: Path = typer.Argument(..., help="Volume to transform."),
    reference: Path = typer.Option(..., "--reference", "-r", help="Volume defining the output grid."),
    output: Path = typer.Option(Path("warped.nii.gz"), "-o", "--output"),
    label: bool = typer.Option(False, "--label", help="Label map: nearest-neighbour interpolation."),
    invert: bool = typer.Option(
        False,
        "--invert",
        help="Invert a transform that has an exact inverse before resampling.",
    ),
) -> None:
    """Apply an already computed transform (resume, contour propagation)."""
    from regix.io.volume import load_volume
    from regix.io.writers import save_image
    from regix.registration.transforms import load_any_transform
    from regix.registration.warp import ElastixAppliedTransform, SitkAppliedTransform

    moving_volume = load_volume(moving, role="labelmap" if label else "image")
    reference_volume = load_volume(reference)
    head = transform.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        if "(Transform " in head and "(TransformParameters " in head:
            # A non-linear elastix chain cannot be represented by SimpleITK; transformix
            # remains the authoritative reader in that case.
            try:
                applied = SitkAppliedTransform(load_any_transform(transform))
            except ValueError:
                applied = ElastixAppliedTransform(transform)
        else:
            applied = SitkAppliedTransform(load_any_transform(transform))
    except Exception as exc:
        raise typer.BadParameter(
            f"could not read transform {transform.name!r}; expected elastix parameters or an ITK transform"
        ) from exc
    if invert:
        transform_value = applied.as_sitk_transform()
        if transform_value is None or "inverse" not in applied.capabilities:
            raise typer.BadParameter("this transform has no exact, representable inverse")
        applied = SitkAppliedTransform(transform_value.GetInverse(), label="inverse")
    warped = applied.resample(
        moving_volume.image, reference_volume.image, is_label=label, default_value=0 if label else None
    )
    save_image(warped, output)
    console.print(
        f"[green]written[/] {output}  {warped.GetSize()} @ "
        f"{tuple(round(v, 3) for v in warped.GetSpacing())} mm"
    )


@app.command("qc")
def recompute_qc(
    output_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        readable=True,
        help="Completed Regix output directory.",
    ),
    set_values: list[str] = typer.Option(
        [],
        "--set",
        help="Override a QC setting, e.g. --set qc.gates.max_tre_mm=3.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the new verdict as JSON."),
) -> None:
    """Re-evaluate QC gates and regenerate the report without rerunning registration.

    Image-derived measurements are immutable inputs from ``run_manifest.json``; this
    command is intended for changing site acceptance thresholds after a completed run.
    """
    import time

    import yaml

    from regix.config import RegistrationConfig
    from regix.layout import EFFECTIVE_CONFIG, MANIFEST, REPORT
    from regix.logging_utils import setup_logging
    from regix.qc.gates import evaluate_gates
    from regix.qc.report import build_html_report

    if as_json:
        # Machine-readable output must remain one JSON document even when a previous
        # command in the same process configured the global Regix logger.
        setup_logging("CRITICAL", quiet=True)

    manifest_path = output_dir / MANIFEST
    config_path = output_dir / EFFECTIVE_CONFIG
    if not manifest_path.is_file() or not config_path.is_file():
        raise typer.BadParameter(
            f"{output_dir} is not a completed Regix run (missing {MANIFEST} or {EFFECTIVE_CONFIG})"
        )
    for assignment in set_values:
        if not assignment.partition("=")[0].startswith("qc."):
            raise typer.BadParameter("regix qc only accepts --set qc.* overrides")
    try:
        cfg = RegistrationConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
        cfg = _apply_sets(cfg, set_values)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(f"could not read the completed run ({exc})") from exc

    metrics = manifest.get("metrics") or {}
    profile = metrics.get("organ_profile") or {}
    stage_summaries = next(
        (step.get("stages", []) for step in manifest.get("steps", []) if step.get("name") == "elastix"),
        [],
    )
    initialization = next(
        (
            {key: step[key] for key in ("mode", "requested", "chosen", "n_candidates") if key in step}
            for step in manifest.get("steps", [])
            if step.get("name") == "initialization"
        ),
        {},
    )
    deformable = any(stage.get("transform") == "BSplineTransform" for stage in stage_summaries)
    gates = cfg.qc.gates
    expected_motion = profile.get("typical_motion_mm")
    displacement_advisory = False
    if gates.max_displacement_ratio is None and expected_motion is not None:
        gates = gates.model_copy(update={"max_displacement_ratio": 2.5})
        displacement_advisory = True
    verdict = evaluate_gates(
        gates,
        similarity=metrics.get("similarity"),
        organ_overlap=metrics.get("organ_overlap"),
        jacobian=metrics.get("jacobian"),
        linear_analysis=metrics.get("linear"),
        landmarks=metrics.get("landmarks"),
        deformable=deformable,
        displacement=metrics.get("displacement"),
        expected_motion_mm=expected_motion,
        stages=stage_summaries,
        initialization=initialization,
        displacement_advisory=displacement_advisory,
    ).to_dict()
    manifest["status"] = verdict["status"]
    metrics["qc"] = verdict
    manifest["metrics"] = metrics
    manifest["config"] = cfg.model_dump(mode="json")
    manifest.setdefault("qc_recomputations", []).append(
        {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "overrides": list(set_values)}
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if cfg.qc.report_html:
        build_html_report(
            output_dir / REPORT,
            {
                "title": "Regix QC re-evaluation",
                "subtitle": f"completed run {manifest.get('run_id', 'unknown')}",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "qc": verdict,
                "similarity": metrics.get("similarity"),
                "organ_overlap": metrics.get("organ_overlap"),
                "landmarks": metrics.get("landmarks"),
                "jacobian": metrics.get("jacobian"),
                "stages": stage_summaries,
                "configuration": {"preset": cfg.name, "QC overrides": set_values or "unchanged"},
                "environment": manifest.get("environment"),
                "warnings": manifest.get("warnings"),
                "degradations": manifest.get("degradations"),
            },
        )
    if as_json:
        typer.echo(json.dumps(verdict, ensure_ascii=False, allow_nan=False))
    else:
        console.print(f"[bold]{verdict['status']}[/] — QC gates re-evaluated in {output_dir}")


@app.command()
def segment(
    image: Path = typer.Argument(..., help="CT volume (file or DICOM series)."),
    output: Path = typer.Option(Path("regix_masks"), "-o", "--output"),
    backend: SegmentBackend = typer.Option(SegmentBackend.TOTALSEGMENTATOR, "--backend"),
    organ: list[str] = typer.Option([], "--organ", help="Restrict to the requested organs."),
    task: str = typer.Option("auto", "--task", help="auto | total | total_mr"),
    fast: bool = typer.Option(True, "--fast/--no-fast", help="Use the lower-resolution TS model."),
    device: str = typer.Option("auto", "--device", help="auto | cuda | cpu"),
    timeout_seconds: int = typer.Option(1800, "--timeout", min=1, max=86_400),
) -> None:
    """Segment the organs with TotalSegmentator and write a label map plus one mask per organ."""
    import SimpleITK as sitk

    from regix.io.volume import load_volume
    from regix.io.writers import save_image
    from regix.organs.labels import resolve_targets
    from regix.organs.segmenter import TotalSegmentatorSegmenter

    volume = load_volume(image)
    targets = resolve_targets(list(organ))
    if backend is not SegmentBackend.TOTALSEGMENTATOR:  # pragma: no cover - Typer validates this
        raise typer.BadParameter(f"unsupported segmentation backend: {backend.value}")
    if task not in {"auto", "total", "total_mr"}:
        raise typer.BadParameter("--task must be auto, total or total_mr")
    if device not in {"auto", "cuda", "cpu"}:
        raise typer.BadParameter("--device must be auto, cuda or cpu")
    resolved_task = (
        "total_mr" if task == "auto" and volume.modality == "MR" else ("total" if task == "auto" else task)
    )
    if resolved_task == "total" and volume.modality == "MR":
        raise typer.BadParameter("--task total is CT-only; use total_mr for an MR volume")
    segmenter = TotalSegmentatorSegmenter(
        task=resolved_task,
        fast=fast,
        roi_subset=targets or None,
        device=device,
        timeout_seconds=timeout_seconds,
    )

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


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_case_name(value: str) -> str:
    """Validate a CSV case name as one portable path component."""
    name = str(value).strip()
    if (
        not name
        or name in {".", ".."}
        or len(name) > 100
        or Path(name).is_absolute()
        or "/" in name
        or "\\" in name
        or ":" in name
        or name.rstrip(" .") != name
        or name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        or not re.fullmatch(r"[\w .-]+", name, flags=re.UNICODE)
    ):
        raise typer.BadParameter(
            f"invalid case name {value!r}: use one portable component (letters, digits, space, _, . or -)"
        )
    return name


def main() -> None:  # console entry point
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[yellow]interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
