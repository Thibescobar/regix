"""Command-line interface tests.

The CLI is the surface most users touch first, and it is the easiest thing to break
without noticing: an option renamed in the configuration silently stops being
forwarded, and nothing fails until someone runs the command. These tests invoke the
real Typer application in-process, so a broken option is a failing test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk
from typer.testing import CliRunner

from regix.cli import app
from tests.conftest import known_rigid, make_phantom, warp

runner = CliRunner()


@pytest.fixture
def phantom_pair(tmp_path):
    """A small CT-CT pair on disk, with a known transform."""
    fixed, labels = make_phantom("CT", shape=(40, 48, 48), noise=3.0)
    truth = known_rigid(fixed, rotation_deg=(2.0, -1.0, 3.0), translation_mm=(4.0, -3.0, 5.0))
    moving = warp(fixed, truth)
    moving_labels = warp(labels, truth, is_label=True)

    paths = {}
    for name, img, dtype in (
        ("fixed", fixed, sitk.sitkFloat32),
        ("moving", moving, sitk.sitkFloat32),
        ("fixed_labels", labels, sitk.sitkUInt16),
        ("moving_labels", moving_labels, sitk.sitkUInt16),
    ):
        path = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(sitk.Cast(img, dtype), str(path), True)
        paths[name] = path
    return paths, truth


def _run(*args: str):
    result = runner.invoke(app, list(args))
    if result.exit_code not in (0, 2):  # 2 = QC FAIL, a legitimate outcome
        raise AssertionError(
            f"regix {' '.join(args)} exited with {result.exit_code}\n{result.output}\n{result.exception!r}"
        )
    return result


# --------------------------------------------------------------------------- #
# Informational commands
# --------------------------------------------------------------------------- #
def test_version_and_help():
    assert "regix" in _run("version").output
    output = _run("--help").output
    for command in ("register", "batch", "apply", "segment", "inspect", "presets", "doctor"):
        assert command in output, f"command {command} is missing from the help"


def test_doctor_reports_the_engine():
    """`regix doctor` must exit 0 when the engine is present, and name it."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "itk-elastix" in result.output
    assert "SimpleITK" in result.output
    # It reports consequences, not just presence: that is the point of the command.
    assert "blocking" in result.output


def test_presets_listing_and_detail():
    listing = _run("presets").output
    assert "ct_mr_abdomen" in listing
    assert "base" in listing

    detail = _run("presets", "ct_mr_abdomen").output
    assert "ct_mr_abdomen" in detail
    assert "bspline" in detail


def test_inspect_a_volume(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    result = _run("inspect", str(paths["fixed"]))
    assert "spacing_mm" in result.output or "spacing" in result.output

    machine = _run("inspect", str(paths["fixed"]), "--json")
    payload = json.loads(machine.output)
    assert payload["size"] == [48, 48, 40]
    assert payload["modality"] in ("CT", "UNKNOWN")


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #
def test_register_writes_every_expected_output(tmp_path, phantom_pair):
    paths, truth = phantom_pair
    out = tmp_path / "out"
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(out),
        "--fixed-modality",
        "CT",
        "--moving-modality",
        "CT",
        "--spacing",
        "2.5",
        "--overwrite",
        "--log-level",
        "WARNING",
    )
    assert "PASS" in result.output or "WARN" in result.output

    for expected in (
        "moving_registered.nii.gz",
        "run_manifest.json",
        "config_effective.yaml",
        "report.html",
        "transform/final_transform.tfm",
        "transform/final_transform.txt",
    ):
        assert (out / expected).exists(), f"{expected} is missing"

    # The transform written must be the one that was measured.
    transform = sitk.ReadTransform(str(out / "transform" / "final_transform.txt"))
    fixed = sitk.ReadImage(str(paths["fixed"]))
    probe = fixed.TransformContinuousIndexToPhysicalPoint([20.0, 20.0, 18.0])
    error = float(
        np.linalg.norm(
            np.asarray(transform.TransformPoint(probe)) - np.asarray(truth.TransformPoint(list(probe)))
        )
    )
    assert error < 2.0, f"{error:.2f} mm away from the ground truth"


def test_dry_run_prints_the_configuration_without_computing(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    out = tmp_path / "never_created"
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(out),
        "--dry-run",
    )
    assert "stages" in result.output
    assert "working_spacing_mm" in result.output
    assert not out.exists(), "--dry-run must not compute anything"


def test_rigid_only_removes_the_other_stages(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(tmp_path / "out"),
        "--rigid-only",
        "--dry-run",
    )
    assert result.output.count("type: rigid") == 1
    assert "type: affine" not in result.output
    assert "deformable_engine: none" in result.output


def test_deformable_flag_appends_a_bspline_stage(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(tmp_path / "out"),
        "--deformable",
        "--dry-run",
    )
    assert "type: bspline" in result.output
    assert "deformable_engine: elastix" in result.output


def test_set_overrides_reach_the_configuration(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(tmp_path / "out"),
        "--dry-run",
        "--set",
        "preprocess.working_spacing_mm=1.25",
        "--set",
        "stages.0.max_iterations=999",
    )
    assert "working_spacing_mm: 1.25" in result.output
    assert "max_iterations: 999" in result.output


def test_invalid_set_is_rejected(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    result = runner.invoke(
        app,
        [
            "register",
            str(paths["fixed"]),
            str(paths["moving"]),
            "-o",
            str(tmp_path / "out"),
            "--dry-run",
            "--set",
            "nonsense",
        ],
    )
    assert result.exit_code != 0, "--set without '=' must be rejected"


def test_organ_targets_and_masks_are_forwarded(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    labels_json = tmp_path / "labels.json"
    labels_json.write_text(json.dumps({"1": "liver", "2": "spleen", "3": "kidney_left"}), encoding="utf-8")

    out = tmp_path / "out"
    result = _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(out),
        "--fixed-modality",
        "CT",
        "--moving-modality",
        "CT",
        "--organ",
        "liver",
        "--fixed-mask",
        str(paths["fixed_labels"]),
        "--moving-mask",
        str(paths["moving_labels"]),
        "--labels",
        str(labels_json),
        "--init",
        "organ_centroid",
        "--spacing",
        "2.5",
        "--overwrite",
        "--log-level",
        "WARNING",
    )
    assert "Dice liver" in result.output, result.output
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["organs"]["targets"] == ["liver"]
    assert manifest["config"]["organs"]["backend"] == "external"
    assert manifest["config"]["init"]["mode"] == "organ_centroid"


def test_existing_output_directory_is_protected(tmp_path, phantom_pair):
    """Without --overwrite, an existing run must not be destroyed."""
    paths, _ = phantom_pair
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("earlier results", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "register",
            str(paths["fixed"]),
            str(paths["moving"]),
            "-o",
            str(out),
            "--fixed-modality",
            "CT",
            "--moving-modality",
            "CT",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code != 0
    assert (out / "precious.txt").exists()


# --------------------------------------------------------------------------- #
# apply / batch
# --------------------------------------------------------------------------- #
def test_apply_propagates_a_label_map(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    out = tmp_path / "out"
    _run(
        "register",
        str(paths["fixed"]),
        str(paths["moving"]),
        "-o",
        str(out),
        "--fixed-modality",
        "CT",
        "--moving-modality",
        "CT",
        "--spacing",
        "2.5",
        "--overwrite",
        "--log-level",
        "WARNING",
    )

    warped = tmp_path / "labels_on_fixed.nii.gz"
    _run(
        "apply",
        str(out / "transform" / "final_transform.tfm"),
        str(paths["moving_labels"]),
        "--reference",
        str(paths["fixed"]),
        "-o",
        str(warped),
        "--label",
    )
    assert warped.exists()

    result_image = sitk.ReadImage(str(warped))
    reference = sitk.ReadImage(str(paths["fixed"]))
    assert result_image.GetSize() == reference.GetSize()
    # Nearest-neighbour interpolation: no label may be invented.
    values = set(np.unique(sitk.GetArrayViewFromImage(result_image)).tolist())
    assert values <= {0, 1, 2, 3}, f"invented labels: {values}"


def test_batch_produces_a_summary(tmp_path, phantom_pair):
    paths, _ = phantom_pair
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "fixed,moving,name\n"
        f"{paths['fixed']},{paths['moving']},case_a\n"
        f"{paths['fixed']},{paths['moving']},case_b\n",
        encoding="utf-8",
    )

    out = tmp_path / "batch"
    result = _run(
        "batch",
        str(csv_path),
        "-o",
        str(out),
        "--set",
        "preprocess.working_spacing_mm=3.0",
        "--set",
        "fixed_modality=CT",
        "--set",
        "moving_modality=CT",
        "--log-level",
        "ERROR",
    )
    assert "case_a" in result.output and "case_b" in result.output

    import csv as csv_module

    with open(out / "summary.csv", encoding="utf-8") as fh:
        rows = list(csv_module.DictReader(fh))
    assert [r["name"] for r in rows] == ["case_a", "case_b"]
    assert all(r["status"] in ("PASS", "WARN") for r in rows), rows
    assert all(float(r["ncc_after"]) > float(r["ncc_before"]) for r in rows)
    assert (out / "case_a" / "run_manifest.json").exists()
    assert (out / "case_b" / "run_manifest.json").exists()


def test_batch_rejects_a_csv_without_the_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("image_a,image_b\nx,y\n", encoding="utf-8")
    result = runner.invoke(app, ["batch", str(csv_path), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "fixed" in result.output or "missing" in result.output.lower()


def test_qc_failure_exits_with_a_distinct_code(tmp_path):
    """A FAIL must be scriptable: exit code 2, not 0 and not a crash."""
    fixed, _ = make_phantom("CT", shape=(32, 40, 40), noise=1.0, seed=21)
    rng = np.random.default_rng(22)
    noise = sitk.GetImageFromArray(
        rng.normal(0.0, 200.0, size=sitk.GetArrayViewFromImage(fixed).shape).astype(np.float32)
    )
    noise.CopyInformation(fixed)

    fixed_path = tmp_path / "f.nii.gz"
    moving_path = tmp_path / "m.nii.gz"
    sitk.WriteImage(fixed, str(fixed_path), True)
    sitk.WriteImage(noise, str(moving_path), True)

    result = runner.invoke(
        app,
        [
            "register",
            str(fixed_path),
            str(moving_path),
            "-o",
            str(tmp_path / "out"),
            "--fixed-modality",
            "CT",
            "--moving-modality",
            "CT",
            "--spacing",
            "3.0",
            "--overwrite",
            "--log-level",
            "ERROR",
            "--set",
            "qc.gates.min_ncc_gain=0.05",
        ],
    )
    assert result.exit_code == 2, f"expected exit code 2, got {result.exit_code}\n{result.output}"
    assert "FAIL" in result.output
