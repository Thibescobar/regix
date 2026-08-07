"""The safety net the correction waves lean on: what a run produces, and what it computes.

Two distinct guarantees live here.

**Numerical.** A golden reference of the final transform, sampled over 400 points inside
the volume. Refactoring the pipeline must not move a single one of them. This is the
only test that can tell a clean-up apart from a silent change of behaviour, and every
subsequent wave of corrections is checked against it.

**Structural.** What a finished run leaves on disk, that its manifest is strict JSON, and
that no input path leaks into the artefacts. These are contracts the project states in
prose -- "a run manifest per run", "no patient identifier in clear text", "replayable
as-is with the elastix binary" -- and that nothing verified.

The golden reference is machine-specific on purpose. elastix's determinism is a property
of the build, not a guarantee Regix makes (the README says so, and is right to), so a
reference captured elsewhere would fail for reasons that have nothing to do with the
code under test. When the recorded engine version does not match the running one, the
comparison skips loudly instead of failing. Regenerate deliberately with:

    REGIX_UPDATE_GOLDEN=1 pytest tests/test_contract.py -k golden
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from tests.conftest import known_rigid, make_phantom, warp

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_transform.json"

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
def _reference_config(out_dir: Path):
    """The configuration the golden reference was captured with. Do not drift.

    Small, deterministic, and deliberately not a preset: a preset is allowed to evolve,
    whereas the golden reference must describe a frozen computation.
    """
    from regix.config import load_preset

    return load_preset("base").with_overrides(
        preprocess={"working_spacing_mm": 2.5},
        output={"dir": str(out_dir), "overwrite": True},
        runtime={"log_level": "WARNING"},
        qc={"report_html": False},
    )


@pytest.fixture(scope="module")
def reference_pair(tmp_path_factory):
    """A fixed CT-CT phantom pair on disk, identical from one session to the next."""
    tmp = tmp_path_factory.mktemp("reference_pair")
    image, labels = make_phantom("CT", shape=(40, 48, 48), noise=3.0, seed=0)
    truth = known_rigid(image, rotation_deg=(3.0, -2.0, 4.0), translation_mm=(5.0, -4.0, 3.0))
    moving = warp(image, truth)

    paths = {}
    for name, img in (("fixed", image), ("moving", moving), ("fixed_labels", labels)):
        dtype = sitk.sitkUInt16 if name.endswith("labels") else sitk.sitkFloat32
        path = tmp / f"{name}.nii.gz"
        sitk.WriteImage(sitk.Cast(img, dtype), str(path), True)
        paths[name] = path
    return paths, truth


def _probe_points(reference: sitk.Image, n: int = 400) -> list[list[float]]:
    """Points spread through the interior of the volume, in physical coordinates.

    Seeded and index-based, so the same physical points come back for a given grid --
    including an oblique one, since the index-to-physical mapping carries the direction.
    """
    rng = np.random.default_rng(20250101)
    size = np.asarray(reference.GetSize(), dtype=float)
    indices = rng.uniform(0.1, 0.9, size=(n, 3)) * (size - 1)
    return [list(reference.TransformContinuousIndexToPhysicalPoint(list(idx))) for idx in indices]


def _engine_version() -> str:
    from regix.registration.itk_bridge import engine_available

    return engine_available()[1]


# --------------------------------------------------------------------------- #
# 0.2 -- numerical non-regression
# --------------------------------------------------------------------------- #
def test_the_final_transform_matches_the_golden_reference(tmp_path, reference_pair):
    """400 points, 1e-6 mm. Any change here is a change of behaviour, not a refactoring.

    The tolerance is deliberately far below clinical relevance: the point is not "close
    enough to be correct", it is "bit-for-bit the same computation". A genuine
    improvement is expected to fail this test and to be recorded by regenerating the
    reference in the same commit.
    """
    from regix.pipeline import RegistrationPipeline

    paths, _ = reference_pair
    out = tmp_path / "golden_run"
    result = RegistrationPipeline(_reference_config(out)).run(paths["fixed"], paths["moving"], out)
    transform = result.applied_transform.as_sitk_transform()
    assert transform is not None, "the reference run must yield a usable sitk transform"

    fixed = sitk.ReadImage(str(paths["fixed"]))
    points = _probe_points(fixed)
    mapped = [list(map(float, transform.TransformPoint(p))) for p in points]

    if os.environ.get("REGIX_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(
            json.dumps(
                {
                    "engine": _engine_version(),
                    "platform": sys.platform,
                    "n_points": len(points),
                    "points": points,
                    "mapped": mapped,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        pytest.skip(f"golden reference regenerated at {GOLDEN}")

    if not GOLDEN.exists():
        pytest.skip(f"no golden reference; create one with REGIX_UPDATE_GOLDEN=1 ({GOLDEN})")

    recorded = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # Both the engine *and* the platform have to match. Two machines can carry the same
    # ITK version and still differ: floating-point contraction, BLAS build and thread
    # count all vary, and elastix samples in parallel. Comparing across them at 1e-6 mm
    # would fail for reasons that have nothing to do with the code under test -- which
    # is exactly what would greet someone resuming this work on another machine.
    captured = (recorded["engine"], recorded.get("platform"))
    current = (_engine_version(), sys.platform)
    if captured != current:
        pytest.skip(
            f"golden reference captured on {captured}, running on {current}: elastix "
            "determinism is a property of the build, so this comparison would be "
            "meaningless. Capture a reference for this machine with REGIX_UPDATE_GOLDEN=1 "
            "on an unmodified checkout, before changing any behaviour."
        )

    # Compare the probes themselves first: if `_probe_points` or the phantom geometry
    # ever changes, the two sets of images below would describe different questions and
    # the comparison would quietly stop meaning anything.
    assert np.allclose(points, recorded["points"], atol=1e-9), (
        "the probe points moved: the golden reference describes a different set of "
        "positions than the one being measured. Regenerate with REGIX_UPDATE_GOLDEN=1."
    )

    deviations = np.linalg.norm(np.asarray(mapped) - np.asarray(recorded["mapped"]), axis=1)
    assert float(deviations.max()) < 1e-6, (
        f"the transform moved: max {deviations.max():.3e} mm, mean {deviations.mean():.3e} mm "
        f"over {len(deviations)} points"
    )


def test_two_runs_of_the_same_configuration_agree(tmp_path, reference_pair):
    """The determinism the README quotes as `0.000 mm point-wise over 400 points`.

    Audit A-16: that figure appears in the README with no test behind it. Run the same
    configuration twice in two separate processes -- same process would share elastix's
    global state and prove less -- and measure. The README is careful to present this as
    a property of the elastix build rather than a Regix guarantee; this test is what
    turns the claim into something a deploying site can re-check on its own build.
    """
    paths, _ = reference_pair
    script = ROOT / "tests" / "data" / "_determinism_probe.py"
    outputs = []
    for run in ("a", "b"):
        proc = subprocess.run(
            [sys.executable, str(script), str(paths["fixed"]), str(paths["moving"]), str(tmp_path / run)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0" if run == "a" else "12345"},
        )
        if proc.returncode != 0:
            pytest.skip(f"determinism probe could not run:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        outputs.append(np.asarray(json.loads(proc.stdout.strip().splitlines()[-1])))

    deviations = np.linalg.norm(outputs[0] - outputs[1], axis=1)
    assert float(deviations.max()) < 1e-6, (
        f"two identical runs disagree by up to {deviations.max():.3e} mm over "
        f"{len(deviations)} points: this elastix build is not deterministic, and the "
        "README's reproducibility paragraph does not hold here"
    )


# --------------------------------------------------------------------------- #
# 0.3 -- artefact contract
# --------------------------------------------------------------------------- #
#: What a base run must leave behind. Hard-coded here on purpose: `regix/layout.py`
#: does not exist yet (audit D-06), and the cleanup that `--overwrite` is missing
#: (audit B-09) needs exactly this inventory to be written safely.
REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "config_effective.yaml",
    "regix.log",
    "moving_registered.nii.gz",
    "transform/final_transform.tfm",
    "transform/final_transform.txt",
    "transform/moving_to_fixed_matrix.txt",
)

#: Top-level entries a base run is allowed to create. An entry appearing here that is
#: not in this list is either a new output nobody documented, or a leftover.
ALLOWED_TOP_LEVEL = {
    "run_manifest.json",
    "config_effective.yaml",
    "regix.log",
    "report.html",
    "moving_registered.nii.gz",
    "transform",
    "elastix",
    "masks",
    "features",
    "cache",
    "transformix",
    "deformation_field.nii.gz",
    "jacobian.nii.gz",
    "dicom_registered",
}


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory, reference_pair):
    """One completed run, reused by the structural tests: they only read from it."""
    from regix.pipeline import RegistrationPipeline

    paths, _ = reference_pair
    out = tmp_path_factory.mktemp("finished") / "out"
    cfg = _reference_config(out).with_overrides(qc={"report_html": True})
    result = RegistrationPipeline(cfg).run(paths["fixed"], paths["moving"], out)
    return out, result


def test_a_run_produces_every_declared_artifact(finished_run):
    out, _ = finished_run
    missing = [name for name in REQUIRED_ARTIFACTS if not (out / name).exists()]
    assert not missing, f"missing artefacts: {missing}"


def test_a_run_creates_nothing_undeclared(finished_run):
    """A new output should be a deliberate decision, not a surprise in the directory."""
    out, _ = finished_run
    unexpected = sorted(p.name for p in out.iterdir() if p.name not in ALLOWED_TOP_LEVEL)
    assert not unexpected, f"undeclared entries in the output directory: {unexpected}"


def test_the_manifest_is_strict_json(finished_run):
    """`NaN` is what json.dumps writes for an unavailable metric, and it is not JSON.

    Audit H-07: `_round` returns float("nan"), `RunManifest.save` leaves allow_nan at its
    default, and any strict consumer -- JSON.parse, jq, most Go/Rust parsers -- rejects
    the file. A nominal run does not produce one, which is precisely why this needs a
    test rather than an inspection.
    """
    out, _ = finished_run
    raw = (out / "run_manifest.json").read_text(encoding="utf-8")

    def _reject(constant: str):
        raise AssertionError(f"run_manifest.json contains the non-JSON constant {constant!r}")

    json.loads(raw, parse_constant=_reject)


@pytest.mark.xfail(
    strict=True,
    reason="audit H-07: metrics._round returns float('nan') for an unavailable measure "
    "and RunManifest.save leaves json.dumps at allow_nan=True, so a degraded run writes "
    "a manifest that strict JSON parsers reject",
)
def test_the_manifest_refuses_to_serialise_a_non_finite_metric(tmp_path):
    """The degraded path the nominal run cannot reach, exercised directly.

    An unavailable NCC (fewer than 64 voxels in the QC mask) becomes float("nan") and
    travels all the way into the manifest. Rather than build that pipeline state, feed
    the manifest what the metrics module would give it.
    """
    from regix.logging_utils import RunManifest

    manifest = RunManifest(run_id="degraded", output_dir=tmp_path)
    manifest.metrics = {"similarity": {"ncc_after": float("nan"), "nmi_after": 1.02}}
    path = manifest.save()

    def _reject(constant: str):
        raise AssertionError(f"the manifest serialised the non-JSON constant {constant!r}")

    json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


@pytest.mark.xfail(
    strict=True,
    reason="audit C-02: run_manifest.json records inputs.<side>.source verbatim, and "
    "config_effective.yaml records every configured path; both are written next to a "
    "carefully pseudonymised subject_id",
)
def test_no_input_path_reaches_the_artifacts(tmp_path):
    """In a department, a path is an identifier: `/studies/DUPONT_Jean_19540312/CT`.

    Audit C-02: Regix pseudonymises `subject_id` carefully and then writes the source
    path next to it, in the manifest and in `config_effective.yaml`. The canary below is
    a directory name that could only come from the input path.
    """
    from regix.pipeline import RegistrationPipeline

    canary = "PATIENTNAMECANARY"
    data = tmp_path / canary
    data.mkdir()
    image, _ = make_phantom("CT", shape=(28, 32, 32), noise=2.0)
    truth = known_rigid(image, rotation_deg=(1.0, 0.0, 1.0), translation_mm=(2.0, -1.0, 1.0))
    for name, img in (("fixed", image), ("moving", warp(image, truth))):
        sitk.WriteImage(sitk.Cast(img, sitk.sitkFloat32), str(data / f"{name}.nii.gz"), True)

    out = tmp_path / "out"
    cfg = _reference_config(out).with_overrides(qc={"report_html": True})
    RegistrationPipeline(cfg).run(data / "fixed.nii.gz", data / "moving.nii.gz", out)

    leaking = [
        path.relative_to(out).as_posix()
        for path in out.rglob("*")
        if path.is_file()
        and path.suffix in (".json", ".yaml", ".html", ".log", ".txt")
        and canary in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not leaking, f"the input path leaked into: {leaking}"


@pytest.mark.xfail(
    strict=True,
    reason="audit A-03: the images handed to elastix (reoriented, clipped, resampled to "
    "the working resolution) are never written, so the `// Replay with:` line in every "
    "parameters.txt points at files that do not exist",
)
def test_the_replay_line_points_at_files_that_exist(finished_run):
    """`replayable as-is with the elastix binary` -- README, and the report footer.

    The generated files say how to replay themselves. Take them at their word.
    """
    out, _ = finished_run
    parameter_files = sorted(out.glob("elastix/*/parameters.txt"))
    assert parameter_files, "no elastix parameter file was written"
    for parameters in parameter_files:
        line = next(
            (ln for ln in parameters.read_text(encoding="utf-8").splitlines() if "Replay with:" in ln),
            None,
        )
        assert line, f"{parameters} carries no replay instruction"
        for token in re.findall(r"[\w./-]+\.nii(?:\.gz)?", line):
            assert (parameters.parent / token).exists(), (
                f"{parameters.relative_to(out)} tells the reader to replay with {token}, "
                "which the run never wrote"
            )
