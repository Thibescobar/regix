"""Tests that hold the documentation to what the code actually does.

Every assertion in this file corresponds to a claim Regix makes about itself -- in the
README, in a module docstring, in a configuration field description or in a preset
comment -- that can be checked mechanically. The project makes a lot of such claims, and
they are the reason it is readable; they are also the thing that silently rots first,
because nothing fails when a decount, an option name or a preset default drifts away
from its description.

Several of these tests are ``xfail(strict=True)`` and carry the identifier of the audit
finding they document. That is deliberate: the claim is wrong *today*, the test records
exactly how, and the day someone fixes the claim the strict xfail turns the suite red so
the marker gets removed with the fix. A passing xfail is a bug in this file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _collected_test_counts() -> dict[str, int]:
    """Tests actually collected, per file. Runs pytest in a subprocess on purpose.

    Importing pytest's collection machinery from inside a test run is fragile; a
    subprocess gives the same answer a developer gets from the command line, which is
    the number the README is supposed to be quoting.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        match = re.fullmatch(r"(tests/[\w_]+\.py):\s*(\d+)", line.strip())
        if match:
            counts[Path(match.group(1)).name] = int(match.group(2))
    if not counts:
        pytest.skip(f"could not collect tests in a subprocess:\n{proc.stdout}\n{proc.stderr}")
    return counts


def _cli_option_names() -> set[str]:
    """Every long option the Typer application actually accepts, across all commands."""
    from typer.testing import CliRunner

    from regix.cli import app

    runner = CliRunner()
    found: set[str] = set()
    texts = [runner.invoke(app, ["--help"]).output]
    for command in ("register", "batch", "apply", "segment", "inspect", "presets", "doctor", "version"):
        texts.append(runner.invoke(app, [command, "--help"]).output)
    for text in texts:
        # `rich` wraps the help table, so an option can be split across lines; the
        # tokens themselves are never split, which is all we need.
        found.update(re.findall(r"--[a-z][a-z0-9-]+", text))
    return found


def _documented_regix_options() -> dict[str, str]:
    """Long options the documentation tells the reader to type, mapped to their source.

    Only text that is unambiguously about the Regix CLI is considered: the README lines
    that invoke ``regix`` (continuations included), the CLI module docstring, and the
    pydantic field descriptions. Options belonging to other tools -- ``uvicorn --port``,
    ``pytest --cov`` -- are therefore never picked up.
    """
    from regix import config as config_module

    sources: dict[str, str] = {}

    def _harvest(text: str, origin: str, only_regix_lines: bool) -> None:
        joined = text.replace("\\\n", " ")
        for line in joined.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if only_regix_lines and not re.match(r"^\$?\s*regix\s", stripped):
                continue
            for option in re.findall(r"--[a-z][a-z0-9-]+", stripped):
                sources.setdefault(option, origin)

    _harvest(README, "README.md", only_regix_lines=True)
    _harvest(config_module.__doc__ or "", "regix/config.py docstring", only_regix_lines=False)

    from regix.cli import __doc__ as cli_doc

    _harvest(cli_doc or "", "regix/cli.py docstring", only_regix_lines=True)

    # Field descriptions are documentation too: they end up in `config_effective.yaml`,
    # in the run manifest and in `regix presets`.
    for model_name in dir(config_module):
        model = getattr(config_module, model_name)
        fields = getattr(model, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        for field_name, field in fields.items():
            _harvest(
                field.description or "",
                f"{model_name}.{field_name} description",
                only_regix_lines=False,
            )
    return sources


# --------------------------------------------------------------------------- #
# Test counts (audit A-07)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit A-07: the README badge and its Testing section disagree with each other "
    "(122 against 90) and both disagree with the collection; no number is quoted here "
    "because adding tests -- as this wave does -- moves it",
)
def test_readme_test_counts_match_the_collection():
    """The README quotes a total and a per-file breakdown. Both must be the real ones.

    The badge and the prose currently disagree with each other, which is the tell: the
    badge was regenerated at some point and the section was not.
    """
    collected = _collected_test_counts()
    total = sum(collected.values())

    badge = re.search(r"tests-(\d+)%20passed", README)
    assert badge, "the README no longer carries a test-count badge"
    assert int(badge.group(1)) == total, f"badge says {badge.group(1)}, collection says {total}"

    prose_total = re.search(r"pytest\s+#\s*(\d+)\s+tests", README)
    assert prose_total, "the Testing section no longer quotes a total"
    assert int(prose_total.group(1)) == total, f"Testing section says {prose_total.group(1)}, got {total}"

    for filename, count in collected.items():
        quoted = re.search(rf"pytest tests/{re.escape(filename)}\s+#\s*(\d+)", README)
        if quoted:
            assert int(quoted.group(1)) == count, f"{filename}: README says {quoted.group(1)}, got {count}"


# --------------------------------------------------------------------------- #
# CLI options (audit A-05, A-13)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit A-13: the cli.py docstring documents `regix segment --backend`, which "
    "does not exist; audit A-05: FeatureConfig.enabled mentions `--allow-cpu-features`, "
    "which does not exist either",
)
def test_every_documented_cli_option_exists():
    """A command copied out of the documentation must not die on `No such option`.

    This is the cheapest possible guard against the whole class of defect, and it covers
    three sources at once: the README, the module docstrings and the field descriptions
    that users read through `regix presets` and `config_effective.yaml`.
    """
    real = _cli_option_names()
    documented = _documented_regix_options()
    missing = {option: origin for option, origin in documented.items() if option not in real}
    assert not missing, "options documented but absent from the CLI: " + ", ".join(
        f"{option} ({origin})" for option, origin in sorted(missing.items())
    )


def test_the_documentation_mentions_the_main_options():
    """Guard against the previous test passing because it harvested nothing at all."""
    documented = _documented_regix_options()
    for expected in ("--preset", "--organ", "--set", "--dry-run"):
        assert expected in documented, f"{expected} is no longer documented anywhere"


# --------------------------------------------------------------------------- #
# Preset table (audit A-02)
# --------------------------------------------------------------------------- #
def _readme_preset_rows() -> dict[str, str]:
    """The `What it encodes` cell of the README preset table, per preset name."""
    rows: dict[str, str] = {}
    for line in README.splitlines():
        match = re.match(r"^\|\s*`([a-z0-9_]+)`\s*\|(.+)\|\s*$", line.strip())
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_the_readme_preset_table_lists_every_bundled_preset():
    from regix.config import available_presets

    rows = _readme_preset_rows()
    for name in available_presets():
        assert name in rows, f"preset {name} is missing from the README table"


def test_the_readme_preset_table_agrees_on_the_stages():
    """The `Stages` column must match the stage list the preset actually builds."""
    from regix.config import load_preset

    for name, cells in _readme_preset_rows().items():
        cfg = load_preset(name)
        # The prose spells it "B-spline"; the enum value is "bspline".
        described = cells.lower().replace("-", "")
        for stage in cfg.stages:
            assert stage.type.value in described, (
                f"{name}: the README does not mention the {stage.type.value} stage"
            )
        if "rigid only" in described.replace("**", ""):
            assert len(cfg.stages) == 1 and cfg.stages[0].type.value == "rigid", (
                f"{name}: the README says rigid only, the preset has {[s.type.value for s in cfg.stages]}"
            )


@pytest.mark.xfail(
    strict=True,
    reason="audit A-02: the README advertises `N4 on the MR` for ct_mr_abdomen while the "
    "preset sets n4_bias_correction: false, with a comment explaining why",
)
def test_the_readme_preset_table_agrees_on_n4():
    """If the table says N4, the preset must actually enable it."""
    from regix.config import load_preset

    for name, cells in _readme_preset_rows().items():
        if "n4" not in cells.lower():
            continue
        cfg = load_preset(name)
        enabled = cfg.preprocess.fixed.n4_bias_correction or cfg.preprocess.moving.n4_bias_correction
        assert enabled, f"{name}: the README advertises N4 but the preset disables it"


# --------------------------------------------------------------------------- #
# `regix presets NAME` (audit A-01)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit A-01: `regix presets NAME` re-serialises the pydantic model, so the "
    "comments carrying the clinical rationale are lost; the README says they are included",
)
def test_presets_command_prints_the_source_comments():
    """The README points at this command as the way to read a preset `comments included`.

    The comments are where the clinical reasoning lives -- why a positioning CBCT is
    rigid, why N4 is off. Dropping them turns the command into a config dump.
    """
    from typer.testing import CliRunner

    from regix.cli import app

    output = CliRunner().invoke(app, ["presets", "ct_mr_abdomen"]).output
    source = (ROOT / "regix" / "presets" / "ct_mr_abdomen.yaml").read_text(encoding="utf-8")
    assert source.count("#") > 0, "the fixture preset no longer has comments to preserve"
    assert "#" in output, "no comment survived `regix presets ct_mr_abdomen`"


# --------------------------------------------------------------------------- #
# Coverage badge (audit I-06)
# --------------------------------------------------------------------------- #
def test_the_coverage_badge_is_backed_by_the_enforced_threshold():
    from_badge = re.search(r"coverage-(\d+)%25", README)
    assert from_badge, "the README no longer carries a coverage badge"
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    enforced = re.search(r"--cov-fail-under=(\d+)", workflow)
    assert enforced, "CI no longer enforces a coverage floor"
    assert int(from_badge.group(1)) >= int(enforced.group(1)), (
        "the badge claims less coverage than CI enforces, which cannot be right"
    )


@pytest.mark.xfail(
    strict=True,
    reason="audit I-06: CI enforces 75 while the badge claims 78, so the badge can drift "
    "three points before anything turns red -- the workflow comment says it `cannot`",
)
def test_the_coverage_badge_cannot_drift():
    """The CI comment claims the badge cannot silently drift. One point of slack, at most."""
    from_badge = int(re.search(r"coverage-(\d+)%25", README).group(1))
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    enforced = int(re.search(r"--cov-fail-under=(\d+)", workflow).group(1))
    assert from_badge - enforced <= 1, (
        f"badge {from_badge} %, floor {enforced} %: {from_badge - enforced} points of slack"
    )


# --------------------------------------------------------------------------- #
# Organ profiles (audit A-06)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit A-06: hu_window, typical_motion_mm, mask_dilate_mm and roi_margin_mm are "
    "declared per organ and propagated by merged_profile, but no consumer ever reads them; "
    "the README says the profile encodes them",
)
def test_every_organ_profile_field_is_consumed_somewhere():
    """A profile attribute nobody reads is a documented feature that does not exist.

    `labels.py` itself is excluded from the search: defining and propagating a value is
    not the same as acting on it.
    """
    import dataclasses

    from regix.organs.labels import OrganProfile

    package = ROOT / "regix"
    haystack = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package.rglob("*.py")) if path.name != "labels.py"
    )
    unused = [
        field.name
        for field in dataclasses.fields(OrganProfile)
        if field.name not in ("name",) and f".{field.name}" not in haystack
    ]
    assert not unused, f"OrganProfile fields nobody consumes: {unused}"


# --------------------------------------------------------------------------- #
# Environment report (audit A-12)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit A-12: the manifest claims to record every library that numerically "
    "influences the result, but the itk-elastix binding version, pydicom and matplotlib "
    "are absent; the `itk` key holds the ITK core version, not the binding's",
)
def test_the_environment_report_covers_the_libraries_that_change_the_result():
    """elastix is the engine: its version is the single most decisive one to record."""
    from regix.logging_utils import environment_report

    report = environment_report()
    for expected in ("itk_elastix", "pydicom", "matplotlib"):
        assert expected in report, f"the environment report does not record {expected}"


def test_the_environment_report_never_raises_without_the_optional_packages():
    """`doctor` and `/health` both call this; it must degrade, not explode."""
    from regix.logging_utils import environment_report

    report = environment_report()
    assert report["python"] and report["simpleitk"] and report["numpy"]
    assert "cuda_available" in report


# --------------------------------------------------------------------------- #
# Packaging (audit E-12, L-01)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="audit E-12: package-data declares `qc/templates/*.html`, a directory that has never existed",
)
def test_declared_package_data_directories_exist():
    block = re.search(r"\[tool\.setuptools\.package-data\](.*?)(\n\[|\Z)", PYPROJECT, re.S)
    assert block, "package-data is no longer declared"
    for pattern in re.findall(r'"([^"]+)"', block.group(1)):
        directory = (ROOT / "regix" / pattern).parent
        assert directory.is_dir(), f"package-data pattern {pattern!r} points at a missing directory"


def test_the_version_is_the_same_in_pyproject_and_in_the_package():
    """Two literals, no check between them: they agree today, and nothing keeps them so."""
    from regix import __version__

    declared = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT, re.M)
    assert declared, "pyproject no longer declares a version"
    assert declared.group(1) == __version__, (
        f"pyproject says {declared.group(1)}, regix.__version__ says {__version__}"
    )


def test_the_ruff_pin_is_the_same_in_pyproject_and_in_ci():
    """The CI comment states the two are kept in step. Nothing verified it."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    in_project = re.search(r'"(ruff>=[\d.]+,<[\d.]+)"', PYPROJECT)
    in_ci = re.search(r'pip install "(ruff>=[\d.]+,<[\d.]+)"', workflow)
    assert in_project and in_ci, "the ruff pin is no longer stated in both places"
    assert in_project.group(1) == in_ci.group(1), (
        f"pyproject pins {in_project.group(1)}, CI installs {in_ci.group(1)}"
    )
