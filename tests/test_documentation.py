"""Tests that hold the documentation to what the code actually does.

Every assertion in this file corresponds to a claim Regix makes about itself -- in the
README, in a module docstring, in a configuration field description or in a preset
comment -- that can be checked mechanically. The project makes a lot of such claims, and
they are the reason it is readable; they are also the thing that silently rots first,
because nothing fails when a decount, an option name or a preset default drifts away
from its description.

The assertions started as strict ``xfail`` reproductions during the audit. They are now
ordinary regression tests: every previously false claim they cover has been corrected.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
VERIFICATION = (ROOT / "VERIFICATION.md").read_text(encoding="utf-8")
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
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:capture",
            "-p",
            "no:cacheprovider",
        ],
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
    from typer.main import get_command
    from typer.testing import CliRunner

    from regix.cli import app

    runner = CliRunner()
    found: set[str] = set()
    # A wide terminal keeps Rich from replacing long option names with an ellipsis.
    # The default test terminal is narrower than ordinary documentation examples.
    texts = [runner.invoke(app, ["--help"], terminal_width=240).output]
    for command in (
        "register",
        "batch",
        "apply",
        "segment",
        "inspect",
        "presets",
        "doctor",
        "version",
        "qc",
    ):
        texts.append(runner.invoke(app, [command, "--help"], terminal_width=240).output)
    for text in texts:
        # `rich` wraps the help table, so an option can be split across lines; the
        # tokens themselves are never split, which is all we need.
        found.update(re.findall(r"--[a-z][a-z0-9-]+", text))
    # Rich intentionally ellipsizes long names even with a wide synthetic terminal.
    # Click's command tree is the authoritative accepted-token list.
    root = get_command(app)
    commands = [root, *getattr(root, "commands", {}).values()]
    for command in commands:
        for parameter in command.params:
            found.update(option for option in getattr(parameter, "opts", ()) if option.startswith("--"))
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
def test_verification_test_count_matches_the_collection():
    """The durable verification record must quote the collection it describes.

    A volatile test-count badge and a long command inventory made the README harder to
    scan. The evidence now lives with the exact environment and conditional skips it
    qualifies, while this test still prevents it from drifting.
    """
    collected = _collected_test_counts()
    total = sum(collected.values())

    quoted = re.search(r"(\d+)\s+collected", VERIFICATION)
    assert quoted, "VERIFICATION.md no longer quotes the collected-test total"
    assert int(quoted.group(1)) == total, f"verification says {quoted.group(1)}, collection says {total}"


# --------------------------------------------------------------------------- #
# CLI options (audit A-05, A-13)
# --------------------------------------------------------------------------- #
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


def test_the_readme_preset_table_agrees_on_n4():
    """An N4 claim must state the preset's actual enabled/disabled state."""
    from regix.config import load_preset

    for name, cells in _readme_preset_rows().items():
        if "n4" not in cells.lower():
            continue
        cfg = load_preset(name)
        enabled = cfg.preprocess.fixed.n4_bias_correction or cfg.preprocess.moving.n4_bias_correction
        described_as_disabled = any(word in cells.lower() for word in ("disabled", "off"))
        assert enabled is not described_as_disabled, (
            f"{name}: the README and preset disagree about whether N4 is enabled"
        )


# --------------------------------------------------------------------------- #
# `regix presets NAME` (audit A-01)
# --------------------------------------------------------------------------- #
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
def test_declared_package_data_directories_exist():
    block = re.search(r"\[tool\.setuptools\.package-data\](.*?)(\n\[|\Z)", PYPROJECT, re.S)
    assert block, "package-data is no longer declared"
    for pattern in re.findall(r'"([^"]+)"', block.group(1)):
        directory = (ROOT / "regix" / pattern).parent
        assert directory.is_dir(), f"package-data pattern {pattern!r} points at a missing directory"


def test_the_version_is_the_same_in_pyproject_and_in_the_package():
    """The package attribute is the single version source used by setuptools."""
    from regix import __version__

    assert re.search(r'^dynamic\s*=\s*\["version"\]', PYPROJECT, re.M)
    assert 'version = { attr = "regix.__version__" }' in PYPROJECT
    assert not re.search(r'^version\s*=\s*"', PYPROJECT, re.M)
    assert __version__ == "0.3.0"


def test_distribution_name_and_install_examples_match():
    assert re.search(r'^name\s*=\s*"regix-medical"', PYPROJECT, re.M)
    assert 'all = ["regix-medical[features,organs,api,report]"]' in PYPROJECT
    for command in (
        "pip install regix-medical",
        'pip install "regix-medical[report]"',
        'pip install "regix-medical[api,report]"',
        'pip install "regix-medical[features]"',
        'pip install "regix-medical[totalsegmentator]"',
    ):
        assert command in README, f"missing documented install: {command}"


def test_documented_feature_providers_are_exactly_the_accepted_values():
    from typing import get_args

    from regix.cli import FeatureProvider
    from regix.config import FeatureConfig

    accepted = set(get_args(FeatureConfig.model_fields["provider"].annotation))
    cli_values = {provider.value for provider in FeatureProvider}
    documented = set(re.findall(r"provider=(auto|anatomix|mind)", README))
    assert accepted == {"auto", "anatomix", "mind"}
    assert cli_values == accepted
    assert documented == accepted


def test_the_ruff_pin_is_the_same_in_pyproject_and_in_ci():
    """The CI comment states the two are kept in step. Nothing verified it."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    in_project = re.search(r'"(ruff>=[\d.]+,<[\d.]+)"', PYPROJECT)
    in_ci = re.search(r'pip install "(ruff>=[\d.]+,<[\d.]+)"', workflow)
    assert in_project and in_ci, "the ruff pin is no longer stated in both places"
    assert in_project.group(1) == in_ci.group(1), (
        f"pyproject pins {in_project.group(1)}, CI installs {in_ci.group(1)}"
    )
