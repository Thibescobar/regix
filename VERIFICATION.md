# Verification record — Regix 0.3.1

- Date: 2026-08-11
- Source: `main` after `v0.3.0`, plus the local presentation-only `0.3.1` patch;
  prepared without commit, tag, GitHub write or PyPI upload
- Scope: PyPI-safe README image/document links and badges, package project URLs, patch
  version, regression suite, package build and clean-wheel validation; runtime behaviour
  is unchanged from `0.3.0`

This file carries exact evidence that would make the project landing page difficult to
scan. A repository or CI result establishes software behaviour only in its stated
environment; it does not establish clinical accuracy or regulatory fitness.

## Audited environment

| Component | Value |
|---|---|
| OS | macOS 15.2 arm64 |
| Python | 3.12.7 |
| Regix | 0.3.1 |
| Distribution | `regix-medical` |
| itk-elastix | 0.25.4 |
| ITK | 5.4.7 |
| SimpleITK | 2.5.6 |
| NumPy | 2.5.2 |
| pydicom | 3.0.2 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.1 |
| Optional execution | Anatomix, monai, torch/CUDA and TotalSegmentator absent |

The source environment and the wheel environment were isolated under `/tmp`. Doctor
reported the elastix engine available. No stable pseudonym key, production DICOM UID
root, API allowlist or API token was configured in the isolated test process.

## Results

| Check | Command | Result |
|---|---|---|
| Full collection | `python -m pytest -p no:capture --collect-only -q --no-header -p no:cacheprovider` | **225 collected** |
| Full suite | `python -m pytest -p no:capture -q --junitxml=/tmp/regix-031-tests.xml` | **223 passed, 2 skipped; 225 collected**, 0 failures/errors |
| Coverage | `python -m pytest -p no:capture -q --cov=regix --cov-report=term-missing:skip-covered --cov-fail-under=81` | **81.85% (82% rounded)**, floor passed; same 223/2 outcomes |
| Formatting | `ruff format --check regix tests` | pass, 47 files already formatted |
| Lint | `ruff check regix tests` | pass, no findings |
| Dead code | `vulture regix .vulture_allowlist.py --min-confidence 90` | pass, no finding |
| Static types | `mypy regix` | progressive/non-blocking by CI policy: **52 existing errors in 13 files**; no new provider-path error remains |
| Patch hygiene | `git diff --check` | pass |
| Package build | `python -m build --no-isolation --outdir /tmp/regix-031-dist.*` from a temporary source copy | `regix_medical-0.3.1.tar.gz` and universal wheel built |
| Metadata | `twine check /tmp/regix-031-dist.*/*` | both artifacts pass; absolute README image and documentation URLs present |
| Archive inspection | wheel/sdist member lists plus cache, patient-file and common private-key scans | eight presets and licence present; no cache, DICOM/NIfTI, patient file or detected secret; tests are source-only in the sdist and absent from the wheel |
| Wheel execution | install wheel with resolved dependencies outside checkout | `import regix`, `regix --version`, `--help`, `doctor`, `presets` and all eight preset loads pass |
| Extras | pip `--dry-run` separately for `features`, `totalsegmentator`, `organs`, `report`, `api`, `dev`, `all` | all resolve; no ML weights downloaded |
| PyPI baseline | User installation from production PyPI | `regix-medical==0.3.0` installs and `regix doctor` reports the registration engine ready |
| README image | `HEAD https://raw.githubusercontent.com/Thibescobar/regix/main/docs/images/qc-overlay.png` | HTTP 200, `Content-Type: image/png` |
| PyPI badges | `HEAD` on the version and supported-Python shields.io URLs | both HTTP 200, `Content-Type: image/svg+xml` |

The coverage badge is one point above the enforced CI floor, which is the maximum slack
allowed by `tests/test_documentation.py`. Optional model/GPU code is not removed from the
denominator merely because the audited environment cannot execute it.

Pytest's standard capture plugin segfaulted during startup on this macOS/Python build.
Disabling only that plugin with `-p no:capture` made collection and both full runs stable;
subprocess output was still collected by the documentation test itself. This is a local
runner workaround, not a change to the CI command or to registration behaviour.

## Collection groups

The 225 collected tests include these explicitly documented groups:

| File | Collected focus |
|---|---|
| `tests/test_units.py` | 67 configuration, parameter, geometry, transform and metric contracts |
| `tests/test_pipeline.py` | 19 end-to-end, embedded-lifecycle and phantom contracts |
| `tests/test_cli.py` | 27 command and option contracts through the Typer application |
| `tests/test_dicom_io.py` | 10 synthetic DICOM, derived-series and DICOM REG contracts |
| `tests/test_registration_internals.py` | 20 initialization and transform-application contracts |
| `tests/test_audit_regressions.py` | 42 security, geometry, cache, optional-import and descriptor-provider regressions |

The remaining 40 are API, golden/numerical-contract and documentation tests. Collection
counts are verified mechanically against this record so they cannot drift silently.

## Numerical expectations

The repository phantoms establish:

| Scenario | Expected result |
|---|---|
| Rigid + affine CT/CT ground truth | transform error below 1.5 mm |
| Liver centroid initialization | Dice above 0.90 |
| B-spline deformation | similarity gain and folded fraction below `1e-3` |
| CT vs inverted-contrast pseudo-MR | MIND fallback error below 3 mm |
| Unrelated volumes | explicit `FAIL` |
| Native-intensity restitution | air intensity remains below -500 HU |

These are synthetic regression thresholds, not patient-data performance claims. Real
pair assessment requires independent landmarks or contours and locally accepted
tolerances.

## Embedded execution benchmark

`python benchmarks/benchmark_execution.py --repeats 3` runs each lifecycle in an
isolated process:

| Mode | Median time | Peak workspace I/O | Persistent I/O |
|---|---:|---:|---:|
| `run()` standalone | 7.34 s | 391 kB / 17 files | 391 kB / 17 files |
| `compute()` full | 7.15 s | 28 kB / 5 files | 0 |
| `compute(..., registered_image=False, qc=False)` | 7.09 s | 28 kB / 5 files | 0 |

Peak resident memory was 1.18–1.19 GB because ITK and elastix dominated it. These
figures compare lifecycle overhead on the bundled rigid case, not clinical accuracy or
all acquisition profiles. Target deployments must repeat them on representative volumes
and hardware.

## Conditional skips

1. The golden-reference test refuses comparison when the complete ITK/elastix build
   identity differs from the checked-in reference. The reference is not recaptured after
   behavioural changes because that would turn the changed implementation into its own
   oracle. Ground-truth phantoms and two-process invariants still run.
2. The external DICOM-conformance test skips when `dciodvfy` is not installed. Structural
   tests still verify source references, required sequences, Type 2 attributes, dates,
   matrix direction, UID policy and derived-instance cleanup.

Upstream wrapper and client deprecation warnings remain visible so dependency upgrades
can remove them deliberately rather than hiding them.

## Not established by this environment

- Anatomix inference, downloaded weights, model variants, CUDA and the optional
  feature-deformable stage;
- TotalSegmentator execution against CT/MR weights;
- external DICOM conformance, PACS/workstation import and round trip;
- peak capacity on site-scale high-resolution volumes;
- accuracy on patient data, clinical acceptance, regulatory compliance or fitness for
  clinical decisions;
- execution of the GitHub-hosted matrix itself: only a completed remote run proves
  runner and network resolution for every job.

Deployment evidence and operational controls are described in
[`docs/OPERATIONS.md`](docs/OPERATIONS.md). Finding-by-finding audit evidence remains in
[`AUDIT_CLOSURE.md`](AUDIT_CLOSURE.md).
