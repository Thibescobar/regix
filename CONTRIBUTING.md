# Contributing to Regix

Regix handles medical-image geometry and potentially identifiable metadata. A change is
complete only when its numerical behaviour, privacy boundary, and documentation agree.

## Development setup

Regix supports Python 3.10–3.12 at runtime. The static-analysis environment uses Python
3.12 because the current NumPy stubs use modern typing syntax.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,api,report]"
regix doctor
```

Install optional GPU or segmentation extras only for work on those paths. Do not download
model weights merely to run the core suite.

## Safety rules

- Never commit patient volumes, DICOM instances, paths that identify a patient, API
  tokens, pseudonym keys, or site UID roots.
- Use synthetic images and DICOM fixtures in tests. `.gitignore` is a backstop, not a
  substitute for inspecting the staged diff.
- Treat transform direction as part of every public contract. Regix uses physical
  coordinates and documents matrices as `p_fixed = M @ p_moving`.
- Preserve native moving intensities in delivered images. Consumer-specific clipping or
  normalization belongs inside that feature or segmentation consumer.
- Do not silently replace an explicitly requested operation after it fails. Return an
  actionable error, or record a visibly degraded optional result.
- Keep initialization, criterion, ROI, and QC masks in their distinct `MaskRole`s.

## Required checks

Run the focused test while developing, then this full set before proposing a change:

```bash
ruff format --check regix tests
ruff check regix tests
vulture regix .vulture_allowlist.py --min-confidence 90
pytest -q
pytest --cov=regix --cov-report=term-missing --cov-fail-under=81
python -m build
twine check dist/*
```

Mypy is intentionally progressive while legacy boundaries are annotated:

```bash
mypy regix
```

CI reports its findings but does not yet block a merge. New or edited public code should
not add errors. Prefer small typed schemas at module boundaries over broad `Any` values.

## Numerical and golden tests

The synthetic phantom is the ordinary regression oracle. The golden elastix comparison
is build-specific and must never be recaptured after changing registration behaviour.
If a new supported platform needs a reference:

1. Start from a clean, behaviourally unchanged commit.
2. Run `REGIX_UPDATE_GOLDEN=1 pytest tests/test_contract.py -k golden`.
3. Run the test again without the variable and confirm it passes.
4. Commit the reference separately with the OS, Python, ITK, and elastix versions.

For geometry changes, include anisotropic, oblique, and out-of-field cases. Assert on
physical points or millimetres, not only array indices.

## Documentation and release checklist

- Add a regression test named for the failed contract, not for the implementation.
- Update CLI help, configuration descriptions, README examples, and operations guidance
  together when a public option or environment variable changes.
- Run `tests/test_documentation.py`; it mechanically checks option names, preset claims,
  package data, versions, verification counts, and the coverage badge.
- Record security-relevant and behaviour-breaking changes in `CHANGELOG.md`.
- Update `AUDIT_CLOSURE.md` only with reproducible evidence.
- Inspect `git diff --check`, `git status --short`, and the contents of the built wheel.

## Pull requests

Explain the user-visible contract, the failure mode being prevented, and the validation
performed. Call out anything that still requires external weights, a GPU, `dciodvfy`, a
PACS, large-volume hardware, or clinical data. Passing synthetic tests is not clinical
validation.
