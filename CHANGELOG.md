# Changelog

All notable changes are recorded here. Regix follows semantic versioning for its Python
and CLI contracts.

## 0.3.0 — 2026-08-08

### Packaging and descriptor selection

- Prepared the PyPI distribution as `regix-medical` while preserving the `regix`
  package, `import regix` and the `regix` command; updated the self-referential `all`
  extra and user installation guidance.
- Added `features.provider=auto|anatomix|mind` and CLI `--feature-provider`. The default
  keeps the existing Anatomix -> MIND-SSC -> intensity fallback; explicit Anatomix and
  MIND modes are strict and record the requested and effective descriptors.
- Kept the MIND-only path independent of Anatomix, torch and GPU imports, and clarified
  that `features.allow_cpu` applies only to Anatomix.
- Removed the obsolete post-audit handoff guide; durable audit and verification evidence
  remains in `AUDIT.md`, `AUDIT_CLOSURE.md` and `VERIFICATION.md`.

### Embedded execution

- Added `RegistrationPipeline.compute()` and the `regix.compute()` shortcut. They run
  the same registration path as `run()` but leave no persistent Regix run artifacts.
- Added a transform-only option (`registered_image=False, qc=False`) that skips native
  restitution and QC when a larger application only needs the mapping.
- Detached returned transforms from elastix parameter files before the disposable
  workspace is removed, including a dense-field fallback for non-convertible chains.
- Temporary workspaces are cleaned after both success and failure; result objects omit
  transient artifact paths.

### Performance and validation

- Embedded runs skip audit-only environment inventory, input descriptions/hashes,
  replay inputs, exports, reports and manifests. Numerical equivalence with standalone
  execution is covered at transform, image and QC-metric levels.
- Added `benchmarks/benchmark_execution.py`, which isolates each mode in a fresh process
  and records elapsed time, peak resident memory, peak workspace I/O and persistent I/O.
- Coverage is 82% on the 216-test CPU suite; CI now enforces an 81% floor.
- Kept `run()` and all standalone artifact contracts backward compatible.

## 0.2.0 — 2026-08-07

This release closes the 134 findings recorded in `AUDIT.md`. It is a security and
correctness release with deliberate breaking changes.

### Security and privacy

- Replaced the public default pseudonym salt with HMAC-SHA256. An absent key now creates
  a random process-local key; stable pseudonyms require a site-managed
  `REGIX_PSEUDONYM_SALT` of at least 16 unpredictable characters.
- Removed source paths from manifests, effective configuration, client-visible API
  failures, and routine logs.
- Confined the HTTP API to canonical paths below `REGIX_API_ALLOWED_ROOTS`, including
  symlink resolution; added optional constant-time token authentication through
  `REGIX_API_TOKEN`.
- Validated batch case names and bounded/paginated the in-memory API job store.
- Made DICOM UID roots configurable and added explicit production warnings for the
  bundled test root.

### Registration correctness

- Fixed duplicate organ aliases so composite organs such as the left lung contain all
  applicable labels.
- Corrected 4D input extraction, `.itk.txt` transform loading, quantization detection,
  explicit-initialization failure handling, null overrides, overwrite cleanup, DICOM
  series merging, multimodal multi-start scoring, and transform-conversion probes.
- Introduced role-bearing masks and a per-run `RunState`; pipeline inputs and
  configuration are no longer mutated or retained across runs.
- Body masks are computed once from native intensities, with physical morphology on a
  coarse binary grid and nearest-neighbour resampling.
- Added physiological displacement gates, explicit initialization-QC contracts, strict
  metric schemas, and `NOT_EVALUATED` when QC is disabled.
- Added native-grid, anisotropic, thick-slice, oblique, and out-of-field regression
  coverage.

### DICOM and interoperability

- Raised the pydicom requirement to 3.x.
- Added explicit series-UID selection and safe merging of a series distributed across
  directories.
- Spatial Registration Objects now reference every source instance and include required
  Type 2 attributes. Derived instances remove stale references/private elements and use
  a controlled SOP-class policy.
- Distinguished Insight transform files (`.itk.txt`) from elastix parameter files
  (`.elastix.txt`) and unified transform loading.

### Reliability and performance

- Hardened elastix parameter parsing and serialization, landmark coordinate handling,
  principal-axis degeneracy checks, segmentation cache identity, optional-import error
  handling, and numeric configuration bounds.
- Added a configurable TotalSegmentator task, fast mode, and subprocess timeout.
- Reduced full-volume copies in intensity QC and volume descriptions; vectorized
  principal axes and surface metrics; bounded MIND memory; avoided dense displacement
  fields for linear transforms; added compact WebP/sidecar report figures.
- Added `regix qc OUT_DIR` to re-evaluate stored metrics without rerunning registration.

### Packaging, CI, and documentation

- Version is now sourced once from `regix.__version__`; dependency ranges have upper
  bounds and a reproducible Python 3.12 numerical lock is provided.
- CI covers Python 3.10–3.12, extras installation, formatting, lint, dead-code detection,
  coverage, package build, wheel installation, and documentation contracts. Mypy is
  reported progressively.
- Added architecture, operations, contribution, audit-closure, and verification guides.
- Expanded the suite to 212 tests and enforced 78% coverage.

### Breaking changes

- Pseudonyms are no longer stable without an explicitly configured secret.
- Empty pseudonym salts, 2D volumes, elastix index-coordinate landmarks, malformed
  parameter files, unsafe batch names, and invalid numeric ranges are rejected.
- The minimum pydicom version is 3.0 and supported Python is 3.10–3.12.
- `WARN` is no longer used as a surrogate for disabled QC; the status is
  `NOT_EVALUATED`.

### Validation still owned by the deployment site

GPU/anatomix/TotalSegmentator execution, `dciodvfy` conformance on an installation that
provides it, PACS interoperability, large-volume capacity, and clinical accuracy were
not established by this CPU-only synthetic test environment.
