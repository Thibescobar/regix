# Regix operations guide

Regix is research software, not a medical device. Deployment is responsible for access
control, key management, DICOM identity, capacity planning, interoperability testing,
and validation on representative local data.

## Installation profiles

Use a dedicated environment and record the resolved dependencies. Core CPU operation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install regix-medical
regix doctor --json
regix version --json
```

The repository provides `requirements-lock.txt` for the audited Linux/Python 3.12
numerical environment. It is not a universal cross-platform lock. Build wheels in CI,
scan their contents, and promote the same artifact through validation and production.

Optional profiles are separate: `features`, `totalsegmentator`, `api`, and `report`.
Anatomix is installed explicitly from its upstream repository after reviewing its code,
weights, and terms. `regix doctor` reports unavailable capabilities and whether a
configured operation will degrade or fail.

## CLI workflows

Inspect a DICOM study before selecting the fixed and moving series:

```bash
regix inspect /data/patient001/
```

The primary commands then cover one registration, a batch, transform application and
policy-only QC re-evaluation:

```bash
regix register day0.nii.gz day90.nii.gz -o out/ \
    --preset ct_ct_liver_followup --roi-crop
regix batch pairs.csv -o batch/ --preset ct_ct_liver_followup
regix apply out/transform/final_transform.tfm contours.nii.gz \
    --reference day0.nii.gz -o propagated.nii.gz --label
regix qc out/ --set qc.gates.max_tre_mm=4.0
```

`regix presets NAME` prints the source preset with comments. Use `--dry-run` to inspect
the effective configuration before a run and `--set path=value` for a deliberate local
override; archive the resulting effective configuration with deployment evidence.

## Environment configuration

| Variable | Purpose | Operational rule |
|---|---|---|
| `REGIX_PSEUDONYM_SALT` | Stable HMAC key for pseudonyms | At least 16 unpredictable characters from a secret manager; never store in the repo or command history |
| `REGIX_DICOM_UID_ROOT` | Registered organizational UID root | Obtain from the site/DICOM authority; never send the bundled test root to a production PACS |
| `REGIX_API_ALLOWED_ROOTS` | Path-separated roots readable/writable by the API | Use narrow mounted study/result roots; canonical paths and symlinks are checked |
| `REGIX_API_TOKEN` | Optional static API token | Use only as an additional control behind TLS and site identity; rotate through the deployment secret store |

Diagnostics and manifests record booleans plus a non-secret pseudonym-key fingerprint,
not secret values or allowlisted paths.

If `REGIX_PSEUDONYM_SALT` is absent, Regix creates a random key once per process. The
result resists trivial enumeration but pseudonyms change after restart. An empty value is
an error; a weak value produces a visible warning. Pseudonymisation does not remove all
identifying pixel content or DICOM metadata and is not anonymisation.

## HTTP service

The API exchanges paths on already mounted storage. A minimal launch is:

```bash
export REGIX_API_ALLOWED_ROOTS=/srv/regix/studies:/srv/regix/results
export REGIX_API_TOKEN='site-managed-secret'
uvicorn regix.api:app --host 127.0.0.1 --port 8000
```

Every request resolves existing inputs and the nearest existing parent of an output,
then verifies containment below an allowed root. Traversal and symlink escapes are
refused. When the token variable is set, send either `Authorization: Bearer ...` or
`X-Regix-Token: ...`; comparison is constant-time.

The built-in executor has one worker because elastix is internally multithreaded. The
job history is bounded to 1,000 entries and `/jobs` is paginated, but state is in memory
and disappears on restart. A real service needs:

- TLS, site authentication/authorization, and access audit logs at a trusted proxy;
- a durable bounded queue and database outside the web process;
- per-user/path authorization stricter than the global filesystem allowlist;
- CPU/memory quotas, request rate limits, and output retention policy;
- a process model tested with the selected ITK thread settings.

Client-visible job failures are opaque. Diagnose with protected server logs; do not add
filesystem paths or exception messages back to HTTP responses.

A minimal authenticated submission is:

```bash
curl -X POST http://localhost:8000/register \
  -H 'Authorization: Bearer site-managed-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "fixed": "/srv/regix/studies/patient001/day0.nii.gz",
    "moving": "/srv/regix/studies/patient001/day90.nii.gz",
    "output_dir": "/srv/regix/results/patient001",
    "preset": "ct_ct_liver_followup"
  }'
```

Poll `GET /jobs/{job_id}` for completion. `GET /health` reports capability state,
`GET /presets` lists configurations, and paginated `GET /jobs` is operational history,
not durable audit storage.

## DICOM identity and export

Input discovery groups all readable instances by `SeriesInstanceUID`, even when a series
is split across subdirectories. If more than one series is eligible, pass
`fixed_series_uid` and `moving_series_uid` (or the matching CLI options) rather than
relying on the reported selection heuristic.

Configure a registered UID root before enabling production DICOM export. Regix creates
new instance/series UIDs and adopts the fixed Frame of Reference for registered output.
The Spatial Registration Object references every source SOP instance and contains the
required empty Type 2 attributes. It represents linear registration only. A nonlinear
result requires a Deformable Spatial Registration object, which Regix does not emit.

Derived series remove private elements and stale reference sequences. The source storage
class is preserved only for source modalities whose pixel module remains valid after
resampling; unsupported classes are refused rather than relabelled optimistically.

Before PACS use:

1. Run the DICOM tests with `dciodvfy` installed and retain its output.
2. Validate every enabled source modality/SOP class against the DICOM standard.
3. Import into a non-production PACS/workstation and verify references, frame of
   reference, matrix direction, display, and round-trip retrieval.
4. Obtain site approval for UID ownership, de-identification, retention, and audit logs.

Synthetic pydicom tests are necessary but not an interoperability certificate.

## Output lifecycle and replay

A run refuses a non-empty output directory unless overwrite is enabled. Overwrite removes
only canonical Regix artifacts from `layout.py`; unrelated review notes remain. Give each
case a dedicated output directory and prevent concurrent writers to it.

The manifest inventory uses relative paths and strict JSON. Linear runs normally contain
`final_transform.tfm`, `final_transform.itk.txt`, and
`moving_to_fixed_matrix.txt`. Stage files use `.elastix.txt` and their parameter
snapshots use `.parameters.txt`. DICOM REG is conditional on two DICOM inputs and a
linear transform; do not build downstream automation that assumes it always exists.

Use `regix qc OUT_DIR --set ...` to apply a revised gate policy to stored metrics. It
does not recompute similarity, contours, landmarks, or fields; rerun registration/QC if
those inputs changed.

`RegistrationPipeline.compute()` is the integration boundary for an application that
owns its own persistence. It leaves no Regix files and its temporary elastix workspace
is cleaned automatically. Consequently it also leaves no Regix manifest, replay bundle,
report or failure artifact: the parent application must provide any required audit log,
retention and failure trace. Use `run()` for a standalone deliverable.

### Python result and caller responsibilities

Both Python entry points return `RegistrationResult`:

| Field | Practical use |
|---|---|
| `status`, `ok`, `warnings` | Gate verdict and visible degradations; `ok` is true only for `PASS` |
| `applied_transform` | Owned moving-to-fixed mapping for points and resampling |
| `registered_image` | Native moving intensities on the fixed grid, unless transform-only mode omitted it |
| `metrics`, `qc` | Similarity, feature-provider trace, geometry/organ/deformation measurements and gate details |
| `stages`, `initialization` | Executed stage summaries and selected initialization |
| `outputs`, `manifest_path`, `seconds` | Persistent bundle locations for `run()` and elapsed time |

| Entry point | Available after return | Not provided |
|---|---|---|
| `run()` | Result, registered image, configured outputs, manifest and replay/review bundle | Only outputs explicitly disabled or inapplicable |
| `compute()` | Result, owned transform, optional registered image and optional QC | Persistent Regix artifacts, report, manifest and replay bundle |
| Transform-only `compute()` | Owned transform, stages, initialization and non-image result data | Registered image, image-dependent QC and every persistent artifact |

Configuration/input errors, unavailable strict descriptor providers, engine failures and
invalid output/work directories raise explicit exceptions. A completed computation can
still return `WARN` or `FAIL`; the caller must interpret the QC policy rather than treat
return as acceptance. An embedding application owns its persistence, access control,
traceability, retry/idempotency policy, cleanup after its own failures and recovery after
process interruption. Use `run()` when Regix should own those standalone artifacts.

### Reviewing a standalone result

The HTML report starts with the gate verdict and shows a fixed-grey/moving-hot overlay
before and after registration. The checkerboard is complementary: seams make local
discontinuities visible when transparency can hide them.

![Checkerboard quality-control view](images/qc-checkerboard.png)

Review the report together with independent landmarks or contours whenever available.
Similarity values alone are the weakest evidence because related values may have driven
the optimizer. The reliability order is landmarks/TRE, organ overlap and surface
distances, deformation plausibility/Jacobian, then NCC or NMI.

## Capacity and large volumes

Working spacing, ROI cropping, feature channels, dense deformation fields, QC figures,
and segmentation determine peak memory. Regix reduces avoidable copies, chunks MIND,
and uses analytical linear statistics, but it does not provide out-of-core registration.

For a new acquisition profile:

1. Benchmark representative largest volumes in an isolated worker with memory limits.
2. Include segmentation/features and report generation, not only elastix.
3. Prefer an anatomically justified ROI and working spacing.
4. Set queue concurrency from measured peak resident memory; do not multiply by CPU count.
5. Exercise cancellation, disk-full, timeout, and worker-restart recovery.

TotalSegmentator has a configurable subprocess timeout (`organs.ts_timeout_seconds` and
CLI `--timeout`). A timeout ends that operation; verify the site's process supervisor
also handles orphaned GPU work.

## Validation layers

| Layer | Repository evidence | Deployment evidence still required |
|---|---|---|
| Software contracts | 225 synthetic/unit/API/CLI/DICOM tests, lint, dead-code scan, package build | Repeat on promoted wheel and target OS/ITK build |
| Numerical registration | Ground-truth phantoms, oblique/thick-slice grids, transform probes | Representative scanner protocols and accepted local tolerances |
| DICOM | Structural pydicom checks; optional `dciodvfy` test | Validator installed, PACS/workstation round trip, site UID policy |
| Optional ML/GPU | Lazy-import and failure/fallback contracts | Exact weights, GPU/driver stack, domain-shift and deterministic-behaviour assessment |
| Clinical | No claim | Independent landmarks/contours, reader review, risk management, regulatory/site approval |

Archive `regix doctor --json`, `regix version --json`, effective configuration,
manifest, QC report, dependency resolution, container/wheel digest, and external
validation records together. A `PASS` is a software gate verdict for configured metrics;
it is not clinical clearance.
