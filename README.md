# Regix

Multimodal, multi-organ medical image registration with elastix.

![CI](https://github.com/Thibescobar/regix/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![Coverage](https://img.shields.io/badge/coverage-82%25-yellowgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Regix turns DICOM series or NIfTI volumes into an auditable registration result. It
combines rigid, affine and deformable elastix stages with organ-aware initialization,
optional modality-invariant features, independent quality checks and clinical-system
exports. The same algorithm can run as a standalone workflow or as an in-memory
component of a larger application.

![Registration quality-control overlay](https://raw.githubusercontent.com/Thibescobar/regix/main/docs/images/qc-overlay.png)

> **Not a medical device.** Regix is research software. It has not been cleared or
> approved by any regulatory authority. No clinical decision should rest on its output
> without qualified review and validation on representative data from the deploying
> site.

## Why Regix

Registration engines provide an optimizer; production workflows need the surrounding
contracts too. Regix adds:

- DICOM series discovery, NIfTI loading and physical-geometry checks;
- rigid, affine and B-spline stages with explicit initialization and transform chaining;
- CT/MR comparison through mutual information, Anatomix features or CPU MIND-SSC;
- organ masks for initialization, criterion support, ROI cropping and Dice measurement;
- native-intensity restitution on the original fixed grid;
- explicit `PASS`, `WARN` or `FAIL` quality gates, a manifest and an HTML report;
- ITK, 3D Slicer and eligible DICOM registration exports;
- a no-artifact Python path for applications that own persistence and traceability.

Three invariants guide the implementation:

1. Preprocessing helps the optimizer but never changes delivered intensities.
2. Quality control does not rely only on the criterion that was optimized.
3. A degraded or unavailable result is reported, never hidden behind a silent fallback.

## Quick start

### Installation

```bash
pip install regix-medical
pip install "regix-medical[report]"
pip install "regix-medical[api,report]"
pip install "regix-medical[features]"
pip install "regix-medical[totalsegmentator]"
pip install "anatomix @ git+https://github.com/neel-dey/anatomix.git"  # optional model
regix doctor
```

`report` adds QC figures, `api` the HTTP service, `features` the torch/Monai tooling used
by Anatomix and feature deformation, and `totalsegmentator` automatic organ masks.
Anatomix remains an explicit upstream install so its code, weights and terms can be
reviewed. Core registration, mutual information and MIND-SSC need no GPU or model
weights. Contributors instead use `pip install -e ".[dev,report]"` from a checkout.

### Command line

```bash
# Inspect the candidate DICOM series and geometry first.
regix inspect /data/patient001/

# Register an abdominal MR onto a CT with independent validation inputs.
regix register /data/ct/ /data/mr/ -o out/ \
    --preset ct_mr_abdomen --organ liver \
    --fixed-mask ct_structures.nii.gz --moving-mask mr_structures.nii.gz \
    --landmarks-fixed landmarks_ct.txt --landmarks-moving landmarks_mr.txt
```

Configuration remains inspectable and overridable without editing YAML:

```bash
regix register fixed.nii.gz moving.nii.gz --dry-run
regix register fixed.nii.gz moving.nii.gz -o out/ --preset base \
    --set preprocess.working_spacing_mm=1.5
```

Use `regix --help` for batch registration, transform application, segmentation and QC
re-evaluation. Deployment and HTTP examples live in the
[operations guide](https://github.com/Thibescobar/regix/blob/main/docs/OPERATIONS.md).

### Python: standalone or embedded

```python
from regix import RegistrationPipeline, load_preset

config = load_preset("ct_mr_abdomen").with_overrides(
    organs={"targets": ["liver"], "roi_crop": True},
    qc={"gates": {"max_tre_mm": 5.0}},
)
pipeline = RegistrationPipeline(config)

# Standalone: persist the image, transforms, report and manifest.
delivered = pipeline.run("ct/", "mr/", "out/")

# Embedded: return the same algorithmic result without persistent Regix artifacts.
in_memory = pipeline.compute("ct/", "mr/")

# Lowest-cost integration when only the transform is needed.
transform_only = pipeline.compute(
    "ct/", "mr/", registered_image=False, qc=False
)
transform = transform_only.applied_transform.as_sitk_transform()
```

`compute()` uses and removes a private temporary elastix workspace, then returns an
owned transform with no stale file dependency. It creates no persistent Regix report,
manifest or replay bundle; the parent application must provide those when required.
Use `run()` when the result itself is the standalone deliverable.

### Choose an execution mode

| Need | Entry point | Persistence and limits |
|---|---|---|
| Standalone CLI deliverable | `regix register ... -o out/` | Persistent image, transforms, manifest and configured QC/report |
| Python deliverable | `RegistrationPipeline.run()` | Returns `RegistrationResult` and writes the same review/replay bundle |
| Embedded Python | `RegistrationPipeline.compute()` | Returns the result without persistent Regix artifacts; its automatically deleted temporary directory is required by elastix |
| Transform only | `compute(..., registered_image=False, qc=False)` | No registered image, image-dependent QC, report, manifest or replay bundle |
| HTTP service | `uvicorn regix.api:app` | Path-based, single-process/in-memory reference service; production needs external identity, queueing and persistence |

## Standalone result

A successful `run()` returns the result in Python and writes a self-contained review
bundle:

| Product | Purpose |
|---|---|
| Registered image | Native moving intensities sampled on the original fixed grid |
| Transform chain | Replayable elastix stages and an owned runtime transform |
| Linear exports | ITK `.tfm`, Insight `.itk.txt` and moving-to-fixed 4x4 matrix |
| Conditional DICOM export | Spatial Registration Object for two DICOM inputs and a linear result |
| `report.html` | Overlays, checkerboard, metrics, gates and environment |
| `run_manifest.json` | Effective configuration, versions, timings, warnings and artifact inventory |

Nonlinear chains retain their elastix files and displacement representation; Regix does
not mislabel them as a linear DICOM registration. See
[output lifecycle and replay](https://github.com/Thibescobar/regix/blob/main/docs/OPERATIONS.md#output-lifecycle-and-replay).

## Presets

| Preset | Pair | Stages | What it encodes |
|---|---|---|---|
| `base` | any | rigid + affine | CPU default, automatic body mask |
| `ct_mr_abdomen` | MR to CT | rigid + affine + B-spline 20 mm | Features, multi-start; N4 intentionally disabled |
| `ct_ct_liver_followup` | CT to CT | rigid + affine + B-spline 20 mm | NCC, liver ROI, Dice >= 0.92 required |
| `mr_ct_prostate` | MR to CT | rigid + affine + B-spline 12 mm | Pelvic mask, TRE <= 3 mm |
| `ct_ct_lung_4d` | CT to CT | rigid + B-spline 15 mm | Lung mask, low bending penalty |
| `ct_cbct_igrt` | CBCT to CT | **rigid only** | Bone window, 40 mm tolerance |
| `pet_ct_wholebody` | PT to CT | **rigid only** | Identity-first initialization, no deformation |
| `mr_mr_brain` | MR to MR | **rigid only** | Scale locked to 2% |

`regix presets NAME` prints the effective YAML with its clinical rationale in comments.
Presets are starting points; deployment acceptance gates must come from the actual use
case and local validation.

### Custom Elastix parameter files

An Elastix parameter file is optional. To use a file from the
[Elastix Model Zoo](https://elastix.lumc.nl/modelzoo/) or a site-validated file, point
the relevant stage at the user-owned file in a YAML configuration:

```yaml
extends: base
stages:
  - type: rigid
  - type: affine
    parameter_file: /absolute/path/to/Parameters.Par0008.affine.txt
```

Then pass the configuration to the normal command:

```bash
regix register fixed.nii.gz moving.nii.gz -o out --config my-zoo-config.yaml
```

Regix does not bundle or download the Zoo. The supplied file remains authoritative for
its optimizer, sampler, pyramids, schedules, metric and internal pixel types; `extra`
can override individual values. Regix re-imposes `UseDirectionCosines=true`,
`HowToCombineTransforms=Compose`, `AutomaticTransformInitialization=false` and
`WriteResultImage=false` because those values are pipeline safety contracts. The
effective file used for every stage is saved under `out/elastix/` for review and replay.
The declared stage `type` must agree with the file's `Transform`.

## How it works

1. Load volumes and validate their physical geometry and field-of-view relationship.
2. Obtain masks, prepare working grids and choose an explicit initialization.
3. Select NCC for monomodal pairs, mutual information for multimodal intensity stages,
   or modality-invariant feature channels when configured and available.
4. Run the elastix stages and restore the original moving intensities on the fixed grid.
5. Measure independent landmarks or contours when supplied, evaluate gates and build
   the optional review bundle.

| Layer | Implementation |
|---|---|
| Registration | [`itk-elastix`](https://github.com/InsightSoftwareConsortium/ITKElastix) |
| Multimodal descriptors | [Anatomix](https://github.com/neel-dey/anatomix) or analytical MIND-SSC |
| Medical-image I/O and transforms | SimpleITK and pydicom |
| Optional organ segmentation | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or supplied masks |

| Comparison | Intended use and operational property |
|---|---|
| Intensities / NCC | Monomodal pairs |
| Intensities / mutual information | Multimodal without descriptors |
| Anatomix | Learned descriptor; GPU recommended; external weights and installation |
| MIND-SSC | Analytical, CPU and deterministic; less discriminative |
| `provider=auto` | Anatomix if executable, then MIND-SSC, then intensities with recorded warnings |
| `provider=anatomix` / `provider=mind` | Strict, reproducible provider selection; failure instead of switching provider |

`features.enabled=false` disables descriptors, `auto` requests them for multimodal or
feature-dependent stages, and `true` requests them explicitly. `features.allow_cpu`
only permits Anatomix on CPU; MIND is always CPU-based. Choose a strict provider from
the CLI when reproducibility matters:

```bash
regix register fixed.nii.gz moving.nii.gz -o out \
  --features --feature-provider mind
regix register fixed.nii.gz moving.nii.gz -o out \
  --features --feature-provider anatomix
```

Anatomix's exact weights, licence, GPU environment and suitability for the intended
imaging domain must be reviewed separately. Regix applies one shared PCA basis to the
fixed and moving descriptor channels; the detailed Anatomix/MIND contracts remain in
[Architecture](https://github.com/Thibescobar/regix/blob/main/docs/ARCHITECTURE.md).

## Validation snapshot

Synthetic phantoms provide known transformations and deliberately failing pairs:

| Scenario | Repository expectation |
|---|---|
| Rigid + affine CT/CT | transform error below 1.5 mm |
| Organ-centroid initialization | liver Dice above 0.90 |
| Deformable phantom | improved similarity and folded fraction below `1e-3` |
| CT vs inverted-contrast pseudo-MR | MIND fallback error below 3 mm |
| Unrelated volumes | explicit `FAIL` |
| Restitution | native air intensity remains below -500 HU |

The complete collection has 225 tests and an enforced 81% coverage floor; the measured
coverage for this archive is 82%. Exact commands, results, conditional skips and
unverified hardware/clinical boundaries are recorded in
[Verification](https://github.com/Thibescobar/regix/blob/main/VERIFICATION.md).
Synthetic success is not clinical validation: patient-data accuracy requires independent
landmarks or contours and qualified review.

### Embedded execution benchmark

Median lifecycle overhead on the bundled synthetic rigid case in the audited CPU
environment:

| Mode | Time | Peak workspace I/O | Persistent I/O |
|---|---:|---:|---:|
| `run()` | 7.34 s | 391 kB / 17 files | 391 kB / 17 files |
| `compute()` | 7.15 s | 28 kB / 5 files | 0 |
| Transform only | 7.09 s | 28 kB / 5 files | 0 |

Peak resident memory remained 1.18–1.19 GB because ITK and elastix dominated it. These
figures measure lifecycle overhead, not clinical accuracy or every acquisition profile.
Reproduce them with `python benchmarks/benchmark_execution.py --repeats 3` on target
hardware.

## Documentation

| Need | Document |
|---|---|
| Architecture, invariants and extension points | [docs/ARCHITECTURE.md](https://github.com/Thibescobar/regix/blob/main/docs/ARCHITECTURE.md) |
| Deployment, API, DICOM, replay and capacity | [docs/OPERATIONS.md](https://github.com/Thibescobar/regix/blob/main/docs/OPERATIONS.md) |
| Exact verification evidence and boundaries | [VERIFICATION.md](https://github.com/Thibescobar/regix/blob/main/VERIFICATION.md) |
| Development and release workflow | [CONTRIBUTING.md](https://github.com/Thibescobar/regix/blob/main/CONTRIBUTING.md) |
| Behavioural history | [CHANGELOG.md](https://github.com/Thibescobar/regix/blob/main/CHANGELOG.md) |
| Audit finding closure | [AUDIT_CLOSURE.md](https://github.com/Thibescobar/regix/blob/main/AUDIT_CLOSURE.md) |

## Limitations

- No 2D/3D, multi-frame or groupwise registration.
- DICOM registration export is linear only; no Deformable Spatial Registration Object.
- Dense inverse transforms are returned only when they can be represented safely.
- Anatomix, TotalSegmentator, GPU execution and site DICOM interoperability require
  validation in the target environment.
- The built-in HTTP executor is intentionally single-process and in-memory; production
  deployment requires external identity, TLS, queueing, persistence and audit controls.
- A Regix `PASS` means the configured software gates passed. It is not regulatory or
  clinical clearance.

## License

Apache-2.0; see [LICENSE](https://github.com/Thibescobar/regix/blob/main/LICENSE). Regix redistributes no model weights. Optional
components retain their own licenses and terms, which must be reviewed before commercial
or clinical use.

## Author

Thibault Escobar — [github.com/Thibescobar](https://github.com/Thibescobar)
