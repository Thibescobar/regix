"""Minimal HTTP service (FastAPI) to plug Regix into an existing workflow.

Designed for the reality of an imaging department: volumes do not travel as HTTP
attachments, they sit on a network share or on locally mounted object storage. The
API therefore takes **paths** and returns **paths**, not bytes. That also keeps
requests well under gateway size limits and avoids duplicating hundreds of
megabytes per call.

Work is asynchronous: a registration takes anywhere from seconds to minutes, well
beyond a reasonable request timeout. We therefore return a job identifier that can
be polled.

    uvicorn regix.api:app --host 127.0.0.1 --port 8000

Assumed limitations: in-memory execution in a single process, no job persistence,
no authentication. For a real deployment, put this service behind a queue
(Celery/RQ) and an authenticated reverse proxy -- and never expose the API
directly on a clinical network.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from regix.logging_utils import get_logger

log = get_logger("api")

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("the API requires fastapi and uvicorn: pip install 'regix[api]'") from exc


# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    fixed: str = Field(description="Path to the fixed volume (file or DICOM directory).")
    moving: str = Field(description="Path to the moving volume.")
    output_dir: str = Field(description="Output directory (must be writable).")
    preset: str = Field(default="base", description="Bundled preset name or path to a YAML file.")
    organs: list[str] = Field(default_factory=list)
    fixed_labelmap: str | None = None
    moving_labelmap: str | None = None
    label_names: dict[int, str] | None = None
    landmarks_fixed: str | None = None
    landmarks_moving: str | None = None
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description='Nested configuration overrides, e.g. {"preprocess": {"working_spacing_mm": 1.5}}',
    )


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "done", "error"]
    submitted_at: float
    finished_at: float | None = None
    seconds: float | None = None
    qc_status: str | None = None
    metrics: dict[str, Any] | None = None
    outputs: dict[str, str] | None = None
    warnings: list[str] | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
_JOBS: dict[str, JobStatus] = {}
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=1)  # elastix is already multi-threaded internally

from regix import DISCLAIMER, __version__  # noqa: E402

app = FastAPI(
    title="Regix",
    description=f"Multimodal / multi-organ registration.\n\n**{DISCLAIMER}**",
    version=__version__,
)


def _update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        current = _JOBS[job_id]
        _JOBS[job_id] = current.model_copy(update=fields)


def _build_config(request: RegisterRequest):
    from regix.config import load_preset

    cfg = load_preset(request.preset)
    overrides: dict[str, Any] = dict(request.overrides)
    if request.organs:
        overrides.setdefault("organs", {})["targets"] = request.organs
    if request.fixed_labelmap or request.moving_labelmap:
        organs = overrides.setdefault("organs", {})
        organs["backend"] = "external"
        if request.fixed_labelmap:
            organs["fixed_labelmap"] = request.fixed_labelmap
        if request.moving_labelmap:
            organs["moving_labelmap"] = request.moving_labelmap
        if request.label_names:
            organs["label_names"] = request.label_names
    if request.landmarks_fixed and request.landmarks_moving:
        qc = overrides.setdefault("qc", {})
        qc["landmarks_fixed"] = request.landmarks_fixed
        qc["landmarks_moving"] = request.landmarks_moving
    overrides.setdefault("output", {})["overwrite"] = True
    return cfg.with_overrides(**overrides)


def _run_job(job_id: str, request: RegisterRequest) -> None:
    from regix.pipeline import RegistrationPipeline

    _update(job_id, state="running")
    try:
        cfg = _build_config(request)
        result = RegistrationPipeline(cfg).run(request.fixed, request.moving, request.output_dir)
        _update(
            job_id,
            state="done",
            finished_at=time.time(),
            seconds=round(result.seconds, 2),
            qc_status=result.status,
            metrics=result.metrics,
            outputs={k: str(v) for k, v in result.outputs.items()},
            warnings=result.warnings,
        )
    except Exception as exc:  # the error is returned to the client, not swallowed
        log.exception("job %s failed", job_id)
        _update(job_id, state="error", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, Any]:
    """Service state and engine availability: poll this before submitting."""
    from regix.logging_utils import environment_report
    from regix.registration.itk_bridge import engine_available

    ok, detail = engine_available()
    return {
        "status": "ok" if ok else "degraded",
        "engine": detail,
        "environment": environment_report(),
        "jobs": len(_JOBS),
        "disclaimer": DISCLAIMER,
    }


@app.get("/presets")
def presets() -> list[dict[str, Any]]:
    from regix.config import available_presets, load_preset

    out = []
    for name in available_presets():
        cfg = load_preset(name)
        out.append(
            {
                "name": name,
                "fixed_modality": cfg.fixed_modality,
                "moving_modality": cfg.moving_modality,
                "stages": [s.type.value for s in cfg.stages],
                "description": (cfg.description or "").strip(),
            }
        )
    return out


@app.post("/register", response_model=JobStatus, status_code=202)
def register(request: RegisterRequest) -> JobStatus:
    """Submit a registration. Returns a job identifier immediately."""
    for label, path in (("fixed", request.fixed), ("moving", request.moving)):
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"{label} volume not found: {path}")
    try:
        _build_config(request)  # validate the configuration before accepting the job
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid configuration: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    status = JobStatus(job_id=job_id, state="queued", submitted_at=time.time())
    with _LOCK:
        _JOBS[job_id] = status
    _POOL.submit(_run_job, job_id, request)
    log.info("job %s submitted (%s -> %s)", job_id, request.moving, request.fixed)
    return status


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job(job_id: str) -> JobStatus:
    with _LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return _JOBS[job_id]


@app.get("/jobs", response_model=list[JobStatus])
def jobs() -> list[JobStatus]:
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda j: j.submitted_at, reverse=True)
